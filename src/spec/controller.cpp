// src/spec/controller.cpp — infccl Speculative Decode Controller implementation
//
// See controller.hpp for design rationale.

#include "infccl/spec/controller.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>

namespace infccl {
namespace spec {

namespace {

// ── CUDA helpers ─────────────────────────────────────────────────────────────

#define INFCCL_CUDA_CHECK(call)                                          \
    do {                                                                  \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                        \
            std::fprintf(stderr, "infccl CUDA error at %s:%d: %s\n",     \
                         __FILE__, __LINE__, cudaGetErrorString(err));    \
            return -1;                                                   \
        }                                                                \
    } while (0)

// Thread-local RNG for rejection sampling
thread_local std::mt19937 g_rng{std::random_device{}()};
thread_local std::uniform_real_distribution<float> g_uniform(0.0f, 1.0f);

// Softmax over a logit vector, in-place
void softmax_inplace(float* logits, int vocab_size) {
    float max_val = *std::max_element(logits, logits + vocab_size);
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        logits[i] = std::exp(logits[i] - max_val);
        sum += logits[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < vocab_size; ++i)
        logits[i] *= inv_sum;
}

// Sample one token from a probability distribution
int32_t sample_from_probs(const float* probs, int vocab_size) {
    float r = g_uniform(g_rng);
    float cumsum = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        cumsum += probs[i];
        if (cumsum >= r) return i;
    }
    return vocab_size - 1;  // Numerical safety
}

// Microsecond clock
uint64_t now_us() {
    using namespace std::chrono;
    return (uint64_t)duration_cast<microseconds>(
        steady_clock::now().time_since_epoch()).count();
}

}  // namespace

// ─── Constructor / Destructor ────────────────────────────────────────────────

Controller::Controller(const ControllerConfig& config)
    : config_(config) {}

Controller::~Controller() {
    if (state_.load(std::memory_order_acquire) != ControllerState::kUninitialized &&
        state_.load(std::memory_order_acquire) != ControllerState::kShutdown) {
        shutdown();
    }
}

// ─── Initialize ──────────────────────────────────────────────────────────────

int Controller::initialize() {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);

    if (state_.load(std::memory_order_acquire) != ControllerState::kUninitialized) {
        std::fprintf(stderr, "infccl: controller already initialized\n");
        return -1;
    }

    gpu_states_.resize(config_.num_gpus);

    for (int i = 0; i < config_.num_gpus; ++i) {
        INFCCL_CUDA_CHECK(cudaSetDevice(i));

        auto& gs = gpu_states_[i];

        // Create stream
        INFCCL_CUDA_CHECK(cudaStreamCreate((cudaStream_t*)&gs.stream));

        // Allocate doorbell in mapped pinned memory
        // This is the Strata Doorbell pattern: host and device share the same physical memory.
        Doorbell* db = nullptr;
        INFCCL_CUDA_CHECK(cudaHostAlloc(&db, sizeof(Doorbell),
                                        cudaHostAllocMapped | cudaHostAllocPortable));
        std::memset((void*)db, 0, sizeof(Doorbell));

        // Get device pointer to the same memory
        Doorbell* db_dev = nullptr;
        INFCCL_CUDA_CHECK(cudaHostGetDevicePointer(&db_dev, db, 0));

        gs.doorbell = db;
        gs.doorbell_dev = db_dev;

        // Create event for polling flush
        // Strata insight: "NVIDIA driver needs cudaEventQuery to make mapped writes host-visible"
        // On BI-V100: same pattern with __threadfence() + cudaEventQuery
        INFCCL_CUDA_CHECK(cudaEventCreate((cudaEvent_t*)&gs.event));
    }

    state_.store(ControllerState::kIdle, std::memory_order_release);
    return 0;
}

// ─── Shutdown ────────────────────────────────────────────────────────────────

int Controller::shutdown() {
    std::lock_guard<std::mutex> lock(lifecycle_mutex_);

    auto current = state_.load(std::memory_order_acquire);
    if (current == ControllerState::kShutdown || current == ControllerState::kUninitialized)
        return 0;

    state_.store(ControllerState::kShutdown, std::memory_order_release);

    for (int i = 0; i < config_.num_gpus; ++i) {
        cudaSetDevice(i);
        auto& gs = gpu_states_[i];

        if (gs.stream) {
            cudaStreamSynchronize((cudaStream_t)gs.stream);
            cudaStreamDestroy((cudaStream_t)gs.stream);
            gs.stream = nullptr;
        }
        if (gs.event) {
            cudaEventDestroy((cudaEvent_t)gs.event);
            gs.event = nullptr;
        }
        if (gs.doorbell) {
            cudaFreeHost(gs.doorbell);
            gs.doorbell = nullptr;
            gs.doorbell_dev = nullptr;
        }
    }

    gpu_states_.clear();
    return 0;
}

// ─── Speculative Decode Step ─────────────────────────────────────────────────

int Controller::speculative_decode_step(
    DraftForwardFn draft_forward,
    TargetForwardFn target_forward,
    KvRollbackFn kv_rollback,
    VerifyWindow* result_out)
{
    auto expected = ControllerState::kIdle;
    if (!state_.compare_exchange_strong(expected, ControllerState::kDrafting,
                                        std::memory_order_acq_rel)) {
        std::fprintf(stderr, "infccl: controller not in Idle state (current: %d)\n",
                     (int)expected);
        return -1;
    }

    VerifyWindow window;
    window.draft_ids.resize(config_.max_spec_tokens);
    window.draft_length = 0;

    // ── Phase 1: Draft ──
    // Generate up to max_spec_tokens tokens using the draft model.
    // Each step: run draft forward → sample → append to context.
    //
    // On BI-V100 with TP=4: each draft forward involves 48 layers of:
    //   GDN/QSA → MoE (router + experts + combine) → HC read/write
    //   with AllReduce after each TP-sharded operation.

    std::vector<int32_t> context;  // Will be populated by the caller's state
    // TODO: pass initial context from vLLM's sequence state

    for (int step = 0; step < config_.max_spec_tokens; ++step) {
        float* logits = nullptr;   // Allocated by callback or from doorbell
        int32_t token = 0;

        draft_forward(step, context.data(), (int)context.size(), logits, &token);

        window.draft_ids[step] = token;
        context.push_back(token);
        window.draft_length = step + 1;
    }

    total_drafted_.fetch_add(window.draft_length, std::memory_order_relaxed);

    // ── Phase 2: Verify ──
    // Run target model on all draft tokens in ONE forward pass (batch).
    // This is the key efficiency gain: one target forward scores N draft tokens.
    state_.store(ControllerState::kVerifying, std::memory_order_release);

    // target_logits has (num_draft + 1) rows: one per draft position + one bonus
    int vocab_size = 0;  // TODO: get from model config
    // For now, the target_forward callback handles allocation
    float* target_logits_buf = nullptr;

    target_forward(window.draft_ids.data(), window.draft_length, target_logits_buf);

    // ── Phase 3: Accept / Reject ──
    state_.store(ControllerState::kAccepting, std::memory_order_release);

    // Rejection sampling: for each draft token i,
    //   accept with probability min(1, p_target(x_i) / p_draft(x_i))
    //   if rejected, sample from max(0, p_target - p_draft) and stop
    //
    // This is Strata's verify.cpp core loop, simplified for our use case.

    // TODO: wire up actual logits from draft_forward and target_forward
    // For now, use the verify_and_accept helper
    // int accepted = verify_and_accept(draft_logits, window.draft_ids.data(),
    //                                  target_logits_buf, window.draft_length,
    //                                  vocab_size, &window);

    // Placeholder: accept all (optimistic)
    window.accept_length = window.draft_length + 1;  // +1 for bonus token
    window.kv_rollback_to = -1;  // No rollback needed

    total_accepted_.fetch_add(window.accept_length, std::memory_order_relaxed);
    total_steps_.fetch_add(1, std::memory_order_relaxed);

    // ── Phase 4: Rollback (if needed) ──
    if (window.accept_length < window.draft_length + 1) {
        state_.store(ControllerState::kRollingBack, std::memory_order_release);

        // Truncate KV cache: remove entries for rejected tokens
        // Position = original_pos + accept_length
        kv_rollback(window.kv_rollback_to);
    }

    // ── Done ──
    if (result_out) *result_out = window;
    state_.store(ControllerState::kIdle, std::memory_order_release);
    return (int)window.accept_length;
}

// ─── Rejection Sampling (Verification) ───────────────────────────────────────

int Controller::verify_and_accept(
    const float* draft_logits,
    const int32_t* draft_tokens,
    const float* target_logits,
    int num_draft,
    int vocab_size,
    VerifyWindow* window)
{
    // Allocate working buffers for softmax
    std::vector<float> draft_probs(vocab_size);
    std::vector<float> target_probs(vocab_size);
    std::vector<float> residual(vocab_size);

    window->target_ids.resize(num_draft + 1);
    window->final_ids.clear();
    window->draft_probs.resize(num_draft);
    window->target_probs.resize(num_draft);

    for (int i = 0; i < num_draft; ++i) {
        int32_t token = draft_tokens[i];

        // Softmax over draft logits at position i
        std::memcpy(draft_probs.data(), draft_logits + (size_t)i * vocab_size,
                    vocab_size * sizeof(float));
        softmax_inplace(draft_probs.data(), vocab_size);

        // Softmax over target logits at position i
        std::memcpy(target_probs.data(), target_logits + (size_t)i * vocab_size,
                    vocab_size * sizeof(float));
        softmax_inplace(target_probs.data(), vocab_size);

        float p_draft = draft_probs[token];
        float p_target = target_probs[token];

        window->draft_probs[i] = p_draft;
        window->target_probs[i] = p_target;

        // Accept with probability min(1, p_target / p_draft)
        float accept_prob = (p_draft > 0.0f) ? std::min(1.0f, p_target / p_draft) : 0.0f;
        float u = g_uniform(g_rng);

        if (u < accept_prob) {
            // Accepted
            window->final_ids.push_back(token);
        } else {
            // Rejected: sample from max(0, p_target - p_draft) normalized
            float residual_sum = 0.0f;
            for (int j = 0; j < vocab_size; ++j) {
                residual[j] = std::max(0.0f, target_probs[j] - draft_probs[j]);
                residual_sum += residual[j];
            }
            if (residual_sum > 0.0f) {
                float inv = 1.0f / residual_sum;
                for (int j = 0; j < vocab_size; ++j) residual[j] *= inv;
                int32_t corrected = sample_from_probs(residual.data(), vocab_size);
                window->final_ids.push_back(corrected);
            } else {
                // Fallback: sample from target
                int32_t fallback = sample_from_probs(target_probs.data(), vocab_size);
                window->final_ids.push_back(fallback);
            }

            // Stop: remaining draft tokens are rejected
            window->accept_length = i + 1;  // Accepted prefix + corrected token
            window->kv_rollback_to = i + 1;  // Roll back KV to this position
            return (int)window->accept_length;
        }
    }

    // All draft tokens accepted → sample bonus token from target at position num_draft
    std::memcpy(target_probs.data(), target_logits + (size_t)num_draft * vocab_size,
                vocab_size * sizeof(float));
    softmax_inplace(target_probs.data(), vocab_size);
    int32_t bonus = sample_from_probs(target_probs.data(), vocab_size);
    window->final_ids.push_back(bonus);
    window->accept_length = num_draft + 1;
    window->kv_rollback_to = -1;  // No rollback needed
    return (int)window->accept_length;
}

// ─── Doorbell Polling ────────────────────────────────────────────────────────

bool Controller::poll_doorbell(int gpu_id, uint64_t timeout_us) {
    auto& gs = gpu_states_[gpu_id];
    uint32_t want = gs.doorbell->sequence + 1;
    uint64_t start = now_us();

    // Strata pattern: spin-poll with periodic cudaEventQuery to flush driver
    // "3 experiments confirmed: driver call necessary for mapped write visibility"
    while (gs.doorbell->sequence < want) {
        if (now_us() - start > timeout_us) return false;

        // Flush driver: cudaEventQuery on a dummy event forces the driver to
        // process pending mapped memory writes
        cudaSetDevice(gpu_id);
        cudaEventQuery((cudaEvent_t)gs.event);

        // Brief pause to avoid saturating PCIe with polling traffic
        // On BI-V100 PCIe 3.0: each cudaEventQuery is a PCIe round-trip (~1-2 us)
        std::this_thread::yield();
    }
    return true;
}

void Controller::ring_doorbell(int gpu_id) {
    auto& gs = gpu_states_[gpu_id];
    gs.doorbell->sequence += 1;
    // The GPU-side ring is a single-thread kernel that writes *seq = *seq + 1
    // and is launched as the LAST operation in the stream (after all data writes).
    // This function is the host-side equivalent for CPU-initiated notifications.
}

// ─── AllReduce ───────────────────────────────────────────────────────────────

int Controller::allreduce(const AllReduceDesc& desc) {
    if (state_.load(std::memory_order_acquire) < ControllerState::kIdle) return -1;

    size_t nbytes = desc.count * desc.dtype_size;

    if (config_.use_infccl_allreduce) {
        // Use infccl's custom AllReduce:
        //   - Small tensors (< threshold): P2P ring AllReduce (infccl_allreduce.so)
        //   - Large tensors (>= threshold): recursive halving / tree (TODO)
        //
        // The P2P ring was already compiled and tested:
        //   infccl_allreduce.so: replaces __threadfence_system → __threadfence (12 occurrences)
        //   Performance: 46.5ms dispatch/AllReduce overhead → target ~20ms with P2P

        if (nbytes < config_.allreduce_threshold) {
            // Small tensor: P2P ring
            // extern int infccl_allreduce_p2p(void* sendbuf, void* recvbuf, size_t count,
            //                                  int dtype_size, void* stream);
            // return infccl_allreduce_p2p(desc.sendbuf, desc.recvbuf, desc.count,
            //                              desc.dtype_size, desc.stream);
        } else {
            // Large tensor: tree AllReduce (TODO: implement)
        }
    }

    // Fallback: torch.distributed (via Python callback)
    return 0;
}

}  // namespace spec
}  // namespace infccl
