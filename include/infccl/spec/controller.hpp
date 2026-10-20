// include/infccl/spec/controller.hpp — infccl Speculative Decode Controller
//
// Coordinates multi-GPU speculative decode across 4× BI-V100 (TP=4, PCIe).
// Design informed by:
//   - Strata's Doorbell protocol (zero-copy GPU→CPU notification via mapped pinned memory)
//   - Mooncake's UB Context lifecycle (state machine: Uninitialized → Active → Draining → Closed)
//   - Strata's verify.cpp (4254 lines): speculative verification window management
//
// The controller owns the decode loop: it drives N speculative tokens through the model,
// then verifies them against the target distribution, accepts a prefix, and rolls back
// the KV cache for rejected tokens. This is the host-side orchestrator; the actual
// kernels (GEMV, attention, MoE, AllReduce) are dispatched through vLLM's model runner.
//
// BI-V100 constraints that shape this design:
//   - No NVLink: all inter-GPU communication is PCIe 3.0 × 16 (~12.8 GB/s per direction)
//   - No __threadfence_system: mapped pinned memory visibility requires cudaEventQuery flush
//   - CUDA 10.2 compat: CUDA Graph capture may be limited
//   - 4 GPUs, 32 GB HBM2 each = 128 GB total (Qwen3.6-35B-A3B fits in FP16 with KV cache)

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

namespace infccl {
namespace spec {

// ─── Configuration ───────────────────────────────────────────────────────────

struct ControllerConfig {
    int num_gpus = 4;                    // TP world size
    int max_spec_tokens = 5;             // Maximum speculative draft tokens per step
    int kv_block_size = 16;              // KV cache block size (tokens per block)
    size_t allreduce_threshold = 8192;   // Bytes below which to use P2P ring instead of tree
    bool use_cuda_graph = false;         // Enable CUDA Graph capture (experimental on BI-V100)
    bool use_infccl_allreduce = true;    // Use custom infccl AllReduce vs torch.distributed
    uint64_t doorbell_timeout_us = 5000; // Timeout for doorbell notification (microseconds)
};

// ─── State Machine ───────────────────────────────────────────────────────────
// Mirrors Mooncake's UbContext::State with spec-decode specific states

enum class ControllerState : uint8_t {
    kUninitialized = 0,  // Created but not yet initialized
    kIdle,               // Waiting for a new decode step
    kDrafting,           // Draft model is generating speculative tokens
    kVerifying,          // Target model is scoring speculative tokens
    kAccepting,          // Prefix accepted, KV cache committed
    kRollingBack,        // Rejected tokens, KV cache being rolled back
    kFailed,             // Unrecoverable error
    kShutdown,           // Orderly shutdown
};

// ─── Doorbell ────────────────────────────────────────────────────────────────
// Adapted from Strata's doorbell protocol:
//   GPU writes to mapped pinned memory, CPU spin-polls with cudaEventQuery flush.
//   On BI-V100: __threadfence() suffices (no __threadfence_system available).

struct Doorbell {
    // All fields are in mapped pinned memory (cudaHostAlloc + cudaHostGetDevicePointer)
    volatile uint32_t sequence;         // Incremented by GPU kernel after writing payload
    volatile uint32_t ack;              // Written by CPU after reading payload
    float* draft_logits;                // [max_spec_tokens × vocab_size] — draft model output
    int32_t* draft_tokens;              // [max_spec_tokens] — sampled draft tokens
    float* target_logits;               // [max_spec_tokens+1 × vocab_size] — target model output
    int32_t* accepted_tokens;           // [max_spec_tokens+1] — final accepted tokens
    int32_t num_drafted;                // How many draft tokens were generated
    int32_t num_accepted;               // How many tokens the verifier accepted

    // Strata insight: "ring in stream is LAST operation (after publish),
    // so host sees ring means data is already there"
    // On BI-V100: need cudaEventQuery calls to flush mapped writes
};

// ─── Verification Window ─────────────────────────────────────────────────────
// From Strata's verify.cpp: manages the speculative token window

struct VerifyWindow {
    int draft_length;                    // Number of draft tokens in this window
    int accept_length;                   // Number accepted after verification (0..draft_length+1)
    int kv_rollback_to;                  // Position to truncate KV cache (on partial accept)

    // Token IDs
    std::vector<int32_t> draft_ids;      // Draft token IDs [draft_length]
    std::vector<int32_t> target_ids;     // Target-sampled token IDs [draft_length + 1]
    std::vector<int32_t> final_ids;      // Accepted prefix + bonus token

    // Probabilities (for rejection sampling)
    std::vector<float> draft_probs;      // p(draft_id[i] | context + draft[:i])
    std::vector<float> target_probs;     // p(draft_id[i] | context + draft[:i]) under target
};

// ─── AllReduce Descriptor ────────────────────────────────────────────────────
// What infccl needs to know for each AllReduce call in the decode step

struct AllReduceDesc {
    void* sendbuf;                       // Device pointer (per-GPU)
    void* recvbuf;                       // Device pointer (per-GPU), may alias sendbuf
    size_t count;                        // Number of elements
    int dtype_size;                      // Bytes per element (2 for FP16, 4 for FP32)
    void* stream;                        // CUDA stream
};

// ─── Controller ──────────────────────────────────────────────────────────────

class Controller {
public:
    explicit Controller(const ControllerConfig& config);
    ~Controller();

    Controller(const Controller&) = delete;
    Controller& operator=(const Controller&) = delete;

    // ── Lifecycle ──

    // Initialize: allocate doorbells, set up per-GPU state, verify TP connectivity.
    // Follows Mooncake's UbContext::initialize pattern.
    int initialize();

    // Shutdown: drain in-flight operations, free doorbells, close streams.
    int shutdown();

    // ── Decode Loop ──

    // Run one speculative decode step:
    //   1. Draft model generates up to max_spec_tokens tokens
    //   2. Target model scores all draft tokens in one forward pass
    //   3. Rejection sampling determines accepted prefix
    //   4. KV cache rolled back for rejected tokens
    //   5. Returns the number of accepted tokens (1..max_spec_tokens+1)
    //
    // Callbacks:
    //   draft_forward: run draft model one step, return logits and sampled token
    //   target_forward: run target model on all draft tokens, return logits
    //   kv_rollback: truncate KV cache to a given position
    //
    // This is the main entry point for the vLLM integration.
    using DraftForwardFn = std::function<void(
        int step_idx,           // 0..max_spec_tokens-1
        const int32_t* context, // Token IDs so far
        int context_len,        // Length of context
        float* logits_out,      // [vocab_size] output
        int32_t* token_out      // Sampled token output
    )>;

    using TargetForwardFn = std::function<void(
        const int32_t* draft_tokens, // [num_draft] draft token IDs
        int num_draft,               // Number of draft tokens
        float* logits_out            // [(num_draft+1) × vocab_size] output
    )>;

    using KvRollbackFn = std::function<void(
        int rollback_to  // Position to truncate KV cache to
    )>;

    int speculative_decode_step(
        DraftForwardFn draft_forward,
        TargetForwardFn target_forward,
        KvRollbackFn kv_rollback,
        VerifyWindow* result_out       // Output: verification result
    );

    // ── AllReduce ──

    // Dispatch an AllReduce through infccl (P2P ring for small tensors,
    // tree/recursive halving for large ones).
    // Called by the model runner during forward passes.
    int allreduce(const AllReduceDesc& desc);

    // ── Query ──

    ControllerState state() const noexcept {
        return state_.load(std::memory_order_acquire);
    }

    int num_gpus() const noexcept { return config_.num_gpus; }
    int max_spec_tokens() const noexcept { return config_.max_spec_tokens; }

    // Statistics
    uint64_t total_accepted() const noexcept { return total_accepted_.load(std::memory_order_relaxed); }
    uint64_t total_drafted() const noexcept { return total_drafted_.load(std::memory_order_relaxed); }
    double acceptance_rate() const noexcept {
        uint64_t d = total_drafted_.load(std::memory_order_relaxed);
        return d > 0 ? (double)total_accepted_.load(std::memory_order_relaxed) / d : 0.0;
    }

private:
    // ── Rejection sampling ──
    // From Strata's verify.cpp: compare draft and target probabilities,
    // accept with probability min(1, target_prob / draft_prob)
    int verify_and_accept(
        const float* draft_logits,     // [num_draft × vocab_size]
        const int32_t* draft_tokens,   // [num_draft]
        const float* target_logits,    // [(num_draft+1) × vocab_size]
        int num_draft,
        int vocab_size,
        VerifyWindow* window
    );

    // ── Doorbell polling ──
    // Adapted from Strata session.cpp: spin-poll with cudaEventQuery flush
    bool poll_doorbell(int gpu_id, uint64_t timeout_us);
    void ring_doorbell(int gpu_id);

    ControllerConfig config_;
    std::atomic<ControllerState> state_{ControllerState::kUninitialized};

    // Per-GPU state
    struct GpuState {
        void* stream;                    // CUDA stream
        Doorbell* doorbell;              // Mapped pinned memory
        Doorbell* doorbell_dev;          // Device-side pointer to same memory
        void* event;                     // cudaEvent for polling flush
    };
    std::vector<GpuState> gpu_states_;

    // Statistics
    std::atomic<uint64_t> total_accepted_{0};
    std::atomic<uint64_t> total_drafted_{0};
    std::atomic<uint64_t> total_steps_{0};

    mutable std::mutex lifecycle_mutex_;
};

}  // namespace spec
}  // namespace infccl
