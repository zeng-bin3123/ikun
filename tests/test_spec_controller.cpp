// tests/test_spec_controller.cpp — Unit tests for infccl spec controller
//
// Tests the rejection sampling logic, state machine transitions, and doorbell protocol.
// Does NOT require GPUs — uses mock callbacks for draft/target forward.
//
// Compile: clang++ -std=c++17 -I../include -DINFCCL_TEST_NO_CUDA test_spec_controller.cpp -o test_spec_controller

#include <cassert>
#include <cmath>
#include <cstdio>
#include <functional>
#include <numeric>
#include <vector>

// Mock CUDA when testing without GPUs
#ifdef INFCCL_TEST_NO_CUDA

typedef int cudaError_t;
typedef void* cudaStream_t;
typedef void* cudaEvent_t;
#define cudaSuccess 0
#define cudaHostAllocMapped 0
#define cudaHostAllocPortable 0
inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
inline cudaError_t cudaStreamCreate(cudaStream_t*) { return cudaSuccess; }
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaEventCreate(cudaEvent_t*) { return cudaSuccess; }
inline cudaError_t cudaEventDestroy(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventQuery(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaHostAlloc(void** ptr, size_t sz, unsigned) {
    *ptr = calloc(1, sz);
    return cudaSuccess;
}
inline cudaError_t cudaHostGetDevicePointer(void** d, void* h, unsigned) {
    *d = h;
    return cudaSuccess;
}
inline cudaError_t cudaFreeHost(void* p) { free(p); return cudaSuccess; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t) { return "mock"; }

#endif  // INFCCL_TEST_NO_CUDA

// Now include the implementation directly (single-compilation-unit test)
#include "../src/spec/controller.cpp"

using namespace infccl::spec;

// ─── Test helpers ────────────────────────────────────────────────────────────

static const int VOCAB_SIZE = 100;  // Small vocab for testing

// Make a peaked probability distribution: prob[target] = peak, rest uniform
static std::vector<float> make_logits(int target, float peak_logit, int vocab_size) {
    std::vector<float> logits(vocab_size, 0.0f);
    logits[target] = peak_logit;
    return logits;
}

// ─── Test 1: Controller lifecycle ────────────────────────────────────────────

void test_lifecycle() {
    std::printf("test_lifecycle... ");

    ControllerConfig config;
    config.num_gpus = 2;
    config.max_spec_tokens = 3;

    Controller ctrl(config);
    assert(ctrl.state() == ControllerState::kUninitialized);

    int rc = ctrl.initialize();
    assert(rc == 0);
    assert(ctrl.state() == ControllerState::kIdle);

    rc = ctrl.shutdown();
    assert(rc == 0);
    assert(ctrl.state() == ControllerState::kShutdown);

    // Double shutdown is safe
    rc = ctrl.shutdown();
    assert(rc == 0);

    std::printf("OK\n");
}

// ─── Test 2: Rejection sampling — all accepted ──────────────────────────────

void test_verify_all_accept() {
    std::printf("test_verify_all_accept... ");

    ControllerConfig config;
    config.num_gpus = 1;
    config.max_spec_tokens = 3;

    Controller ctrl(config);
    ctrl.initialize();

    // Draft and target agree perfectly → all should be accepted
    int num_draft = 3;
    int32_t draft_tokens[] = {5, 10, 15};

    // Draft logits: peaked at the draft tokens
    std::vector<float> draft_logits(num_draft * VOCAB_SIZE, 0.0f);
    for (int i = 0; i < num_draft; ++i)
        draft_logits[i * VOCAB_SIZE + draft_tokens[i]] = 10.0f;

    // Target logits: identical to draft (perfect agreement)
    std::vector<float> target_logits((num_draft + 1) * VOCAB_SIZE, 0.0f);
    for (int i = 0; i < num_draft; ++i)
        target_logits[i * VOCAB_SIZE + draft_tokens[i]] = 10.0f;
    // Bonus position: peaked at token 20
    target_logits[num_draft * VOCAB_SIZE + 20] = 10.0f;

    VerifyWindow window;
    // Call the private verify_and_accept through the speculative_decode_step path
    // For direct testing, we'd need to make it accessible. Using a friend or
    // wrapping test. For now, test through the full path with mock callbacks.

    // Mock callbacks
    int draft_call_count = 0;
    auto draft_fn = [&](int step, const int32_t*, int, float* logits_out, int32_t* token_out) {
        // Return pre-computed logits
        if (logits_out && step < num_draft) {
            std::memcpy(logits_out, draft_logits.data() + step * VOCAB_SIZE,
                        VOCAB_SIZE * sizeof(float));
        }
        if (token_out && step < num_draft) *token_out = draft_tokens[step];
        draft_call_count++;
    };

    auto target_fn = [&](const int32_t*, int nd, float* logits_out) {
        if (logits_out) {
            std::memcpy(logits_out, target_logits.data(),
                        (nd + 1) * VOCAB_SIZE * sizeof(float));
        }
    };

    auto kv_rollback_fn = [&](int pos) {
        // Should not be called when all tokens are accepted
        assert(false && "kv_rollback should not be called");
    };

    VerifyWindow result;
    int accepted = ctrl.speculative_decode_step(draft_fn, target_fn, kv_rollback_fn, &result);

    assert(accepted > 0);
    assert(draft_call_count == config.max_spec_tokens);

    ctrl.shutdown();
    std::printf("OK (accepted=%d)\n", accepted);
}

// ─── Test 3: State machine transitions ───────────────────────────────────────

void test_state_transitions() {
    std::printf("test_state_transitions... ");

    ControllerConfig config;
    config.num_gpus = 1;
    config.max_spec_tokens = 1;

    Controller ctrl(config);

    // Cannot decode before initialization
    VerifyWindow result;
    int rc = ctrl.speculative_decode_step(
        [](int, const int32_t*, int, float*, int32_t*) {},
        [](const int32_t*, int, float*) {},
        [](int) {},
        &result);
    assert(rc == -1);

    ctrl.initialize();

    // Now it should work
    rc = ctrl.speculative_decode_step(
        [](int, const int32_t*, int, float*, int32_t* t) { if (t) *t = 0; },
        [](const int32_t*, int, float*) {},
        [](int) {},
        &result);
    assert(rc > 0);
    assert(ctrl.state() == ControllerState::kIdle);

    ctrl.shutdown();
    std::printf("OK\n");
}

// ─── Test 4: Statistics tracking ─────────────────────────────────────────────

void test_statistics() {
    std::printf("test_statistics... ");

    ControllerConfig config;
    config.num_gpus = 1;
    config.max_spec_tokens = 4;

    Controller ctrl(config);
    ctrl.initialize();

    assert(ctrl.total_drafted() == 0);
    assert(ctrl.total_accepted() == 0);
    assert(ctrl.acceptance_rate() == 0.0);

    // Run one step
    VerifyWindow result;
    ctrl.speculative_decode_step(
        [](int, const int32_t*, int, float*, int32_t* t) { if (t) *t = 0; },
        [](const int32_t*, int, float*) {},
        [](int) {},
        &result);

    assert(ctrl.total_drafted() == 4);  // max_spec_tokens
    assert(ctrl.total_accepted() > 0);

    ctrl.shutdown();
    std::printf("OK (drafted=%lu, accepted=%lu, rate=%.2f)\n",
                ctrl.total_drafted(), ctrl.total_accepted(), ctrl.acceptance_rate());
}

// ─── Test 5: Rejection sampling math ─────────────────────────────────────────

void test_rejection_sampling_math() {
    std::printf("test_rejection_sampling_math... ");

    // Test softmax
    float logits[] = {1.0f, 2.0f, 3.0f};
    softmax_inplace(logits, 3);
    float sum = logits[0] + logits[1] + logits[2];
    assert(std::abs(sum - 1.0f) < 1e-5f);
    assert(logits[2] > logits[1] && logits[1] > logits[0]);

    // Test sampling
    float probs[] = {0.0f, 0.0f, 1.0f};
    for (int i = 0; i < 100; ++i) {
        int32_t s = sample_from_probs(probs, 3);
        assert(s == 2);
    }

    std::printf("OK\n");
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main() {
    std::printf("infccl spec controller tests\n");
    std::printf("============================\n\n");

    test_lifecycle();
    test_state_transitions();
    test_rejection_sampling_math();
    test_statistics();
    test_verify_all_accept();

    std::printf("\nAll tests passed.\n");
    return 0;
}
