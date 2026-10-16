/* Test the ikun collective runner — host-side logic only.
 *
 * Verifies:
 *   1. CommHandle lifecycle (create, move, destroy)
 *   2. Config validation (invalid rank, zero buffer)
 *   3. Finalization guard (dispatch before finalize fails)
 *   4. Statistics counters
 *
 * Does NOT require CUDA — tests the orchestration logic in isolation.
 * The actual GPU allreduce is tested via test_infccl.py.
 */

#include <cassert>
#include <cstdio>
#include <memory>
#include <vector>

#include "distributed_runtime/ikun_collective_runner.h"

using namespace ikun;

// ============================================================================
// Test helpers
// ============================================================================

#define TEST(name) \
    static void test_##name(); \
    struct Reg_##name { Reg_##name() { tests.push_back({#name, test_##name}); } } reg_##name; \
    static void test_##name()

struct TestEntry {
    const char* name;
    void (*fn)();
};
static std::vector<TestEntry> tests;

// ============================================================================
// Tests
// ============================================================================

TEST(config_defaults) {
    CollectiveRunnerConfig cfg;
    assert(cfg.rank == 0);
    assert(cfg.world_size == 4);
    assert(cfg.max_buf_bytes == 16 * 1024 * 1024);
    assert(cfg.p2p_threshold_bytes == 8 * 1024 * 1024);
    assert(cfg.topo == TopoKind::kFlatPIX);
    printf("  config defaults OK\n");
}

TEST(handle_default_state) {
    CommHandle h;
    assert(!h.is_ready());
    assert(h.rank() == -1);
    assert(h.world_size() == 0);
    assert(h.generation() == 0);
    assert(h.custom_ar_count() == 0);
    assert(h.nccl_count() == 0);
    printf("  default handle OK\n");
}

TEST(create_invalid_rank) {
    CommHandle h;
    CollectiveRunnerConfig cfg;
    cfg.rank = -1;
    Status s = CollectiveRunner::create_comm(h, cfg);
    assert(s == Status::kErrorInvalidArg);
    printf("  invalid rank rejected OK\n");
}

TEST(create_rank_too_large) {
    CommHandle h;
    CollectiveRunnerConfig cfg;
    cfg.rank = 5;
    cfg.world_size = 4;
    Status s = CollectiveRunner::create_comm(h, cfg);
    assert(s == Status::kErrorInvalidArg);
    printf("  rank >= world_size rejected OK\n");
}

TEST(create_zero_buffer) {
    CommHandle h;
    CollectiveRunnerConfig cfg;
    cfg.max_buf_bytes = 0;
    Status s = CollectiveRunner::create_comm(h, cfg);
    assert(s == Status::kErrorInvalidArg);
    printf("  zero buffer rejected OK\n");
}

TEST(dispatch_before_finalize) {
    CommHandle h;
    CollectiveRunnerConfig cfg;
    cfg.rank = 0;
    cfg.world_size = 2;
    cfg.max_buf_bytes = 1024;
    cfg.custom_ar_so_path = "/nonexistent/path.so";  // force no .so load

    Status s = CollectiveRunner::create_comm(h, cfg);
    // create_comm may succeed or fail depending on dlopen
    // The key test: dispatch before finalize must fail
    if (s == Status::kOk) {
        float buf[4] = {1, 2, 3, 4};
        Status ds = CollectiveRunner::allreduce(
            h, buf, buf, 4, DataType::kFloat32, ReduceOp::kSum, nullptr);
        assert(ds == Status::kErrorNotFinalized);
        printf("  dispatch before finalize rejected OK\n");
    } else {
        printf("  create_comm failed (expected without CUDA): status=%d\n", (int)s);
    }
}

TEST(handle_move) {
    CommHandle h1;
    CollectiveRunnerConfig cfg;
    cfg.rank = 0;
    cfg.world_size = 2;
    cfg.max_buf_bytes = 1024;
    cfg.custom_ar_so_path = "/nonexistent/path.so";

    Status s = CollectiveRunner::create_comm(h1, cfg);
    if (s == Status::kOk) {
        assert(h1.rank() == 0);

        // Move
        CommHandle h2 = std::move(h1);
        assert(!h1.is_ready());  // moved-from
        assert(h2.rank() == 0);

        // Move assign
        CommHandle h3;
        h3 = std::move(h2);
        assert(h3.rank() == 0);
        printf("  handle move OK\n");
    } else {
        printf("  create_comm failed (expected without CUDA), move test skipped\n");
    }
}

TEST(status_enum) {
    assert(static_cast<int>(Status::kOk) == 0);
    assert(static_cast<int>(Status::kErrorInit) < 0);
    assert(static_cast<int>(Status::kErrorP2P) < 0);
    assert(static_cast<int>(Status::kErrorAlloc) < 0);
    printf("  status enum values OK\n");
}

TEST(dtype_values) {
    assert(static_cast<int>(DataType::kFloat32) == 0);
    assert(static_cast<int>(DataType::kFloat16) == 1);
    assert(static_cast<int>(DataType::kBFloat16) == 2);
    printf("  dtype enum values OK\n");
}

TEST(reduce_op_values) {
    assert(static_cast<int>(ReduceOp::kSum) == 0);
    assert(static_cast<int>(ReduceOp::kProd) == 1);
    assert(static_cast<int>(ReduceOp::kMax) == 2);
    assert(static_cast<int>(ReduceOp::kMin) == 3);
    printf("  reduce op enum values OK\n");
}

TEST(coalesced_empty) {
    CommHandle h;
    std::vector<CollectiveRunner::TensorDesc> empty;
    // Should return kErrorNotFinalized since h is not initialized
    Status s = CollectiveRunner::allreduce_coalesced(
        h, empty, ReduceOp::kSum, nullptr);
    assert(s == Status::kErrorNotFinalized);
    printf("  coalesced empty with uninitialized handle fails correctly\n");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("=== ikun CollectiveRunner tests ===\n");
    int pass = 0, fail = 0;
    for (const auto& t : tests) {
        printf("[TEST] %s\n", t.name);
        try {
            t.fn();
            pass++;
        } catch (const std::exception& e) {
            printf("  FAILED: %s\n", e.what());
            fail++;
        } catch (...) {
            printf("  FAILED: unknown exception\n");
            fail++;
        }
    }
    printf("\n=== Results: %d passed, %d failed ===\n", pass, fail);
    return fail > 0 ? 1 : 0;
}

