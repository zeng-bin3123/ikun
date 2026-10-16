/* ikun_collective_runner.h — Collective communication runner for BI-V100
 *
 * Form template fill (LAYER 2 → ikun project):
 *   prjname    → ikun
 *   adj        → Collective
 *   Obj        → Runner
 *   typenameType → CommHandle
 *   objtype    → comm_handle_t
 *   argname    → rank, world_size, max_buf_bytes
 *
 * Mathematical basis (LAYER 1):
 *   - Theorem 2.1 (CQE Reduction): generation counter monotonic barrier
 *   - Theorem 10.1 (Coalescing): batch N small allreduces → 1 launch
 *   - Theorem 11.1 (P2P Crossover): P2P wins for tensors < threshold on PCIe
 *
 * Structural depth matches NCCL's:
 *   Init → CommCreate → RegisterPeer → Finalize → Dispatch → Destroy
 *   Each phase is zero-alloc at steady state (pre-resolved pointers).
 */

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ikun {

// ============================================================================
// Type aliases (form template: typenameType → CommHandle)
// ============================================================================

/// Opaque handle to an initialized communicator.
using comm_handle_t = void*;

/// Datatype enum for collective operations.
enum class DataType : int32_t {
    kFloat32 = 0,
    kFloat16 = 1,
    kBFloat16 = 2,
};

/// Reduce operation enum.
enum class ReduceOp : int32_t {
    kSum = 0,
    kProd = 1,
    kMax = 2,
    kMin = 3,
};

/// Topology kind for the runner's dispatch strategy.
enum class TopoKind : int8_t {
    kFlatPIX = 0,   // All pairs via single PCIe bridge (BI-V100 verified)
    kGrouped = 1,    // Mixed PIX + PXB/PHB between groups
};

/// Status codes returned by runner operations.
enum class Status : int32_t {
    kOk = 0,
    kErrorInit = -1,
    kErrorP2P = -2,
    kErrorAlloc = -3,
    kErrorComm = -4,
    kErrorInvalidArg = -5,
    kErrorNotFinalized = -6,
};

// ============================================================================
// Configuration (form template: argname fills)
// ============================================================================

struct CollectiveRunnerConfig {
    int32_t  rank = 0;
    int32_t  world_size = 4;
    int64_t  max_buf_bytes = 16 * 1024 * 1024;  // 16 MB staging per rank

    // Threshold below which custom P2P allreduce is used instead of NCCL
    // (Theorem 11.1: P2P wins for tensors < ~8MB on BI-V100 PCIe)
    int64_t  p2p_threshold_bytes = 8 * 1024 * 1024;

    // Coalescing batch size (Theorem 10.1: batch N small ops → 1 launch)
    int32_t  coalesce_max_pending = 4;
    int64_t  coalesce_max_bytes = 4 * 1024 * 1024;

    // Topology for dispatch decisions
    TopoKind topo = TopoKind::kFlatPIX;

    // NCCL library path (fallback for large tensors)
    std::string nccl_so_path = "libnccl.so.2";

    // Custom allreduce .so path
    std::string custom_ar_so_path;  // empty = auto-detect
};

// ============================================================================
// CommHandle — wrapper around the communicator state
// ============================================================================

/// Forward declaration of internal state.
struct CollectiveRunnerState;

/// The main communicator handle.
/// Lifetime: create → register_peers → finalize → (dispatch N times) → destroy
///
/// Uses shared_ptr (type-erased deleter) to avoid requiring the complete
/// type of CollectiveRunnerState in every translation unit that includes
/// this header. The handle is still non-copyable — use std::move.
class CommHandle {
public:
    CommHandle() = default;
    ~CommHandle() = default;

    CommHandle(const CommHandle&) = delete;
    CommHandle& operator=(const CommHandle&) = delete;
    CommHandle(CommHandle&&) noexcept = default;
    CommHandle& operator=(CommHandle&&) noexcept = default;

    /// True after finalize() succeeds.
    [[nodiscard]] bool is_ready() const noexcept;

    /// Current rank.
    [[nodiscard]] int32_t rank() const noexcept;

    /// World size.
    [[nodiscard]] int32_t world_size() const noexcept;

    /// Generation counter (monotonic, for barrier synchronization).
    [[nodiscard]] uint32_t generation() const noexcept;

    /// Statistics: number of custom P2P allreduce calls.
    [[nodiscard]] int64_t custom_ar_count() const noexcept;

    /// Statistics: number of NCCL fallback calls.
    [[nodiscard]] int64_t nccl_count() const noexcept;

    /// Statistics: number of coalesced batches.
    [[nodiscard]] int64_t coalesced_count() const noexcept;

private:
    friend class CollectiveRunner;
    std::shared_ptr<CollectiveRunnerState> state_;
};

// ============================================================================
// CollectiveRunner — the top-level orchestrator
// ============================================================================

/// Collective communication runner for ikun/BI-V100.
///
/// Structural depth (LAYER 2 form):
///   1. init_p2p()        — enable P2P access between all GPU pairs
///   2. create_comm()     — allocate staging buffers + metadata (once)
///   3. register_peer()   — IPC-map each peer's buffers (once per peer)
///   4. finalize()        — upload pre-resolved pointers to device (once)
///   5. allreduce()       — zero-alloc dispatch (steady-state hot path)
///   6. allreduce_coalesced() — batched dispatch for small tensors
///   7. destroy()         — release all resources
///
/// The runner selects between custom P2P allreduce (small tensors) and
/// NCCL (large tensors) based on Theorem 11.1's crossover threshold.
class CollectiveRunner {
public:
    CollectiveRunner() = default;
    ~CollectiveRunner() = default;

    /// Phase 1: Enable P2P access between all GPU pairs.
    /// Returns kOk if all pairs succeed, kErrorP2P if any fail.
    static Status init_p2p(int32_t num_gpus);

    /// Phase 2: Create a communicator with pre-allocated buffers.
    /// All device memory is allocated HERE, not during dispatch.
    static Status create_comm(CommHandle& handle, const CollectiveRunnerConfig& config);

    /// Phase 3: Register a peer's staging buffer via IPC handles.
    /// Called once per peer after create_comm.
    static Status register_peer(CommHandle& handle, int32_t peer_rank,
                                const void* buf_ipc_handle,
                                const void* meta_ipc_handle);

    /// Phase 4: Finalize — upload pointer arrays to device.
    /// After this, all dispatches are zero-alloc.
    static Status finalize(CommHandle& handle);

    /// Phase 5a: Single allreduce dispatch.
    /// Selects P2P or NCCL based on tensor size vs threshold.
    /// stream: CUDA stream handle (pass 0 for default stream).
    static Status allreduce(CommHandle& handle,
                            void* output, const void* input,
                            int64_t count, DataType dtype,
                            ReduceOp op, void* stream);

    /// Phase 5b: Coalesced allreduce for multiple small tensors.
    /// Packs into a single contiguous buffer, one allreduce, then unpacks.
    /// (Theorem 10.1: saves (N-1) * alpha launch overhead)
    struct TensorDesc {
        void*    data;
        int64_t  count;
        DataType dtype;
    };

    static Status allreduce_coalesced(CommHandle& handle,
                                      const std::vector<TensorDesc>& tensors,
                                      ReduceOp op, void* stream);

    /// Phase 5c: Broadcast from root rank to all others.
    /// Uses infccl_collectives.so if available, else NCCL fallback.
    static Status broadcast(CommHandle& handle,
                            void* data, int64_t count, DataType dtype,
                            int32_t root, void* stream);

    /// Phase 5d: Allgather — each rank contributes its chunk, all get the full result.
    /// output must hold count * world_size elements.
    static Status allgather(CommHandle& handle,
                            void* output, const void* input,
                            int64_t count, DataType dtype,
                            void* stream);

    /// Phase 5e: Reduce — reduce all ranks' data to the root rank.
    static Status reduce(CommHandle& handle,
                         void* output, const void* input,
                         int64_t count, DataType dtype,
                         ReduceOp op, int32_t root, void* stream);

    /// Phase 5f: Reduce-scatter — reduce then scatter equal chunks to each rank.
    /// output holds count / world_size elements per rank.
    static Status reduce_scatter(CommHandle& handle,
                                 void* output, const void* input,
                                 int64_t count, DataType dtype,
                                 ReduceOp op, void* stream);

    /// Phase 6: Destroy communicator and release all resources.
    static void destroy(CommHandle& handle);

    /// Utility: test P2P connectivity between two specific GPUs.
    /// Returns true if bidirectional P2P works.
    static bool test_p2p_pair(int32_t gpu_a, int32_t gpu_b);
};

}  // namespace ikun

