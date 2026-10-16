/*
 * infccl_allreduce.cu — Ring AllReduce for Iluvatar BI-V100 (4× PCIe)
 *
 * v2: Added ring reduce-scatter + allgather for medium tensors (256KB~8MB).
 *     Keeps the fused one-shot kernel for tiny tensors (≤256KB).
 *
 * Three execution paths:
 *   1. Fused one-shot (≤ 256KB): all-to-all copy → barrier → local reduce
 *      Best for very small tensors where latency α dominates.
 *   2. Ring AllReduce (256KB ~ 8MB): reduce-scatter (P-1 steps) →
 *      allgather (P-1 steps). Each step transfers only size/P bytes.
 *      Bandwidth-optimal: total wire traffic = 2*(P-1)/P * size.
 *   3. NCCL fallback (≥ 8MB or non-SUM ops): system libnccl.so.2.
 *
 * Ring protocol (for P=4 GPUs, rank=r):
 *   Reduce-Scatter: 3 steps, each rank sends shard[(r-step)%P] to successor
 *     Step 0: send own shard, recv from predecessor, reduce into shard
 *     Step 1: send reduced shard, recv next, reduce again
 *     Step 2: final recv completes one fully-reduced shard per rank
 *   AllGather: 3 steps, each rank sends its completed shard around the ring
 *     Step 0: send completed shard to successor
 *     Step 1: forward received shard
 *     Step 2: receive last shard — all ranks now have all data
 *
 * Mooncake reference: mooncake-pg/src/device_comm/device_collective/
 *   algorithms/ring/ring_all_reduce.cu — adapted for CUDA 10.2 + PCIe P2P.
 *
 * CUDA 10.2 constraints:
 *   - No cooperative_groups::grid_group (no cross-block sync in kernel)
 *   - No __nv_bfloat16 intrinsics (use bit-manipulation)
 *   - sm_70 architecture target
 *
 * Compile for BI-V100:
 *   /usr/local/corex/bin/nvcc -arch=sm_70 -O3 -shared -Xcompiler -fPIC \
 *       -o lib/infccl_allreduce.so csrc/infccl_allreduce.cu -lcudart
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define INFCCL_MAX_RANKS 8
#define INFCCL_BLOCK_SIZE 256
#define INFCCL_MAX_GRID 1024
#define INFCCL_ONESHOT_THRESHOLD (256 * 1024)   // ≤256KB: fused one-shot
#define INFCCL_RING_MAX_BYTES   (8 * 1024 * 1024) // ≤8MB: ring AR

// ============================================================================
// Data structures
// ============================================================================

struct InfcclMeta {
    // flags[peer] — written by peer to signal this rank
    volatile uint32_t flags[INFCCL_MAX_RANKS];
    uint32_t _pad0[INFCCL_MAX_RANKS];
    uint32_t rank;
    uint32_t world_size;
    uint64_t max_size;
    char _align[64 - 24];
};

struct InfcclComm {
    void*              staging[INFCCL_MAX_RANKS];
    volatile uint32_t* barriers[INFCCL_MAX_RANKS];
    int                rank;
    int                world_size;
    int64_t            max_size;
    uint32_t           generation;

    void**             d_staging;
    volatile uint32_t** d_barriers;

    // Ring topology: ring_next[r] = successor, ring_prev[r] = predecessor
    // Ordered to avoid non-P2P pairs (e.g., skip GPU 0↔3 direct)
    int                ring_order[INFCCL_MAX_RANKS]; // rank → ring position
    int                ring_rank[INFCCL_MAX_RANKS];  // ring position → rank
    int*               d_ring_order;
    int*               d_ring_rank;
};

// ============================================================================
// BF16 helpers (CUDA 10.2 — no native __nv_bfloat16)
// ============================================================================

__device__ __forceinline__ float bf16_to_float(uint16_t v) {
    uint32_t u32 = ((uint32_t)v) << 16;
    float f;
    memcpy(&f, &u32, 4);
    return f;
}

__device__ __forceinline__ uint16_t float_to_bf16(float f) {
    uint32_t u32;
    memcpy(&u32, &f, 4);
    u32 += 0x7FFF + ((u32 >> 16) & 1); // round to nearest even
    return (uint16_t)(u32 >> 16);
}

// ============================================================================
// Barrier: signal-and-wait using generation counter
//   Only block 0 thread 0 does the signaling/waiting.
//   All threads wait at __syncthreads() after.
// ============================================================================

__device__ __forceinline__ void infccl_barrier_signal_wait(
    volatile uint32_t** peer_barriers,
    int rank, int world_size, uint32_t gen)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int r = 0; r < world_size; r++)
            if (r != rank)
                peer_barriers[r][rank] = gen;
        for (int r = 0; r < world_size; r++)
            if (r != rank)
                while (peer_barriers[rank][r] < gen);
    }
    __syncthreads();
}

// Pairwise signal: signal just one peer, wait for just one peer
__device__ __forceinline__ void infccl_pair_signal_wait(
    volatile uint32_t** peer_barriers,
    int rank, int peer, uint32_t gen)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        peer_barriers[peer][rank] = gen;
        while (peer_barriers[rank][peer] < gen);
    }
    __syncthreads();
}

// ============================================================================
// Kernel 1: Fused one-shot AllReduce (≤ 256KB)
//   Same as v1 — all-to-all copy → barrier → local reduce
// ============================================================================

template<typename T, int WORLD_SIZE>
__global__ void infccl_fused_allreduce(
    T* __restrict__             output,
    const T* __restrict__       input,
    void** __restrict__         peer_staging,
    volatile uint32_t** __restrict__ peer_barriers,
    const int                   rank,
    const int64_t               count,
    const uint32_t              gen)
{
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    T* my_buf = (T*)peer_staging[rank];
    for (int64_t i = tid; i < count; i += stride)
        my_buf[i] = input[i];

    __threadfence_system();
    infccl_barrier_signal_wait(peer_barriers, rank, WORLD_SIZE, gen);

    for (int64_t i = tid; i < count; i += stride) {
        T acc = ((T*)peer_staging[0])[i];
        #pragma unroll
        for (int r = 1; r < WORLD_SIZE; r++)
            acc = acc + ((T*)peer_staging[r])[i];
        output[i] = acc;
    }
}

// FP16 specialization: vectorized copy, FP32 accumulation
template<int WORLD_SIZE>
__global__ void infccl_fused_allreduce_fp16(
    half* __restrict__          output,
    const half* __restrict__    input,
    void** __restrict__         peer_staging,
    volatile uint32_t** __restrict__ peer_barriers,
    const int                   rank,
    const int64_t               count,
    const uint32_t              gen)
{
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    half* my_buf = (half*)peer_staging[rank];
    const int64_t count_v = count / 2;
    for (int64_t i = tid; i < count_v; i += stride)
        ((float*)my_buf)[i] = ((const float*)input)[i];
    if ((count & 1) && tid == 0) my_buf[count - 1] = input[count - 1];

    __threadfence_system();
    infccl_barrier_signal_wait(peer_barriers, rank, WORLD_SIZE, gen);

    for (int64_t i = tid; i < count; i += stride) {
        float acc = __half2float(((half*)peer_staging[0])[i]);
        #pragma unroll
        for (int r = 1; r < WORLD_SIZE; r++)
            acc += __half2float(((half*)peer_staging[r])[i]);
        output[i] = __float2half(acc);
    }
}

// BF16 specialization
template<int WORLD_SIZE>
__global__ void infccl_fused_allreduce_bf16(
    uint16_t* __restrict__      output,
    const uint16_t* __restrict__ input,
    void** __restrict__         peer_staging,
    volatile uint32_t** __restrict__ peer_barriers,
    const int                   rank,
    const int64_t               count,
    const uint32_t              gen)
{
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    uint16_t* my_buf = (uint16_t*)peer_staging[rank];
    const int64_t count_v = count / 2;
    for (int64_t i = tid; i < count_v; i += stride)
        ((uint32_t*)my_buf)[i] = ((const uint32_t*)input)[i];
    if ((count & 1) && tid == 0) my_buf[count - 1] = input[count - 1];

    __threadfence_system();
    infccl_barrier_signal_wait(peer_barriers, rank, WORLD_SIZE, gen);

    for (int64_t i = tid; i < count; i += stride) {
        float acc = bf16_to_float(((uint16_t*)peer_staging[0])[i]);
        #pragma unroll
        for (int r = 1; r < WORLD_SIZE; r++)
            acc += bf16_to_float(((uint16_t*)peer_staging[r])[i]);
        output[i] = float_to_bf16(acc);
    }
}

// ============================================================================
// Kernel 2: Ring AllReduce (256KB ~ 8MB)
//
// Protocol: reduce-scatter (P-1 steps) + allgather (P-1 steps)
//
// Each rank's staging buffer is divided into P shards of size count/P.
// Ring order is topology-aware (avoids non-P2P pairs).
//
// This kernel is launched P-1 times for reduce-scatter + P-1 for allgather.
// We use a multi-phase approach with host-side generation management to avoid
// needing cross-block sync (which CUDA 10.2 can't do on BI-V100).
//
// Actually, to minimize kernel launch overhead (the #1 bottleneck per PRD),
// we implement the full ring in a SINGLE kernel using per-step barriers.
// Each step uses a different generation counter value.
// Only block 0, thread 0 does the barrier; all blocks wait on __syncthreads().
//
// IMPORTANT: This only works with gridDim.x == 1 (single block) to ensure
// __syncthreads() is a full device barrier. For multi-block parallelism,
// we'd need cooperative_groups::grid_group (CUDA 11+) or multi-kernel launch.
// On BI-V100 with PCIe bottleneck, single-block is fine because the
// limiter is PCIe bandwidth, not compute.
// ============================================================================

// FP16 ring allreduce with FP32 accumulation (the hot path for BI-V100)
template<int WORLD_SIZE>
__global__ void infccl_ring_allreduce_fp16(
    half* __restrict__          output,        // in-place: same as input
    const half* __restrict__    input,
    void** __restrict__         peer_staging,
    volatile uint32_t** __restrict__ peer_barriers,
    const int* __restrict__     ring_rank,     // ring_position → GPU rank
    const int                   rank,
    const int                   ring_pos,      // this rank's position in ring
    const int64_t               count,         // total elements
    const uint32_t              gen_base)       // base generation for steps
{
    // Single-block kernel for ring — all threads cooperate per step
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    // Shard layout: P shards, each of shard_count elements
    const int64_t shard_count = (count + WORLD_SIZE - 1) / WORLD_SIZE;

    // Predecessor and successor in ring
    const int pred_pos = (ring_pos + WORLD_SIZE - 1) % WORLD_SIZE;
    const int succ_pos = (ring_pos + 1) % WORLD_SIZE;
    const int pred_rank = ring_rank[pred_pos];
    const int succ_rank = ring_rank[succ_pos];

    // My staging buffer, predecessor's staging buffer (for reading)
    half* my_staging = (half*)peer_staging[rank];
    half* pred_staging = (half*)peer_staging[pred_rank];

    // ── Phase 0: Copy input to staging ──
    for (int64_t i = tid; i < count; i += nthreads)
        my_staging[i] = input[i];
    __threadfence_system();

    // Global barrier: all ranks must finish copying before any reduce starts
    uint32_t gen = gen_base;
    if (threadIdx.x == 0) {
        for (int r = 0; r < WORLD_SIZE; r++)
            if (r != rank) peer_barriers[r][rank] = gen;
        for (int r = 0; r < WORLD_SIZE; r++)
            if (r != rank) while (peer_barriers[rank][r] < gen);
    }
    __syncthreads();

    // ── Phase 1: Reduce-Scatter (P-1 steps) ──
    // After step s, rank at ring_pos has a partial sum of (s+1) contributors
    // for shard index (ring_pos - s) % P.
    for (int step = 0; step < WORLD_SIZE - 1; step++) {
        // Shard to process: the one we just received data for
        int shard_idx = ((ring_pos - step) % WORLD_SIZE + WORLD_SIZE) % WORLD_SIZE;
        int64_t shard_start = (int64_t)shard_idx * shard_count;
        int64_t shard_end = shard_start + shard_count;
        if (shard_end > count) shard_end = count;
        int64_t shard_len = shard_end - shard_start;

        if (shard_len > 0) {
            // Read from predecessor's staging, reduce into our staging
            // FP32 accumulation for numerical stability
            for (int64_t i = tid; i < shard_len; i += nthreads) {
                int64_t idx = shard_start + i;
                float a = __half2float(my_staging[idx]);
                float b = __half2float(pred_staging[idx]);
                my_staging[idx] = __float2half(a + b);
            }
        }

        __threadfence_system();

        // Pairwise barrier with predecessor and successor
        // Use step-specific generation to avoid stale matches
        gen = gen_base + (uint32_t)(step + 1);
        if (threadIdx.x == 0) {
            // Signal successor that our shard is ready
            peer_barriers[succ_rank][rank] = gen;
            // Wait for predecessor to finish its reduce
            while (peer_barriers[rank][pred_rank] < gen);
        }
        __syncthreads();
    }

    // After reduce-scatter: each rank has one fully-reduced shard.
    // The completed shard for this rank is at index:
    // (ring_pos - (P-1)) % P == (ring_pos + 1) % P
    int my_completed_shard = (ring_pos + 1) % WORLD_SIZE;

    // ── Phase 2: AllGather (P-1 steps) ──
    // Each rank sends its completed shard around the ring.
    for (int step = 0; step < WORLD_SIZE - 1; step++) {
        // Shard to forward: the one we received last step (or our completed one)
        int shard_idx;
        if (step == 0)
            shard_idx = my_completed_shard;
        else
            shard_idx = ((my_completed_shard - step) % WORLD_SIZE + WORLD_SIZE) % WORLD_SIZE;

        int64_t shard_start = (int64_t)shard_idx * shard_count;
        int64_t shard_end = shard_start + shard_count;
        if (shard_end > count) shard_end = count;
        int64_t shard_len = shard_end - shard_start;

        // Read from predecessor's staging into our staging (copy the shard)
        // The predecessor just wrote their completed/forwarded shard
        int recv_shard = ((my_completed_shard - step - 1) % WORLD_SIZE + WORLD_SIZE) % WORLD_SIZE;
        int64_t recv_start = (int64_t)recv_shard * shard_count;
        int64_t recv_end = recv_start + shard_count;
        if (recv_end > count) recv_end = count;
        int64_t recv_len = recv_end - recv_start;

        if (recv_len > 0) {
            for (int64_t i = tid; i < recv_len; i += nthreads) {
                int64_t idx = recv_start + i;
                my_staging[idx] = pred_staging[idx];
            }
        }

        __threadfence_system();

        gen = gen_base + (uint32_t)(WORLD_SIZE + step);
        if (threadIdx.x == 0) {
            peer_barriers[succ_rank][rank] = gen;
            while (peer_barriers[rank][pred_rank] < gen);
        }
        __syncthreads();
    }

    // ── Phase 3: Copy result from staging to output ──
    for (int64_t i = tid; i < count; i += nthreads)
        output[i] = my_staging[i];
}

// BF16 ring allreduce
template<int WORLD_SIZE>
__global__ void infccl_ring_allreduce_bf16(
    uint16_t* __restrict__       output,
    const uint16_t* __restrict__ input,
    void** __restrict__          peer_staging,
    volatile uint32_t** __restrict__ peer_barriers,
    const int* __restrict__      ring_rank,
    const int                    rank,
    const int                    ring_pos,
    const int64_t                count,
    const uint32_t               gen_base)
{
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int64_t shard_count = (count + WORLD_SIZE - 1) / WORLD_SIZE;

    const int pred_pos = (ring_pos + WORLD_SIZE - 1) % WORLD_SIZE;
    const int succ_pos = (ring_pos + 1) % WORLD_SIZE;
    const int pred_rank = ring_rank[pred_pos];
    const int succ_rank = ring_rank[succ_pos];

    uint16_t* my_staging = (uint16_t*)peer_staging[rank];
    uint16_t* pred_staging = (uint16_t*)peer_staging[pred_rank];

    // Copy input to staging (vectorized: 2 bf16 per uint32)
    const int64_t count_v = count / 2;
    for (int64_t i = tid; i < count_v; i += nthreads)
        ((uint32_t*)my_staging)[i] = ((const uint32_t*)input)[i];
    if ((count & 1) && tid == 0) my_staging[count - 1] = input[count - 1];
    __threadfence_system();

    uint32_t gen = gen_base;
    if (threadIdx.x == 0) {
        for (int r = 0; r < WORLD_SIZE; r++)
            if (r != rank) peer_barriers[r][rank] = gen;
        for (int r = 0; r < WORLD_SIZE; r++)
            if (r != rank) while (peer_barriers[rank][r] < gen);
    }
    __syncthreads();

    // Reduce-scatter
    for (int step = 0; step < WORLD_SIZE - 1; step++) {
        int shard_idx = ((ring_pos - step) % WORLD_SIZE + WORLD_SIZE) % WORLD_SIZE;
        int64_t shard_start = (int64_t)shard_idx * shard_count;
        int64_t shard_end = shard_start + shard_count;
        if (shard_end > count) shard_end = count;
        int64_t shard_len = shard_end - shard_start;

        if (shard_len > 0) {
            for (int64_t i = tid; i < shard_len; i += nthreads) {
                int64_t idx = shard_start + i;
                float a = bf16_to_float(my_staging[idx]);
                float b = bf16_to_float(pred_staging[idx]);
                my_staging[idx] = float_to_bf16(a + b);
            }
        }
        __threadfence_system();

        gen = gen_base + (uint32_t)(step + 1);
        if (threadIdx.x == 0) {
            peer_barriers[succ_rank][rank] = gen;
            while (peer_barriers[rank][pred_rank] < gen);
        }
        __syncthreads();
    }

    int my_completed_shard = (ring_pos + 1) % WORLD_SIZE;

    // AllGather
    for (int step = 0; step < WORLD_SIZE - 1; step++) {
        int recv_shard = ((my_completed_shard - step - 1) % WORLD_SIZE + WORLD_SIZE) % WORLD_SIZE;
        int64_t recv_start = (int64_t)recv_shard * shard_count;
        int64_t recv_end = recv_start + shard_count;
        if (recv_end > count) recv_end = count;
        int64_t recv_len = recv_end - recv_start;

        if (recv_len > 0) {
            for (int64_t i = tid; i < recv_len; i += nthreads) {
                int64_t idx = recv_start + i;
                my_staging[idx] = pred_staging[idx];
            }
        }
        __threadfence_system();

        gen = gen_base + (uint32_t)(WORLD_SIZE + step);
        if (threadIdx.x == 0) {
            peer_barriers[succ_rank][rank] = gen;
            while (peer_barriers[rank][pred_rank] < gen);
        }
        __syncthreads();
    }

    // Copy result to output
    for (int64_t i = tid; i < count_v; i += nthreads)
        ((uint32_t*)output)[i] = ((uint32_t*)my_staging)[i];
    if ((count & 1) && tid == 0) output[count - 1] = my_staging[count - 1];
}

// FP32 ring allreduce
template<int WORLD_SIZE>
__global__ void infccl_ring_allreduce_fp32(
    float* __restrict__         output,
    const float* __restrict__   input,
    void** __restrict__         peer_staging,
    volatile uint32_t** __restrict__ peer_barriers,
    const int* __restrict__     ring_rank,
    const int                   rank,
    const int                   ring_pos,
    const int64_t               count,
    const uint32_t              gen_base)
{
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int64_t shard_count = (count + WORLD_SIZE - 1) / WORLD_SIZE;

    const int pred_pos = (ring_pos + WORLD_SIZE - 1) % WORLD_SIZE;
    const int succ_pos = (ring_pos + 1) % WORLD_SIZE;
    const int pred_rank = ring_rank[pred_pos];
    const int succ_rank = ring_rank[succ_pos];

    float* my_staging = (float*)peer_staging[rank];
    float* pred_staging = (float*)peer_staging[pred_rank];

    for (int64_t i = tid; i < count; i += nthreads)
        my_staging[i] = input[i];
    __threadfence_system();

    uint32_t gen = gen_base;
    if (threadIdx.x == 0) {
        for (int r = 0; r < WORLD_SIZE; r++)
            if (r != rank) peer_barriers[r][rank] = gen;
        for (int r = 0; r < WORLD_SIZE; r++)
            if (r != rank) while (peer_barriers[rank][r] < gen);
    }
    __syncthreads();

    // Reduce-scatter
    for (int step = 0; step < WORLD_SIZE - 1; step++) {
        int shard_idx = ((ring_pos - step) % WORLD_SIZE + WORLD_SIZE) % WORLD_SIZE;
        int64_t shard_start = (int64_t)shard_idx * shard_count;
        int64_t shard_end = shard_start + shard_count;
        if (shard_end > count) shard_end = count;
        int64_t shard_len = shard_end - shard_start;

        if (shard_len > 0) {
            for (int64_t i = tid; i < shard_len; i += nthreads) {
                int64_t idx = shard_start + i;
                my_staging[idx] = my_staging[idx] + pred_staging[idx];
            }
        }
        __threadfence_system();

        gen = gen_base + (uint32_t)(step + 1);
        if (threadIdx.x == 0) {
            peer_barriers[succ_rank][rank] = gen;
            while (peer_barriers[rank][pred_rank] < gen);
        }
        __syncthreads();
    }

    int my_completed_shard = (ring_pos + 1) % WORLD_SIZE;

    // AllGather
    for (int step = 0; step < WORLD_SIZE - 1; step++) {
        int recv_shard = ((my_completed_shard - step - 1) % WORLD_SIZE + WORLD_SIZE) % WORLD_SIZE;
        int64_t recv_start = (int64_t)recv_shard * shard_count;
        int64_t recv_end = recv_start + shard_count;
        if (recv_end > count) recv_end = count;
        int64_t recv_len = recv_end - recv_start;

        if (recv_len > 0) {
            for (int64_t i = tid; i < recv_len; i += nthreads) {
                int64_t idx = recv_start + i;
                my_staging[idx] = pred_staging[idx];
            }
        }
        __threadfence_system();

        gen = gen_base + (uint32_t)(WORLD_SIZE + step);
        if (threadIdx.x == 0) {
            peer_barriers[succ_rank][rank] = gen;
            while (peer_barriers[rank][pred_rank] < gen);
        }
        __syncthreads();
    }

    for (int64_t i = tid; i < count; i += nthreads)
        output[i] = my_staging[i];
}


// ============================================================================
// Host-callable C interface
// ============================================================================

extern "C" {

int infccl_init_p2p(int num_gpus) {
    int failures = 0;
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        for (int j = 0; j < num_gpus; j++) {
            if (i == j) continue;
            int can = 0;
            cudaDeviceCanAccessPeer(&can, i, j);
            if (can) {
                cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
                if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
                    fprintf(stderr, "[infccl] GPU %d->%d P2P enable failed: %s\n",
                            i, j, cudaGetErrorString(err));
                    failures++;
                }
            } else {
                fprintf(stderr, "[infccl] GPU %d cannot P2P access GPU %d\n", i, j);
                failures++;
            }
        }
    }
    return failures ? -1 : 0;
}

// Compute optimal ring order: prefer consecutive GPUs that can P2P.
// For 4 GPUs where 0↔3 might not P2P: order = [0,1,2,3] → ring 0→1→2→3→0
static void compute_ring_order(int world_size, int* ring_order, int* ring_rank) {
    // Default: identity order (0,1,2,...,P-1)
    // This works well when all consecutive pairs can P2P (typical for BI-V100).
    // Enhancement: detect P2P matrix and reorder to skip non-P2P pairs.
    
    int used[INFCCL_MAX_RANKS] = {0};
    int order[INFCCL_MAX_RANKS];
    
    // Greedy: start from GPU 0, pick nearest P2P-reachable unused GPU
    order[0] = 0;
    used[0] = 1;
    
    for (int pos = 1; pos < world_size; pos++) {
        int prev = order[pos - 1];
        int best = -1;
        
        // First try: find unused GPU that can P2P with prev
        for (int g = 0; g < world_size; g++) {
            if (used[g]) continue;
            cudaSetDevice(prev);
            int can = 0;
            cudaDeviceCanAccessPeer(&can, prev, g);
            if (can) {
                best = g;
                break; // take first available P2P neighbor
            }
        }
        
        // Fallback: any unused GPU
        if (best < 0) {
            for (int g = 0; g < world_size; g++) {
                if (!used[g]) { best = g; break; }
            }
        }
        
        order[pos] = best;
        used[best] = 1;
    }
    
    // Fill ring_rank (position → GPU rank) and ring_order (GPU rank → position)
    for (int pos = 0; pos < world_size; pos++) {
        ring_rank[pos] = order[pos];
        ring_order[order[pos]] = pos;
    }
    
    fprintf(stderr, "[infccl] Ring order:");
    for (int pos = 0; pos < world_size; pos++)
        fprintf(stderr, " %d", ring_rank[pos]);
    fprintf(stderr, "\n");
}

InfcclComm* infccl_comm_create(int rank, int world_size, int64_t max_size) {
    InfcclComm* comm = (InfcclComm*)calloc(1, sizeof(InfcclComm));
    if (!comm) return NULL;

    comm->rank = rank;
    comm->world_size = world_size;
    comm->max_size = max_size;
    comm->generation = 1;

    cudaSetDevice(rank);

    // Allocate staging buffer
    void* local_buf = NULL;
    cudaMalloc(&local_buf, max_size);
    cudaMemset(local_buf, 0, max_size);
    comm->staging[rank] = local_buf;

    // Allocate metadata (barrier flags)
    InfcclMeta* local_meta = NULL;
    cudaMalloc(&local_meta, sizeof(InfcclMeta));
    cudaMemset(local_meta, 0, sizeof(InfcclMeta));
    comm->barriers[rank] = (volatile uint32_t*)&local_meta->flags[0];

    // Device pointer arrays
    cudaMalloc(&comm->d_staging, sizeof(void*) * INFCCL_MAX_RANKS);
    cudaMalloc(&comm->d_barriers, sizeof(volatile uint32_t*) * INFCCL_MAX_RANKS);

    // Ring topology arrays
    cudaMalloc(&comm->d_ring_order, sizeof(int) * INFCCL_MAX_RANKS);
    cudaMalloc(&comm->d_ring_rank, sizeof(int) * INFCCL_MAX_RANKS);

    // Compute ring order (P2P-aware)
    compute_ring_order(world_size, comm->ring_order, comm->ring_rank);

    return comm;
}

int infccl_comm_get_ipc(InfcclComm* comm,
                        cudaIpcMemHandle_t* buf_handle,
                        cudaIpcMemHandle_t* meta_handle) {
    cudaSetDevice(comm->rank);
    cudaError_t e1 = cudaIpcGetMemHandle(buf_handle, comm->staging[comm->rank]);
    cudaError_t e2 = cudaIpcGetMemHandle(meta_handle, (void*)comm->barriers[comm->rank]);
    return (e1 == cudaSuccess && e2 == cudaSuccess) ? 0 : -1;
}

int infccl_comm_register_peer(InfcclComm* comm, int peer_rank,
                              cudaIpcMemHandle_t* buf_handle,
                              cudaIpcMemHandle_t* meta_handle) {
    cudaSetDevice(comm->rank);
    void* peer_buf = NULL;
    void* peer_meta = NULL;

    cudaError_t e1 = cudaIpcOpenMemHandle(&peer_buf, *buf_handle,
                                          cudaIpcMemLazyEnablePeerAccess);
    cudaError_t e2 = cudaIpcOpenMemHandle(&peer_meta, *meta_handle,
                                          cudaIpcMemLazyEnablePeerAccess);
    if (e1 != cudaSuccess || e2 != cudaSuccess) {
        fprintf(stderr, "[infccl] IPC open failed for peer %d: %s / %s\n",
                peer_rank, cudaGetErrorString(e1), cudaGetErrorString(e2));
        return -1;
    }

    comm->staging[peer_rank] = peer_buf;
    comm->barriers[peer_rank] = (volatile uint32_t*)peer_meta;
    return 0;
}

int infccl_comm_finalize(InfcclComm* comm) {
    cudaSetDevice(comm->rank);
    cudaMemcpy(comm->d_staging, comm->staging,
               sizeof(void*) * comm->world_size, cudaMemcpyHostToDevice);
    cudaMemcpy(comm->d_barriers, comm->barriers,
               sizeof(volatile uint32_t*) * comm->world_size, cudaMemcpyHostToDevice);
    cudaMemcpy(comm->d_ring_order, comm->ring_order,
               sizeof(int) * comm->world_size, cudaMemcpyHostToDevice);
    cudaMemcpy(comm->d_ring_rank, comm->ring_rank,
               sizeof(int) * comm->world_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    return 0;
}

// Launch allreduce — dispatches to fused or ring based on tensor size
// dtype: 0=fp32, 1=fp16, 2=bf16
// Returns: 0 on success, 1 if tensor too large (caller should use NCCL)
int infccl_allreduce(InfcclComm* comm,
                     void* output, const void* input,
                     int64_t count, int dtype,
                     cudaStream_t stream) {
    const int rank = comm->rank;
    const int ws = comm->world_size;
    const int elem_size = (dtype == 0) ? 4 : 2;
    const int64_t nbytes = count * elem_size;

    // Consume 2*P generations per ring call (P-1 reduce-scatter + P-1 allgather + 1 init)
    const uint32_t gen_base = comm->generation;
    comm->generation += (uint32_t)(2 * ws + 1);

    // ── Path 1: Fused one-shot for small tensors ──
    if (nbytes <= INFCCL_ONESHOT_THRESHOLD) {
        const uint32_t gen = gen_base;
        int grid = (int)((count + INFCCL_BLOCK_SIZE - 1) / INFCCL_BLOCK_SIZE);
        if (grid > INFCCL_MAX_GRID) grid = INFCCL_MAX_GRID;

        #define LAUNCH_FUSED(T, WS) \
            infccl_fused_allreduce<T, WS><<<grid, INFCCL_BLOCK_SIZE, 0, stream>>>( \
                (T*)output, (const T*)input, \
                comm->d_staging, comm->d_barriers, rank, count, gen)

        #define LAUNCH_FP16_FUSED(WS) \
            infccl_fused_allreduce_fp16<WS><<<grid, INFCCL_BLOCK_SIZE, 0, stream>>>( \
                (half*)output, (const half*)input, \
                comm->d_staging, comm->d_barriers, rank, count, gen)

        #define LAUNCH_BF16_FUSED(WS) \
            infccl_fused_allreduce_bf16<WS><<<grid, INFCCL_BLOCK_SIZE, 0, stream>>>( \
                (uint16_t*)output, (const uint16_t*)input, \
                comm->d_staging, comm->d_barriers, rank, count, gen)

        if (ws == 4) {
            if (dtype == 0)      LAUNCH_FUSED(float, 4);
            else if (dtype == 1) LAUNCH_FP16_FUSED(4);
            else if (dtype == 2) LAUNCH_BF16_FUSED(4);
        } else if (ws == 2) {
            if (dtype == 0)      LAUNCH_FUSED(float, 2);
            else if (dtype == 1) LAUNCH_FP16_FUSED(2);
            else if (dtype == 2) LAUNCH_BF16_FUSED(2);
        } else {
            if (dtype == 0)      LAUNCH_FUSED(float, 8);
            else if (dtype == 1) LAUNCH_FP16_FUSED(8);
            else if (dtype == 2) LAUNCH_BF16_FUSED(8);
        }

        #undef LAUNCH_FUSED
        #undef LAUNCH_FP16_FUSED
        #undef LAUNCH_BF16_FUSED
        return 0;
    }

    // ── Path 2: Ring allreduce for medium tensors ──
    if (nbytes <= INFCCL_RING_MAX_BYTES) {
        const int ring_pos = comm->ring_order[rank];

        // Single block: we need __syncthreads() to act as device barrier.
        // Use maximum threads for bandwidth.
        const int block = INFCCL_BLOCK_SIZE;

        #define LAUNCH_RING_FP16(WS) \
            infccl_ring_allreduce_fp16<WS><<<1, block, 0, stream>>>( \
                (half*)output, (const half*)input, \
                comm->d_staging, comm->d_barriers, comm->d_ring_rank, \
                rank, ring_pos, count, gen_base)

        #define LAUNCH_RING_BF16(WS) \
            infccl_ring_allreduce_bf16<WS><<<1, block, 0, stream>>>( \
                (uint16_t*)output, (const uint16_t*)input, \
                comm->d_staging, comm->d_barriers, comm->d_ring_rank, \
                rank, ring_pos, count, gen_base)

        #define LAUNCH_RING_FP32(WS) \
            infccl_ring_allreduce_fp32<WS><<<1, block, 0, stream>>>( \
                (float*)output, (const float*)input, \
                comm->d_staging, comm->d_barriers, comm->d_ring_rank, \
                rank, ring_pos, count, gen_base)

        if (ws == 4) {
            if (dtype == 0)      LAUNCH_RING_FP32(4);
            else if (dtype == 1) LAUNCH_RING_FP16(4);
            else if (dtype == 2) LAUNCH_RING_BF16(4);
        } else if (ws == 2) {
            if (dtype == 0)      LAUNCH_RING_FP32(2);
            else if (dtype == 1) LAUNCH_RING_FP16(2);
            else if (dtype == 2) LAUNCH_RING_BF16(2);
        } else {
            if (dtype == 0)      LAUNCH_RING_FP32(8);
            else if (dtype == 1) LAUNCH_RING_FP16(8);
            else if (dtype == 2) LAUNCH_RING_BF16(8);
        }

        #undef LAUNCH_RING_FP16
        #undef LAUNCH_RING_BF16
        #undef LAUNCH_RING_FP32
        return 0;
    }

    // ── Path 3: Too large for custom AR → signal caller to use NCCL ──
    return 1;
}

// Query which allreduce path will be used for a given size
// Returns: 0=fused, 1=ring, 2=nccl(too large)
int infccl_query_path(int64_t nbytes) {
    if (nbytes <= INFCCL_ONESHOT_THRESHOLD) return 0;
    if (nbytes <= INFCCL_RING_MAX_BYTES) return 1;
    return 2;
}

// ============================================================================
// ReduceScatter: each rank gets 1/P of the reduced data
//   input: count elements on each rank (all same data to reduce)
//   output: count/P elements on each rank (this rank's shard of the result)
//
// This is the first half of ring allreduce. Used in TP column-parallel Linear:
//   output = Linear(input)  [each rank computes partial]
//   reduced_shard = reduce_scatter(output)  [sum partials, scatter result]
//
// dtype: 0=fp32, 1=fp16, 2=bf16
// Returns: 0 success, 1 too large for custom path
// ============================================================================
int infccl_reduce_scatter(InfcclComm* comm,
                          void* output, const void* input,
                          int64_t input_count, int dtype,
                          cudaStream_t stream) {
    const int rank = comm->rank;
    const int ws = comm->world_size;
    const int elem_size = (dtype == 0) ? 4 : 2;
    const int64_t nbytes = input_count * elem_size;
    const int64_t shard_count = (input_count + ws - 1) / ws;
    const int ring_pos = comm->ring_order[rank];

    if (nbytes > INFCCL_RING_MAX_BYTES) return 1;

    const uint32_t gen_base = comm->generation;
    comm->generation += (uint32_t)(ws + 1);

    // We reuse the ring allreduce infrastructure but only run the
    // reduce-scatter phase, then copy this rank's completed shard to output.
    // For simplicity we launch the full ring allreduce kernel and then
    // just extract this rank's shard. This wastes the allgather bandwidth
    // but keeps the code simple for the first version.
    // TODO: implement a reduce-scatter-only kernel for better efficiency.

    // For now, run infccl_allreduce in-place on the staging buffer
    // then extract the shard. But since the ring kernel is single-block,
    // we can't easily re-enter it. Instead, use the fused path for small
    // and the full ring for medium, then copy the shard out.

    // Simple approach: allreduce the full input, then extract shard
    int rc = infccl_allreduce(comm, (void*)input, input, input_count, dtype, stream);
    if (rc != 0) return rc;

    // Copy this rank's shard from the allreduced result to output
    // The shard index for this rank in the ring is ring_pos
    int64_t shard_start = (int64_t)ring_pos * shard_count;
    int64_t shard_end = shard_start + shard_count;
    if (shard_end > input_count) shard_end = input_count;
    int64_t shard_len = shard_end - shard_start;

    if (shard_len > 0) {
        const char* src = (const char*)input + shard_start * elem_size;
        cudaMemcpyAsync(output, src, shard_len * elem_size,
                        cudaMemcpyDeviceToDevice, stream);
    }

    return 0;
}

// ============================================================================
// AllGather: each rank contributes count/P elements, output has count elements
//   input: count/P elements on each rank (this rank's shard)
//   output: count elements on each rank (all shards concatenated)
//
// This is the second half of ring allreduce. Used in TP row-parallel Linear:
//   shard = Linear(input_shard)  [each rank computes its shard]
//   full_output = all_gather(shard)  [gather all shards]
//
// dtype: 0=fp32, 1=fp16, 2=bf16
// Returns: 0 success, 1 too large for custom path
// ============================================================================
int infccl_all_gather(InfcclComm* comm,
                      void* output, const void* input,
                      int64_t shard_count, int dtype,
                      cudaStream_t stream) {
    const int rank = comm->rank;
    const int ws = comm->world_size;
    const int elem_size = (dtype == 0) ? 4 : 2;
    const int64_t total_count = shard_count * ws;
    const int64_t nbytes = total_count * elem_size;
    const int ring_pos = comm->ring_order[rank];

    if (nbytes > INFCCL_RING_MAX_BYTES) return 1;

    const uint32_t gen_base = comm->generation;
    comm->generation += (uint32_t)(ws + 1);

    // Step 1: Each rank copies its shard into the correct position
    // in its staging buffer
    int64_t my_offset = (int64_t)ring_pos * shard_count;
    char* staging = (char*)comm->staging[rank];
    cudaMemcpyAsync(staging + my_offset * elem_size, input,
                    shard_count * elem_size,
                    cudaMemcpyDeviceToDevice, stream);

    // Step 2: Barrier — all ranks have placed their shard
    cudaStreamSynchronize(stream);
    // Use a simple flag-based barrier (host-side for now)
    // TODO: GPU-side barrier for lower latency
    uint32_t gen = gen_base;
    volatile uint32_t* my_barrier = comm->barriers[rank];
    // Signal all peers
    for (int r = 0; r < ws; r++) {
        if (r != rank) {
            volatile uint32_t* peer_flags = comm->barriers[r];
            peer_flags[rank] = gen;
        }
    }
    // Wait for all peers
    for (int r = 0; r < ws; r++) {
        if (r != rank) {
            while (my_barrier[r] < gen);
        }
    }

    // Step 3: Each rank copies all shards from all peers' staging to output
    for (int r = 0; r < ws; r++) {
        int r_ring_pos = comm->ring_order[r];
        int64_t offset = (int64_t)r_ring_pos * shard_count;
        const char* peer_staging = (const char*)comm->staging[r];
        cudaMemcpyAsync((char*)output + offset * elem_size,
                        peer_staging + offset * elem_size,
                        shard_count * elem_size,
                        cudaMemcpyDeviceToDevice, stream);
    }

    return 0;
}

void infccl_comm_destroy(InfcclComm* comm) {
    if (!comm) return;
    cudaSetDevice(comm->rank);

    if (comm->staging[comm->rank])
        cudaFree(comm->staging[comm->rank]);
    if (comm->barriers[comm->rank])
        cudaFree((void*)comm->barriers[comm->rank]);

    for (int r = 0; r < comm->world_size; r++) {
        if (r != comm->rank) {
            if (comm->staging[r]) cudaIpcCloseMemHandle(comm->staging[r]);
            if (comm->barriers[r]) cudaIpcCloseMemHandle((void*)comm->barriers[r]);
        }
    }

    if (comm->d_staging)    cudaFree(comm->d_staging);
    if (comm->d_barriers)   cudaFree((void*)comm->d_barriers);
    if (comm->d_ring_order) cudaFree(comm->d_ring_order);
    if (comm->d_ring_rank)  cudaFree(comm->d_ring_rank);

    free(comm);
}

}  // extern "C"
