/*
 * ikun_allreduce.cu — Ring-based allreduce for larger payloads
 *
 * Layer 1 math specs:
 *   - Ring step k: rank r sends chunk[(r-k) % W] to rank (r+1) % W
 *   - Reduce-scatter phase: W-1 steps, accumulating reductions
 *   - Allgather phase: W-1 steps, propagating final values
 *   - Termination: total 2*(W-1) ring steps
 *   - Slice invariant: each chunk = ceil(count / world_size) elements
 *
 * Layer 2 form template:
 *   prjname = ikun
 *   Obj = RingAllReducer
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define IKUN_RING_BLOCK_SIZE 256

/*
 * Ring reduce-scatter kernel: accumulate src chunk into dst chunk
 */
__global__ void ikun_ring_reduce_f32(float* dst, const float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] += src[idx];
    }
}

/*
 * Ring copy kernel: copy chunk from src to dst
 */
__global__ void ikun_ring_copy_f32(float* dst, const float* src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

/*
 * Layer 1 spec: Ring allreduce
 *
 * Given: world_size = W, count = N, chunk_size = ceil(N / W)
 *
 * Phase 1 (reduce-scatter):
 *   For step k in [0, W-1):
 *     send_chunk_idx = (rank - k + W) % W
 *     recv_chunk_idx = (rank - k - 1 + W) % W
 *     Each rank sends its send_chunk to (rank+1)%W
 *     and receives + reduces recv_chunk from (rank-1+W)%W
 *
 * Phase 2 (allgather):
 *   For step k in [0, W-1):
 *     send_chunk_idx = (rank - k + 1 + W) % W
 *     recv_chunk_idx = (rank - k + W) % W
 *     Each rank sends its send_chunk to (rank+1)%W
 *     and receives recv_chunk from (rank-1+W)%W (no reduce, just copy)
 *
 * This file provides the kernel implementations.
 * The orchestration uses cudaMemcpyPeerAsync for cross-GPU transfers.
 */

typedef struct IkunRingState {
    int rank;
    int world_size;
    int device_id;
    float** gpu_bufs;         /* device buffer pointers, one per GPU */
    float** staging_bufs;     /* staging buffer pointers, one per GPU */
    size_t count;
    size_t chunk_size;
    cudaStream_t stream;
} IkunRingState;

/*
 * ikun_ring_allreduce_execute: execute ring allreduce
 *
 * Preconditions:
 *   - All GPUs have their data in gpu_bufs[rank]
 *   - staging_bufs[rank] has at least count * sizeof(float) bytes
 *   - P2P access enabled between consecutive ring neighbors
 *
 * Postcondition: all gpu_bufs[rank] contain the reduced result
 *
 * Layer 1 termination proof:
 *   Loop variable: step k goes from 0 to W-2 (inclusive)
 *   Progress: k increments by 1 each iteration
 *   Bound: loop exits when k == W-1
 *   Total iterations: 2 * (W-1)
 */
extern "C"
int ikun_ring_allreduce_execute(
    int rank,
    int world_size,
    int device_id,
    float** all_bufs,        /* array of world_size device buffer pointers */
    size_t count,
    cudaStream_t stream
) {
    if (world_size <= 1) {
        /* Nothing to do for single GPU */
        return 0;
    }

    size_t chunk_size = (count + world_size - 1) / world_size;

    cudaSetDevice(device_id);

    /* Phase 1: Reduce-scatter */
    for (int step = 0; step < world_size - 1; step++) {
        /* Chunk index this rank will receive and reduce */
        int recv_chunk = ((rank - step - 1) % world_size + world_size) % world_size;
        int send_rank = ((rank - 1) % world_size + world_size) % world_size;

        /* Offset into the buffer for this chunk */
        size_t chunk_offset = (size_t)recv_chunk * chunk_size;
        size_t actual_chunk = chunk_size;
        if (chunk_offset + actual_chunk > count) {
            actual_chunk = count - chunk_offset;
        }
        if (actual_chunk == 0) continue;

        /* Read from predecessor's buffer (P2P) and reduce into own */
        float* my_chunk = all_bufs[rank] + chunk_offset;
        const float* peer_chunk = all_bufs[send_rank] + chunk_offset;

        int cb = (actual_chunk + IKUN_RING_BLOCK_SIZE - 1) / IKUN_RING_BLOCK_SIZE;
        ikun_ring_reduce_f32<<<cb, IKUN_RING_BLOCK_SIZE, 0, stream>>>(
            my_chunk, peer_chunk, (int)actual_chunk
        );
        cudaStreamSynchronize(stream);
    }

    /* Phase 2: Allgather */
    for (int step = 0; step < world_size - 1; step++) {
        int recv_chunk = ((rank - step) % world_size + world_size) % world_size;
        int send_rank = ((rank - 1) % world_size + world_size) % world_size;

        size_t chunk_offset = (size_t)recv_chunk * chunk_size;
        size_t actual_chunk = chunk_size;
        if (chunk_offset + actual_chunk > count) {
            actual_chunk = count - chunk_offset;
        }
        if (actual_chunk == 0) continue;

        /* Copy from predecessor (who now has the fully reduced chunk) */
        float* my_chunk = all_bufs[rank] + chunk_offset;
        const float* peer_chunk = all_bufs[send_rank] + chunk_offset;

        int cb = (actual_chunk + IKUN_RING_BLOCK_SIZE - 1) / IKUN_RING_BLOCK_SIZE;
        ikun_ring_copy_f32<<<cb, IKUN_RING_BLOCK_SIZE, 0, stream>>>(
            my_chunk, peer_chunk, (int)actual_chunk
        );
        cudaStreamSynchronize(stream);
    }

    return 0;
}

