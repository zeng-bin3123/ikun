/*
 * infccl_collectives.cu — Full Collective Operations for Iluvatar BI-V100
 *
 * Implements AllGather, Broadcast, Reduce, ReduceScatter using the same
 * P2P shared-memory architecture as infccl_allreduce.cu.
 *
 * Mathematical foundations (from NCCL THEORY Layer 1):
 *
 *   AllGather:    Each rank contributes count/world_size elements.
 *                 After completion, every rank holds all N*chunk elements.
 *                 Ring step count = world_size - 1.
 *
 *   Broadcast:    Root copies data to staging; all ranks read from root's buffer.
 *                 Single-phase: copy + barrier + read.
 *
 *   Reduce:       All ranks contribute; root holds the sum.
 *                 Inverse of Broadcast: all write staging, root reduces.
 *
 *   ReduceScatter: Each rank gets a reduced slice of the data.
 *                  Dual of AllGather: reduce + scatter in one pass.
 *
 * Compile for BI-V100:
 *   /usr/local/corex/bin/nvcc -arch=sm_70 -O3 -shared -Xcompiler -fPIC \
 *       -o lib/infccl_collectives.so csrc/infccl_collectives.cu -lcudart
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define INFCCL_MAX_RANKS  8
#define INFCCL_BLOCK_SIZE 256
#define INFCCL_MAX_GRID   1024

// ============================================================================
// Shared data structures (same layout as infccl_allreduce.cu for interop)
// ============================================================================

struct InfcclCollComm {
    void*              staging[INFCCL_MAX_RANKS];
    volatile uint32_t* barriers[INFCCL_MAX_RANKS];
    int                rank;
    int                world_size;
    int64_t            max_size;
    uint32_t           generation;

    void**             d_staging;
    volatile uint32_t** d_barriers;
};

// ============================================================================
// BF16 helpers (CUDA 10.2 compatible — no __nv_bfloat16)
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
    u32 += 0x7FFF + ((u32 >> 16) & 1);
    return (uint16_t)(u32 >> 16);
}

// ============================================================================
// Device barrier: signal + wait using monotonic generation counter
//
// Layer 1 (Theorem): signal_wait_monotone
//   ∀ gen g, ∀ r ∈ [0, world_size), barrier[rank][r] is monotonically
//   non-decreasing. Wait terminates when barrier[rank][r] >= g.
//
// Note: Uses __threadfence() (device-level) instead of __threadfence_system()
// which is not supported by the Iluvatar CoreX clang LLVM backend (ivcore10).
// On PCIe-attached BI-V100 GPUs, volatile + __threadfence() + PCIe ordering
// provides the required visibility guarantees for P2P mapped memory.
// ============================================================================

__device__ __forceinline__ void infccl_barrier_signal_wait(
    volatile uint32_t** barriers,
    int rank, int world_size, uint32_t gen)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Signal: write gen to each peer's slot for us
        for (int r = 0; r < world_size; r++) {
            if (r != rank)
                barriers[r][rank] = gen;
        }
        __threadfence();  // ensure writes visible before spinning
        // Wait: spin until all peers have signaled gen
        for (int r = 0; r < world_size; r++) {
            if (r != rank) {
                while (barriers[rank][r] < gen)
                    ;  // volatile ensures re-read
            }
        }
    }
    __syncthreads();
}

// ============================================================================
// Helper kernel: copy data to staging buffer (separate launch for P2P ordering)
// ============================================================================

template<typename T>
__global__ void infccl_copy_to_staging(
    T* __restrict__         staging,
    const T* __restrict__   input,
    const int64_t           count)
{
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = tid; i < count; i += stride)
        staging[i] = input[i];
}

// Helper: gather from all staging buffers into output (after P2P sync)
template<typename T, int WORLD_SIZE>
__global__ void infccl_gather_from_staging(
    T* __restrict__         output,
    void** __restrict__     peer_staging,
    volatile uint32_t** __restrict__ peer_barriers,
    const int               rank,
    const int64_t           send_count,
    const uint32_t          gen)
{
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    // Barrier: wait for all peers to finish copying
    infccl_barrier_signal_wait(peer_barriers, rank, WORLD_SIZE, gen);

    // Gather
    for (int r = 0; r < WORLD_SIZE; r++) {
        const T* peer_buf = (const T*)peer_staging[r];
        const int64_t out_offset = (int64_t)r * send_count;
        for (int64_t i = tid; i < send_count; i += stride)
            output[out_offset + i] = peer_buf[i];
    }
}

// Helper: reduce-scatter from all staging buffers (after P2P sync)
template<typename T, int WORLD_SIZE>
__global__ void infccl_scatter_reduce_from_staging(
    T* __restrict__         output,
    void** __restrict__     peer_staging,
    volatile uint32_t** __restrict__ peer_barriers,
    const int               rank,
    const int64_t           total_count,
    const uint32_t          gen)
{
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const int64_t chunk_size = total_count / WORLD_SIZE;

    // Barrier
    infccl_barrier_signal_wait(peer_barriers, rank, WORLD_SIZE, gen);

    // Reduce my slice
    const int64_t my_offset = (int64_t)rank * chunk_size;
    for (int64_t i = tid; i < chunk_size; i += stride) {
        T acc = ((T*)peer_staging[0])[my_offset + i];
        #pragma unroll
        for (int r = 1; r < WORLD_SIZE; r++)
            acc = acc + ((T*)peer_staging[r])[my_offset + i];
        output[i] = acc;
    }
}


// ============================================================================
// AllGather Kernel
//
// Layer 1 (Theorem): allgather_chunk_coverage
//   Let chunk = ceil(total_count / world_size).
//   ∀ r ∈ [0, world_size), rank r writes input[0..chunk) to
//   staging[r][0..chunk). After barrier, every rank reads all chunks
//   into output[r*chunk .. (r+1)*chunk).
//   Post: output = concat(staging[0], staging[1], ..., staging[ws-1])
//
// Layer 2 form: GetBlock(slotIdx, nSlots) → slotIdx % nSlots
//   Filled: slotIdx → rank, nSlots → world_size
// ============================================================================

template<typename T, int WORLD_SIZE>
__global__ void infccl_allgather_kernel(
    T* __restrict__             output,       // [total_count] = ws * send_count
    const T* __restrict__       input,        // [send_count] per rank
    void** __restrict__         peer_staging,
    volatile uint32_t** __restrict__ peer_barriers,
    const int                   rank,
    const int64_t               send_count,   // elements per rank
    const uint32_t              gen)
{
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    // Phase 1: Copy my input to my staging buffer
    T* my_buf = (T*)peer_staging[rank];
    for (int64_t i = tid; i < send_count; i += stride)
        my_buf[i] = input[i];

    __threadfence();  // device-level fence (ivcore10 compat)

    // Phase 2: Barrier
    infccl_barrier_signal_wait(peer_barriers, rank, WORLD_SIZE, gen);

    // Phase 3: Gather all chunks into output
    // output layout: [rank0_chunk | rank1_chunk | ... | rank(ws-1)_chunk]
    for (int r = 0; r < WORLD_SIZE; r++) {
        const T* peer_buf = (const T*)peer_staging[r];
        const int64_t out_offset = (int64_t)r * send_count;
        for (int64_t i = tid; i < send_count; i += stride)
            output[out_offset + i] = peer_buf[i];
    }
}

// ============================================================================
// Broadcast Kernel
//
// Layer 1 (Theorem): bcast_pipeline_length
//   Total steps = 1 (root copies) + 1 (barrier) + 1 (all read).
//   Post: ∀ r, output_r = input_root
//
// Layer 2 form: BcastRole(rank, root) →
//   if rank == root: SEND (write staging)
//   else:            RECV (read from root's staging)
// ============================================================================

template<typename T, int WORLD_SIZE>
__global__ void infccl_broadcast_kernel(
    T* __restrict__             output,
    const T* __restrict__       input,
    void** __restrict__         peer_staging,
    volatile uint32_t** __restrict__ peer_barriers,
    const int                   rank,
    const int                   root,
    const int64_t               count,
    const uint32_t              gen)
{
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    // Phase 1: Root copies input to its staging buffer
    if (rank == root) {
        T* my_buf = (T*)peer_staging[root];
        for (int64_t i = tid; i < count; i += stride)
            my_buf[i] = input[i];
    }

    __threadfence();  // device-level fence (ivcore10 compat)

    // Phase 2: Barrier (all ranks synchronize)
    infccl_barrier_signal_wait(peer_barriers, rank, WORLD_SIZE, gen);

    // Phase 3: All ranks read from root's staging buffer
    const T* root_buf = (const T*)peer_staging[root];
    for (int64_t i = tid; i < count; i += stride)
        output[i] = root_buf[i];
}

// ============================================================================
// Reduce Kernel (SUM)
//
// Layer 1 (Theorem): reduce_roles_distinct
//   root ≠ leaf ≠ intermediate for any rank assignment.
//   In flat P2P: root reduces, non-root only writes.
//   Post: output_root[i] = Σ_{r=0}^{ws-1} input_r[i]
//
// Layer 2 form: ReduceRole(rank, root, ws) →
//   if rank == root: ROOT (read all staging, reduce, write output)
//   else:            LEAF (write staging only)
// ============================================================================

template<typename T, int WORLD_SIZE>
__global__ void infccl_reduce_kernel(
    T* __restrict__             output,       // valid on root only
    const T* __restrict__       input,
    void** __restrict__         peer_staging,
    volatile uint32_t** __restrict__ peer_barriers,
    const int                   rank,
    const int                   root,
    const int64_t               count,
    const uint32_t              gen)
{
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    // Phase 1: All ranks copy input to their staging buffer
    T* my_buf = (T*)peer_staging[rank];
    for (int64_t i = tid; i < count; i += stride)
        my_buf[i] = input[i];

    __threadfence();  // device-level fence (ivcore10 compat)

    // Phase 2: Barrier
    infccl_barrier_signal_wait(peer_barriers, rank, WORLD_SIZE, gen);

    // Phase 3: Root reduces all staging buffers
    if (rank == root) {
        for (int64_t i = tid; i < count; i += stride) {
            T acc = ((T*)peer_staging[0])[i];
            #pragma unroll
            for (int r = 1; r < WORLD_SIZE; r++)
                acc = acc + ((T*)peer_staging[r])[i];
            output[i] = acc;
        }
    }
}

// FP16 Reduce with FP32 accumulation
template<int WORLD_SIZE>
__global__ void infccl_reduce_fp16_kernel(
    half* __restrict__          output,
    const half* __restrict__    input,
    void** __restrict__         peer_staging,
    volatile uint32_t** __restrict__ peer_barriers,
    const int                   rank,
    const int                   root,
    const int64_t               count,
    const uint32_t              gen)
{
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    half* my_buf = (half*)peer_staging[rank];
    const int64_t count_v = count / 2;
    for (int64_t i = tid; i < count_v; i += stride)
        ((float*)my_buf)[i] = ((const float*)input)[i];
    if ((count & 1) && tid == 0)
        my_buf[count - 1] = input[count - 1];

    __threadfence();  // device-level fence (ivcore10 compat)
    infccl_barrier_signal_wait(peer_barriers, rank, WORLD_SIZE, gen);

    if (rank == root) {
        for (int64_t i = tid; i < count; i += stride) {
            float acc = __half2float(((half*)peer_staging[0])[i]);
            #pragma unroll
            for (int r = 1; r < WORLD_SIZE; r++)
                acc += __half2float(((half*)peer_staging[r])[i]);
            output[i] = __float2half(acc);
        }
    }
}

// BF16 Reduce with FP32 accumulation
template<int WORLD_SIZE>
__global__ void infccl_reduce_bf16_kernel(
    uint16_t* __restrict__      output,
    const uint16_t* __restrict__ input,
    void** __restrict__         peer_staging,
    volatile uint32_t** __restrict__ peer_barriers,
    const int                   rank,
    const int                   root,
    const int64_t               count,
    const uint32_t              gen)
{
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;

    uint16_t* my_buf = (uint16_t*)peer_staging[rank];
    const int64_t count_v = count / 2;
    for (int64_t i = tid; i < count_v; i += stride)
        ((uint32_t*)my_buf)[i] = ((const uint32_t*)input)[i];
    if ((count & 1) && tid == 0)
        my_buf[count - 1] = input[count - 1];

    __threadfence();  // device-level fence (ivcore10 compat)
    infccl_barrier_signal_wait(peer_barriers, rank, WORLD_SIZE, gen);

    if (rank == root) {
        for (int64_t i = tid; i < count; i += stride) {
            float acc = bf16_to_float(((uint16_t*)peer_staging[0])[i]);
            #pragma unroll
            for (int r = 1; r < WORLD_SIZE; r++)
                acc += bf16_to_float(((uint16_t*)peer_staging[r])[i]);
            output[i] = float_to_bf16(acc);
        }
    }
}

// ============================================================================
// ReduceScatter Kernel
//
// Layer 1 (Theorem): reduce_scatter_steps = world_size - 1
//   Total elements = count. chunk = count / world_size.
//   Rank r gets reduced slice [r*chunk .. (r+1)*chunk).
//   Post: output_r[i] = Σ_{k=0}^{ws-1} input_k[r*chunk + i]
//
// Layer 2 form: GetBlock(slotIdx, nSlots) → slotIdx % nSlots
//   Filled: output offset = rank * chunk_size
// ============================================================================

template<typename T, int WORLD_SIZE>
__global__ void infccl_reduce_scatter_kernel(
    T* __restrict__             output,       // [chunk_size] per rank
    const T* __restrict__       input,        // [total_count] = ws * chunk_size
    void** __restrict__         peer_staging,
    volatile uint32_t** __restrict__ peer_barriers,
    const int                   rank,
    const int64_t               total_count,
    const uint32_t              gen)
{
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const int64_t chunk_size = total_count / WORLD_SIZE;

    // Phase 1: Copy full input to staging
    T* my_buf = (T*)peer_staging[rank];
    for (int64_t i = tid; i < total_count; i += stride)
        my_buf[i] = input[i];

    __threadfence();  // device-level fence (ivcore10 compat)

    // Phase 2: Barrier
    infccl_barrier_signal_wait(peer_barriers, rank, WORLD_SIZE, gen);

    // Phase 3: Each rank reduces its own slice from all staging buffers
    const int64_t my_offset = (int64_t)rank * chunk_size;
    for (int64_t i = tid; i < chunk_size; i += stride) {
        T acc = ((T*)peer_staging[0])[my_offset + i];
        #pragma unroll
        for (int r = 1; r < WORLD_SIZE; r++)
            acc = acc + ((T*)peer_staging[r])[my_offset + i];
        output[i] = acc;
    }
}

// FP16 ReduceScatter with FP32 accumulation
template<int WORLD_SIZE>
__global__ void infccl_reduce_scatter_fp16_kernel(
    half* __restrict__          output,
    const half* __restrict__    input,
    void** __restrict__         peer_staging,
    volatile uint32_t** __restrict__ peer_barriers,
    const int                   rank,
    const int64_t               total_count,
    const uint32_t              gen)
{
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const int64_t chunk_size = total_count / WORLD_SIZE;

    half* my_buf = (half*)peer_staging[rank];
    const int64_t count_v = total_count / 2;
    for (int64_t i = tid; i < count_v; i += stride)
        ((float*)my_buf)[i] = ((const float*)input)[i];
    if ((total_count & 1) && tid == 0)
        my_buf[total_count - 1] = input[total_count - 1];

    __threadfence();  // device-level fence (ivcore10 compat)
    infccl_barrier_signal_wait(peer_barriers, rank, WORLD_SIZE, gen);

    const int64_t my_offset = (int64_t)rank * chunk_size;
    for (int64_t i = tid; i < chunk_size; i += stride) {
        float acc = __half2float(((half*)peer_staging[0])[my_offset + i]);
        #pragma unroll
        for (int r = 1; r < WORLD_SIZE; r++)
            acc += __half2float(((half*)peer_staging[r])[my_offset + i]);
        output[i] = __float2half(acc);
    }
}

// BF16 ReduceScatter with FP32 accumulation
template<int WORLD_SIZE>
__global__ void infccl_reduce_scatter_bf16_kernel(
    uint16_t* __restrict__      output,
    const uint16_t* __restrict__ input,
    void** __restrict__         peer_staging,
    volatile uint32_t** __restrict__ peer_barriers,
    const int                   rank,
    const int64_t               total_count,
    const uint32_t              gen)
{
    const int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x;
    const int64_t chunk_size = total_count / WORLD_SIZE;

    uint16_t* my_buf = (uint16_t*)peer_staging[rank];
    const int64_t count_v = total_count / 2;
    for (int64_t i = tid; i < count_v; i += stride)
        ((uint32_t*)my_buf)[i] = ((const uint32_t*)input)[i];
    if ((total_count & 1) && tid == 0)
        my_buf[total_count - 1] = input[total_count - 1];

    __threadfence();  // device-level fence (ivcore10 compat)
    infccl_barrier_signal_wait(peer_barriers, rank, WORLD_SIZE, gen);

    const int64_t my_offset = (int64_t)rank * chunk_size;
    for (int64_t i = tid; i < chunk_size; i += stride) {
        float acc = bf16_to_float(((uint16_t*)peer_staging[0])[my_offset + i]);
        #pragma unroll
        for (int r = 1; r < WORLD_SIZE; r++)
            acc += bf16_to_float(((uint16_t*)peer_staging[r])[my_offset + i]);
        output[i] = float_to_bf16(acc);
    }
}

// ============================================================================
// Host-callable C interface
// ============================================================================

extern "C" {

// Communicator lifecycle — reuses same memory layout as allreduce comm
InfcclCollComm* infccl_coll_comm_create(int rank, int world_size, int64_t max_size) {
    InfcclCollComm* comm = (InfcclCollComm*)calloc(1, sizeof(InfcclCollComm));
    if (!comm) return NULL;

    comm->rank = rank;
    comm->world_size = world_size;
    comm->max_size = max_size;
    comm->generation = 1;

    cudaSetDevice(rank);

    void* local_buf = NULL;
    cudaMalloc(&local_buf, max_size);
    cudaMemset(local_buf, 0, max_size);
    comm->staging[rank] = local_buf;

    // Barrier flags
    volatile uint32_t* flags = NULL;
    cudaMalloc((void**)&flags, sizeof(uint32_t) * INFCCL_MAX_RANKS * 2);
    cudaMemset((void*)flags, 0, sizeof(uint32_t) * INFCCL_MAX_RANKS * 2);
    comm->barriers[rank] = flags;

    cudaMalloc(&comm->d_staging, sizeof(void*) * INFCCL_MAX_RANKS);
    cudaMalloc(&comm->d_barriers, sizeof(volatile uint32_t*) * INFCCL_MAX_RANKS);

    return comm;
}

int infccl_coll_comm_get_ipc(InfcclCollComm* comm,
                              cudaIpcMemHandle_t* buf_handle,
                              cudaIpcMemHandle_t* barrier_handle) {
    cudaSetDevice(comm->rank);
    cudaError_t e1 = cudaIpcGetMemHandle(buf_handle, comm->staging[comm->rank]);
    cudaError_t e2 = cudaIpcGetMemHandle(barrier_handle, (void*)comm->barriers[comm->rank]);
    return (e1 == cudaSuccess && e2 == cudaSuccess) ? 0 : -1;
}

int infccl_coll_register_peer(InfcclCollComm* comm, int peer_rank,
                               cudaIpcMemHandle_t* buf_handle,
                               cudaIpcMemHandle_t* barrier_handle) {
    cudaSetDevice(comm->rank);
    void* peer_buf = NULL;
    void* peer_barrier = NULL;

    cudaError_t e1 = cudaIpcOpenMemHandle(&peer_buf, *buf_handle,
                                           cudaIpcMemLazyEnablePeerAccess);
    cudaError_t e2 = cudaIpcOpenMemHandle(&peer_barrier, *barrier_handle,
                                           cudaIpcMemLazyEnablePeerAccess);
    if (e1 != cudaSuccess || e2 != cudaSuccess) {
        fprintf(stderr, "[infccl_coll] IPC open failed for peer %d: %s / %s\n",
                peer_rank, cudaGetErrorString(e1), cudaGetErrorString(e2));
        return -1;
    }

    comm->staging[peer_rank] = peer_buf;
    comm->barriers[peer_rank] = (volatile uint32_t*)peer_barrier;
    return 0;
}

int infccl_coll_finalize(InfcclCollComm* comm) {
    cudaSetDevice(comm->rank);
    cudaMemcpy(comm->d_staging, comm->staging,
               sizeof(void*) * comm->world_size, cudaMemcpyHostToDevice);
    cudaMemcpy(comm->d_barriers, comm->barriers,
               sizeof(volatile uint32_t*) * comm->world_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    return 0;
}

static int compute_grid(int64_t count) {
    int grid = (int)((count + INFCCL_BLOCK_SIZE - 1) / INFCCL_BLOCK_SIZE);
    if (grid > INFCCL_MAX_GRID) grid = INFCCL_MAX_GRID;
    if (grid < 1) grid = 1;
    return grid;
}

// ---- AllGather ----
// Uses two-phase launch: (1) copy to staging, (2) barrier + gather
// This ensures P2P writes are visible across GPUs on PCIe (BI-V100)
// without __threadfence_system which is unavailable on ivcore10.
int infccl_allgather(InfcclCollComm* comm,
                     void* output, const void* input,
                     int64_t send_count, int dtype,
                     cudaStream_t stream) {
    const int rank = comm->rank;
    const int ws = comm->world_size;

    if (ws == 1) {
        int elem_size = (dtype == 0) ? 4 : 2;
        cudaMemcpyAsync(output, input, send_count * elem_size,
                        cudaMemcpyDeviceToDevice, stream);
        return 0;
    }

    const uint32_t gen = comm->generation++;
    int grid = compute_grid(send_count);

    // Phase 1: Copy input to this rank's staging buffer
    void* my_staging = comm->staging[rank];
    if (dtype == 0) {
        infccl_copy_to_staging<float><<<grid, INFCCL_BLOCK_SIZE, 0, stream>>>(
            (float*)my_staging, (const float*)input, send_count);
    } else if (dtype == 1) {
        infccl_copy_to_staging<half><<<grid, INFCCL_BLOCK_SIZE, 0, stream>>>(
            (half*)my_staging, (const half*)input, send_count);
    } else {
        infccl_copy_to_staging<uint16_t><<<grid, INFCCL_BLOCK_SIZE, 0, stream>>>(
            (uint16_t*)my_staging, (const uint16_t*)input, send_count);
    }

    // Synchronize: ensures P2P staging writes are complete before barrier
    cudaStreamSynchronize(stream);

    // Phase 2: Barrier + gather from all peers
    #define LAUNCH_GATHER(T, WS) \
        infccl_gather_from_staging<T, WS><<<grid, INFCCL_BLOCK_SIZE, 0, stream>>>( \
            (T*)output, comm->d_staging, comm->d_barriers, rank, send_count, gen)

    if (dtype == 0) {
        if (ws == 2) LAUNCH_GATHER(float, 2);
        else if (ws == 4) LAUNCH_GATHER(float, 4);
        else LAUNCH_GATHER(float, 8);
    } else if (dtype == 1) {
        if (ws == 2) LAUNCH_GATHER(half, 2);
        else if (ws == 4) LAUNCH_GATHER(half, 4);
        else LAUNCH_GATHER(half, 8);
    } else if (dtype == 2) {
        if (ws == 2) LAUNCH_GATHER(uint16_t, 2);
        else if (ws == 4) LAUNCH_GATHER(uint16_t, 4);
        else LAUNCH_GATHER(uint16_t, 8);
    }

    #undef LAUNCH_GATHER
    return 0;
}

// ---- Broadcast ----
int infccl_broadcast(InfcclCollComm* comm,
                     void* output, const void* input,
                     int64_t count, int dtype, int root,
                     cudaStream_t stream) {
    const int rank = comm->rank;
    const int ws = comm->world_size;

    if (ws == 1) {
        int elem_size = (dtype == 0) ? 4 : 2;
        cudaMemcpyAsync(output, input, count * elem_size,
                        cudaMemcpyDeviceToDevice, stream);
        return 0;
    }

    const uint32_t gen = comm->generation++;
    int grid = compute_grid(count);

    #define LAUNCH_BCAST(T, WS) \
        infccl_broadcast_kernel<T, WS><<<grid, INFCCL_BLOCK_SIZE, 0, stream>>>( \
            (T*)output, (const T*)input, \
            comm->d_staging, comm->d_barriers, rank, root, count, gen)

    if (dtype == 0) {
        if (ws == 2) LAUNCH_BCAST(float, 2);
        else if (ws == 4) LAUNCH_BCAST(float, 4);
        else LAUNCH_BCAST(float, 8);
    } else if (dtype == 1) {
        if (ws == 2) LAUNCH_BCAST(half, 2);
        else if (ws == 4) LAUNCH_BCAST(half, 4);
        else LAUNCH_BCAST(half, 8);
    } else if (dtype == 2) {
        if (ws == 2) LAUNCH_BCAST(uint16_t, 2);
        else if (ws == 4) LAUNCH_BCAST(uint16_t, 4);
        else LAUNCH_BCAST(uint16_t, 8);
    }

    #undef LAUNCH_BCAST
    return 0;
}

// ---- Reduce ----
int infccl_reduce(InfcclCollComm* comm,
                  void* output, const void* input,
                  int64_t count, int dtype, int root,
                  cudaStream_t stream) {
    const int rank = comm->rank;
    const int ws = comm->world_size;

    if (ws == 1) {
        int elem_size = (dtype == 0) ? 4 : 2;
        cudaMemcpyAsync(output, input, count * elem_size,
                        cudaMemcpyDeviceToDevice, stream);
        return 0;
    }

    const uint32_t gen = comm->generation++;
    int grid = compute_grid(count);

    #define LAUNCH_REDUCE_GENERIC(T, WS) \
        infccl_reduce_kernel<T, WS><<<grid, INFCCL_BLOCK_SIZE, 0, stream>>>( \
            (T*)output, (const T*)input, \
            comm->d_staging, comm->d_barriers, rank, root, count, gen)

    #define LAUNCH_REDUCE_FP16(WS) \
        infccl_reduce_fp16_kernel<WS><<<grid, INFCCL_BLOCK_SIZE, 0, stream>>>( \
            (half*)output, (const half*)input, \
            comm->d_staging, comm->d_barriers, rank, root, count, gen)

    #define LAUNCH_REDUCE_BF16(WS) \
        infccl_reduce_bf16_kernel<WS><<<grid, INFCCL_BLOCK_SIZE, 0, stream>>>( \
            (uint16_t*)output, (const uint16_t*)input, \
            comm->d_staging, comm->d_barriers, rank, root, count, gen)

    if (dtype == 0) {
        if (ws == 2) LAUNCH_REDUCE_GENERIC(float, 2);
        else if (ws == 4) LAUNCH_REDUCE_GENERIC(float, 4);
        else LAUNCH_REDUCE_GENERIC(float, 8);
    } else if (dtype == 1) {
        if (ws == 2) LAUNCH_REDUCE_FP16(2);
        else if (ws == 4) LAUNCH_REDUCE_FP16(4);
        else LAUNCH_REDUCE_FP16(8);
    } else if (dtype == 2) {
        if (ws == 2) LAUNCH_REDUCE_BF16(2);
        else if (ws == 4) LAUNCH_REDUCE_BF16(4);
        else LAUNCH_REDUCE_BF16(8);
    }

    #undef LAUNCH_REDUCE_GENERIC
    #undef LAUNCH_REDUCE_FP16
    #undef LAUNCH_REDUCE_BF16
    return 0;
}

// ---- ReduceScatter ----
// Uses two-phase launch like AllGather for P2P ordering.
int infccl_reduce_scatter(InfcclCollComm* comm,
                          void* output, const void* input,
                          int64_t total_count, int dtype,
                          cudaStream_t stream) {
    const int rank = comm->rank;
    const int ws = comm->world_size;

    if (ws == 1) {
        int elem_size = (dtype == 0) ? 4 : 2;
        cudaMemcpyAsync(output, input, total_count * elem_size,
                        cudaMemcpyDeviceToDevice, stream);
        return 0;
    }

    const uint32_t gen = comm->generation++;
    const int64_t chunk_size = total_count / ws;
    int grid = compute_grid(chunk_size);

    // Phase 1: Copy full input to staging
    void* my_staging = comm->staging[rank];
    int copy_grid = compute_grid(total_count);
    if (dtype == 0) {
        infccl_copy_to_staging<float><<<copy_grid, INFCCL_BLOCK_SIZE, 0, stream>>>(
            (float*)my_staging, (const float*)input, total_count);
    } else if (dtype == 1) {
        infccl_copy_to_staging<half><<<copy_grid, INFCCL_BLOCK_SIZE, 0, stream>>>(
            (half*)my_staging, (const half*)input, total_count);
    } else {
        infccl_copy_to_staging<uint16_t><<<copy_grid, INFCCL_BLOCK_SIZE, 0, stream>>>(
            (uint16_t*)my_staging, (const uint16_t*)input, total_count);
    }

    // Synchronize: ensures P2P staging writes complete
    cudaStreamSynchronize(stream);

    // Phase 2: Barrier + scatter-reduce
    #define LAUNCH_SR(T, WS) \
        infccl_scatter_reduce_from_staging<T, WS><<<grid, INFCCL_BLOCK_SIZE, 0, stream>>>( \
            (T*)output, comm->d_staging, comm->d_barriers, rank, total_count, gen)

    if (dtype == 0) {
        if (ws == 2) LAUNCH_SR(float, 2);
        else if (ws == 4) LAUNCH_SR(float, 4);
        else LAUNCH_SR(float, 8);
    } else if (dtype == 1) {
        if (ws == 2) LAUNCH_SR(half, 2);
        else if (ws == 4) LAUNCH_SR(half, 4);
        else LAUNCH_SR(half, 8);
    } else if (dtype == 2) {
        if (ws == 2) LAUNCH_SR(uint16_t, 2);
        else if (ws == 4) LAUNCH_SR(uint16_t, 4);
        else LAUNCH_SR(uint16_t, 8);
    }

    #undef LAUNCH_SR
    return 0;
}

// ---- Destroy ----
void infccl_coll_comm_destroy(InfcclCollComm* comm) {
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

    if (comm->d_staging) cudaFree(comm->d_staging);
    if (comm->d_barriers) cudaFree((void*)comm->d_barriers);

    free(comm);
}

// ---- P2P Init (reuse from allreduce) ----
int infccl_coll_init_p2p(int num_gpus) {
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
                    fprintf(stderr, "[infccl_coll] GPU %d->%d P2P enable failed: %s\n",
                            i, j, cudaGetErrorString(err));
                    failures++;
                }
            } else {
                fprintf(stderr, "[infccl_coll] GPU %d cannot P2P access GPU %d\n", i, j);
                failures++;
            }
        }
    }
    return failures ? -1 : 0;
}

}  // extern "C"

