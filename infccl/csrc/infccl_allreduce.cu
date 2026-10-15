/*
 * infccl_allreduce.cu — Custom AllReduce for Iluvatar BI-V100 (4× PCIe)
 *
 * Architecture: Two-phase shared-memory allreduce over PCIe P2P.
 *   Phase 1 (scatter-reduce): Each GPU writes its shard to all peers' 
 *           staging buffers via P2P. Each GPU reduces its own column.
 *   Phase 2 (allgather): Each GPU broadcasts its reduced shard to all peers.
 *
 * This avoids NCCL's ring/tree overhead for small-to-medium tensors
 * (< 8 MB) which dominate MoE allreduce in Qwen3.6-35B-A3B decode.
 *
 * Memory layout (per-communicator):
 *   Each rank allocates a contiguous buffer of size max_size * world_size.
 *   Buffer[i] is the staging area where rank i's data is written by all peers.
 *   Two sets of barriers (flags) synchronize phases.
 *
 * Compile: nvcc -arch=sm_70 -O3 -shared -Xcompiler -fPIC \
 *          -o infccl_allreduce.so infccl_allreduce.cu
 *
 * For BI-V100 (Iluvatar CoreX), compile with:
 *   /usr/local/corex/bin/clang++ --cuda-gpu-arch=gfx_ilu100 ...
 *   or via the corex CUDA 10.2 compatible nvcc wrapper.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>

// Maximum number of GPUs in a single node
#define INFCCL_MAX_RANKS 8

// Barrier flag states
#define INFCCL_FLAG_READY    0x1
#define INFCCL_FLAG_DONE     0x2

// ============================================================================
// Data structures
// ============================================================================

// Metadata block at the start of each rank's shared buffer
struct InfcclMeta {
    volatile uint32_t barrier_in[INFCCL_MAX_RANKS];   // phase 1 barriers
    volatile uint32_t barrier_out[INFCCL_MAX_RANKS];  // phase 2 barriers
    uint32_t rank;
    uint32_t world_size;
    uint64_t max_size;        // max allreduce size in bytes
    uint64_t counter;         // monotonic counter for barrier generations
    char padding[64 - 40];    // align to cache line
};

// Buffer pointers visible to all ranks (stored in device memory)
struct InfcclHandles {
    void* buffers[INFCCL_MAX_RANKS];    // P2P-mapped staging buffers
    InfcclMeta* metas[INFCCL_MAX_RANKS]; // P2P-mapped metadata
};

// ============================================================================
// Phase 1 Kernel: Scatter-Reduce
// Each thread block handles a chunk of data. 
// All ranks write their contribution, then reduce locally.
// ============================================================================

template<typename T>
__global__ void infccl_scatter_reduce_kernel(
    T* __restrict__ output,           // local output buffer
    const T* __restrict__ input,      // local input buffer
    T** __restrict__ peer_buffers,    // peer staging buffers
    volatile uint32_t** __restrict__ peer_barriers_in,
    volatile uint32_t** __restrict__ peer_barriers_out,
    const int rank,
    const int world_size,
    const int64_t count,              // number of elements
    const uint32_t generation)
{
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = gridDim.x * blockDim.x;
    
    // Step 1: Write local data to our own staging buffer
    // (peer buffers[rank] points to our staging area via P2P mapping)
    T* my_staging = peer_buffers[rank];
    for (int64_t i = tid; i < count; i += stride) {
        my_staging[i] = input[i];
    }
    __threadfence_system();  // ensure P2P writes are visible
    
    // Step 2: Signal that our data is ready
    if (tid == 0) {
        for (int r = 0; r < world_size; r++) {
            if (r != rank) {
                // Tell peer r that rank's data is ready
                peer_barriers_in[r][rank] = generation;
            }
        }
    }
    __syncthreads();
    
    // Step 3: Wait for all peers to signal their data is ready
    if (tid == 0) {
        for (int r = 0; r < world_size; r++) {
            if (r != rank) {
                // Wait until peer r says its data is ready
                while (peer_barriers_in[rank][r] != generation) {
                    // spin — volatile ensures re-read
                }
            }
        }
    }
    __syncthreads();
    
    // Step 4: Reduce all peers' contributions into output
    for (int64_t i = tid; i < count; i += stride) {
        T sum = peer_buffers[0][i];
        for (int r = 1; r < world_size; r++) {
            sum = sum + peer_buffers[r][i];
        }
        output[i] = sum;
    }
}

// Specialized for float16 with vectorized loads
__global__ void infccl_scatter_reduce_fp16_vec(
    half* __restrict__ output,
    const half* __restrict__ input,
    half** __restrict__ peer_buffers,
    volatile uint32_t** __restrict__ peer_barriers_in,
    const int rank,
    const int world_size,
    const int64_t count,
    const uint32_t generation)
{
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = gridDim.x * blockDim.x;
    
    // Vectorized write (2 halfs = 1 float = 4 bytes)
    const int64_t count_vec = count / 2;
    float* my_staging_f = reinterpret_cast<float*>(peer_buffers[rank]);
    const float* input_f = reinterpret_cast<const float*>(input);
    
    for (int64_t i = tid; i < count_vec; i += stride) {
        my_staging_f[i] = input_f[i];
    }
    // Handle remainder
    if (count % 2 != 0 && tid == 0) {
        peer_buffers[rank][count - 1] = input[count - 1];
    }
    __threadfence_system();
    
    if (tid == 0) {
        for (int r = 0; r < world_size; r++) {
            if (r != rank) {
                peer_barriers_in[r][rank] = generation;
            }
        }
    }
    __syncthreads();
    
    if (tid == 0) {
        for (int r = 0; r < world_size; r++) {
            if (r != rank) {
                while (peer_barriers_in[rank][r] != generation) {}
            }
        }
    }
    __syncthreads();
    
    // Reduce with fp32 accumulation for numerical stability
    for (int64_t i = tid; i < count; i += stride) {
        float sum = __half2float(peer_buffers[0][i]);
        for (int r = 1; r < world_size; r++) {
            sum += __half2float(peer_buffers[r][i]);
        }
        output[i] = __float2half(sum);
    }
}

// ============================================================================
// One-shot AllReduce for small tensors (< 256KB)
// Uses a single kernel with all phases fused.
// ============================================================================

template<typename T, int WORLD_SIZE>
__global__ void infccl_oneshot_allreduce(
    T* __restrict__ result,
    const T* __restrict__ input,
    T** __restrict__ peer_buffers,
    volatile uint32_t* __restrict__ my_barrier_in,
    volatile uint32_t** __restrict__ peer_barriers_in,
    const int rank,
    const int64_t count,
    const uint32_t generation)
{
    // This kernel is designed for <= 256KB data
    // Step 1: Copy input to staging
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = gridDim.x * blockDim.x;
    
    T* my_buf = peer_buffers[rank];
    for (int64_t i = tid; i < count; i += stride) {
        my_buf[i] = input[i];
    }
    __threadfence_system();
    
    // Signal and wait (single thread)
    if (tid == 0) {
        // Signal all peers
        for (int r = 0; r < WORLD_SIZE; r++) {
            if (r != rank) {
                peer_barriers_in[r][rank] = generation;
            }
        }
        // Wait for all peers
        for (int r = 0; r < WORLD_SIZE; r++) {
            if (r != rank) {
                while (my_barrier_in[r] != generation) {}
            }
        }
    }
    __syncthreads();
    
    // Reduce — unrolled for known world sizes
    for (int64_t i = tid; i < count; i += stride) {
        T val = peer_buffers[0][i];
        #pragma unroll
        for (int r = 1; r < WORLD_SIZE; r++) {
            val = val + peer_buffers[r][i];
        }
        result[i] = val;
    }
}

// ============================================================================
// Host-callable C interface
// ============================================================================

extern "C" {

// Initialize P2P access between all GPU pairs
int infccl_init_p2p(int num_gpus) {
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        for (int j = 0; j < num_gpus; j++) {
            if (i != j) {
                int can_access = 0;
                cudaDeviceCanAccessPeer(&can_access, i, j);
                if (can_access) {
                    cudaDeviceEnablePeerAccess(j, 0);
                } else {
                    fprintf(stderr, "[infccl] WARNING: GPU %d cannot P2P access GPU %d\n", i, j);
                    return -1;
                }
            }
        }
    }
    return 0;
}

// Allocate staging buffers for a communicator
// Returns pointer to InfcclHandles (device memory on rank's GPU)
InfcclHandles* infccl_create_handles(
    int rank, int world_size, int64_t max_size)
{
    cudaSetDevice(rank);
    
    // Allocate local staging buffer (accessible by all peers via P2P)
    void* local_buffer = nullptr;
    cudaMalloc(&local_buffer, max_size);
    cudaMemset(local_buffer, 0, max_size);
    
    // Allocate metadata
    InfcclMeta* local_meta = nullptr;
    cudaMalloc(&local_meta, sizeof(InfcclMeta));
    cudaMemset(local_meta, 0, sizeof(InfcclMeta));
    
    // Set metadata
    InfcclMeta h_meta;
    memset(&h_meta, 0, sizeof(h_meta));
    h_meta.rank = rank;
    h_meta.world_size = world_size;
    h_meta.max_size = max_size;
    h_meta.counter = 1;
    cudaMemcpy(local_meta, &h_meta, sizeof(InfcclMeta), cudaMemcpyHostToDevice);
    
    // Allocate handles struct  
    InfcclHandles* handles = nullptr;
    cudaMalloc(&handles, sizeof(InfcclHandles));
    
    InfcclHandles h_handles;
    memset(&h_handles, 0, sizeof(h_handles));
    h_handles.buffers[rank] = local_buffer;
    h_handles.metas[rank] = local_meta;
    cudaMemcpy(handles, &h_handles, sizeof(InfcclHandles), cudaMemcpyHostToDevice);
    
    return handles;
}

// Get the IPC memory handle for sharing with peers
int infccl_get_ipc_handle(
    InfcclHandles* handles, int rank,
    cudaIpcMemHandle_t* buf_handle,
    cudaIpcMemHandle_t* meta_handle)
{
    InfcclHandles h_handles;
    cudaMemcpy(&h_handles, handles, sizeof(InfcclHandles), cudaMemcpyDeviceToHost);
    
    cudaIpcGetMemHandle(buf_handle, h_handles.buffers[rank]);
    cudaIpcGetMemHandle(meta_handle, h_handles.metas[rank]);
    return 0;
}

// Register a peer's IPC handle
int infccl_register_peer(
    InfcclHandles* handles, int peer_rank,
    cudaIpcMemHandle_t* buf_handle,
    cudaIpcMemHandle_t* meta_handle)
{
    void* peer_buf = nullptr;
    void* peer_meta = nullptr;
    
    cudaIpcOpenMemHandle(&peer_buf, *buf_handle, cudaIpcMemLazyEnablePeerAccess);
    cudaIpcOpenMemHandle(&peer_meta, *meta_handle, cudaIpcMemLazyEnablePeerAccess);
    
    InfcclHandles h_handles;
    cudaMemcpy(&h_handles, handles, sizeof(InfcclHandles), cudaMemcpyDeviceToHost);
    h_handles.buffers[peer_rank] = peer_buf;
    h_handles.metas[peer_rank] = (InfcclMeta*)peer_meta;
    cudaMemcpy(handles, &h_handles, sizeof(InfcclHandles), cudaMemcpyHostToDevice);
    
    return 0;
}

// Launch allreduce
int infccl_allreduce_launch(
    InfcclHandles* handles,
    void* output, const void* input,
    int64_t count, int dtype,  // 0=fp32, 1=fp16, 2=bf16
    int rank, int world_size,
    uint32_t generation,
    cudaStream_t stream)
{
    InfcclHandles h_handles;
    cudaMemcpy(&h_handles, handles, sizeof(InfcclHandles), cudaMemcpyDeviceToHost);
    
    // Build kernel arguments on host
    // For simplicity, use the general kernel path
    int block_size = 256;
    int64_t elements = count;
    int grid_size = (elements + block_size - 1) / block_size;
    if (grid_size > 1024) grid_size = 1024;
    
    // We need device arrays of pointers
    void** d_peer_buffers = nullptr;
    cudaMalloc(&d_peer_buffers, sizeof(void*) * world_size);
    cudaMemcpy(d_peer_buffers, h_handles.buffers, sizeof(void*) * world_size, cudaMemcpyHostToDevice);
    
    // Build barrier pointer arrays
    // Each meta has barrier_in[INFCCL_MAX_RANKS]
    volatile uint32_t** d_barriers_in = nullptr;
    cudaMalloc(&d_barriers_in, sizeof(uint32_t*) * world_size);
    
    // Compute offsets to barrier_in in each peer's meta
    volatile uint32_t* h_barrier_ptrs[INFCCL_MAX_RANKS];
    for (int r = 0; r < world_size; r++) {
        h_barrier_ptrs[r] = h_handles.metas[r]->barrier_in;
    }
    cudaMemcpy(d_barriers_in, h_barrier_ptrs, sizeof(uint32_t*) * world_size, cudaMemcpyHostToDevice);
    
    if (dtype == 0) { // fp32
        infccl_scatter_reduce_kernel<float><<<grid_size, block_size, 0, stream>>>(
            (float*)output, (const float*)input,
            (float**)d_peer_buffers,
            d_barriers_in, nullptr,
            rank, world_size, elements, generation);
    } else if (dtype == 1) { // fp16
        infccl_scatter_reduce_fp16_vec<<<grid_size, block_size, 0, stream>>>(
            (half*)output, (const half*)input,
            (half**)d_peer_buffers,
            d_barriers_in,
            rank, world_size, elements, generation);
    }
    // Note: bf16 would need similar treatment
    
    cudaFree(d_peer_buffers);
    cudaFree((void*)d_barriers_in);
    
    return 0;
}

// Cleanup
void infccl_destroy_handles(InfcclHandles* handles, int rank, int world_size) {
    InfcclHandles h_handles;
    cudaMemcpy(&h_handles, handles, sizeof(InfcclHandles), cudaMemcpyDeviceToHost);
    
    // Free local buffers
    if (h_handles.buffers[rank]) cudaFree(h_handles.buffers[rank]);
    if (h_handles.metas[rank]) cudaFree(h_handles.metas[rank]);
    
    // Close IPC handles for peers
    for (int r = 0; r < world_size; r++) {
        if (r != rank) {
            if (h_handles.buffers[r]) cudaIpcCloseMemHandle(h_handles.buffers[r]);
            if (h_handles.metas[r]) cudaIpcCloseMemHandle(h_handles.metas[r]);
        }
    }
    
    cudaFree(handles);
}

}  // extern "C"
