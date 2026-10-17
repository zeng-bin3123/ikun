/*
 * common_kernel.h — GPU primitives for BI-V100
 *
 * BI-V100 specifics:
 *   - warpSize = 64 (NOT 32)
 *   - 128 KB shared memory per SM (vs 96KB on NVIDIA sm_70)
 *   - 16 SMs
 *   - No __nv_bfloat16 intrinsics (CUDA 10.2)
 *   - __threadfence_system not supported on ivcore10
 *
 * This file provides:
 *   1. BF16 ↔ FP32 conversion (bit manipulation)
 *   2. Vectorized load/store (float4, 16 bytes per thread)
 *   3. Warp-level reduction (warpSize=64, 6 shuffle steps)
 *   4. Block-level reduction via shared memory
 *   5. ReduceOrCopy kernel: the fundamental building block
 */

#ifndef INFCCL_COMMON_KERNEL_H_
#define INFCCL_COMMON_KERNEL_H_

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

/* ── BI-V100 constants ── */
#define INFCCL_WARP_SIZE    64
#define INFCCL_BLOCK_DIM   256
#define INFCCL_MAX_BLOCKS  256          /* 16 SMs × 16 resident blocks */

/* ── BF16 helpers (no __nv_bfloat16 on CUDA 10.2) ── */
__device__ __forceinline__ float bf16_to_f32(uint16_t v) {
    uint32_t u = (uint32_t)v << 16;
    float f;
    __builtin_memcpy(&f, &u, 4);
    return f;
}

__device__ __forceinline__ uint16_t f32_to_bf16(float f) {
    uint32_t u;
    __builtin_memcpy(&u, &f, 4);
    /* round-to-nearest-even */
    u += 0x7FFF + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}

/* ── dtype-generic element size ── */
__host__ __device__ __forceinline__
int infcclTypeSize(int dtype) {
    /* 0=fp16, 1=fp32, 2=bf16, 3=int32 */
    return (dtype == 1 || dtype == 3) ? 4 : 2;
}

/* ── Load one element as float (for FP32-accumulation reduce) ── */
__device__ __forceinline__
float loadAsFloat(const void* ptr, int64_t idx, int dtype) {
    switch (dtype) {
        case 0:  return __half2float(((const half*)ptr)[idx]);
        case 1:  return ((const float*)ptr)[idx];
        case 2:  return bf16_to_f32(((const uint16_t*)ptr)[idx]);
        case 3:  return (float)((const int32_t*)ptr)[idx];
        default: return 0.0f;
    }
}

/* ── Store one float back to native dtype ── */
__device__ __forceinline__
void storeFromFloat(void* ptr, int64_t idx, float val, int dtype) {
    switch (dtype) {
        case 0:  ((half*)ptr)[idx]     = __float2half(val); break;
        case 1:  ((float*)ptr)[idx]    = val;               break;
        case 2:  ((uint16_t*)ptr)[idx] = f32_to_bf16(val);  break;
        case 3:  ((int32_t*)ptr)[idx]  = (int32_t)val;      break;
    }
}

/* ── Reduction functor applied in FP32 ── */
__device__ __forceinline__
float applyOp(float a, float b, int op) {
    switch (op) {
        case 0: /* sum  */ return a + b;
        case 1: /* max  */ return fmaxf(a, b);
        case 2: /* min  */ return fminf(a, b);
        case 3: /* prod */ return a * b;
        default: return a + b;
    }
}

/* ── Warp-level reduce (warpSize=64 → 6 shuffle steps) ── */
__device__ __forceinline__
float warpReduce(float val, int op) {
    #pragma unroll
    for (int offset = INFCCL_WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFFFFFFFFFFULL, val, offset);
        val = applyOp(val, other, op);
    }
    return val;
}

/* ── Block-level reduce via shared memory ── */
__device__ __forceinline__
float blockReduce(float val, int op) {
    __shared__ float shared[INFCCL_BLOCK_DIM / INFCCL_WARP_SIZE];
    const int lane   = threadIdx.x % INFCCL_WARP_SIZE;
    const int warpId = threadIdx.x / INFCCL_WARP_SIZE;

    val = warpReduce(val, op);

    if (lane == 0) shared[warpId] = val;
    __syncthreads();

    /* first warp reduces across warps */
    const int nwarps = INFCCL_BLOCK_DIM / INFCCL_WARP_SIZE;
    val = (threadIdx.x < nwarps) ? shared[threadIdx.x] : 0.0f;
    if (warpId == 0) val = warpReduce(val, op);
    return val;
}

/*
 * ── ReduceOrCopy kernel ──
 *
 * The fundamental building block for all collectives:
 *   dst[i] = reduce(src0[i], src1[i], ..., srcN[i])
 *
 * When nsrc == 1 and op is irrelevant, this is a vectorized copy.
 * When nsrc > 1, it reduces all sources into dst with FP32 accumulation.
 *
 * Parameters:
 *   dst     — output buffer (device pointer on this rank)
 *   srcs    — array of nsrc source pointers (may be on other GPUs via IPC)
 *   nsrc    — number of sources to reduce
 *   count   — number of elements
 *   dtype   — infcclDataType_t
 *   op      — infcclRedOp_t
 */
__global__ void infcclReduceOrCopyKernel(
    void*        dst,
    const void** srcs,
    int          nsrc,
    int64_t      count,
    int          dtype,
    int          op)
{
    const int64_t tid    = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x  * blockDim.x;

    for (int64_t i = tid; i < count; i += stride) {
        float acc = loadAsFloat(srcs[0], i, dtype);
        for (int s = 1; s < nsrc; s++) {
            float v = loadAsFloat(srcs[s], i, dtype);
            acc = applyOp(acc, v, op);
        }
        storeFromFloat(dst, i, acc, dtype);
    }
}

/*
 * ── Vectorized copy kernel ──
 *
 * Copies data using float4 (16 bytes per thread) for PCIe efficiency.
 * Used after cudaMemcpyPeer stages data into the local staging buffer.
 */
__global__ void infcclCopyKernel(
    void*       dst,
    const void* src,
    int64_t     nbytes)
{
    const int64_t tid    = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x  * blockDim.x;

    /* float4 path: 16 bytes per iteration */
    const int64_t n16 = nbytes / 16;
    float4*       d4  = (float4*)dst;
    const float4* s4  = (const float4*)src;
    for (int64_t i = tid; i < n16; i += stride)
        d4[i] = s4[i];

    /* tail bytes */
    const int64_t tail_start = n16 * 16;
    char*       dc = (char*)dst;
    const char* sc = (const char*)src;
    for (int64_t i = tail_start + tid; i < nbytes; i += stride)
        dc[i] = sc[i];
}

/* ── Grid sizing helper ── */
static inline int infcclGridSize(int64_t count) {
    int blocks = (int)((count + INFCCL_BLOCK_DIM - 1) / INFCCL_BLOCK_DIM);
    return (blocks > INFCCL_MAX_BLOCKS) ? INFCCL_MAX_BLOCKS : blocks;
}

#endif /* INFCCL_COMMON_KERNEL_H_ */
