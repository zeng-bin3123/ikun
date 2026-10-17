/*
 * reduce_kernel.h — Reduction op templates
 *
 * Type-dispatched reduction: sum, max, min, prod across
 * fp16, fp32, bf16, int32. All reduction happens in FP32
 * for numerical stability (critical for mixed-precision inference).
 *
 * Mirrors NCCL's reduce_kernel.h but with:
 *   - BF16 via manual bit-shift (no __nv_bfloat16 on CUDA 10.2)
 *   - warpSize=64 shuffle reduction
 */

#ifndef INFCCL_REDUCE_KERNEL_H_
#define INFCCL_REDUCE_KERNEL_H_

#include "common_kernel.h"

/*
 * In-place reduce: dst[i] = op(dst[i], src[i])
 * Both pointers are on the local GPU.
 * Uses FP32 accumulation regardless of the storage dtype.
 */
__global__ void infcclReduceInPlace(
    void*       dst,
    const void* src,
    int64_t     count,
    int         dtype,
    int         op)
{
    const int64_t tid    = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x  * blockDim.x;

    for (int64_t i = tid; i < count; i += stride) {
        float a = loadAsFloat(dst, i, dtype);
        float b = loadAsFloat(src, i, dtype);
        storeFromFloat(dst, i, applyOp(a, b, op), dtype);
    }
}

/*
 * Multi-source reduce: dst[i] = op(src0[i], src1[i], ..., srcN[i])
 * Used when all peer staging buffers are locally visible.
 */
__global__ void infcclMultiReduce(
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
        #pragma unroll 4
        for (int s = 1; s < nsrc; s++)
            acc = applyOp(acc, loadAsFloat(srcs[s], i, dtype), op);
        storeFromFloat(dst, i, acc, dtype);
    }
}

#endif /* INFCCL_REDUCE_KERNEL_H_ */
