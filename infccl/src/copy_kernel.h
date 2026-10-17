/*
 * copy_kernel.h — Copy primitives for staged transport
 *
 * On BI-V100, all cross-GPU data movement goes through cudaMemcpyPeer
 * (host DMA). These kernels handle the local GPU-side copies:
 *   - Packing into staging buffer before cudaMemcpyPeer
 *   - Unpacking from staging buffer after cudaMemcpyPeer
 *   - Shard extraction for reduce-scatter
 */

#ifndef INFCCL_COPY_KERNEL_H_
#define INFCCL_COPY_KERNEL_H_

#include "common_kernel.h"

/*
 * Copy a shard (sub-range) from src to dst.
 * Used in reduce-scatter and all-gather to move one chunk.
 */
__global__ void infcclCopyShard(
    void*       dst,
    const void* src,
    int64_t     dst_offset_elems,
    int64_t     src_offset_elems,
    int64_t     count,
    int         elem_size)
{
    const int64_t nbytes = count * elem_size;
    const int64_t tid    = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.x  * blockDim.x;

    const char* s = (const char*)src + src_offset_elems * elem_size;
    char*       d = (char*)dst       + dst_offset_elems * elem_size;

    /* float4 vectorized path */
    const int64_t n16 = nbytes / 16;
    for (int64_t i = tid; i < n16; i += stride)
        ((float4*)d)[i] = ((const float4*)s)[i];

    const int64_t tail = n16 * 16;
    for (int64_t i = tail + tid; i < nbytes; i += stride)
        d[i] = s[i];
}

#endif /* INFCCL_COPY_KERNEL_H_ */
