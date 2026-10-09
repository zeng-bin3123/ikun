#ifndef INFCCL_COPY_KERNEL_H_
#define INFCCL_COPY_KERNEL_H_

#include "common_kernel.h"

template<typename T>
struct FuncPassA {
    __device__ __forceinline__ T operator()(const T x, const T y) const { return x; }
};

template<int UNROLL, int THREADS, typename T>
__global__ void CopyKernel(volatile T* __restrict__ dest,
    const volatile T* __restrict__ src, int N) {
    ReduceOrCopy<UNROLL, THREADS, FuncPassA<T>, T, false, false>(
        threadIdx.x, dest, nullptr, src, nullptr, N);
}

template<int UNROLL, int THREADS, typename T>
__global__ void DoubleCopyKernel(volatile T* __restrict__ dest0,
    volatile T* __restrict__ dest1,
    const volatile T* __restrict__ src, int N) {
    ReduceOrCopy<UNROLL, THREADS, FuncPassA<T>, T, true, false>(
        threadIdx.x, dest0, dest1, src, nullptr, N);
}

template<typename T>
__global__ void SimpleCopy(T* dst, const T* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

#endif
