#ifndef INFCCL_REDUCE_KERNEL_H_
#define INFCCL_REDUCE_KERNEL_H_

#include "common_kernel.h"

template<typename T> struct FuncSum {
    __device__ __forceinline__ T operator()(const T x, const T y) const { return x + y; }
};
template<typename T> struct FuncProd {
    __device__ __forceinline__ T operator()(const T x, const T y) const { return x * y; }
};
template<typename T> struct FuncMax {
    __device__ __forceinline__ T operator()(const T x, const T y) const { return (x < y) ? y : x; }
};
template<typename T> struct FuncMin {
    __device__ __forceinline__ T operator()(const T x, const T y) const { return (x < y) ? x : y; }
};

template<> struct FuncSum<half> {
    __device__ __forceinline__ half operator()(half x, half y) const { return __hadd(x, y); }
};
template<> struct FuncProd<half> {
    __device__ __forceinline__ half operator()(half x, half y) const { return __hmul(x, y); }
};

template<int UNROLL, int THREADS, class FUNC, typename T>
__global__ void ReduceKernel(volatile T* dst,
    const volatile T* src0, const volatile T* src1, int N) {
    ReduceOrCopy<UNROLL, THREADS, FUNC, T, false, true>(
        threadIdx.x, dst, nullptr, src0, src1, N);
}

template<int UNROLL, int THREADS, class FUNC, typename T>
__global__ void ReduceInplaceKernel(volatile T* dst,
    const volatile T* src, int N) {
    ReduceOrCopy<UNROLL, THREADS, FUNC, T, false, true>(
        threadIdx.x, dst, nullptr, dst, src, N);
}

template<int UNROLL, int THREADS, class FUNC, typename T>
__global__ void ReduceAndCopyKernel(volatile T* dst0, volatile T* dst1,
    const volatile T* src0, const volatile T* src1, int N) {
    ReduceOrCopy<UNROLL, THREADS, FUNC, T, true, true>(
        threadIdx.x, dst0, dst1, src0, src1, N);
}

template<typename T, class FUNC>
__global__ void ReduceInplaceSimple(T* dst, const T* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = FUNC()(dst[i], src[i]);
}

template<typename T>
__global__ void FillKernel(T* buf, int n, T val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = val;
}

#endif
