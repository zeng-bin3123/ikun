#ifndef INFCCL_COMMON_KERNEL_H_
#define INFCCL_COMMON_KERNEL_H_

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define INFCCL_WARP_SIZE_K 64
#define INFCCL_THREADS 256
#define INFCCL_UNROLL 8
#define INFCCL_BLOCKS(n) (((n) + INFCCL_THREADS - 1) / INFCCL_THREADS)
#define INFCCL_ROUNDUP(x, y) (((((x) + (y) - 1) / (y))) * (y))

typedef uint64_t PackType;

template<typename T> inline __device__
T vFetch(const volatile T* ptr) { return *ptr; }

template<typename T> inline __device__
void vStore(volatile T* ptr, const T val) { *ptr = val; }

#define ALIGNUP(x, a) ((((x)-1) & ~((a)-1)) + (a))

template<typename T>
__device__ inline volatile T* AlignUp(volatile T* ptr, size_t align) {
    size_t ptrval = reinterpret_cast<size_t>(ptr);
    return reinterpret_cast<volatile T*>(ALIGNUP(ptrval, align));
}

template<class FUNC, typename T>
struct MultiOp {
    __device__ __forceinline__ PackType operator()(PackType a, PackType b) const;
};

template<class FUNC>
struct MultiOp<FUNC, float> {
    __device__ __forceinline__ PackType operator()(PackType a, PackType b) const {
        union { PackType p; float f[2]; } ua, ub, ur;
        ua.p = a; ub.p = b;
        ur.f[0] = FUNC()(ua.f[0], ub.f[0]);
        ur.f[1] = FUNC()(ua.f[1], ub.f[1]);
        return ur.p;
    }
};

template<class FUNC>
struct MultiOp<FUNC, half> {
    __device__ __forceinline__ PackType operator()(PackType a, PackType b) const {
        union { PackType p; half h[4]; } ua, ub, ur;
        ua.p = a; ub.p = b;
        ur.h[0] = FUNC()(ua.h[0], ub.h[0]);
        ur.h[1] = FUNC()(ua.h[1], ub.h[1]);
        ur.h[2] = FUNC()(ua.h[2], ub.h[2]);
        ur.h[3] = FUNC()(ua.h[3], ub.h[3]);
        return ur.p;
    }
};

template<class FUNC>
struct MultiOp<FUNC, int> {
    __device__ __forceinline__ PackType operator()(PackType a, PackType b) const {
        union { PackType p; int i[2]; } ua, ub, ur;
        ua.p = a; ub.p = b;
        ur.i[0] = FUNC()(ua.i[0], ub.i[0]);
        ur.i[1] = FUNC()(ua.i[1], ub.i[1]);
        return ur.p;
    }
};

template<class FUNC>
struct MultiOp<FUNC, double> {
    __device__ __forceinline__ PackType operator()(PackType a, PackType b) const {
        union { PackType p; double d; } ua, ub, ur;
        ua.p = a; ub.p = b;
        ur.d = FUNC()(ua.d, ub.d);
        return ur.p;
    }
};

template<class FUNC>
struct MultiOp<FUNC, long long> {
    __device__ __forceinline__ PackType operator()(PackType a, PackType b) const {
        union { PackType p; long long l; } ua, ub, ur;
        ua.p = a; ub.p = b;
        ur.l = FUNC()(ua.l, ub.l);
        return ur.p;
    }
};

template<class FUNC>
struct MultiOp<FUNC, unsigned long long> {
    __device__ __forceinline__ PackType operator()(PackType a, PackType b) const {
        union { PackType p; unsigned long long l; } ua, ub, ur;
        ua.p = a; ub.p = b;
        ur.l = FUNC()(ua.l, ub.l);
        return ur.p;
    }
};

template<int UNROLL, int THREADS, class FUNC, typename T, bool HAS_DEST1, bool HAS_SRC1>
__device__ inline void ReduceOrCopy(const int tid,
    volatile T* __restrict__ dest0, volatile T* __restrict__ dest1,
    const volatile T* __restrict__ src0, const volatile T* __restrict__ src1,
    int N) {
    if (N == 0) return;

    int Npreamble = AlignUp(dest0, alignof(PackType)) - dest0;
    bool alignable = ((AlignUp(src0, alignof(PackType)) == src0 + Npreamble) &&
        (!HAS_DEST1 || (AlignUp(dest1, alignof(PackType)) == dest1 + Npreamble)) &&
        (!HAS_SRC1  || (AlignUp(src1, alignof(PackType)) == src1  + Npreamble)));
    if (!alignable) Npreamble = N;

    for (int idx = tid; idx < Npreamble; idx += THREADS) {
        T val = vFetch(src0 + idx);
        if (HAS_SRC1) val = FUNC()(val, vFetch(src1 + idx));
        vStore(dest0 + idx, val);
        if (HAS_DEST1) vStore(dest1 + idx, val);
    }

    int Ndone = Npreamble;
    int Nrem = N - Ndone;

    if (alignable && Nrem > 0) {
        const volatile T* s0 = src0 + Ndone;
        const volatile T* s1 = HAS_SRC1 ? src1 + Ndone : nullptr;
        volatile T* d0 = dest0 + Ndone;
        volatile T* d1 = HAS_DEST1 ? dest1 + Ndone : nullptr;

        int elemsPerPack = sizeof(PackType) / sizeof(T);
        int Nalign = Nrem / elemsPerPack;

        #pragma unroll 4
        for (int idx = tid; idx < Nalign; idx += THREADS) {
            PackType p0 = ((const volatile PackType*)s0)[idx];
            PackType p1;
            if (HAS_SRC1) p1 = ((const volatile PackType*)s1)[idx];
            PackType pr = HAS_SRC1 ? MultiOp<FUNC, T>()(p0, p1) : p0;
            ((volatile PackType*)d0)[idx] = pr;
            if (HAS_DEST1) ((volatile PackType*)d1)[idx] = pr;
        }

        int Ndone2 = Nalign * elemsPerPack;
        int tailStart = Ndone + Ndone2;
        int tailRem = N - tailStart;
        for (int idx = tid; idx < tailRem; idx += THREADS) {
            T val = vFetch(src0 + tailStart + idx);
            if (HAS_SRC1) val = FUNC()(val, vFetch(src1 + tailStart + idx));
            vStore(dest0 + tailStart + idx, val);
            if (HAS_DEST1) vStore(dest1 + tailStart + idx, val);
        }
    }
}

#endif
