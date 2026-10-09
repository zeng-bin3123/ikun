#include "core.h"
#include "reduce_kernel.h"
#include "enqueue.h"
#include <cuda_fp16.h>

template<typename T, class FUNC>
static infcclResult_t stagedReduceChunked(void** buffs, int count, int root,
    infcclComm_t comm, cudaStream_t stream) {
    int dev0 = comm->devs[0];
    size_t elemSize = sizeof(T);
    int savedDev; cudaGetDevice(&savedDev);
    CUDACHECK(cudaSetDevice(dev0));

    int chunkMax = INFCCL_CHUNK_ELEMS;
    infcclResult_t r = infcclEnsureStaged(comm, (size_t)chunkMax * elemSize);
    if (r != infcclSuccess) { cudaSetDevice(savedDev); return r; }

    for (int off = 0; off < count; off += chunkMax) {
        int cnt = (off + chunkMax > count) ? (count - off) : chunkMax;
        size_t bytes = cnt * elemSize;
        char* buf0Ptr = (char*)buffs[0] + off * elemSize;

        for (int p = 1; p < comm->nDev; p++) {
            char* peerPtr = (char*)buffs[p] + off * elemSize;
            CUDACHECK(cudaMemcpyPeer(comm->staged, dev0, peerPtr, comm->devs[p], bytes));
            ReduceInplaceKernel<INFCCL_UNROLL, INFCCL_THREADS, FUNC, T>
                <<<1, INFCCL_THREADS, 0, stream>>>((volatile T*)buf0Ptr, (const volatile T*)comm->staged, cnt);
            CUDACHECK(cudaStreamSynchronize(stream));
        }

        if (root != 0) {
            char* rootPtr = (char*)buffs[root] + off * elemSize;
            CUDACHECK(cudaMemcpyPeer(rootPtr, comm->devs[root], buf0Ptr, dev0, bytes));
        }
    }

    cudaSetDevice(savedDev);
    return infcclSuccess;
}

template<typename T>
static infcclResult_t reduceWithType(void** buffs, int count,
    infcclRedOp_t op, int root, infcclComm_t comm, cudaStream_t stream) {
    switch (op) {
        case infcclSum:  return stagedReduceChunked<T, FuncSum<T>>(buffs, count, root, comm, stream);
        case infcclProd: return stagedReduceChunked<T, FuncProd<T>>(buffs, count, root, comm, stream);
        case infcclMax:  return stagedReduceChunked<T, FuncMax<T>>(buffs, count, root, comm, stream);
        case infcclMin:  return stagedReduceChunked<T, FuncMin<T>>(buffs, count, root, comm, stream);
        default: return infcclInvalidOperation;
    }
}

infcclResult_t infcclReduce(void** buffs, int count,
    infcclDataType_t datatype, infcclRedOp_t op, int root,
    infcclComm_t comm, cudaStream_t stream) {
    if (root < 0 || root >= comm->nDev) return infcclInvalidRank;

    infcclResult_t r = infcclEnqueueCheck(comm, stream);
    if (r != infcclSuccess) return r;

    switch (datatype) {
        case infcclChar:   r = reduceWithType<char>(buffs, count, op, root, comm, stream); break;
        case infcclInt:    r = reduceWithType<int>(buffs, count, op, root, comm, stream); break;
        case infcclHalf:   r = reduceWithType<half>(buffs, count, op, root, comm, stream); break;
        case infcclFloat:  r = reduceWithType<float>(buffs, count, op, root, comm, stream); break;
        case infcclInt64:  r = reduceWithType<long long>(buffs, count, op, root, comm, stream); break;
        case infcclUint64: r = reduceWithType<unsigned long long>(buffs, count, op, root, comm, stream); break;
        default: return infcclInvalidType;
    }

    if (r == infcclSuccess) infcclEnqueueRecord(comm, stream);
    return r;
}
