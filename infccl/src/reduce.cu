#include "core.h"
#include "reduce_kernel.h"
#include "enqueue.h"
#include <cuda_fp16.h>

template<typename T, class FUNC>
static infcclResult_t ipcReduce(void** buffs, int count, int root, infcclComm_t comm, cudaStream_t stream) {
    int ndev = comm->nDev;
    size_t bytes = count * sizeof(T);
    int savedDev; cudaGetDevice(&savedDev);
    int dev0 = comm->devs[0];

    CUDACHECK(cudaSetDevice(dev0));
    infcclResult_t r = infcclEnsureStaged(comm, bytes);
    if (r != infcclSuccess) { cudaSetDevice(savedDev); return r; }

    cudaStream_t s0 = comm->ipc.streams[0];

    for (int p = 1; p < ndev; p++) {
        CUDACHECK(cudaMemcpyPeerAsync((T*)comm->staged, dev0, (T*)buffs[p], comm->devs[p], bytes, s0));
        CUDACHECK(cudaStreamSynchronize(s0));
        ReduceInplaceSimple<T, FUNC><<<INFCCL_BLOCKS(count), INFCCL_THREADS, 0, s0>>>(
            (T*)buffs[0], (const T*)comm->staged, count);
        CUDACHECK(cudaStreamSynchronize(s0));
    }

    if (root != 0) {
        cudaStream_t sr = comm->ipc.streams[root];
        CUDACHECK(cudaMemcpyPeerAsync((T*)buffs[root], comm->devs[root], (T*)buffs[0], dev0, bytes, sr));
        CUDACHECK(cudaStreamSynchronize(sr));
    }

    cudaSetDevice(savedDev);
    return infcclSuccess;
}

template<typename T>
static infcclResult_t reduceWithType(void** buffs, int count,
    infcclRedOp_t op, int root, infcclComm_t comm, cudaStream_t stream) {
    switch (op) {
        case infcclSum:  return ipcReduce<T, FuncSum<T>>(buffs, count, root, comm, stream);
        case infcclProd: return ipcReduce<T, FuncProd<T>>(buffs, count, root, comm, stream);
        case infcclMax:  return ipcReduce<T, FuncMax<T>>(buffs, count, root, comm, stream);
        case infcclMin:  return ipcReduce<T, FuncMin<T>>(buffs, count, root, comm, stream);
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
