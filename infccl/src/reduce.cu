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

    infcclResult_t r = infcclIpcExchangeBuffers(comm, buffs, bytes);
    if (r != infcclSuccess) { cudaSetDevice(savedDev); return r; }

    CUDACHECK(cudaSetDevice(dev0));
    r = infcclEnsureStaged(comm, bytes);
    if (r != infcclSuccess) { cudaSetDevice(savedDev); return r; }

    cudaStream_t s = comm->ipc.streams[0];

    for (int p = 1; p < ndev; p++) {
        void* src_on_0 = comm->ipc.mapped[p][0];
        CUDACHECK(cudaMemcpyAsync(comm->staged, src_on_0, bytes, cudaMemcpyDeviceToDevice, s));
        CUDACHECK(cudaStreamSynchronize(s));
        ReduceInplaceSimple<T, FUNC><<<INFCCL_BLOCKS(count), INFCCL_THREADS, 0, s>>>(
            (T*)buffs[0], (const T*)comm->staged, count);
        CUDACHECK(cudaStreamSynchronize(s));
    }

    if (root != 0) {
        void* dst_on_root = comm->ipc.mapped[0][root];
        CUDACHECK(cudaSetDevice(comm->devs[root]));
        cudaStream_t sr = comm->ipc.streams[root];
        CUDACHECK(cudaMemcpyAsync(buffs[root], dst_on_root, bytes, cudaMemcpyDeviceToDevice, sr));
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
