#include "core.h"
#include "reduce_kernel.h"
#include "enqueue.h"
#include <cuda_fp16.h>

template<typename T, class FUNC>
static infcclResult_t ipcAllReduce(void** buffs, int count, infcclComm_t comm, cudaStream_t stream) {
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

    for (int p = 1; p < ndev; p++) {
        cudaStream_t sp = comm->ipc.streams[p];
        CUDACHECK(cudaMemcpyPeerAsync((T*)buffs[p], comm->devs[p], (T*)buffs[0], dev0, bytes, sp));
    }
    for (int p = 1; p < ndev; p++)
        CUDACHECK(cudaStreamSynchronize(comm->ipc.streams[p]));

    cudaSetDevice(savedDev);
    return infcclSuccess;
}

template<class FUNC, typename T>
static infcclResult_t allReduceDispatch(void** buffs, int count,
    infcclComm_t comm, cudaStream_t stream) {
    if (count == 0) return infcclSuccess;
    return ipcAllReduce<T, FUNC>(buffs, count, comm, stream);
}

template<typename T>
static infcclResult_t allReduceWithType(void** buffs, int count,
    infcclRedOp_t op, infcclComm_t comm, cudaStream_t stream) {
    switch (op) {
        case infcclSum:  return allReduceDispatch<FuncSum<T>,  T>(buffs, count, comm, stream);
        case infcclProd: return allReduceDispatch<FuncProd<T>, T>(buffs, count, comm, stream);
        case infcclMax:  return allReduceDispatch<FuncMax<T>,  T>(buffs, count, comm, stream);
        case infcclMin:  return allReduceDispatch<FuncMin<T>,  T>(buffs, count, comm, stream);
        default: return infcclInvalidOperation;
    }
}

infcclResult_t infcclAllReduce(void** buffs, int count,
    infcclDataType_t datatype, infcclRedOp_t op,
    infcclComm_t comm, cudaStream_t stream) {
    infcclResult_t r = infcclEnqueueCheck(comm, stream);
    if (r != infcclSuccess) return r;
    switch (datatype) {
        case infcclChar:   r = allReduceWithType<char>(buffs, count, op, comm, stream); break;
        case infcclInt:    r = allReduceWithType<int>(buffs, count, op, comm, stream); break;
        case infcclHalf:   r = allReduceWithType<half>(buffs, count, op, comm, stream); break;
        case infcclFloat:  r = allReduceWithType<float>(buffs, count, op, comm, stream); break;
        case infcclInt64:  r = allReduceWithType<long long>(buffs, count, op, comm, stream); break;
        case infcclUint64: r = allReduceWithType<unsigned long long>(buffs, count, op, comm, stream); break;
        default: return infcclInvalidType;
    }
    if (r == infcclSuccess) infcclEnqueueRecord(comm, stream);
    return r;
}
