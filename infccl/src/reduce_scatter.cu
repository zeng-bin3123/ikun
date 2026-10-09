#include "core.h"
#include "reduce_kernel.h"
#include "enqueue.h"
#include <cuda_fp16.h>

template<typename T, class FUNC>
static infcclResult_t ipcReduceScatter(void** sendbufs, void** recvbufs,
    int recvcount, infcclComm_t comm, cudaStream_t stream) {
    int ndev = comm->nDev;
    size_t chunkBytes = recvcount * sizeof(T);
    size_t totalBytes = ndev * chunkBytes;
    int totalCount = ndev * recvcount;
    int savedDev; cudaGetDevice(&savedDev);
    int dev0 = comm->devs[0];

    infcclResult_t r = infcclIpcExchangeBuffers(comm, sendbufs, totalBytes);
    if (r != infcclSuccess) { cudaSetDevice(savedDev); return r; }

    CUDACHECK(cudaSetDevice(dev0));
    r = infcclEnsureStaged(comm, totalBytes);
    if (r != infcclSuccess) { cudaSetDevice(savedDev); return r; }

    T* full = NULL;
    CUDACHECK(cudaMalloc(&full, totalBytes));
    cudaStream_t s0 = comm->ipc.streams[0];
    CUDACHECK(cudaMemcpyAsync(full, sendbufs[0], totalBytes, cudaMemcpyDeviceToDevice, s0));
    CUDACHECK(cudaStreamSynchronize(s0));

    for (int p = 1; p < ndev; p++) {
        void* src_on_0 = comm->ipc.mapped[p][0];
        CUDACHECK(cudaMemcpyAsync(comm->staged, src_on_0, totalBytes, cudaMemcpyDeviceToDevice, s0));
        CUDACHECK(cudaStreamSynchronize(s0));
        ReduceInplaceSimple<T, FUNC><<<INFCCL_BLOCKS(totalCount), INFCCL_THREADS, 0, s0>>>(
            full, (const T*)comm->staged, totalCount);
        CUDACHECK(cudaStreamSynchronize(s0));
    }

    CUDACHECK(cudaMemcpyAsync(recvbufs[0], full, chunkBytes, cudaMemcpyDeviceToDevice, s0));

    infcclResult_t r2 = infcclIpcExchangeBuffers(comm, recvbufs, chunkBytes);
    if (r2 == infcclSuccess) {
        for (int g = 1; g < ndev; g++) {
            CUDACHECK(cudaSetDevice(comm->devs[g]));
            cudaStream_t sg = comm->ipc.streams[g];
            void* chunk_on_0_mapped = NULL;

            cudaIpcMemHandle_t handle;
            CUDACHECK(cudaSetDevice(dev0));
            CUDACHECK(cudaIpcGetMemHandle(&handle, (char*)full + g * chunkBytes));
            CUDACHECK(cudaSetDevice(comm->devs[g]));
            CUDACHECK(cudaMemcpyPeer(recvbufs[g], comm->devs[g], (char*)full + g * chunkBytes, dev0, chunkBytes));
        }
    }

    CUDACHECK(cudaSetDevice(dev0));
    CUDACHECK(cudaStreamSynchronize(s0));
    cudaFree(full);
    cudaSetDevice(savedDev);
    return infcclSuccess;
}

template<typename T>
static infcclResult_t reduceScatterWithType(void** sendbufs, void** recvbufs,
    int recvcount, infcclRedOp_t op, infcclComm_t comm, cudaStream_t stream) {
    switch (op) {
        case infcclSum:  return ipcReduceScatter<T, FuncSum<T>>(sendbufs, recvbufs, recvcount, comm, stream);
        case infcclProd: return ipcReduceScatter<T, FuncProd<T>>(sendbufs, recvbufs, recvcount, comm, stream);
        case infcclMax:  return ipcReduceScatter<T, FuncMax<T>>(sendbufs, recvbufs, recvcount, comm, stream);
        case infcclMin:  return ipcReduceScatter<T, FuncMin<T>>(sendbufs, recvbufs, recvcount, comm, stream);
        default: return infcclInvalidOperation;
    }
}

infcclResult_t infcclReduceScatter(void** sendbufs, void** recvbufs,
    int recvcount, infcclDataType_t datatype, infcclRedOp_t op,
    infcclComm_t comm, cudaStream_t stream) {
    if (recvcount == 0) return infcclSuccess;
    infcclResult_t r = infcclEnqueueCheck(comm, stream);
    if (r != infcclSuccess) return r;
    switch (datatype) {
        case infcclChar:   r = reduceScatterWithType<char>(sendbufs, recvbufs, recvcount, op, comm, stream); break;
        case infcclInt:    r = reduceScatterWithType<int>(sendbufs, recvbufs, recvcount, op, comm, stream); break;
        case infcclHalf:   r = reduceScatterWithType<half>(sendbufs, recvbufs, recvcount, op, comm, stream); break;
        case infcclFloat:  r = reduceScatterWithType<float>(sendbufs, recvbufs, recvcount, op, comm, stream); break;
        case infcclInt64:  r = reduceScatterWithType<long long>(sendbufs, recvbufs, recvcount, op, comm, stream); break;
        case infcclUint64: r = reduceScatterWithType<unsigned long long>(sendbufs, recvbufs, recvcount, op, comm, stream); break;
        default: return infcclInvalidType;
    }
    if (r == infcclSuccess) infcclEnqueueRecord(comm, stream);
    return r;
}
