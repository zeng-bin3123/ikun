#include "core.h"
#include "reduce_kernel.h"
#include "enqueue.h"
#include <cuda_fp16.h>

template<typename T, class FUNC>
static infcclResult_t stagedReduceScatterImpl(void** sendbufs, void** recvbufs,
    int recvcount, infcclComm_t comm, cudaStream_t stream) {
    int ndev = comm->nDev;
    int rootDev = comm->devs[0];
    size_t chunkBytes = recvcount * sizeof(T);
    size_t totalBytes = ndev * chunkBytes;
    int totalCount = ndev * recvcount;

    int savedDev; cudaGetDevice(&savedDev);
    CUDACHECK(cudaSetDevice(rootDev));

    infcclResult_t r = infcclEnsureStaged(comm, totalBytes);
    if (r != infcclSuccess) { cudaSetDevice(savedDev); return r; }

    T* full = NULL;
    CUDACHECK(cudaMalloc(&full, totalBytes));
    CUDACHECK(cudaMemcpy(full, sendbufs[0], totalBytes, cudaMemcpyDeviceToDevice));

    for (int p = 1; p < ndev; p++) {
        CUDACHECK(cudaMemcpyPeer(comm->staged, rootDev, sendbufs[p], comm->devs[p], totalBytes));
        ReduceInplaceKernel<INFCCL_UNROLL, INFCCL_THREADS, FUNC, T>
            <<<1, INFCCL_THREADS, 0, stream>>>((volatile T*)full, (const volatile T*)comm->staged, totalCount);
        CUDACHECK(cudaStreamSynchronize(stream));
    }

    for (int g = 0; g < ndev; g++) {
        CUDACHECK(cudaMemcpyPeer(
            recvbufs[g], comm->devs[g],
            (char*)full + g * chunkBytes, rootDev, chunkBytes));
    }

    cudaFree(full);
    cudaSetDevice(savedDev);
    return infcclSuccess;
}

template<typename T>
static infcclResult_t reduceScatterWithType(void** sendbufs, void** recvbufs,
    int recvcount, infcclRedOp_t op, infcclComm_t comm, cudaStream_t stream) {
    switch (op) {
        case infcclSum:  return stagedReduceScatterImpl<T, FuncSum<T>>(sendbufs, recvbufs, recvcount, comm, stream);
        case infcclProd: return stagedReduceScatterImpl<T, FuncProd<T>>(sendbufs, recvbufs, recvcount, comm, stream);
        case infcclMax:  return stagedReduceScatterImpl<T, FuncMax<T>>(sendbufs, recvbufs, recvcount, comm, stream);
        case infcclMin:  return stagedReduceScatterImpl<T, FuncMin<T>>(sendbufs, recvbufs, recvcount, comm, stream);
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
