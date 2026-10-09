#include "core.h"
#include "enqueue.h"
#include <cuda_fp16.h>

template<typename T>
static infcclResult_t stagedAllGatherImpl(void** sendbufs, void** recvbufs,
    int sendcount, infcclComm_t comm, cudaStream_t stream) {
    int ndev = comm->nDev;
    int rootDev = comm->devs[0];
    size_t chunkBytes = sendcount * sizeof(T);
    size_t totalBytes = ndev * chunkBytes;

    int savedDev; cudaGetDevice(&savedDev);
    CUDACHECK(cudaSetDevice(rootDev));

    infcclResult_t r = infcclEnsureStaged(comm, totalBytes);
    if (r != infcclSuccess) { cudaSetDevice(savedDev); return r; }

    for (int g = 0; g < ndev; g++) {
        CUDACHECK(cudaMemcpyPeer(
            (char*)comm->staged + g * chunkBytes, rootDev,
            sendbufs[g], comm->devs[g], chunkBytes));
    }

    for (int g = 0; g < ndev; g++) {
        CUDACHECK(cudaMemcpyPeer(
            recvbufs[g], comm->devs[g],
            comm->staged, rootDev, totalBytes));
    }

    cudaSetDevice(savedDev);
    return infcclSuccess;
}

infcclResult_t infcclAllGather(void** sendbufs, void** recvbufs,
    int sendcount, infcclDataType_t datatype,
    infcclComm_t comm, cudaStream_t stream) {
    if (sendcount == 0) return infcclSuccess;

    infcclResult_t r = infcclEnqueueCheck(comm, stream);
    if (r != infcclSuccess) return r;

    switch (datatype) {
        case infcclChar:   r = stagedAllGatherImpl<char>(sendbufs, recvbufs, sendcount, comm, stream); break;
        case infcclInt:    r = stagedAllGatherImpl<int>(sendbufs, recvbufs, sendcount, comm, stream); break;
        case infcclHalf:   r = stagedAllGatherImpl<half>(sendbufs, recvbufs, sendcount, comm, stream); break;
        case infcclFloat:  r = stagedAllGatherImpl<float>(sendbufs, recvbufs, sendcount, comm, stream); break;
        case infcclInt64:  r = stagedAllGatherImpl<long long>(sendbufs, recvbufs, sendcount, comm, stream); break;
        case infcclUint64: r = stagedAllGatherImpl<unsigned long long>(sendbufs, recvbufs, sendcount, comm, stream); break;
        default: return infcclInvalidType;
    }

    if (r == infcclSuccess) infcclEnqueueRecord(comm, stream);
    return r;
}
