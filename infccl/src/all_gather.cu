#include "core.h"
#include <cuda_fp16.h>
#include "enqueue.h"

template<typename T>
struct AllGatherArgs {
    void** sendbufs;
    void** recvbufs;
    int sendcount;
    int nDev;
    int rootDev;
    int* devs;
    void* staged;
};

template<typename T>
static infcclResult_t stagedAllGather(AllGatherArgs<T> args, cudaStream_t stream) {
    int ndev = args.nDev;
    int rootDev = args.rootDev;
    size_t chunkBytes = args.sendcount * sizeof(T);
    size_t totalBytes = ndev * chunkBytes;

    for (int g = 0; g < ndev; g++) {
        CUDACHECK(cudaMemcpyPeer(
            (char*)args.staged + g * chunkBytes, rootDev,
            args.sendbufs[g], args.devs[g], chunkBytes));
    }

    for (int g = 0; g < ndev; g++) {
        CUDACHECK(cudaMemcpyPeer(
            args.recvbufs[g], args.devs[g],
            args.staged, rootDev, totalBytes));
    }
    return infcclSuccess;
}

template<typename T>
static infcclResult_t allGatherWithType(void** sendbufs, void** recvbufs,
    int sendcount, infcclComm_t comm, cudaStream_t stream) {
    if (sendcount == 0) return infcclSuccess;
    int savedDev; cudaGetDevice(&savedDev);
    int rootDev = comm->devs[0];
    CUDACHECK(cudaSetDevice(rootDev));

    size_t totalBytes = (size_t)comm->nDev * sendcount * sizeof(T);
    infcclResult_t r = infcclEnsureStaged(comm, totalBytes);
    if (r != infcclSuccess) { cudaSetDevice(savedDev); return r; }

    AllGatherArgs<T> args;
    args.sendbufs = sendbufs; args.recvbufs = recvbufs;
    args.sendcount = sendcount; args.nDev = comm->nDev;
    args.rootDev = rootDev; args.devs = comm->devs;
    args.staged = comm->staged;

    r = stagedAllGather<T>(args, stream);
    cudaSetDevice(savedDev);
    return r;
}

infcclResult_t infcclAllGather(void** sendbufs, void** recvbufs,
    int sendcount, infcclDataType_t datatype,
    infcclComm_t comm, cudaStream_t stream) {
    switch (datatype) {
        case infcclChar:   return allGatherWithType<char>(sendbufs, recvbufs, sendcount, comm, stream);
        case infcclInt:    return allGatherWithType<int>(sendbufs, recvbufs, sendcount, comm, stream);
        case infcclHalf:   return allGatherWithType<half>(sendbufs, recvbufs, sendcount, comm, stream);
        case infcclFloat:  return allGatherWithType<float>(sendbufs, recvbufs, sendcount, comm, stream);
        case infcclInt64:  return allGatherWithType<long long>(sendbufs, recvbufs, sendcount, comm, stream);
        case infcclUint64: return allGatherWithType<unsigned long long>(sendbufs, recvbufs, sendcount, comm, stream);
        default: return infcclInvalidType;
    }
}
