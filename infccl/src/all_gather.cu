#include "core.h"
#include "enqueue.h"
#include <cuda_fp16.h>

template<typename T>
static infcclResult_t ipcAllGather(void** sendbufs, void** recvbufs,
    int sendcount, infcclComm_t comm, cudaStream_t stream) {
    int ndev = comm->nDev;
    int dev0 = comm->devs[0];
    size_t chunkBytes = sendcount * sizeof(T);
    size_t totalBytes = ndev * chunkBytes;
    int savedDev; cudaGetDevice(&savedDev);

    CUDACHECK(cudaSetDevice(dev0));
    infcclResult_t r = infcclEnsureStaged(comm, totalBytes);
    if (r != infcclSuccess) { cudaSetDevice(savedDev); return r; }

    cudaStream_t s0 = comm->ipc.streams[0];
    CUDACHECK(cudaMemcpyAsync(comm->staged, sendbufs[0], chunkBytes, cudaMemcpyDeviceToDevice, s0));
    for (int g = 1; g < ndev; g++) {
        CUDACHECK(cudaMemcpyPeerAsync((char*)comm->staged + g * chunkBytes, dev0,
            sendbufs[g], comm->devs[g], chunkBytes, s0));
    }
    CUDACHECK(cudaStreamSynchronize(s0));

    CUDACHECK(cudaMemcpyAsync(recvbufs[0], comm->staged, totalBytes, cudaMemcpyDeviceToDevice, s0));
    for (int p = 1; p < ndev; p++) {
        cudaStream_t sp = comm->ipc.streams[p];
        CUDACHECK(cudaMemcpyPeerAsync(recvbufs[p], comm->devs[p], comm->staged, dev0, totalBytes, sp));
    }
    for (int g = 0; g < ndev; g++)
        CUDACHECK(cudaStreamSynchronize(comm->ipc.streams[g]));

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
        case infcclChar:   r = ipcAllGather<char>(sendbufs, recvbufs, sendcount, comm, stream); break;
        case infcclInt:    r = ipcAllGather<int>(sendbufs, recvbufs, sendcount, comm, stream); break;
        case infcclHalf:   r = ipcAllGather<half>(sendbufs, recvbufs, sendcount, comm, stream); break;
        case infcclFloat:  r = ipcAllGather<float>(sendbufs, recvbufs, sendcount, comm, stream); break;
        case infcclInt64:  r = ipcAllGather<long long>(sendbufs, recvbufs, sendcount, comm, stream); break;
        case infcclUint64: r = ipcAllGather<unsigned long long>(sendbufs, recvbufs, sendcount, comm, stream); break;
        default: return infcclInvalidType;
    }
    if (r == infcclSuccess) infcclEnqueueRecord(comm, stream);
    return r;
}
