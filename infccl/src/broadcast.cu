#include "core.h"
#include "enqueue.h"
#include <cuda_fp16.h>

template<typename T>
static infcclResult_t ipcBcast(void** buffs, int count, int root, infcclComm_t comm, cudaStream_t stream) {
    int ndev = comm->nDev;
    size_t bytes = count * sizeof(T);
    int rootDev = comm->devs[root];

    for (int p = 0; p < ndev; p++) {
        if (p == root) continue;
        cudaStream_t sp = comm->ipc.streams[p];
        CUDACHECK(cudaMemcpyPeerAsync((T*)buffs[p], comm->devs[p], (T*)buffs[root], rootDev, bytes, sp));
    }
    for (int p = 0; p < ndev; p++) {
        if (p == root) continue;
        CUDACHECK(cudaStreamSynchronize(comm->ipc.streams[p]));
    }
    return infcclSuccess;
}

infcclResult_t infcclBcast(void** buffs, int count,
    infcclDataType_t datatype, int root,
    infcclComm_t comm, cudaStream_t stream) {
    if (root < 0 || root >= comm->nDev) return infcclInvalidRank;
    if (count == 0) return infcclSuccess;
    infcclResult_t r = infcclEnqueueCheck(comm, stream);
    if (r != infcclSuccess) return r;
    switch (datatype) {
        case infcclChar:   r = ipcBcast<char>(buffs, count, root, comm, stream); break;
        case infcclInt:    r = ipcBcast<int>(buffs, count, root, comm, stream); break;
        case infcclHalf:   r = ipcBcast<half>(buffs, count, root, comm, stream); break;
        case infcclFloat:  r = ipcBcast<float>(buffs, count, root, comm, stream); break;
        case infcclInt64:  r = ipcBcast<long long>(buffs, count, root, comm, stream); break;
        case infcclUint64: r = ipcBcast<unsigned long long>(buffs, count, root, comm, stream); break;
        default: return infcclInvalidType;
    }
    if (r == infcclSuccess) infcclEnqueueRecord(comm, stream);
    return r;
}
