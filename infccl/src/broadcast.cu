#include "core.h"
#include "copy_kernel.h"
#include "enqueue.h"
#include <cuda_fp16.h>

template<typename T>
static infcclResult_t stagedBcastChunked(void** buffs, int count, int root,
    infcclComm_t comm, cudaStream_t stream) {
    size_t elemSize = sizeof(T);
    int rootDev = comm->devs[root];
    int chunkMax = INFCCL_CHUNK_ELEMS;

    for (int off = 0; off < count; off += chunkMax) {
        int cnt = (off + chunkMax > count) ? (count - off) : chunkMax;
        size_t bytes = cnt * elemSize;
        char* rootPtr = (char*)buffs[root] + off * elemSize;

        for (int p = 0; p < comm->nDev; p++) {
            if (p == root) continue;
            char* peerPtr = (char*)buffs[p] + off * elemSize;
            CUDACHECK(cudaMemcpyPeer(peerPtr, comm->devs[p], rootPtr, rootDev, bytes));
        }
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
        case infcclChar:   r = stagedBcastChunked<char>(buffs, count, root, comm, stream); break;
        case infcclInt:    r = stagedBcastChunked<int>(buffs, count, root, comm, stream); break;
        case infcclHalf:   r = stagedBcastChunked<half>(buffs, count, root, comm, stream); break;
        case infcclFloat:  r = stagedBcastChunked<float>(buffs, count, root, comm, stream); break;
        case infcclInt64:  r = stagedBcastChunked<long long>(buffs, count, root, comm, stream); break;
        case infcclUint64: r = stagedBcastChunked<unsigned long long>(buffs, count, root, comm, stream); break;
        default: return infcclInvalidType;
    }

    if (r == infcclSuccess) infcclEnqueueRecord(comm, stream);
    return r;
}
