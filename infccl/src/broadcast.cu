#include "core.h"
#include "enqueue.h"

template<typename T>
struct BroadcastArgs {
    void** buffs;
    int N;
    int nDev;
    int root;
    int rootDev;
    int* devs;
    int chunkElems;
};

template<typename T>
static infcclResult_t stagedBroadcastChunked(BroadcastArgs<T> args, cudaStream_t stream) {
    int rootDev = args.rootDev;
    int root = args.root;
    size_t elemSize = sizeof(T);

    for (int off = 0; off < args.N; off += args.chunkElems) {
        int cnt = (off + args.chunkElems > args.N) ? (args.N - off) : args.chunkElems;
        size_t bytes = cnt * elemSize;
        char* rootPtr = (char*)args.buffs[root] + off * elemSize;

        for (int p = 0; p < args.nDev; p++) {
            if (p == root) continue;
            char* peerPtr = (char*)args.buffs[p] + off * elemSize;
            CUDACHECK(cudaMemcpyPeer(peerPtr, args.devs[p], rootPtr, rootDev, bytes));
        }
    }
    return infcclSuccess;
}

template<typename T>
static infcclResult_t bcastWithType(void** buffs, int count, int root,
    infcclComm_t comm, cudaStream_t stream) {
    if (count == 0) return infcclSuccess;
    BroadcastArgs<T> args;
    args.buffs = buffs; args.N = count; args.nDev = comm->nDev;
    args.root = root; args.rootDev = comm->devs[root];
    args.devs = comm->devs; args.chunkElems = INFCCL_CHUNK_ELEMS;
    return stagedBroadcastChunked<T>(args, stream);
}

infcclResult_t infcclBcast(void** buffs, int count,
    infcclDataType_t datatype, int root,
    infcclComm_t comm, cudaStream_t stream) {
    if (root < 0 || root >= comm->nDev) return infcclInvalidRank;
    switch (datatype) {
        case infcclChar:   return bcastWithType<char>(buffs, count, root, comm, stream);
        case infcclInt:    return bcastWithType<int>(buffs, count, root, comm, stream);
        case infcclHalf:   return bcastWithType<half>(buffs, count, root, comm, stream);
        case infcclFloat:  return bcastWithType<float>(buffs, count, root, comm, stream);
        case infcclDouble: return bcastWithType<double>(buffs, count, root, comm, stream);
        case infcclInt64:  return bcastWithType<long long>(buffs, count, root, comm, stream);
        case infcclUint64: return bcastWithType<unsigned long long>(buffs, count, root, comm, stream);
        default: return infcclInvalidType;
    }
}
