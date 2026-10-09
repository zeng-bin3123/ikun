#include "core.h"
#include "reduce_kernel.h"
#include "enqueue.h"

template<typename T>
struct ReduceArgs {
    void** buffs;
    int N;
    int nDev;
    int root;
    int rootDev;
    int* devs;
    void* staged;
    int chunkElems;
};

template<typename T, class FUNC>
static infcclResult_t stagedReduceChunked(ReduceArgs<T> args, cudaStream_t stream) {
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
            CUDACHECK(cudaMemcpyPeer(args.staged, rootDev, peerPtr, args.devs[p], bytes));
            ReduceInplaceSimple<T, FUNC><<<INFCCL_BLOCKS(cnt), INFCCL_THREADS, 0, stream>>>(
                (T*)rootPtr, (const T*)args.staged, cnt);
            CUDACHECK(cudaStreamSynchronize(stream));
        }
    }
    return infcclSuccess;
}

template<class FUNC, typename T>
static infcclResult_t reduceWithTypeAndFunc(void** buffs, int count, int root,
    infcclComm_t comm, cudaStream_t stream) {
    if (count == 0) return infcclSuccess;
    int savedDev; cudaGetDevice(&savedDev);
    int rootDev = comm->devs[root];
    CUDACHECK(cudaSetDevice(rootDev));

    int chunkMax = INFCCL_CHUNK_ELEMS;
    infcclResult_t r = infcclEnsureStaged(comm, (size_t)chunkMax * sizeof(T));
    if (r != infcclSuccess) { cudaSetDevice(savedDev); return r; }

    ReduceArgs<T> args;
    args.buffs = buffs; args.N = count; args.nDev = comm->nDev;
    args.root = root; args.rootDev = rootDev;
    args.devs = comm->devs; args.staged = comm->staged;
    args.chunkElems = chunkMax;

    r = stagedReduceChunked<T, FUNC>(args, stream);
    cudaSetDevice(savedDev);
    return r;
}

template<typename T>
static infcclResult_t reduceWithType(void** buffs, int count,
    infcclRedOp_t op, int root, infcclComm_t comm, cudaStream_t stream) {
    switch (op) {
        case infcclSum:  return reduceWithTypeAndFunc<FuncSum<T>,  T>(buffs, count, root, comm, stream);
        case infcclProd: return reduceWithTypeAndFunc<FuncProd<T>, T>(buffs, count, root, comm, stream);
        case infcclMax:  return reduceWithTypeAndFunc<FuncMax<T>,  T>(buffs, count, root, comm, stream);
        case infcclMin:  return reduceWithTypeAndFunc<FuncMin<T>,  T>(buffs, count, root, comm, stream);
        default: return infcclInvalidOperation;
    }
}

infcclResult_t infcclReduce(void** buffs, int count,
    infcclDataType_t datatype, infcclRedOp_t op, int root,
    infcclComm_t comm, cudaStream_t stream) {
    if (root < 0 || root >= comm->nDev) return infcclInvalidRank;
    switch (datatype) {
        case infcclChar:   return reduceWithType<char>(buffs, count, op, root, comm, stream);
        case infcclInt:    return reduceWithType<int>(buffs, count, op, root, comm, stream);
        case infcclHalf:   return reduceWithType<half>(buffs, count, op, root, comm, stream);
        case infcclFloat:  return reduceWithType<float>(buffs, count, op, root, comm, stream);
        case infcclDouble: return reduceWithType<double>(buffs, count, op, root, comm, stream);
        case infcclInt64:  return reduceWithType<long long>(buffs, count, op, root, comm, stream);
        case infcclUint64: return reduceWithType<unsigned long long>(buffs, count, op, root, comm, stream);
        default: return infcclInvalidType;
    }
}
