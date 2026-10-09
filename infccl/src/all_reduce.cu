#include "core.h"
#include "reduce_kernel.h"
#include "enqueue.h"

template<typename T>
struct AllReduceArgs {
    void** buffs;
    int N;
    int nDev;
    int rootDev;
    int* devs;
    void* staged;
    int chunkElems;
};

template<typename T, class FUNC>
static infcclResult_t stagedAllReduceChunked(AllReduceArgs<T> args, cudaStream_t stream) {
    int rootDev = args.rootDev;
    size_t elemSize = sizeof(T);

    for (int off = 0; off < args.N; off += args.chunkElems) {
        int cnt = (off + args.chunkElems > args.N) ? (args.N - off) : args.chunkElems;
        size_t bytes = cnt * elemSize;
        char* rootPtr = (char*)args.buffs[0] + off * elemSize;

        for (int p = 1; p < args.nDev; p++) {
            char* peerPtr = (char*)args.buffs[p] + off * elemSize;
            CUDACHECK(cudaMemcpyPeer(args.staged, rootDev, peerPtr, args.devs[p], bytes));
            ReduceInplaceSimple<T, FUNC><<<INFCCL_BLOCKS(cnt), INFCCL_THREADS, 0, stream>>>(
                (T*)rootPtr, (const T*)args.staged, cnt);
            CUDACHECK(cudaStreamSynchronize(stream));
        }

        for (int p = 1; p < args.nDev; p++) {
            char* peerPtr = (char*)args.buffs[p] + off * elemSize;
            CUDACHECK(cudaMemcpyPeer(peerPtr, args.devs[p], rootPtr, rootDev, bytes));
        }
    }
    return infcclSuccess;
}

template<class FUNC, typename T>
static infcclResult_t allReduceWithTypeAndFunc(void** buffs, int count,
    infcclComm_t comm, cudaStream_t stream) {
    if (count == 0) return infcclSuccess;
    int savedDev; cudaGetDevice(&savedDev);
    int rootDev = comm->devs[0];
    CUDACHECK(cudaSetDevice(rootDev));

    int chunkMax = INFCCL_CHUNK_ELEMS;
    size_t chunkBytes = (size_t)chunkMax * sizeof(T);
    infcclResult_t r = infcclEnsureStaged(comm, chunkBytes);
    if (r != infcclSuccess) { cudaSetDevice(savedDev); return r; }

    AllReduceArgs<T> args;
    args.buffs = buffs;
    args.N = count;
    args.nDev = comm->nDev;
    args.rootDev = rootDev;
    args.devs = comm->devs;
    args.staged = comm->staged;
    args.chunkElems = chunkMax;

    r = stagedAllReduceChunked<T, FUNC>(args, stream);
    cudaSetDevice(savedDev);
    return r;
}

template<typename T>
static infcclResult_t allReduceWithType(void** buffs, int count,
    infcclRedOp_t op, infcclComm_t comm, cudaStream_t stream) {
    switch (op) {
        case infcclSum:  return allReduceWithTypeAndFunc<FuncSum<T>,  T>(buffs, count, comm, stream);
        case infcclProd: return allReduceWithTypeAndFunc<FuncProd<T>, T>(buffs, count, comm, stream);
        case infcclMax:  return allReduceWithTypeAndFunc<FuncMax<T>,  T>(buffs, count, comm, stream);
        case infcclMin:  return allReduceWithTypeAndFunc<FuncMin<T>,  T>(buffs, count, comm, stream);
        default: return infcclInvalidOperation;
    }
}

infcclResult_t infcclAllReduce(void** buffs, int count,
    infcclDataType_t datatype, infcclRedOp_t op,
    infcclComm_t comm, cudaStream_t stream) {
    switch (datatype) {
        case infcclChar:   return allReduceWithType<char>(buffs, count, op, comm, stream);
        case infcclInt:    return allReduceWithType<int>(buffs, count, op, comm, stream);
        case infcclHalf:   return allReduceWithType<half>(buffs, count, op, comm, stream);
        case infcclFloat:  return allReduceWithType<float>(buffs, count, op, comm, stream);
        case infcclInt64:  return allReduceWithType<long long>(buffs, count, op, comm, stream);
        case infcclUint64: return allReduceWithType<unsigned long long>(buffs, count, op, comm, stream);
        default: return infcclInvalidType;
    }
}
