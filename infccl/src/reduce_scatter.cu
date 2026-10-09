#include "core.h"
#include "reduce_kernel.h"
#include "enqueue.h"

template<typename T>
struct ReduceScatterArgs {
    void** sendbufs;
    void** recvbufs;
    int recvcount;
    int nDev;
    int rootDev;
    int* devs;
    void* staged;
};

template<typename T, class FUNC>
static infcclResult_t stagedReduceScatter(ReduceScatterArgs<T> args, cudaStream_t stream) {
    int ndev = args.nDev;
    int rootDev = args.rootDev;
    size_t chunkBytes = args.recvcount * sizeof(T);
    size_t totalBytes = ndev * chunkBytes;
    int totalCount = ndev * args.recvcount;

    T* full = NULL;
    CUDACHECK(cudaMalloc(&full, totalBytes));
    CUDACHECK(cudaMemcpy(full, args.sendbufs[0], totalBytes, cudaMemcpyDeviceToDevice));

    for (int p = 1; p < ndev; p++) {
        CUDACHECK(cudaMemcpyPeer(args.staged, rootDev, args.sendbufs[p], args.devs[p], totalBytes));
        ReduceInplaceSimple<T, FUNC><<<INFCCL_BLOCKS(totalCount), INFCCL_THREADS, 0, stream>>>(
            (T*)full, (const T*)args.staged, totalCount);
        CUDACHECK(cudaStreamSynchronize(stream));
    }

    for (int g = 0; g < ndev; g++) {
        CUDACHECK(cudaMemcpyPeer(
            args.recvbufs[g], args.devs[g],
            (char*)full + g * chunkBytes, rootDev, chunkBytes));
    }

    cudaFree(full);
    return infcclSuccess;
}

template<class FUNC, typename T>
static infcclResult_t reduceScatterWithTypeAndFunc(void** sendbufs, void** recvbufs,
    int recvcount, infcclComm_t comm, cudaStream_t stream) {
    if (recvcount == 0) return infcclSuccess;
    int savedDev; cudaGetDevice(&savedDev);
    int rootDev = comm->devs[0];
    CUDACHECK(cudaSetDevice(rootDev));

    size_t totalBytes = (size_t)comm->nDev * recvcount * sizeof(T);
    infcclResult_t r = infcclEnsureStaged(comm, totalBytes);
    if (r != infcclSuccess) { cudaSetDevice(savedDev); return r; }

    ReduceScatterArgs<T> args;
    args.sendbufs = sendbufs; args.recvbufs = recvbufs;
    args.recvcount = recvcount; args.nDev = comm->nDev;
    args.rootDev = rootDev; args.devs = comm->devs;
    args.staged = comm->staged;

    r = stagedReduceScatter<T, FUNC>(args, stream);
    cudaSetDevice(savedDev);
    return r;
}

template<typename T>
static infcclResult_t reduceScatterWithType(void** sendbufs, void** recvbufs,
    int recvcount, infcclRedOp_t op, infcclComm_t comm, cudaStream_t stream) {
    switch (op) {
        case infcclSum:  return reduceScatterWithTypeAndFunc<FuncSum<T>,  T>(sendbufs, recvbufs, recvcount, comm, stream);
        case infcclProd: return reduceScatterWithTypeAndFunc<FuncProd<T>, T>(sendbufs, recvbufs, recvcount, comm, stream);
        case infcclMax:  return reduceScatterWithTypeAndFunc<FuncMax<T>,  T>(sendbufs, recvbufs, recvcount, comm, stream);
        case infcclMin:  return reduceScatterWithTypeAndFunc<FuncMin<T>,  T>(sendbufs, recvbufs, recvcount, comm, stream);
        default: return infcclInvalidOperation;
    }
}

infcclResult_t infcclReduceScatter(void** sendbufs, void** recvbufs,
    int recvcount, infcclDataType_t datatype, infcclRedOp_t op,
    infcclComm_t comm, cudaStream_t stream) {
    switch (datatype) {
        case infcclChar:   return reduceScatterWithType<char>(sendbufs, recvbufs, recvcount, op, comm, stream);
        case infcclInt:    return reduceScatterWithType<int>(sendbufs, recvbufs, recvcount, op, comm, stream);
        case infcclHalf:   return reduceScatterWithType<half>(sendbufs, recvbufs, recvcount, op, comm, stream);
        case infcclFloat:  return reduceScatterWithType<float>(sendbufs, recvbufs, recvcount, op, comm, stream);
        case infcclDouble: return reduceScatterWithType<double>(sendbufs, recvbufs, recvcount, op, comm, stream);
        case infcclInt64:  return reduceScatterWithType<long long>(sendbufs, recvbufs, recvcount, op, comm, stream);
        case infcclUint64: return reduceScatterWithType<unsigned long long>(sendbufs, recvbufs, recvcount, op, comm, stream);
        default: return infcclInvalidType;
    }
}
