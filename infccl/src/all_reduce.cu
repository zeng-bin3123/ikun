#include "core.h"
#include "reduce_kernel.h"
#include "copy_kernel.h"
#include "enqueue.h"
#include <cuda_fp16.h>

#define NUM_SUBCHUNKS 2

template<typename T>
struct AllReduceArgs {
    void** buffs;
    int N;
    int nDev;
    int* devs;

    void* recvBuf[INFCCL_MAX_DEVS];
    size_t recvBufBytes;

    int sliceSize;
    int chunkSize;
    int numChunks;
};

template<typename T>
static infcclResult_t setupArgs(AllReduceArgs<T>& args, void** buffs, int count,
    infcclComm_t comm) {
    args.buffs = buffs;
    args.N = count;
    args.nDev = comm->nDev;
    args.devs = comm->devs;

    int bufferNPerSlice = INFCCL_CHUNK_ELEMS / (NUM_SUBCHUNKS * args.nDev);
    int unrollSize = INFCCL_THREADS * INFCCL_UNROLL;
    args.sliceSize = (bufferNPerSlice / unrollSize) * unrollSize;
    if (args.sliceSize < unrollSize) args.sliceSize = unrollSize;

    int subchunkSize = args.nDev * args.sliceSize;
    args.chunkSize = NUM_SUBCHUNKS * subchunkSize;

    int remainder = args.N % args.chunkSize;
    if ((args.N > args.chunkSize) && (remainder > 0) &&
        (args.N < 5 * args.chunkSize) && (2 * remainder < args.chunkSize)) {
        args.sliceSize /= 2;
        subchunkSize = args.nDev * args.sliceSize;
        args.chunkSize = NUM_SUBCHUNKS * subchunkSize;
        args.numChunks = (args.N + args.chunkSize - 1) / args.chunkSize;
    } else {
        args.numChunks = (args.N + args.chunkSize - 1) / args.chunkSize;
    }

    size_t recvBytes = (size_t)args.sliceSize * NUM_SUBCHUNKS * sizeof(T);
    args.recvBufBytes = recvBytes;

    int savedDev; cudaGetDevice(&savedDev);
    for (int g = 0; g < args.nDev; g++) {
        CUDACHECK(cudaSetDevice(args.devs[g]));
        CUDACHECK(cudaMalloc(&args.recvBuf[g], recvBytes));
    }
    cudaSetDevice(savedDev);
    return infcclSuccess;
}

template<typename T>
static void cleanupArgs(AllReduceArgs<T>& args) {
    int savedDev; cudaGetDevice(&savedDev);
    for (int g = 0; g < args.nDev; g++) {
        cudaSetDevice(args.devs[g]);
        if (args.recvBuf[g]) cudaFree(args.recvBuf[g]);
        args.recvBuf[g] = NULL;
    }
    cudaSetDevice(savedDev);
}

static inline void getSliceSizeAndOffset(int* size, int* offset, int slice,
    int numSlices, int sliceSize, int N, int chunkOffset) {
    *offset = slice * sliceSize;
    int remaining = N - chunkOffset - *offset;
    *size = (remaining < sliceSize) ? remaining : sliceSize;
    if (*size < 0) *size = 0;
}

template<typename T, class FUNC>
static infcclResult_t ringAllReduce(AllReduceArgs<T>& args, cudaStream_t stream) {
    int nDev = args.nDev;
    size_t elemSize = sizeof(T);

    for (int chunk = 0; chunk < args.numChunks; chunk++) {
        int chunkOffset = chunk * args.chunkSize;
        int chunkRemaining = args.N - chunkOffset;
        int thisChunkSize = (chunkRemaining < args.chunkSize) ? chunkRemaining : args.chunkSize;
        int numSlices = nDev;
        int sliceN = (thisChunkSize + numSlices - 1) / numSlices;

        for (int step = 0; step < nDev - 1; step++) {
            for (int g = 0; g < nDev; g++) {
                int sendSlice = (g - step + nDev) % nDev;
                int next = (g + 1) % nDev;

                int offset, size;
                getSliceSizeAndOffset(&size, &offset, sendSlice, numSlices, sliceN, args.N, chunkOffset);
                if (size <= 0) continue;

                size_t bytes = size * elemSize;
                char* sendPtr = (char*)args.buffs[g] + (chunkOffset + offset) * elemSize;

                CUDACHECK(cudaMemcpyPeer(args.recvBuf[next], args.devs[next],
                    sendPtr, args.devs[g], bytes));
            }

            for (int g = 0; g < nDev; g++) {
                int recvSlice = (g - step - 1 + nDev) % nDev;
                int offset, size;
                getSliceSizeAndOffset(&size, &offset, recvSlice, numSlices, sliceN, args.N, chunkOffset);
                if (size <= 0) continue;

                CUDACHECK(cudaSetDevice(args.devs[g]));
                T* dst = (T*)((char*)args.buffs[g] + (chunkOffset + offset) * elemSize);
                T* src = (T*)args.recvBuf[g];
                ReduceInplaceSimple<T, FUNC><<<INFCCL_BLOCKS(size), INFCCL_THREADS, 0, stream>>>(
                    dst, src, size);
                CUDACHECK(cudaStreamSynchronize(stream));
            }
        }

        for (int step = 0; step < nDev - 1; step++) {
            for (int g = 0; g < nDev; g++) {
                int sendSlice = (g - step + 1 + nDev) % nDev;
                int next = (g + 1) % nDev;

                int offset, size;
                getSliceSizeAndOffset(&size, &offset, sendSlice, numSlices, sliceN, args.N, chunkOffset);
                if (size <= 0) continue;

                size_t bytes = size * elemSize;
                char* sendPtr = (char*)args.buffs[g] + (chunkOffset + offset) * elemSize;

                CUDACHECK(cudaMemcpyPeer(
                    (char*)args.buffs[next] + (chunkOffset + offset) * elemSize,
                    args.devs[next], sendPtr, args.devs[g], bytes));
            }
        }
    }

    return infcclSuccess;
}

template<typename T, class FUNC>
static infcclResult_t stagedAllReduce(void** buffs, int count, infcclComm_t comm, cudaStream_t stream) {
    int savedDev; cudaGetDevice(&savedDev);
    int rootDev = comm->devs[0];
    CUDACHECK(cudaSetDevice(rootDev));

    size_t chunkBytes = (size_t)INFCCL_CHUNK_ELEMS * sizeof(T);
    infcclResult_t r = infcclEnsureStaged(comm, chunkBytes);
    if (r != infcclSuccess) { cudaSetDevice(savedDev); return r; }

    int chunkMax = INFCCL_CHUNK_ELEMS;
    for (int off = 0; off < count; off += chunkMax) {
        int cnt = (off + chunkMax > count) ? (count - off) : chunkMax;
        size_t bytes = cnt * sizeof(T);
        char* rootPtr = (char*)buffs[0] + off * sizeof(T);

        for (int p = 1; p < comm->nDev; p++) {
            char* peerPtr = (char*)buffs[p] + off * sizeof(T);
            CUDACHECK(cudaMemcpyPeer(comm->staged, rootDev, peerPtr, comm->devs[p], bytes));
            ReduceInplaceSimple<T, FUNC><<<INFCCL_BLOCKS(cnt), INFCCL_THREADS, 0, stream>>>(
                (T*)rootPtr, (const T*)comm->staged, cnt);
            CUDACHECK(cudaStreamSynchronize(stream));
        }

        for (int p = 1; p < comm->nDev; p++) {
            char* peerPtr = (char*)buffs[p] + off * sizeof(T);
            CUDACHECK(cudaMemcpyPeer(peerPtr, comm->devs[p], rootPtr, rootDev, bytes));
        }
    }

    cudaSetDevice(savedDev);
    return infcclSuccess;
}

template<class FUNC, typename T>
static infcclResult_t allReduceDispatch(void** buffs, int count,
    infcclComm_t comm, cudaStream_t stream) {
    if (count == 0) return infcclSuccess;

    int crossover = comm->nDev * 1048576;

    int savedDev; cudaGetDevice(&savedDev);

    if (count > crossover && comm->nDev > 1) {
        AllReduceArgs<T> args;
        memset(&args, 0, sizeof(args));
        infcclResult_t r = setupArgs<T>(args, buffs, count, comm);
        if (r != infcclSuccess) { cudaSetDevice(savedDev); return r; }
        r = ringAllReduce<T, FUNC>(args, stream);
        cleanupArgs(args);
        cudaSetDevice(savedDev);
        return r;
    }

    infcclResult_t r = stagedAllReduce<T, FUNC>(buffs, count, comm, stream);
    cudaSetDevice(savedDev);
    return r;
}

template<typename T>
static infcclResult_t allReduceWithType(void** buffs, int count,
    infcclRedOp_t op, infcclComm_t comm, cudaStream_t stream) {
    switch (op) {
        case infcclSum:  return allReduceDispatch<FuncSum<T>,  T>(buffs, count, comm, stream);
        case infcclProd: return allReduceDispatch<FuncProd<T>, T>(buffs, count, comm, stream);
        case infcclMax:  return allReduceDispatch<FuncMax<T>,  T>(buffs, count, comm, stream);
        case infcclMin:  return allReduceDispatch<FuncMin<T>,  T>(buffs, count, comm, stream);
        default: return infcclInvalidOperation;
    }
}

infcclResult_t infcclAllReduce(void** buffs, int count,
    infcclDataType_t datatype, infcclRedOp_t op,
    infcclComm_t comm, cudaStream_t stream) {
    infcclResult_t r = infcclEnqueueCheck(comm, stream);
    if (r != infcclSuccess) return r;

    switch (datatype) {
        case infcclChar:   r = allReduceWithType<char>(buffs, count, op, comm, stream); break;
        case infcclInt:    r = allReduceWithType<int>(buffs, count, op, comm, stream); break;
        case infcclHalf:   r = allReduceWithType<half>(buffs, count, op, comm, stream); break;
        case infcclFloat:  r = allReduceWithType<float>(buffs, count, op, comm, stream); break;
        case infcclInt64:  r = allReduceWithType<long long>(buffs, count, op, comm, stream); break;
        case infcclUint64: r = allReduceWithType<unsigned long long>(buffs, count, op, comm, stream); break;
        default: return infcclInvalidType;
    }

    if (r == infcclSuccess)
        infcclEnqueueRecord(comm, stream);
    return r;
}
