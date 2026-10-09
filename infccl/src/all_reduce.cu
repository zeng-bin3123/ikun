#include "core.h"
#include "reduce_kernel.h"
#include "copy_kernel.h"
#include "enqueue.h"
#include <cuda_fp16.h>

template<typename T, class FUNC>
static infcclResult_t ipcAllReduce(void** buffs, int count, infcclComm_t comm, cudaStream_t stream) {
    int ndev = comm->nDev;
    size_t bytes = count * sizeof(T);
    int savedDev; cudaGetDevice(&savedDev);

    infcclResult_t r = infcclIpcExchangeBuffers(comm, buffs, bytes);
    if (r != infcclSuccess) { cudaSetDevice(savedDev); return r; }

    CUDACHECK(cudaSetDevice(comm->devs[0]));
    r = infcclEnsureStaged(comm, bytes);
    if (r != infcclSuccess) { cudaSetDevice(savedDev); return r; }

    for (int p = 1; p < ndev; p++) {
        void* src_on_0 = comm->ipc.mapped[p][0];
        cudaStream_t s = comm->ipc.streams[0];
        CUDACHECK(cudaMemcpyAsync(comm->staged, src_on_0, bytes, cudaMemcpyDeviceToDevice, s));
        CUDACHECK(cudaStreamSynchronize(s));
        ReduceInplaceSimple<T, FUNC><<<INFCCL_BLOCKS(count), INFCCL_THREADS, 0, s>>>(
            (T*)buffs[0], (const T*)comm->staged, count);
        CUDACHECK(cudaStreamSynchronize(s));
    }

    for (int p = 1; p < ndev; p++) {
        void* dst_on_p = comm->ipc.mapped[0][p];
        CUDACHECK(cudaSetDevice(comm->devs[p]));
        cudaStream_t s = comm->ipc.streams[p];
        CUDACHECK(cudaMemcpyAsync(buffs[p], dst_on_p, bytes, cudaMemcpyDeviceToDevice, s));
    }
    for (int p = 1; p < ndev; p++) {
        CUDACHECK(cudaStreamSynchronize(comm->ipc.streams[p]));
    }

    cudaSetDevice(savedDev);
    return infcclSuccess;
}

template<typename T, class FUNC>
static infcclResult_t ipcRingAllReduce(void** buffs, int count, infcclComm_t comm, cudaStream_t stream) {
    int ndev = comm->nDev;
    size_t elemSize = sizeof(T);
    int savedDev; cudaGetDevice(&savedDev);

    infcclResult_t r = infcclIpcExchangeBuffers(comm, buffs, count * elemSize);
    if (r != infcclSuccess) { cudaSetDevice(savedDev); return r; }

    size_t sliceBytes = ((count + ndev - 1) / ndev) * elemSize;
    for (int g = 0; g < ndev; g++) {
        CUDACHECK(cudaSetDevice(comm->devs[g]));
        r = infcclEnsureStaged(comm, sliceBytes);
    }
    CUDACHECK(cudaSetDevice(comm->devs[0]));
    r = infcclEnsureStaged(comm, sliceBytes);
    if (r != infcclSuccess) { cudaSetDevice(savedDev); return r; }

    int sliceN = (count + ndev - 1) / ndev;

    for (int step = 0; step < ndev - 1; step++) {
        for (int g = 0; g < ndev; g++) {
            int sendSlice = (g - step + ndev) % ndev;
            int next = (g + 1) % ndev;
            int off = sendSlice * sliceN;
            int cnt = (off + sliceN > count) ? (count - off) : sliceN;
            if (cnt <= 0) continue;

            void* src_on_next = comm->ipc.mapped[g][next];
            CUDACHECK(cudaSetDevice(comm->devs[next]));
            cudaStream_t s = comm->ipc.streams[next];
            CUDACHECK(cudaMemcpyAsync(
                (char*)buffs[next] + count * elemSize,
                (char*)src_on_next + off * elemSize,
                cnt * elemSize, cudaMemcpyDeviceToDevice, s));
        }

        for (int g = 0; g < ndev; g++)
            CUDACHECK(cudaStreamSynchronize(comm->ipc.streams[g]));

        for (int g = 0; g < ndev; g++) {
            int recvSlice = (g - step - 1 + ndev) % ndev;
            int off = recvSlice * sliceN;
            int cnt = (off + sliceN > count) ? (count - off) : sliceN;
            if (cnt <= 0) continue;

            CUDACHECK(cudaSetDevice(comm->devs[g]));
            cudaStream_t s = comm->ipc.streams[g];
            T* dst = (T*)((char*)buffs[g] + off * elemSize);
            T* src = (T*)((char*)buffs[g] + count * elemSize);
            ReduceInplaceSimple<T, FUNC><<<INFCCL_BLOCKS(cnt), INFCCL_THREADS, 0, s>>>(dst, src, cnt);
        }
        for (int g = 0; g < ndev; g++)
            CUDACHECK(cudaStreamSynchronize(comm->ipc.streams[g]));
    }

    for (int step = 0; step < ndev - 1; step++) {
        for (int g = 0; g < ndev; g++) {
            int sendSlice = (g - step + 1 + ndev) % ndev;
            int next = (g + 1) % ndev;
            int off = sendSlice * sliceN;
            int cnt = (off + sliceN > count) ? (count - off) : sliceN;
            if (cnt <= 0) continue;

            void* src_on_next = comm->ipc.mapped[g][next];
            CUDACHECK(cudaSetDevice(comm->devs[next]));
            cudaStream_t s = comm->ipc.streams[next];
            CUDACHECK(cudaMemcpyAsync(
                (char*)buffs[next] + off * elemSize,
                (char*)src_on_next + off * elemSize,
                cnt * elemSize, cudaMemcpyDeviceToDevice, s));
        }
        for (int g = 0; g < ndev; g++)
            CUDACHECK(cudaStreamSynchronize(comm->ipc.streams[g]));
    }

    cudaSetDevice(savedDev);
    return infcclSuccess;
}

template<class FUNC, typename T>
static infcclResult_t allReduceDispatch(void** buffs, int count,
    infcclComm_t comm, cudaStream_t stream) {
    if (count == 0) return infcclSuccess;
    return ipcAllReduce<T, FUNC>(buffs, count, comm, stream);
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

    if (r == infcclSuccess) infcclEnqueueRecord(comm, stream);
    return r;
}
