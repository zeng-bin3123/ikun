#ifndef INFCCL_ENQUEUE_H_
#define INFCCL_ENQUEUE_H_

#include "core.h"

static inline infcclResult_t infcclEnqueueCheck(infcclComm_t comm, cudaStream_t stream) {
    InfcclEventQueue* eq = &comm->events;
    cudaError_t flag = cudaEventQuery(eq->isDone[eq->back]);
    if (flag == cudaErrorNotReady) {
        CUDACHECK(cudaStreamWaitEvent(stream, eq->isDone[eq->back], 0));
    } else if (flag != cudaSuccess && flag != cudaErrorNotReady) {
        cudaGetLastError();
    }
    return infcclSuccess;
}

static inline infcclResult_t infcclEnqueueRecord(infcclComm_t comm, cudaStream_t stream) {
    InfcclEventQueue* eq = &comm->events;
    eq->back = (eq->back + 1) % INFCCL_MAX_QUEUE;
    CUDACHECK(cudaEventRecord(eq->isDone[eq->back], stream));
    return infcclSuccess;
}

#endif
