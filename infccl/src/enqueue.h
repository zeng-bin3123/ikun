/*
 * enqueue.h — Event queue for batched collective dispatch
 *
 * Placeholder for Mooncake §4 coalescing (future optimization):
 *   Pack N small collectives into one call to amortize
 *   the 5.5μs kernel launch overhead.
 *
 * Currently unused: collectives execute immediately.
 */

#ifndef INFCCL_ENQUEUE_H_
#define INFCCL_ENQUEUE_H_

#include "infccl.h"

typedef struct {
    infcclRedOp_t    op;
    infcclDataType_t dtype;
    size_t           count;
    const void*      sendbuff;
    void*            recvbuff;
    int              root;       /* -1 for all-to-all ops */
} infcclTask;

/* enqueue returns immediately; flush executes all pending tasks */
/* infcclResult_t infcclEnqueue(infcclComm_t comm, infcclTask task); */
/* infcclResult_t infcclFlush(infcclComm_t comm, cudaStream_t stream); */

#endif /* INFCCL_ENQUEUE_H_ */
