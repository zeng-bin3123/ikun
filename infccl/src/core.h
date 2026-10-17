/*
 * core.h — Internal communicator structures
 *
 * BI-V100 staged transport model:
 *   Each rank allocates a staging buffer on GPU.
 *   Cross-GPU data moves via cudaMemcpyPeer (host-DMA).
 *   Kernel P2P direct write FAILS on this hardware.
 *
 * Ring topology is computed at init time, avoiding
 * GPU pairs that cannot do P2P (e.g. 0↔3 cross-switch).
 */

#ifndef INFCCL_CORE_H_
#define INFCCL_CORE_H_

#include "infccl.h"
#include <cuda_runtime.h>
#include <pthread.h>

#define INFCCL_MAX_RANKS   8
#define INFCCL_STAGING_MB  32           /* per-rank staging buffer (MB)  */
#define INFCCL_STAGING_BYTES ((size_t)INFCCL_STAGING_MB << 20)

/* ── P2P reachability matrix ── */
typedef struct {
    int can_p2p[INFCCL_MAX_RANKS][INFCCL_MAX_RANKS]; /* 1 = direct P2P OK */
} infcclP2PMatrix;

/* ── Ring ordering ── */
typedef struct {
    int order[INFCCL_MAX_RANKS];   /* rank  → ring position */
    int rank[INFCCL_MAX_RANKS];    /* ring position → rank  */
    int prev[INFCCL_MAX_RANKS];    /* ring predecessor rank for each rank */
    int next[INFCCL_MAX_RANKS];    /* ring successor  rank for each rank */
} infcclRing;

/* ── Per-rank communicator (opaque to user) ── */
struct infcclComm {
    int           rank;
    int           nranks;

    /* staged transport: each rank has a staging buffer visible to this rank */
    void*         staging[INFCCL_MAX_RANKS]; /* device ptr per rank          */
    size_t        staging_bytes;             /* bytes per staging buffer     */

    /* synchronization: host-side barrier via shared flag array             */
    /* flag[r] is written by rank r, polled by all others                   */
    volatile int* barrier_flags;             /* host-pinned, nranks ints     */
    uint32_t      generation;                /* monotonic barrier generation */

    /* ring topology */
    infcclRing    ring;
    infcclP2PMatrix p2p;

    /* CUDA device for this rank */
    int           cudaDev;
};

/* ── Internal helpers (core.cu) ── */
infcclResult_t infcclProbeP2P(int nranks, infcclP2PMatrix* mat);
void           infcclBuildRing(int nranks, const infcclP2PMatrix* mat,
                               infcclRing* ring);

/* Host-side barrier: signal this rank is done, wait for all others. */
static inline void infcclBarrier(struct infcclComm* comm) {
    const int r  = comm->rank;
    const int n  = comm->nranks;
    const uint32_t gen = ++comm->generation;
    __atomic_store_n(&comm->barrier_flags[r], (int)gen, __ATOMIC_RELEASE);
    for (int i = 0; i < n; i++) {
        if (i == r) continue;
        while ((uint32_t)__atomic_load_n(&comm->barrier_flags[i],
                                          __ATOMIC_ACQUIRE) < gen)
            ;                              /* spin — single-node, fast */
    }
}

#endif /* INFCCL_CORE_H_ */
