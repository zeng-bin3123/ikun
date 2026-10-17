/*
 * core.cu — Bootstrap, P2P probe, ring topology, communicator lifecycle
 *
 * Staged transport model:
 *   1. Each rank cudaMalloc's a staging buffer on its own GPU.
 *   2. Cross-GPU moves use cudaMemcpyPeer(dst_dev, dst_ptr, src_dev, src_ptr).
 *      This goes through host DMA — the only reliable path on BI-V100.
 *   3. A host-side barrier (pinned memory atomic flag) synchronizes ranks.
 *
 * No CUDA IPC is used because kernel P2P direct write fails on this HW.
 * Instead, each collective explicitly calls cudaMemcpyPeer for each chunk.
 */

#include "core.h"
#include "infccl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

/* ═══════ P2P probe ═══════ */

infcclResult_t infcclProbeP2P(int nranks, infcclP2PMatrix* mat) {
    memset(mat, 0, sizeof(*mat));
    for (int i = 0; i < nranks; i++) {
        mat->can_p2p[i][i] = 1;
        cudaSetDevice(i);
        for (int j = 0; j < nranks; j++) {
            if (i == j) continue;
            int can = 0;
            cudaDeviceCanAccessPeer(&can, i, j);
            mat->can_p2p[i][j] = can;
            if (can) {
                cudaError_t e = cudaDeviceEnablePeerAccess(j, 0);
                if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled)
                    mat->can_p2p[i][j] = 0;
            }
        }
    }
    return infcclSuccess;
}

/* ═══════ Ring topology builder ═══════ */

/*
 * Greedy ring construction:
 *   Start at GPU 0, pick the nearest unused GPU that can P2P.
 *   Avoids non-P2P pairs (e.g. GPU 0↔3 on some BI-V100 boards).
 *   Falls back to any unused GPU if no P2P neighbor available.
 */
void infcclBuildRing(int nranks, const infcclP2PMatrix* mat,
                     infcclRing* ring) {
    int used[INFCCL_MAX_RANKS] = {0};
    int path[INFCCL_MAX_RANKS];

    path[0] = 0;
    used[0] = 1;

    for (int pos = 1; pos < nranks; pos++) {
        int prev = path[pos - 1];
        int best = -1;
        /* prefer P2P-reachable unused neighbor */
        for (int g = 0; g < nranks; g++) {
            if (!used[g] && mat->can_p2p[prev][g]) {
                best = g; break;
            }
        }
        /* fallback: any unused */
        if (best < 0)
            for (int g = 0; g < nranks; g++)
                if (!used[g]) { best = g; break; }
        path[pos] = best;
        used[best] = 1;
    }

    /* fill ring arrays */
    for (int pos = 0; pos < nranks; pos++) {
        int r = path[pos];
        ring->rank[pos]  = r;             /* position → rank */
        ring->order[r]   = pos;           /* rank → position */
    }
    for (int pos = 0; pos < nranks; pos++) {
        int r = path[pos];
        ring->prev[r] = path[(pos + nranks - 1) % nranks];
        ring->next[r] = path[(pos + 1)          % nranks];
    }

    fprintf(stderr, "[infccl] ring:");
    for (int p = 0; p < nranks; p++)
        fprintf(stderr, " %d", ring->rank[p]);
    fprintf(stderr, "\n");
}

/* ═══════ Unique ID (single-node: trivial) ═══════ */

infcclResult_t infcclGetUniqueId(infcclUniqueId* uid) {
    memset(uid, 0, sizeof(*uid));
    /* single-node: just stamp with pid + time for uniqueness */
    snprintf(uid->internal, INFCCL_UNIQUE_ID_BYTES,
             "infccl-%d-%ld", (int)getpid(), (long)time(NULL));
    return infcclSuccess;
}

/* ═══════ Communicator init ═══════ */

infcclResult_t infcclCommInitRank(infcclComm_t* comm_out, int nranks,
                                   infcclUniqueId commId, int rank) {
    if (rank < 0 || rank >= nranks || nranks > INFCCL_MAX_RANKS)
        return infcclInvalidArgument;

    struct infcclComm* c = (struct infcclComm*)calloc(1, sizeof(*c));
    if (!c) return infcclInternalError;

    c->rank         = rank;
    c->nranks       = nranks;
    c->staging_bytes = INFCCL_STAGING_BYTES;
    c->generation   = 0;
    c->cudaDev      = rank;              /* assume rank == device ordinal */

    /* allocate this rank's staging buffer */
    cudaSetDevice(c->cudaDev);
    cudaError_t e = cudaMalloc(&c->staging[rank], c->staging_bytes);
    if (e != cudaSuccess) { free(c); return infcclInternalError; }
    cudaMemset(c->staging[rank], 0, c->staging_bytes);

    /* host-pinned barrier flags (shared across ranks via fork/mmap in real
       multi-process; for single-process multi-GPU tests, the pointer is
       shared directly via infcclCommInitAll helper below). */
    e = cudaHostAlloc((void**)&c->barrier_flags,
                      sizeof(int) * INFCCL_MAX_RANKS,
                      cudaHostAllocMapped | cudaHostAllocPortable);
    if (e != cudaSuccess) {
        cudaFree(c->staging[rank]);
        free(c);
        return infcclInternalError;
    }
    memset((void*)c->barrier_flags, 0, sizeof(int) * INFCCL_MAX_RANKS);

    /* probe P2P */
    infcclProbeP2P(nranks, &c->p2p);

    /* build ring */
    infcclBuildRing(nranks, &c->p2p, &c->ring);

    *comm_out = c;
    fprintf(stderr, "[infccl] rank %d/%d init on GPU %d, staging %zu MB\n",
            rank, nranks, c->cudaDev, c->staging_bytes >> 20);
    return infcclSuccess;
}

/*
 * Helper: init all ranks in a single process (for testing).
 * Shares the same barrier_flags and records peer staging pointers.
 */
infcclResult_t infcclCommInitAll(infcclComm_t* comms, int nranks) {
    /* first, create all communicators */
    infcclUniqueId uid;
    infcclGetUniqueId(&uid);

    for (int r = 0; r < nranks; r++) {
        infcclResult_t res = infcclCommInitRank(&comms[r], nranks, uid, r);
        if (res != infcclSuccess) return res;
    }

    /* share barrier flags: use rank 0's allocation for all */
    volatile int* shared_flags = comms[0]->barrier_flags;
    for (int r = 1; r < nranks; r++) {
        cudaFreeHost((void*)comms[r]->barrier_flags);
        comms[r]->barrier_flags = shared_flags;
    }

    /* record peer staging pointers */
    for (int r = 0; r < nranks; r++)
        for (int p = 0; p < nranks; p++)
            comms[r]->staging[p] = comms[p]->staging[p];

    return infcclSuccess;
}

/* ═══════ Query ═══════ */

infcclResult_t infcclCommCount(const infcclComm_t comm, int* count) {
    *count = comm->nranks;
    return infcclSuccess;
}

infcclResult_t infcclCommUserRank(const infcclComm_t comm, int* rank) {
    *rank = comm->rank;
    return infcclSuccess;
}

/* ═══════ Destroy ═══════ */

infcclResult_t infcclCommDestroy(infcclComm_t comm) {
    if (!comm) return infcclSuccess;
    cudaSetDevice(comm->cudaDev);
    if (comm->staging[comm->rank])
        cudaFree(comm->staging[comm->rank]);
    /* only rank 0 owns the barrier_flags allocation */
    if (comm->rank == 0 && comm->barrier_flags)
        cudaFreeHost((void*)comm->barrier_flags);
    free(comm);
    return infcclSuccess;
}
