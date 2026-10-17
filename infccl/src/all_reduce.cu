/*
 * all_reduce.cu — AllReduce via staged transport
 *
 * Algorithm (since kernel P2P direct write fails on BI-V100):
 *   1. Each rank copies sendbuff into its staging buffer (GPU-local)
 *   2. Host barrier: all ranks have staged
 *   3. Each rank pulls data from every peer via cudaMemcpyPeer
 *      into local staging area at offset [peer * chunk_size]
 *   4. Local reduce kernel: reduce all nranks chunks into recvbuff
 *
 * This is the "broadcast-reduce" approach — bandwidth cost is
 *   nranks × size × 2 (copy in + read from peers)
 * which is worse than a ring's 2*(P-1)/P × size, but avoids
 * kernel P2P writes entirely. For small tensors (< 1MB) the
 * latency advantage of fewer barriers dominates.
 *
 * For larger tensors the ring algorithm would be better, but
 * requires kernel-to-kernel signaling which BI-V100 cannot do
 * (no __threadfence_system on ivcore10, no cooperative_groups).
 * Future: multi-kernel-launch ring with host-side step barriers.
 */

#include "core.h"
#include "common_kernel.h"
#include "reduce_kernel.h"

infcclResult_t infcclAllReduce(const void* sendbuff, void* recvbuff,
                                size_t count, infcclDataType_t datatype,
                                infcclRedOp_t op, infcclComm_t comm,
                                cudaStream_t stream) {
    const int rank   = comm->rank;
    const int nranks = comm->nranks;
    const int esize  = infcclTypeSize((int)datatype);
    const size_t nbytes = count * esize;

    if (nbytes > comm->staging_bytes)
        return infcclInvalidArgument;

    if (nranks == 1) {
        if (sendbuff != recvbuff)
            cudaMemcpyAsync(recvbuff, sendbuff, nbytes,
                            cudaMemcpyDeviceToDevice, stream);
        return infcclSuccess;
    }

    cudaSetDevice(comm->cudaDev);

    /* Step 1: copy sendbuff → own staging slot */
    void* my_staging = comm->staging[rank];
    cudaMemcpyAsync(my_staging, sendbuff, nbytes,
                    cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    /* Step 2: host barrier — all ranks have staged their data */
    infcclBarrier(comm);

    /* Step 3: pull every peer's staging data into local temp area.
       We use recvbuff as scratch for the first peer, then reduce
       subsequent peers into it. */

    /* Initialize recvbuff from rank 0's data */
    int first = comm->ring.rank[0];   /* start with ring head */
    if (first == rank) {
        /* our own data — copy from staging */
        cudaMemcpyAsync(recvbuff, my_staging, nbytes,
                        cudaMemcpyDeviceToDevice, stream);
    } else {
        /* peer data — cudaMemcpyPeer (host DMA) */
        cudaMemcpyPeerAsync(recvbuff, comm->cudaDev,
                            comm->staging[first], first,
                            nbytes, stream);
    }

    /* Reduce remaining peers one at a time */
    /* Use a temporary buffer carved from unused staging region */
    void* temp = (char*)my_staging + nbytes;  /* after our own data */
    if (nbytes * 2 > comm->staging_bytes) {
        /* staging too small for temp — use recvbuff as accumulator directly */
        temp = NULL;
    }

    for (int p = 1; p < nranks; p++) {
        int peer = comm->ring.rank[p];
        void* peer_data;
        if (peer == rank) {
            peer_data = my_staging;
        } else {
            /* pull peer's staging into temp (or directly) */
            void* dst = temp ? temp : recvbuff;
            cudaMemcpyPeerAsync(dst, comm->cudaDev,
                                comm->staging[peer], peer,
                                nbytes, stream);
            peer_data = dst;
        }

        /* reduce peer_data into recvbuff */
        int grid = infcclGridSize(count);
        infcclReduceInPlace<<<grid, INFCCL_BLOCK_DIM, 0, stream>>>(
            recvbuff, peer_data, (int64_t)count,
            (int)datatype, (int)op);
    }

    /* Step 4: barrier — ensure all ranks finished reading staging */
    cudaStreamSynchronize(stream);
    infcclBarrier(comm);

    return infcclSuccess;
}
