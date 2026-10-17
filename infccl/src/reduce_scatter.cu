/*
 * reduce_scatter.cu — ReduceScatter via staged transport
 *
 * Input:  each rank has recvcount × nranks elements (full tensor)
 * Output: each rank gets recvcount elements (its reduced shard)
 *
 * Algorithm:
 *   1. Each rank stages its full sendbuff
 *   2. Host barrier
 *   3. Each rank pulls the relevant shard from every peer,
 *      reduces them locally into recvbuff
 *   4. Host barrier
 *
 * Shard assignment: rank r gets shard r (elements [r*recvcount .. (r+1)*recvcount))
 */

#include "core.h"
#include "common_kernel.h"
#include "reduce_kernel.h"

infcclResult_t infcclReduceScatter(const void* sendbuff, void* recvbuff,
                                    size_t recvcount, infcclDataType_t datatype,
                                    infcclRedOp_t op, infcclComm_t comm,
                                    cudaStream_t stream) {
    const int rank   = comm->rank;
    const int nranks = comm->nranks;
    const int esize  = infcclTypeSize((int)datatype);
    const size_t shard_bytes = recvcount * esize;
    const size_t total_bytes = shard_bytes * nranks;

    if (total_bytes > comm->staging_bytes)
        return infcclInvalidArgument;

    if (nranks == 1) {
        if (sendbuff != recvbuff)
            cudaMemcpyAsync(recvbuff, sendbuff, shard_bytes,
                            cudaMemcpyDeviceToDevice, stream);
        return infcclSuccess;
    }

    cudaSetDevice(comm->cudaDev);

    /* Step 1: stage full sendbuff */
    cudaMemcpyAsync(comm->staging[rank], sendbuff, total_bytes,
                    cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    /* Step 2: barrier */
    infcclBarrier(comm);

    /* Step 3: this rank reads shard[rank] from every peer, reduces locally */
    /* My shard offset within each rank's staged buffer */
    size_t shard_offset = (size_t)rank * shard_bytes;

    /* Initialize recvbuff from first peer's shard */
    int first = 0;
    const char* first_src = (const char*)comm->staging[first] + shard_offset;
    if (first == rank) {
        cudaMemcpyAsync(recvbuff, first_src, shard_bytes,
                        cudaMemcpyDeviceToDevice, stream);
    } else {
        cudaMemcpyPeerAsync(recvbuff, comm->cudaDev,
                            (void*)first_src, first,
                            shard_bytes, stream);
    }

    /* Temp buffer for pulling each subsequent peer's shard */
    void* temp = (char*)comm->staging[rank] + total_bytes;
    int has_temp = ((total_bytes + shard_bytes) <= comm->staging_bytes);

    for (int p = 1; p < nranks; p++) {
        const char* peer_shard = (const char*)comm->staging[p] + shard_offset;

        void* local_copy;
        if (p == rank) {
            /* own data — already local */
            local_copy = (void*)peer_shard;
        } else if (has_temp) {
            /* pull into temp */
            cudaMemcpyPeerAsync(temp, comm->cudaDev,
                                (void*)peer_shard, p,
                                shard_bytes, stream);
            local_copy = temp;
        } else {
            /* no temp space: pull directly into recvbuff then reduce in-place
               This is a correctness fallback (overwrites partial result). */
            cudaMemcpyPeerAsync(recvbuff, comm->cudaDev,
                                (void*)peer_shard, p,
                                shard_bytes, stream);
            local_copy = recvbuff;
        }

        /* reduce peer's shard into recvbuff */
        if (local_copy != recvbuff) {
            int grid = infcclGridSize(recvcount);
            infcclReduceInPlace<<<grid, INFCCL_BLOCK_DIM, 0, stream>>>(
                recvbuff, local_copy, (int64_t)recvcount,
                (int)datatype, (int)op);
        }
    }

    /* Step 4: barrier */
    cudaStreamSynchronize(stream);
    infcclBarrier(comm);

    return infcclSuccess;
}
