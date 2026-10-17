/*
 * broadcast.cu — Broadcast from root via staged transport
 *
 * Root's sendbuff is copied to all ranks' recvbuff.
 *
 * Algorithm:
 *   1. Root stages sendbuff
 *   2. Host barrier
 *   3. Non-root ranks pull from root's staging via cudaMemcpyPeer
 *   4. Host barrier
 */

#include "core.h"
#include "common_kernel.h"

infcclResult_t infcclBroadcast(const void* sendbuff, void* recvbuff,
                                size_t count, infcclDataType_t datatype,
                                int root, infcclComm_t comm,
                                cudaStream_t stream) {
    const int rank   = comm->rank;
    const int nranks = comm->nranks;
    const size_t nbytes = count * infcclTypeSize((int)datatype);

    if (nbytes > comm->staging_bytes)
        return infcclInvalidArgument;

    if (nranks == 1) {
        if (sendbuff != recvbuff)
            cudaMemcpyAsync(recvbuff, sendbuff, nbytes,
                            cudaMemcpyDeviceToDevice, stream);
        return infcclSuccess;
    }

    cudaSetDevice(comm->cudaDev);

    /* Step 1: root stages sendbuff */
    if (rank == root) {
        cudaMemcpyAsync(comm->staging[rank], sendbuff, nbytes,
                        cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    /* Step 2: barrier */
    infcclBarrier(comm);

    /* Step 3: everyone reads from root's staging */
    if (rank == root) {
        /* root: copy from staging to recvbuff */
        if (sendbuff != recvbuff)
            cudaMemcpyAsync(recvbuff, comm->staging[root], nbytes,
                            cudaMemcpyDeviceToDevice, stream);
    } else {
        /* non-root: pull from root's staging via host DMA */
        cudaMemcpyPeerAsync(recvbuff, comm->cudaDev,
                            comm->staging[root], root,
                            nbytes, stream);
    }

    /* Step 4: barrier */
    cudaStreamSynchronize(stream);
    infcclBarrier(comm);

    return infcclSuccess;
}
