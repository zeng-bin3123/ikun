/*
 * reduce.cu — Reduce to root via staged transport
 *
 * All ranks contribute sendbuff, root rank receives the reduced result.
 * Non-root ranks' recvbuff is untouched.
 *
 * Algorithm:
 *   1. All ranks stage their sendbuff
 *   2. Host barrier
 *   3. Root pulls all peers' staging via cudaMemcpyPeer and reduces locally
 *   4. Host barrier (all done reading)
 */

#include "core.h"
#include "common_kernel.h"
#include "reduce_kernel.h"

infcclResult_t infcclReduce(const void* sendbuff, void* recvbuff,
                             size_t count, infcclDataType_t datatype,
                             infcclRedOp_t op, int root,
                             infcclComm_t comm, cudaStream_t stream) {
    const int rank   = comm->rank;
    const int nranks = comm->nranks;
    const int esize  = infcclTypeSize((int)datatype);
    const size_t nbytes = count * esize;

    if (nbytes > comm->staging_bytes)
        return infcclInvalidArgument;
    if (root < 0 || root >= nranks)
        return infcclInvalidArgument;

    if (nranks == 1) {
        if (sendbuff != recvbuff)
            cudaMemcpyAsync(recvbuff, sendbuff, nbytes,
                            cudaMemcpyDeviceToDevice, stream);
        return infcclSuccess;
    }

    cudaSetDevice(comm->cudaDev);

    /* Step 1: stage sendbuff */
    void* my_staging = comm->staging[rank];
    cudaMemcpyAsync(my_staging, sendbuff, nbytes,
                    cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    /* Step 2: barrier */
    infcclBarrier(comm);

    /* Step 3: root reduces */
    if (rank == root) {
        /* initialize recvbuff from own staging */
        cudaMemcpyAsync(recvbuff, my_staging, nbytes,
                        cudaMemcpyDeviceToDevice, stream);

        void* temp = (char*)my_staging + nbytes;

        for (int p = 0; p < nranks; p++) {
            if (p == root) continue;
            /* pull peer p's staging into temp */
            cudaMemcpyPeerAsync(temp, comm->cudaDev,
                                comm->staging[p], p,
                                nbytes, stream);
            /* reduce into recvbuff */
            int grid = infcclGridSize(count);
            infcclReduceInPlace<<<grid, INFCCL_BLOCK_DIM, 0, stream>>>(
                recvbuff, temp, (int64_t)count,
                (int)datatype, (int)op);
        }
    }

    /* Step 4: barrier */
    cudaStreamSynchronize(stream);
    infcclBarrier(comm);

    return infcclSuccess;
}
