/*
 * all_gather.cu — AllGather via staged transport
 *
 * Each rank provides sendcount elements. Output is nranks × sendcount
 * with all ranks' data concatenated in rank order.
 *
 * Algorithm:
 *   1. Each rank stages its sendbuff
 *   2. Host barrier
 *   3. Each rank pulls every peer's staging into the correct offset
 *      of recvbuff via cudaMemcpyPeer
 *   4. Host barrier
 */

#include "core.h"
#include "common_kernel.h"

infcclResult_t infcclAllGather(const void* sendbuff, void* recvbuff,
                                size_t sendcount, infcclDataType_t datatype,
                                infcclComm_t comm, cudaStream_t stream) {
    const int rank   = comm->rank;
    const int nranks = comm->nranks;
    const int esize  = infcclTypeSize((int)datatype);
    const size_t shard_bytes = sendcount * esize;

    if (shard_bytes > comm->staging_bytes)
        return infcclInvalidArgument;

    if (nranks == 1) {
        if (sendbuff != recvbuff)
            cudaMemcpyAsync(recvbuff, sendbuff, shard_bytes,
                            cudaMemcpyDeviceToDevice, stream);
        return infcclSuccess;
    }

    cudaSetDevice(comm->cudaDev);

    /* Step 1: stage sendbuff */
    cudaMemcpyAsync(comm->staging[rank], sendbuff, shard_bytes,
                    cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    /* Step 2: barrier */
    infcclBarrier(comm);

    /* Step 3: gather all shards into recvbuff */
    for (int p = 0; p < nranks; p++) {
        char* dst = (char*)recvbuff + (size_t)p * shard_bytes;
        if (p == rank) {
            cudaMemcpyAsync(dst, comm->staging[rank], shard_bytes,
                            cudaMemcpyDeviceToDevice, stream);
        } else {
            cudaMemcpyPeerAsync(dst, comm->cudaDev,
                                comm->staging[p], p,
                                shard_bytes, stream);
        }
    }

    /* Step 4: barrier */
    cudaStreamSynchronize(stream);
    infcclBarrier(comm);

    return infcclSuccess;
}
