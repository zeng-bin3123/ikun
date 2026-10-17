/*
 * infccl.h — Public API for Iluvatar Inference CCL
 *
 * Mirrors NCCL's public API shape but adapted for BI-V100:
 *   - All collectives use staged transport (cudaMemcpyPeer)
 *   - No kernel-level P2P direct writes
 *   - warpSize=64
 *
 * Build: clang++ -x cuda --cuda-gpu-arch=ivcore10
 */

#ifndef INFCCL_H_
#define INFCCL_H_

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Result codes ── */
typedef enum {
    infcclSuccess         = 0,
    infcclUnhandledError   = 1,
    infcclInvalidArgument  = 2,
    infcclInvalidUsage     = 3,
    infcclInternalError    = 4,
    infcclNoP2P            = 5,
} infcclResult_t;

/* ── Data types ── */
typedef enum {
    infcclFloat16  = 0,
    infcclFloat32  = 1,
    infcclBfloat16 = 2,
    infcclInt32    = 3,
} infcclDataType_t;

/* ── Reduction ops ── */
typedef enum {
    infcclSum  = 0,
    infcclMax  = 1,
    infcclMin  = 2,
    infcclProd = 3,
} infcclRedOp_t;

/* ── Opaque communicator handle ── */
typedef struct infcclComm* infcclComm_t;

/* ── Unique ID for bootstrap ── */
#define INFCCL_UNIQUE_ID_BYTES 128
typedef struct { char internal[INFCCL_UNIQUE_ID_BYTES]; } infcclUniqueId;

/* ── Lifecycle ── */
infcclResult_t infcclGetUniqueId(infcclUniqueId* uniqueId);
infcclResult_t infcclCommInitRank(infcclComm_t* comm, int nranks,
                                   infcclUniqueId commId, int rank);
infcclResult_t infcclCommDestroy(infcclComm_t comm);

/* ── Query ── */
infcclResult_t infcclCommCount(const infcclComm_t comm, int* count);
infcclResult_t infcclCommUserRank(const infcclComm_t comm, int* rank);

/* ── Collectives ── */
infcclResult_t infcclAllReduce(const void* sendbuff, void* recvbuff,
                                size_t count, infcclDataType_t datatype,
                                infcclRedOp_t op, infcclComm_t comm,
                                cudaStream_t stream);

infcclResult_t infcclReduce(const void* sendbuff, void* recvbuff,
                             size_t count, infcclDataType_t datatype,
                             infcclRedOp_t op, int root,
                             infcclComm_t comm, cudaStream_t stream);

infcclResult_t infcclBroadcast(const void* sendbuff, void* recvbuff,
                                size_t count, infcclDataType_t datatype,
                                int root, infcclComm_t comm,
                                cudaStream_t stream);

infcclResult_t infcclAllGather(const void* sendbuff, void* recvbuff,
                                size_t sendcount, infcclDataType_t datatype,
                                infcclComm_t comm, cudaStream_t stream);

infcclResult_t infcclReduceScatter(const void* sendbuff, void* recvbuff,
                                    size_t recvcount, infcclDataType_t datatype,
                                    infcclRedOp_t op, infcclComm_t comm,
                                    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif /* INFCCL_H_ */
