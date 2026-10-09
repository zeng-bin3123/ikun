#ifndef INFCCL_H_
#define INFCCL_H_

#include <cuda_runtime.h>
#include <stddef.h>

#define INFCCL_MAJOR 0
#define INFCCL_MINOR 1
#define INFCCL_PATCH 0
#define INFCCL_MAX_DEVS 8
#define INFCCL_MAX_QUEUE 4
#define INFCCL_DEFAULT_BUFFER_SIZE (1 << 21)

typedef enum {
    infcclSuccess               = 0,
    infcclUnhandledCudaError    = 1,
    infcclSystemError           = 2,
    infcclInternalError         = 3,
    infcclInvalidDevicePointer  = 4,
    infcclInvalidRank           = 5,
    infcclUnsupportedDeviceCount= 6,
    infcclDeviceNotFound        = 7,
    infcclInvalidDeviceIndex    = 8,
    infcclCudaMallocFailed      = 9,
    infcclRankMismatch          = 10,
    infcclInvalidArgument       = 11,
    infcclInvalidType           = 12,
    infcclInvalidOperation      = 13,
    infcclNumResults            = 14
} infcclResult_t;

typedef enum {
    infcclSum  = 0,
    infcclProd = 1,
    infcclMax  = 2,
    infcclMin  = 3,
    infcclNumOps = 4
} infcclRedOp_t;

typedef enum {
    infcclChar   = 0,
    infcclInt    = 1,
    infcclHalf   = 2,
    infcclFloat  = 3,
    infcclDouble = 4,
    infcclInt64  = 5,
    infcclUint64 = 6,
    infcclNumTypes = 7
} infcclDataType_t;

struct infcclComm;
typedef struct infcclComm* infcclComm_t;

infcclResult_t infcclGetVersion(int* version);
infcclResult_t infcclCommInitAll(infcclComm_t* comms, int ndev, const int* devlist);
infcclResult_t infcclCommDestroy(infcclComm_t comm);
infcclResult_t infcclCommCount(const infcclComm_t comm, int* count);
infcclResult_t infcclCommCuDevice(const infcclComm_t comm, int* device);
infcclResult_t infcclCommUserRank(const infcclComm_t comm, int* rank);

infcclResult_t infcclAllReduce(void** buffs, int count,
    infcclDataType_t datatype, infcclRedOp_t op,
    infcclComm_t comm, cudaStream_t stream);

infcclResult_t infcclReduce(void** buffs, int count,
    infcclDataType_t datatype, infcclRedOp_t op, int root,
    infcclComm_t comm, cudaStream_t stream);

infcclResult_t infcclBcast(void** buffs, int count,
    infcclDataType_t datatype, int root,
    infcclComm_t comm, cudaStream_t stream);

infcclResult_t infcclAllGather(void** sendbufs, void** recvbufs,
    int sendcount, infcclDataType_t datatype,
    infcclComm_t comm, cudaStream_t stream);

infcclResult_t infcclReduceScatter(void** sendbufs, void** recvbufs,
    int recvcount, infcclDataType_t datatype, infcclRedOp_t op,
    infcclComm_t comm, cudaStream_t stream);

const char* infcclGetErrorString(infcclResult_t result);

#endif
