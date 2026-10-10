#ifndef INFCCL_H_
#define INFCCL_H_

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifndef INFCCL_MAX_DEVS
#define INFCCL_MAX_DEVS 8
#endif

namespace infccl { struct CommState; }
typedef infccl::CommState* infcclComm_t;

#ifdef __cplusplus
extern "C" {
#endif

const char* infcclGetErrorString(int code);
int infcclCommInitAll(infcclComm_t* comms, int ndev, const int* devlist);
int infcclCommDestroy(infcclComm_t comm);
int infcclCommCount(const infcclComm_t comm, int* count);
int infcclCommCuDevice(const infcclComm_t comm, int* device);
int infcclCommUserRank(const infcclComm_t comm, int* rank);
int infcclRegisterMemory(infcclComm_t comm, int gpu, void* addr, size_t bytes);
int infcclUnregisterMemory(infcclComm_t comm, int gpu, void* addr);
int infcclAllReduce(void** buffs, int count, int datatype, int op, infcclComm_t comm, cudaStream_t stream);
int infcclReduce(void** buffs, int count, int datatype, int op, int root, infcclComm_t comm, cudaStream_t stream);
int infcclBcast(void** buffs, int count, int datatype, int root, infcclComm_t comm, cudaStream_t stream);
int infcclAllGather(void** sbufs, void** rbufs, int sendcount, int datatype, infcclComm_t comm, cudaStream_t stream);
int infcclReduceScatter(void** sbufs, void** rbufs, int recvcount, int datatype, int op, infcclComm_t comm, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

typedef int infcclResult_t;
typedef int infcclRedOp_t;
typedef int infcclDataType_t;

enum { infcclSuccess=0 };
enum { infcclSum=0, infcclProd=1, infcclMax=2, infcclMin=3 };
enum { infcclChar=0, infcclInt=1, infcclHalf=2, infcclFloat=3, infcclInt64=4, infcclUint64=5 };

#endif
