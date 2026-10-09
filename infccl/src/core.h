#ifndef INFCCL_CORE_H_
#define INFCCL_CORE_H_

#include "infccl.h"
#include <cstdio>
#include <cuda_runtime.h>

#define INFCCL_WARP_SIZE 64
#define INFCCL_MEM_PAD_ALIGN 4096
#define INFCCL_CHUNK_ELEMS 65536

#define CUDACHECK(cmd) do { \
    cudaError_t e = (cmd); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "infccl CUDA %s:%d '%s'\n", \
            __FILE__, __LINE__, cudaGetErrorString(e)); \
        return infcclUnhandledCudaError; \
    } \
} while(0)

typedef enum { INFCCL_NONE=0, INFCCL_WARN=1, INFCCL_INFO=2, INFCCL_ABORT=3 } InfcclDebugLevel;
extern InfcclDebugLevel infcclDebugLevel;

#define WARN(...) do { \
    if (infcclDebugLevel >= INFCCL_WARN) { \
        fprintf(stderr, "INFCCL WARN %s:%d ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); \
        if (infcclDebugLevel >= INFCCL_ABORT) abort(); \
    } \
} while(0)

#define INFO(...) do { \
    if (infcclDebugLevel >= INFCCL_INFO) { \
        fprintf(stderr, "INFCCL INFO "); \
        fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); \
    } \
} while(0)

typedef enum {
    INFCCL_PATH_STAGED = 0,
    INFCCL_PATH_P2P_DIRECT = 1,
    INFCCL_PATH_HOST_SHM = 2
} InfcclTransportPath;

typedef struct {
    int rank;
    int cudaDev;
    char pciBusId[16];
    int numaNode;
} InfcclRankInfo;

typedef struct {
    cudaEvent_t isDone[INFCCL_MAX_QUEUE];
    int back;
} InfcclEventQueue;

struct infcclIpcConn {
    void* base[INFCCL_MAX_DEVS];
    size_t baseBytes[INFCCL_MAX_DEVS];
    void* mapped[INFCCL_MAX_DEVS][INFCCL_MAX_DEVS];
    cudaStream_t streams[INFCCL_MAX_DEVS];
};

struct infcclComm {
    int nDev;
    int rank;
    int cudaDev;

    InfcclRankInfo rankInfo[INFCCL_MAX_DEVS];
    int userFromRing[INFCCL_MAX_DEVS];
    int ringFromUser[INFCCL_MAX_DEVS];
    int devs[INFCCL_MAX_DEVS];

    int peerMatrix[INFCCL_MAX_DEVS][INFCCL_MAX_DEVS];
    int p2pDirectWorks;
    InfcclTransportPath transportPath;

    void* staged;
    size_t stagedBytes;

    infcclIpcConn ipc;

    InfcclEventQueue events;
    size_t buffSize;
};

infcclResult_t infcclIpcRegister(infcclComm_t comm, int gpu, void* ptr, size_t bytes);
void* infcclIpcGetMapped(infcclComm_t comm, int fromGpu, int onGpu);
cudaStream_t infcclGetStream(infcclComm_t comm, int gpu);

static inline size_t infcclTypeSize(infcclDataType_t type) {
    switch(type) {
        case infcclChar:   return 1;
        case infcclInt:    return 4;
        case infcclHalf:   return 2;
        case infcclFloat:  return 4;
        case infcclDouble: return 8;
        case infcclInt64:  return 8;
        case infcclUint64: return 8;
        default: return 0;
    }
}

infcclResult_t infcclEnsureStaged(infcclComm_t comm, size_t bytes);

#endif
