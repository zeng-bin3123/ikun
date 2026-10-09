#include "core.h"
#include "reduce_kernel.h"
#include <cstdlib>
#include <cstring>

static infcclResult_t infcclIpcSetup(infcclComm_t comm);
static void infcclIpcCleanup(infcclComm_t comm);

InfcclDebugLevel infcclDebugLevel = INFCCL_WARN;

static void initDebug() {
    const char* env = getenv("INFCCL_DEBUG");
    if (!env) return;
    if (strcmp(env, "WARN") == 0) infcclDebugLevel = INFCCL_WARN;
    else if (strcmp(env, "INFO") == 0) infcclDebugLevel = INFCCL_INFO;
    else if (strcmp(env, "ABORT") == 0) infcclDebugLevel = INFCCL_ABORT;
}

infcclResult_t infcclGetVersion(int* version) {
    *version = INFCCL_MAJOR * 10000 + INFCCL_MINOR * 100 + INFCCL_PATCH;
    return infcclSuccess;
}

const char* infcclGetErrorString(infcclResult_t code) {
    switch (code) {
        case infcclSuccess:               return "no error";
        case infcclUnhandledCudaError:     return "unhandled cuda error";
        case infcclSystemError:            return "system error";
        case infcclInternalError:          return "internal error";
        case infcclInvalidDevicePointer:   return "invalid device pointer";
        case infcclInvalidRank:            return "invalid rank";
        case infcclUnsupportedDeviceCount: return "unsupported device count";
        case infcclDeviceNotFound:         return "device not found";
        case infcclInvalidDeviceIndex:     return "invalid device index";
        case infcclCudaMallocFailed:       return "cuda malloc failed";
        case infcclRankMismatch:           return "rank mismatch";
        case infcclInvalidArgument:        return "invalid argument";
        case infcclInvalidType:            return "invalid type";
        case infcclInvalidOperation:       return "invalid operation";
        default:                           return "unknown error";
    }
}

infcclResult_t infcclEnsureStaged(infcclComm_t comm, size_t bytes) {
    if (comm->stagedBytes >= bytes) return infcclSuccess;
    int savedDev; cudaGetDevice(&savedDev);
    cudaSetDevice(comm->devs[0]);
    if (comm->staged) cudaFree(comm->staged);
    cudaError_t e = cudaMalloc(&comm->staged, bytes);
    cudaSetDevice(savedDev);
    if (e != cudaSuccess) { comm->staged = NULL; comm->stagedBytes = 0; return infcclCudaMallocFailed; }
    comm->stagedBytes = bytes;
    return infcclSuccess;
}

static infcclResult_t buildTopology(infcclComm_t comm) {
    int savedDev; cudaGetDevice(&savedDev);
    for (int r = 0; r < comm->nDev; r++) {
        InfcclRankInfo* ri = &comm->rankInfo[r];
        ri->rank = r;
        ri->cudaDev = comm->devs[r];
        cudaSetDevice(ri->cudaDev);
        if (cudaDeviceGetPCIBusId(ri->pciBusId, 16, ri->cudaDev) != cudaSuccess)
            snprintf(ri->pciBusId, 16, "unknown");
        char numaPath[256];
        snprintf(numaPath, sizeof(numaPath), "/sys/bus/pci/devices/%s/numa_node", ri->pciBusId);
        FILE* f = fopen(numaPath, "r");
        if (f) { fscanf(f, "%d", &ri->numaNode); fclose(f); }
        else ri->numaNode = -1;
        INFO("rank %d: dev %d PCI %s NUMA %d", r, ri->cudaDev, ri->pciBusId, ri->numaNode);
    }
    for (int i = 0; i < comm->nDev; i++) {
        comm->userFromRing[i] = i;
        comm->ringFromUser[i] = i;
    }
    cudaSetDevice(savedDev);
    return infcclSuccess;
}

static infcclResult_t probeP2P(infcclComm_t comm) {
    int ndev = comm->nDev;
    int savedDev; cudaGetDevice(&savedDev);

    for (int i = 0; i < ndev; i++)
        for (int j = 0; j < ndev; j++)
            comm->peerMatrix[i][j] = (i == j) ? 1 : 0;

    for (int i = 0; i < ndev; i++) {
        cudaSetDevice(comm->devs[i]);
        for (int j = 0; j < ndev; j++) {
            if (i == j) continue;
            int can = 0;
            cudaDeviceCanAccessPeer(&can, comm->devs[i], comm->devs[j]);
            if (can) {
                cudaError_t e = cudaDeviceEnablePeerAccess(comm->devs[j], 0);
                if (e == cudaSuccess || e == cudaErrorPeerAccessAlreadyEnabled) {
                    if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
                    comm->peerMatrix[i][j] = 1;
                }
            }
        }
    }

    comm->p2pDirectWorks = 0;
    if (ndev >= 2 && comm->peerMatrix[0][1]) {
        const int TN = 256;
        float* srcBuf = NULL;
        float* dstBuf = NULL;
        float* hostCheck = (float*)malloc(TN * sizeof(float));

        cudaSetDevice(comm->devs[0]);
        cudaMalloc(&srcBuf, TN * sizeof(float));
        FillKernel<float><<<1, 256>>>(srcBuf, TN, 7.0f);
        cudaDeviceSynchronize();

        cudaSetDevice(comm->devs[1]);
        cudaMalloc(&dstBuf, TN * sizeof(float));
        cudaMemset(dstBuf, 0, TN * sizeof(float));

        cudaSetDevice(comm->devs[0]);
        FillKernel<float><<<1, 256>>>(dstBuf, TN, 7.0f);
        cudaDeviceSynchronize();

        cudaSetDevice(comm->devs[1]);
        cudaMemcpy(hostCheck, dstBuf, TN * sizeof(float), cudaMemcpyDeviceToHost);

        int errors = 0;
        for (int i = 0; i < TN; i++)
            if (hostCheck[i] != 7.0f) errors++;

        comm->p2pDirectWorks = (errors == 0) ? 1 : 0;

        if (errors > 0) {
            INFO("P2P kernel direct: %d/%d errors -> STAGED path", errors, TN);
        } else {
            INFO("P2P kernel direct: OK");
        }

        cudaSetDevice(comm->devs[0]); cudaFree(srcBuf);
        cudaSetDevice(comm->devs[1]); cudaFree(dstBuf);
        free(hostCheck);
    }

    comm->transportPath = comm->p2pDirectWorks ? INFCCL_PATH_P2P_DIRECT : INFCCL_PATH_STAGED;
    INFO("transport: %s", comm->p2pDirectWorks ? "P2P_DIRECT" : "STAGED");
    cudaSetDevice(savedDev);
    return infcclSuccess;
}

static infcclResult_t commAlloc(infcclComm_t* out, int ndev, int rank, int cudaDev) {
    infcclComm_t c = (infcclComm_t)calloc(1, sizeof(struct infcclComm));
    if (!c) return infcclSystemError;
    c->nDev = ndev;
    c->rank = rank;
    c->cudaDev = cudaDev;
    c->staged = NULL;
    c->stagedBytes = 0;
    c->buffSize = INFCCL_DEFAULT_BUFFER_SIZE;
    c->p2pDirectWorks = 0;
    c->transportPath = INFCCL_PATH_STAGED;
    memset(&c->events, 0, sizeof(InfcclEventQueue));
    *out = c;
    return infcclSuccess;
}

static infcclResult_t commInitEvents(infcclComm_t comm) {
    int savedDev; cudaGetDevice(&savedDev);
    cudaSetDevice(comm->cudaDev);
    for (int i = 0; i < INFCCL_MAX_QUEUE; i++) {
        CUDACHECK(cudaEventCreateWithFlags(&comm->events.isDone[i], cudaEventDisableTiming));
        CUDACHECK(cudaEventRecord(comm->events.isDone[i], 0));
    }
    comm->events.back = 0;
    cudaSetDevice(savedDev);
    return infcclSuccess;
}

static void commFree(infcclComm_t comm) {
    if (!comm) return;
    infcclIpcCleanup(comm);
    int savedDev; cudaGetDevice(&savedDev);
    cudaSetDevice(comm->cudaDev);
    if (comm->staged) cudaFree(comm->staged);
    for (int i = 0; i < INFCCL_MAX_QUEUE; i++)
        if (comm->events.isDone[i]) cudaEventDestroy(comm->events.isDone[i]);
    cudaSetDevice(savedDev);
    free(comm);
}

infcclResult_t infcclCommInitAll(infcclComm_t* comms, int ndev, const int* devlist) {
    initDebug();
    if (ndev < 1 || ndev > INFCCL_MAX_DEVS) {
        WARN("invalid device count %d", ndev);
        return infcclUnsupportedDeviceCount;
    }
    int savedDev; cudaGetDevice(&savedDev);
    for (int r = 0; r < ndev; r++) comms[r] = NULL;

    for (int r = 0; r < ndev; r++) {
        int dev = devlist ? devlist[r] : r;
        if (cudaSetDevice(dev) != cudaSuccess) {
            WARN("rank %d: bad device %d", r, dev);
            for (int i = 0; i < r; i++) { commFree(comms[i]); comms[i] = NULL; }
            cudaSetDevice(savedDev);
            return infcclInvalidDeviceIndex;
        }
        infcclResult_t res = commAlloc(&comms[r], ndev, r, dev);
        if (res != infcclSuccess) {
            for (int i = 0; i < r; i++) { commFree(comms[i]); comms[i] = NULL; }
            cudaSetDevice(savedDev);
            return res;
        }
        for (int i = 0; i < ndev; i++)
            comms[r]->devs[i] = devlist ? devlist[i] : i;
        res = commInitEvents(comms[r]);
        if (res != infcclSuccess) {
            for (int i = 0; i <= r; i++) { commFree(comms[i]); comms[i] = NULL; }
            cudaSetDevice(savedDev);
            return res;
        }
    }

    infcclResult_t res = buildTopology(comms[0]);
    if (res != infcclSuccess) goto fail;
    res = probeP2P(comms[0]);
    if (res != infcclSuccess) goto fail;

    for (int r = 0; r < ndev; r++) {
        memcpy(comms[r]->rankInfo, comms[0]->rankInfo, sizeof(InfcclRankInfo) * ndev);
        memcpy(comms[r]->userFromRing, comms[0]->userFromRing, sizeof(int) * ndev);
        memcpy(comms[r]->ringFromUser, comms[0]->ringFromUser, sizeof(int) * ndev);
        memcpy(comms[r]->peerMatrix, comms[0]->peerMatrix, sizeof(comms[0]->peerMatrix));
        comms[r]->p2pDirectWorks = comms[0]->p2pDirectWorks;
        comms[r]->transportPath = comms[0]->transportPath;
    }

    for (int r = 0; r < ndev; r++) {
        res = infcclIpcSetup(comms[r]);
        if (res != infcclSuccess) goto fail;
    }

    INFO("init %d devs, transport=%s, IPC streams ready", ndev, comms[0]->p2pDirectWorks ? "P2P" : "STAGED");
    cudaSetDevice(savedDev);
    return infcclSuccess;

fail:
    for (int i = 0; i < ndev; i++) { if (comms[i]) { commFree(comms[i]); comms[i] = NULL; } }
    cudaSetDevice(savedDev);
    return res;
}

infcclResult_t infcclCommDestroy(infcclComm_t comm) {
    commFree(comm);
    return infcclSuccess;
}

infcclResult_t infcclCommCount(const infcclComm_t comm, int* count) { *count = comm->nDev; return infcclSuccess; }
infcclResult_t infcclCommCuDevice(const infcclComm_t comm, int* device) { *device = comm->cudaDev; return infcclSuccess; }
infcclResult_t infcclCommUserRank(const infcclComm_t comm, int* rank) { *rank = comm->rank; return infcclSuccess; }

static infcclResult_t infcclIpcSetup(infcclComm_t comm) {
    int savedDev; cudaGetDevice(&savedDev);
    memset(&comm->ipc, 0, sizeof(infcclIpcConn));

    for (int g = 0; g < comm->nDev; g++) {
        CUDACHECK(cudaSetDevice(comm->devs[g]));
        CUDACHECK(cudaStreamCreateWithFlags(&comm->ipc.streams[g], cudaStreamNonBlocking));
    }

    cudaSetDevice(savedDev);
    return infcclSuccess;
}

static void infcclIpcCleanup(infcclComm_t comm) {
    int savedDev; cudaGetDevice(&savedDev);
    for (int i = 0; i < comm->nDev; i++) {
        for (int j = 0; j < comm->nDev; j++) {
            if (i != j && comm->ipc.mapped[i][j]) {
                cudaSetDevice(comm->devs[j]);
                cudaIpcCloseMemHandle(comm->ipc.mapped[i][j]);
                comm->ipc.mapped[i][j] = NULL;
            }
        }
    }
    for (int g = 0; g < comm->nDev; g++) {
        if (comm->ipc.streams[g]) {
            cudaSetDevice(comm->devs[g]);
            cudaStreamDestroy(comm->ipc.streams[g]);
            comm->ipc.streams[g] = NULL;
        }
        if (comm->ipc.base[g]) {
            cudaSetDevice(comm->devs[g]);
            cudaFree(comm->ipc.base[g]);
            comm->ipc.base[g] = NULL;
        }
    }
    cudaSetDevice(savedDev);
}

infcclResult_t infcclIpcExchangeBuffers(infcclComm_t comm, void** buffs, size_t bytes) {
    int savedDev; cudaGetDevice(&savedDev);

    for (int i = 0; i < comm->nDev; i++) {
        for (int j = 0; j < comm->nDev; j++) {
            if (i != j && comm->ipc.mapped[i][j]) {
                cudaSetDevice(comm->devs[j]);
                cudaIpcCloseMemHandle(comm->ipc.mapped[i][j]);
                comm->ipc.mapped[i][j] = NULL;
            }
        }
    }

    for (int g = 0; g < comm->nDev; g++) {
        cudaIpcMemHandle_t handle;
        CUDACHECK(cudaSetDevice(comm->devs[g]));
        CUDACHECK(cudaIpcGetMemHandle(&handle, buffs[g]));

        for (int other = 0; other < comm->nDev; other++) {
            if (other == g) continue;
            CUDACHECK(cudaSetDevice(comm->devs[other]));
            void* mapped = NULL;
            CUDACHECK(cudaIpcOpenMemHandle(&mapped, handle, cudaIpcMemLazyEnablePeerAccess));
            comm->ipc.mapped[g][other] = mapped;
        }
    }

    cudaSetDevice(savedDev);
    return infcclSuccess;
}

cudaStream_t infcclGetStream(infcclComm_t comm, int gpu) {
    return comm->ipc.streams[gpu];
}
