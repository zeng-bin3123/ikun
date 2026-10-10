#include "transfer_engine.h"
#include "kernels.h"
#include <cuda_fp16.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>

namespace infccl {

struct CommState {
    int nDev;
    int rank;
    int cudaDev;
    int devs[INFCCL_MAX_DEVS];
    TransferEngine* engine;
    PeerTransport* peer;
};

}

typedef infccl::CommState* infcclComm_t;
typedef int infcclResult_t;
#define infcclSuccess 0

static infccl::TransferEngine* g_engine = nullptr;

extern "C" {

infcclResult_t infcclCommInitAll(infcclComm_t* comms, int ndev, const int* devlist) {
    using namespace infccl;
    if (ndev < 1 || ndev > INFCCL_MAX_DEVS) return -1;
    auto meta = std::make_shared<TransferMetadata>();
    g_engine = new TransferEngine(meta);
    int dl[INFCCL_MAX_DEVS];
    for (int i = 0; i < ndev; i++) dl[i] = devlist ? devlist[i] : i;
    g_engine->init("local", ndev, dl);
    Transport* xport = g_engine->installOrGetTransport("peer", nullptr);
    if (!xport) { delete g_engine; g_engine = nullptr; return -1; }
    PeerTransport* peer = g_engine->getPeerTransport();

    for (int rank = 0; rank < ndev; rank++) {
        auto* c = (CommState*)calloc(1, sizeof(CommState));
        c->nDev = ndev; c->rank = rank; c->cudaDev = dl[rank];
        memcpy(c->devs, dl, sizeof(int) * ndev);
        c->engine = g_engine; c->peer = peer;
        comms[rank] = c;
    }
    return 0;
}

infcclResult_t infcclCommDestroy(infcclComm_t comm) {
    if (!comm) return 0;
    if (comm->rank == 0 && g_engine) { delete g_engine; g_engine = nullptr; }
    free(comm);
    return 0;
}

infcclResult_t infcclCommCount(const infcclComm_t c, int* n) { *n = c->nDev; return 0; }
infcclResult_t infcclCommCuDevice(const infcclComm_t c, int* d) { *d = c->cudaDev; return 0; }
infcclResult_t infcclCommUserRank(const infcclComm_t c, int* r) { *r = c->rank; return 0; }

const char* infcclGetErrorString(int code) {
    switch (code) {
        case 0: return "no error";
        default: return "error";
    }
}

infcclResult_t infcclRegisterMemory(infcclComm_t comm, int gpu, void* addr, size_t bytes) {
    char loc[16]; snprintf(loc, 16, "gpu%d", gpu);
    return comm->engine->registerLocalMemory(addr, bytes, loc, true) == 0 ? 0 : -1;
}

infcclResult_t infcclUnregisterMemory(infcclComm_t comm, int gpu, void* addr) {
    (void)gpu;
    return comm->engine->unregisterLocalMemory(addr) == 0 ? 0 : -1;
}

}

namespace infccl {

template<typename T, class F>
static int starAllReduce(void** buffs, int count, CommState* comm) {
    int ndev = comm->nDev;
    size_t bytes = count * sizeof(T);
    int saved; cudaGetDevice(&saved);
    int blk = INFCCL_BLOCKS(count / (sizeof(Pack64) / sizeof(T)));
    if (blk < 1) blk = 1;

    cudaSetDevice(comm->devs[0]);
    cudaStream_t s0 = comm->peer->stream(0);
    for (int p = 1; p < ndev; p++) {
        comm->peer->stageFrom(0, buffs[p], p, 0, bytes);
        cudaStreamSynchronize(s0);
        reduce_local_kernel<T,F><<<blk, INFCCL_THREADS, 0, s0>>>(
            (T*)buffs[0], (const T*)comm->peer->stagingBuffer(0), count);
        cudaStreamSynchronize(s0);
    }
    for (int g = 1; g < ndev; g++) {
        comm->peer->stageFrom(g, buffs[0], 0, 0, bytes);
        cudaSetDevice(comm->devs[g]);
        cudaStreamSynchronize(comm->peer->stream(g));
        cudaMemcpyAsync(buffs[g], comm->peer->stagingBuffer(g), bytes,
            cudaMemcpyDeviceToDevice, comm->peer->stream(g));
    }
    for (int g = 1; g < ndev; g++) {
        cudaSetDevice(comm->devs[g]);
        cudaStreamSynchronize(comm->peer->stream(g));
    }
    cudaSetDevice(saved);
    return OK;
}

template<typename T, class F>
static int ringAllReduce(void** buffs, int count, CommState* comm) {
    int ndev = comm->nDev;
    if (count == 0) return OK;
    int chunk = count / ndev;
    if (chunk < 1) return starAllReduce<T, F>(buffs, count, comm);
    int rem = count - chunk * ndev;
    int saved; cudaGetDevice(&saved);

    for (int step = 0; step < ndev - 1; step++) {
        for (int g = 0; g < ndev; g++) {
            int ci = (g - step - 1 + 2 * ndev) % ndev;
            int off = ci * chunk + (ci < rem ? ci : rem);
            int len = chunk + (ci < rem ? 1 : 0);
            if (len < 1) continue;
            int src = (g - 1 + ndev) % ndev;
            comm->peer->stageFrom(g, buffs[src], src, off * sizeof(T), len * sizeof(T));
        }
        for (int g = 0; g < ndev; g++) {
            cudaSetDevice(comm->devs[g]);
            cudaStreamSynchronize(comm->peer->stream(g));
        }
        for (int g = 0; g < ndev; g++) {
            int ci = (g - step - 1 + 2 * ndev) % ndev;
            int off = ci * chunk + (ci < rem ? ci : rem);
            int len = chunk + (ci < rem ? 1 : 0);
            if (len < 1) continue;
            cudaSetDevice(comm->devs[g]);
            int blk = INFCCL_BLOCKS(len / (sizeof(Pack64) / sizeof(T)));
            if (blk < 1) blk = 1;
            reduce_local_kernel<T,F><<<blk, INFCCL_THREADS, 0, comm->peer->stream(g)>>>(
                (T*)buffs[g] + off, (const T*)comm->peer->stagingBuffer(g), len);
        }
        for (int g = 0; g < ndev; g++) {
            cudaSetDevice(comm->devs[g]);
            cudaStreamSynchronize(comm->peer->stream(g));
        }
    }

    for (int step = 0; step < ndev - 1; step++) {
        for (int g = 0; g < ndev; g++) {
            int ci = (g - step + 2 * ndev) % ndev;
            int off = ci * chunk + (ci < rem ? ci : rem);
            int len = chunk + (ci < rem ? 1 : 0);
            if (len < 1) continue;
            int src = (g - 1 + ndev) % ndev;
            comm->peer->stageFrom(g, buffs[src], src, off * sizeof(T), len * sizeof(T));
        }
        for (int g = 0; g < ndev; g++) {
            cudaSetDevice(comm->devs[g]);
            cudaStreamSynchronize(comm->peer->stream(g));
        }
        for (int g = 0; g < ndev; g++) {
            int ci = (g - step + 2 * ndev) % ndev;
            int off = ci * chunk + (ci < rem ? ci : rem);
            int len = chunk + (ci < rem ? 1 : 0);
            if (len < 1) continue;
            cudaSetDevice(comm->devs[g]);
            cudaMemcpyAsync((T*)buffs[g] + off, comm->peer->stagingBuffer(g),
                len * sizeof(T), cudaMemcpyDeviceToDevice, comm->peer->stream(g));
        }
        for (int g = 0; g < ndev; g++) {
            cudaSetDevice(comm->devs[g]);
            cudaStreamSynchronize(comm->peer->stream(g));
        }
    }

    cudaSetDevice(saved);
    return OK;
}

template<typename T, class F>
static int doReduce(void** buffs, int count, int root, CommState* comm) {
    int ndev = comm->nDev, saved; cudaGetDevice(&saved);
    size_t bytes = count * sizeof(T);
    int blk = INFCCL_BLOCKS(count / (sizeof(Pack64) / sizeof(T)));
    if (blk < 1) blk = 1;
    cudaSetDevice(comm->devs[root]);
    cudaStream_t sr = comm->peer->stream(root);
    for (int p = 0; p < ndev; p++) {
        if (p == root) continue;
        comm->peer->stageFrom(root, buffs[p], p, 0, bytes);
        cudaStreamSynchronize(sr);
        reduce_local_kernel<T,F><<<blk, INFCCL_THREADS, 0, sr>>>(
            (T*)buffs[root], (const T*)comm->peer->stagingBuffer(root), count);
        cudaStreamSynchronize(sr);
    }
    cudaSetDevice(saved);
    return OK;
}

template<typename T>
static int doBcast(void** buffs, int count, int root, CommState* comm) {
    int ndev = comm->nDev, saved; cudaGetDevice(&saved);
    size_t bytes = count * sizeof(T);
    for (int g = 0; g < ndev; g++) {
        if (g == root) continue;
        comm->peer->stageFrom(g, buffs[root], root, 0, bytes);
    }
    for (int g = 0; g < ndev; g++) {
        if (g == root) continue;
        cudaSetDevice(comm->devs[g]);
        cudaStreamSynchronize(comm->peer->stream(g));
        cudaMemcpyAsync(buffs[g], comm->peer->stagingBuffer(g), bytes,
            cudaMemcpyDeviceToDevice, comm->peer->stream(g));
    }
    for (int g = 0; g < ndev; g++) {
        if (g != root) { cudaSetDevice(comm->devs[g]); cudaStreamSynchronize(comm->peer->stream(g)); }
    }
    cudaSetDevice(saved);
    return OK;
}

template<typename T>
static int doAllGather(void** sbufs, void** rbufs, int sc, CommState* comm) {
    int ndev = comm->nDev, saved; cudaGetDevice(&saved);
    size_t bytes = sc * sizeof(T);
    for (int g = 0; g < ndev; g++) {
        cudaSetDevice(comm->devs[g]);
        cudaStream_t s = comm->peer->stream(g);
        cudaMemcpyAsync((T*)rbufs[g] + g * sc, sbufs[g], bytes, cudaMemcpyDeviceToDevice, s);
        for (int src = 0; src < ndev; src++) {
            if (src == g) continue;
            comm->peer->stageFrom(g, sbufs[src], src, 0, bytes);
            cudaStreamSynchronize(s);
            cudaMemcpyAsync((T*)rbufs[g] + src * sc, comm->peer->stagingBuffer(g),
                bytes, cudaMemcpyDeviceToDevice, s);
        }
    }
    for (int g = 0; g < ndev; g++) {
        cudaSetDevice(comm->devs[g]);
        cudaStreamSynchronize(comm->peer->stream(g));
    }
    cudaSetDevice(saved);
    return OK;
}

template<typename T, class F>
static int doReduceScatter(void** sbufs, void** rbufs, int rc, CommState* comm) {
    int ndev = comm->nDev, saved; cudaGetDevice(&saved);
    size_t chunk_bytes = rc * sizeof(T);
    int blk = INFCCL_BLOCKS(rc / (sizeof(Pack64) / sizeof(T)));
    if (blk < 1) blk = 1;
    for (int g = 0; g < ndev; g++) {
        cudaSetDevice(comm->devs[g]);
        cudaStream_t s = comm->peer->stream(g);
        cudaMemcpyAsync(rbufs[g], (T*)sbufs[g] + g * rc, chunk_bytes, cudaMemcpyDeviceToDevice, s);
        for (int src = 0; src < ndev; src++) {
            if (src == g) continue;
            comm->peer->stageFrom(g, sbufs[src], src, g * chunk_bytes, chunk_bytes);
            cudaStreamSynchronize(s);
            reduce_local_kernel<T,F><<<blk, INFCCL_THREADS, 0, s>>>(
                (T*)rbufs[g], (const T*)comm->peer->stagingBuffer(g), rc);
            cudaStreamSynchronize(s);
        }
    }
    cudaSetDevice(saved);
    return OK;
}

}

extern "C" {

#define DISPATCH_OP(T,buffs,count,op,comm) switch(op){\
    case 0:return infccl::ringAllReduce<T,FuncSum<T>>(buffs,count,comm);\
    case 1:return infccl::ringAllReduce<T,FuncProd<T>>(buffs,count,comm);\
    case 2:return infccl::ringAllReduce<T,FuncMax<T>>(buffs,count,comm);\
    case 3:return infccl::ringAllReduce<T,FuncMin<T>>(buffs,count,comm);\
    default:return -1;}

int infcclAllReduce(void** buffs, int count, int datatype, int op, infcclComm_t comm, cudaStream_t stream) {
    (void)stream;
    switch (datatype) {
        case 3: DISPATCH_OP(float, buffs, count, op, comm);
        case 2: DISPATCH_OP(half, buffs, count, op, comm);
        case 1: DISPATCH_OP(int, buffs, count, op, comm);
        case 0: DISPATCH_OP(char, buffs, count, op, comm);
        case 4: DISPATCH_OP(long long, buffs, count, op, comm);
        case 5: DISPATCH_OP(unsigned long long, buffs, count, op, comm);
        default: return -1;
    }
}

int infcclReduce(void** buffs, int count, int datatype, int op, int root, infcclComm_t comm, cudaStream_t stream) {
    (void)stream;
    if (datatype == 3) { switch(op) {
        case 0: return infccl::doReduce<float,FuncSum<float>>(buffs,count,root,comm);
        case 1: return infccl::doReduce<float,FuncProd<float>>(buffs,count,root,comm);
        case 2: return infccl::doReduce<float,FuncMax<float>>(buffs,count,root,comm);
        case 3: return infccl::doReduce<float,FuncMin<float>>(buffs,count,root,comm);
        default: return -1;
    }}
    if (datatype == 2) return infccl::doReduce<half,FuncSum<half>>(buffs,count,root,comm);
    if (datatype == 1) return infccl::doReduce<int,FuncSum<int>>(buffs,count,root,comm);
    return -1;
}

int infcclBcast(void** buffs, int count, int datatype, int root, infcclComm_t comm, cudaStream_t stream) {
    (void)stream;
    switch (datatype) {
        case 3: return infccl::doBcast<float>(buffs,count,root,comm);
        case 2: return infccl::doBcast<half>(buffs,count,root,comm);
        case 1: return infccl::doBcast<int>(buffs,count,root,comm);
        default: return -1;
    }
}

int infcclAllGather(void** sbufs, void** rbufs, int sendcount, int datatype, infcclComm_t comm, cudaStream_t stream) {
    (void)stream;
    switch (datatype) {
        case 3: return infccl::doAllGather<float>(sbufs,rbufs,sendcount,comm);
        case 2: return infccl::doAllGather<half>(sbufs,rbufs,sendcount,comm);
        case 1: return infccl::doAllGather<int>(sbufs,rbufs,sendcount,comm);
        default: return -1;
    }
}

int infcclReduceScatter(void** sbufs, void** rbufs, int recvcount, int datatype, int op, infcclComm_t comm, cudaStream_t stream) {
    (void)stream;
    if (datatype == 3) { switch(op) {
        case 0: return infccl::doReduceScatter<float,FuncSum<float>>(sbufs,rbufs,recvcount,comm);
        case 2: return infccl::doReduceScatter<float,FuncMax<float>>(sbufs,rbufs,recvcount,comm);
        case 3: return infccl::doReduceScatter<float,FuncMin<float>>(sbufs,rbufs,recvcount,comm);
        default: return -1;
    }}
    if (datatype == 2) return infccl::doReduceScatter<half,FuncSum<half>>(sbufs,rbufs,recvcount,comm);
    return -1;
}

}
