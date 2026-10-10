#include "transfer_engine.h"
#include "infccl.h"
#include "peer_transport.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

using namespace infccl;

__global__ void fill_kernel(float*b,int n,float v){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)b[i]=v;}

struct BenchConfig {
    int block_sizes[16];
    int num_block_sizes;
    int iters;
    int warmup;
    int bidirectional;
};

static void benchPairwise(PeerTransport* xport, TransferEngine* engine,
    int ndev, int src, int dst, const BenchConfig& cfg) {

    SegmentID seg_id = engine->openSegment("local");
    printf("  %d->%d: ", src, dst);

    for (int bs = 0; bs < cfg.num_block_sizes; bs++) {
        size_t bytes = cfg.block_sizes[bs];
        int N = bytes / 4;

        cudaSetDevice(xport->devId(src));
        float* sbuf; cudaMalloc(&sbuf, bytes);
        fill_kernel<<<(N+255)/256,256>>>(sbuf, N, 1.0f);
        cudaDeviceSynchronize();
        char sloc[16]; snprintf(sloc, 16, "gpu%d", src);
        engine->registerLocalMemory(sbuf, bytes, sloc);

        cudaSetDevice(xport->devId(dst));
        float* dbuf; cudaMalloc(&dbuf, bytes);
        cudaMemset(dbuf, 0, bytes);
        char dloc[16]; snprintf(dloc, 16, "gpu%d", dst);
        engine->registerLocalMemory(dbuf, bytes, dloc);

        for (int w = 0; w < cfg.warmup; w++) {
            Transport::BatchID bid = xport->allocateBatchID(1);
            std::vector<Transport::TransferRequest> reqs(1);
            reqs[0].opcode = Transport::TransferRequest::READ;
            reqs[0].source = dbuf;
            reqs[0].target_id = seg_id;
            reqs[0].target_offset = 0;
            reqs[0].length = bytes;
            xport->submitTransfer(bid, reqs);
            cudaSetDevice(xport->devId(dst));
            cudaDeviceSynchronize();
            xport->freeBatchID(bid);
        }

        cudaSetDevice(xport->devId(dst));
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);

        for (int it = 0; it < cfg.iters; it++) {
            Transport::BatchID bid = xport->allocateBatchID(1);
            std::vector<Transport::TransferRequest> reqs(1);
            reqs[0].opcode = Transport::TransferRequest::READ;
            reqs[0].source = dbuf;
            reqs[0].target_id = seg_id;
            reqs[0].target_offset = 0;
            reqs[0].length = bytes;
            xport->submitTransfer(bid, reqs);
            cudaDeviceSynchronize();
            xport->freeBatchID(bid);
        }

        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);
        float us_per = ms * 1000.0f / cfg.iters;
        float gbps = bytes * 1e-6f / (ms / cfg.iters);
        printf("%zuKB:%.1fus/%.1fGB/s ", bytes/1024, us_per, gbps);

        cudaEventDestroy(t0); cudaEventDestroy(t1);
        engine->unregisterLocalMemory(dbuf);
        engine->unregisterLocalMemory(sbuf);
        cudaSetDevice(xport->devId(src)); cudaFree(sbuf);
        cudaSetDevice(xport->devId(dst)); cudaFree(dbuf);
    }
    printf("\n");
}

static void benchAllReduce(PeerTransport* peer, int ndev, const int* devs, const BenchConfig& cfg) {
    printf("\nAllReduce benchmark (%d GPUs):\n", ndev);
    printf("  %-10s %8s %10s %10s\n", "size", "us/call", "algoBW", "busBW");

    infcclComm_t comms[INFCCL_MAX_DEVS];

    infcclCommInitAll(comms, ndev, devs);

    for (int bs = 0; bs < cfg.num_block_sizes; bs++) {
        size_t bytes = cfg.block_sizes[bs];
        int N = bytes / 4;

        float* bufs[INFCCL_MAX_DEVS];
        for (int g = 0; g < ndev; g++) {
            cudaSetDevice(devs[g]);
            cudaMalloc(&bufs[g], bytes);
            fill_kernel<<<(N+255)/256,256>>>(bufs[g], N, 1.0f);
            cudaDeviceSynchronize();
        }

        for (int w = 0; w < cfg.warmup; w++)
            infcclAllReduce((void**)bufs, N, 3, 0, comms[0], 0);

        cudaSetDevice(devs[0]);
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);
        for (int it = 0; it < cfg.iters; it++)
            infcclAllReduce((void**)bufs, N, 3, 0, comms[0], 0);
        cudaEventRecord(t1); cudaEventSynchronize(t1);

        float ms; cudaEventElapsedTime(&ms, t0, t1);
        float us = ms * 1000.0f / cfg.iters;
        float algo_bw = bytes * 2e-6f / (ms / cfg.iters);
        float bus_bw = algo_bw * (ndev - 1.0f) / ndev;
        printf("  %-10zu %8.1f %8.1fGB/s %8.1fGB/s\n", bytes, us, algo_bw, bus_bw);

        cudaEventDestroy(t0); cudaEventDestroy(t1);
        for (int g = 0; g < ndev; g++) { cudaSetDevice(devs[g]); cudaFree(bufs[g]); }
    }

    for (int r = 0; r < ndev; r++) infcclCommDestroy(comms[r]);
}

int main(int argc, char** argv) {
    int ngpu; cudaGetDeviceCount(&ngpu);
    printf("=== infccl Transfer Bench (%d GPUs) ===\n", ngpu);
    if (ngpu < 2) { printf("need 2+\n"); return 1; }

    int devs[INFCCL_MAX_DEVS];
    for (int i = 0; i < ngpu; i++) devs[i] = i;

    BenchConfig cfg;
    cfg.num_block_sizes = 0;
    int defaults[] = {4096, 65536, 262144, 1048576, 4194304, 16777216};
    for (int i = 0; i < 6; i++) cfg.block_sizes[cfg.num_block_sizes++] = defaults[i];
    cfg.iters = 100;
    cfg.warmup = 10;
    cfg.bidirectional = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--iters") == 0 && i+1 < argc) cfg.iters = atoi(argv[++i]);
        if (strcmp(argv[i], "--warmup") == 0 && i+1 < argc) cfg.warmup = atoi(argv[++i]);
    }

    auto meta = std::make_shared<TransferMetadata>();
    TransferEngine engine(meta);
    engine.init("local", ngpu, devs);
    Transport* xport = engine.installOrGetTransport("peer", nullptr);
    PeerTransport* peer = engine.getPeerTransport();

    peer->context().probeBandwidth();
    peer->context().probeLatency();
    peer->context().probeOptimalSlice();
    printf("\n");
    peer->context().printTopology();

    printf("\nPairwise READ bandwidth:\n");
    for (int i = 0; i < ngpu; i++)
        for (int j = 0; j < ngpu; j++) {
            if (i == j) continue;
            benchPairwise(peer, &engine, ngpu, i, j, cfg);
        }

    benchAllReduce(peer, ngpu, devs, cfg);

    printf("\nWorker stats: submitted=%lu completed=%lu failed=%lu bytes=%lu\n",
        (unsigned long)peer->worker().stats().slices_submitted.load(),
        (unsigned long)peer->worker().stats().slices_completed.load(),
        (unsigned long)peer->worker().stats().slices_failed.load(),
        (unsigned long)peer->worker().stats().bytes_transferred.load());

    printf("=== done ===\n");
    return 0;
}
