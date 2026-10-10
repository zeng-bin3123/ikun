#include "peer_transport.h"
#include "transfer_engine.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
using namespace infccl;
#define CCHK(x) do{cudaError_t e=(x);if(e){fprintf(stderr,"%s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)

struct BenchConfig {
    int warmup_iters;
    int measure_iters;
    int verify_data;
    int print_per_pair;
    int test_sizes_count;
    size_t test_sizes[16];

    static BenchConfig standard() {
        BenchConfig c;
        c.warmup_iters = 10;
        c.measure_iters = 50;
        c.verify_data = 1;
        c.print_per_pair = 1;
        c.test_sizes_count = 6;
        c.test_sizes[0] = 4096;
        c.test_sizes[1] = 65536;
        c.test_sizes[2] = 262144;
        c.test_sizes[3] = 1048576;
        c.test_sizes[4] = 4194304;
        c.test_sizes[5] = 16777216;
        return c;
    }

    static BenchConfig quick() {
        BenchConfig c;
        c.warmup_iters = 5;
        c.measure_iters = 20;
        c.verify_data = 0;
        c.print_per_pair = 0;
        c.test_sizes_count = 3;
        c.test_sizes[0] = 65536;
        c.test_sizes[1] = 1048576;
        c.test_sizes[2] = 16777216;
        return c;
    }

    static BenchConfig fromEnv() {
        BenchConfig c = standard();
        const char* e;
        e = getenv("BENCH_ITERS"); if (e) c.measure_iters = atoi(e);
        e = getenv("BENCH_WARMUP"); if (e) c.warmup_iters = atoi(e);
        e = getenv("BENCH_VERIFY"); if (e) c.verify_data = atoi(e);
        e = getenv("BENCH_QUICK"); if (e && atoi(e)) c = quick();
        return c;
    }
};

struct BenchResult {
    int src_gpu;
    int dst_gpu;
    size_t bytes;
    float raw_bw_gbps;
    float engine_bw_gbps;
    float latency_us;
    float overhead_pct;
    int data_ok;
};

static float measure_raw_bw(int src, int dst, size_t bytes, int iters) {
    float *s_buf, *d_buf;
    CCHK(cudaSetDevice(src)); CCHK(cudaMalloc(&s_buf, bytes)); CCHK(cudaMemset(s_buf, 1, bytes));
    CCHK(cudaSetDevice(dst)); CCHK(cudaMalloc(&d_buf, bytes));
    CCHK(cudaSetDevice(src));
    cudaStream_t s; cudaStreamCreate(&s);
    cudaMemcpyPeerAsync(d_buf, dst, s_buf, src, bytes, s);
    cudaStreamSynchronize(s);
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0, s);
    for (int i = 0; i < iters; i++)
        cudaMemcpyPeerAsync(d_buf, dst, s_buf, src, bytes, s);
    cudaEventRecord(t1, s); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    float bw = bytes * (float)iters * 1e-6f / ms;
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaStreamDestroy(s);
    CCHK(cudaSetDevice(src)); cudaFree(s_buf);
    CCHK(cudaSetDevice(dst)); cudaFree(d_buf);
    return bw;
}

static float measure_raw_latency(int src, int dst, int iters) {
    float *s_buf, *d_buf;
    CCHK(cudaSetDevice(src)); CCHK(cudaMalloc(&s_buf, 4));
    CCHK(cudaSetDevice(dst)); CCHK(cudaMalloc(&d_buf, 4));
    CCHK(cudaSetDevice(src));
    cudaStream_t s; cudaStreamCreate(&s);
    cudaMemcpyPeerAsync(d_buf, dst, s_buf, src, 4, s);
    cudaStreamSynchronize(s);
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0, s);
    for (int i = 0; i < iters; i++)
        cudaMemcpyPeerAsync(d_buf, dst, s_buf, src, 4, s);
    cudaEventRecord(t1, s); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    float lat = ms * 1000.0f / iters;
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaStreamDestroy(s);
    CCHK(cudaSetDevice(src)); cudaFree(s_buf);
    CCHK(cudaSetDevice(dst)); cudaFree(d_buf);
    return lat;
}

static BenchResult bench_pair(PeerTransport* peer, Transport* xport,
    int src, int dst, size_t bytes, const BenchConfig& cfg) {
    BenchResult r;
    r.src_gpu = src; r.dst_gpu = dst; r.bytes = bytes;
    r.data_ok = 1;

    float *s_buf, *d_buf;
    CCHK(cudaSetDevice(peer->devId(src))); CCHK(cudaMalloc(&s_buf, bytes));
    CCHK(cudaSetDevice(peer->devId(dst))); CCHK(cudaMalloc(&d_buf, bytes));

    if (cfg.verify_data) {
        float *h = (float*)malloc(bytes);
        for (size_t i = 0; i < bytes/4; i++) h[i] = (float)i * 0.001f + (float)(src*10+dst);
        CCHK(cudaSetDevice(peer->devId(src)));
        CCHK(cudaMemcpy(s_buf, h, bytes, cudaMemcpyHostToDevice));
        free(h);
    } else {
        CCHK(cudaSetDevice(peer->devId(src))); CCHK(cudaMemset(s_buf, 1, bytes));
    }

    for (int i = 0; i < cfg.warmup_iters; i++) {
        auto bid = xport->allocateBatchID(1);
        xport->submitTransfer(bid, {{Transport::TransferRequest::WRITE,
            s_buf, d_buf, bytes, src, dst, 0, 0}});
        xport->waitBatch(bid);
        xport->freeBatchID(bid);
    }

    int64_t start = now_us();
    for (int i = 0; i < cfg.measure_iters; i++) {
        auto bid = xport->allocateBatchID(1);
        xport->submitTransfer(bid, {{Transport::TransferRequest::WRITE,
            s_buf, d_buf, bytes, src, dst, 0, 0}});
        xport->waitBatch(bid);
        xport->freeBatchID(bid);
    }
    int64_t elapsed = now_us() - start;
    r.engine_bw_gbps = bytes * (float)cfg.measure_iters * 1e-3f / (float)elapsed;

    r.raw_bw_gbps = measure_raw_bw(peer->devId(src), peer->devId(dst),
        bytes, cfg.measure_iters);
    r.overhead_pct = 100.0f * (1.0f - r.engine_bw_gbps / r.raw_bw_gbps);
    if (r.overhead_pct < 0) r.overhead_pct = 0;

    if (cfg.verify_data) {
        float *h = (float*)malloc(bytes);
        CCHK(cudaSetDevice(peer->devId(dst)));
        CCHK(cudaMemcpy(h, d_buf, bytes, cudaMemcpyDeviceToHost));
        int errs = 0;
        for (size_t i = 0; i < bytes/4 && errs < 10; i++) {
            float expected = (float)i * 0.001f + (float)(src*10+dst);
            if (fabsf(h[i] - expected) > 0.01f) errs++;
        }
        r.data_ok = (errs == 0) ? 1 : 0;
        free(h);
    }

    CCHK(cudaSetDevice(peer->devId(src))); cudaFree(s_buf);
    CCHK(cudaSetDevice(peer->devId(dst))); cudaFree(d_buf);
    return r;
}

static void print_bw_matrix(PeerTransport* peer, Transport* xport,
    int ndev, size_t bytes, const BenchConfig& cfg) {
    printf("\n--- %zuKB transfer bandwidth (GB/s) ---\n", bytes/1024);
    printf("         ");
    for (int j = 0; j < ndev; j++) printf("  GPU%-2d   ", j);
    printf("\n");
    for (int i = 0; i < ndev; i++) {
        printf("GPU%d ->  ", i);
        for (int j = 0; j < ndev; j++) {
            if (i == j) { printf("    --    "); continue; }
            auto r = bench_pair(peer, xport, i, j, bytes, cfg);
            printf(" %5.1f%s   ", r.engine_bw_gbps, r.data_ok ? "" : "!");
        }
        printf("\n");
    }
}

static void print_latency_matrix(int ndev, const int* devs) {
    printf("\n--- latency matrix (us) ---\n");
    printf("         ");
    for (int j = 0; j < ndev; j++) printf("  GPU%-2d  ", j);
    printf("\n");
    for (int i = 0; i < ndev; i++) {
        printf("GPU%d ->  ", i);
        for (int j = 0; j < ndev; j++) {
            if (i == j) { printf("   --    "); continue; }
            float lat = measure_raw_latency(devs[i], devs[j], 500);
            printf(" %5.1f   ", lat);
        }
        printf("\n");
    }
}

static void print_overhead_sweep(PeerTransport* peer, Transport* xport,
    int src, int dst, const BenchConfig& cfg) {
    printf("\n--- overhead sweep %d->%d ---\n", src, dst);
    printf("%-12s  %-10s  %-10s  %-8s  %-6s\n", "size", "raw", "engine", "overhead", "ok");
    for (int si = 0; si < cfg.test_sizes_count; si++) {
        size_t bytes = cfg.test_sizes[si];
        auto r = bench_pair(peer, xport, src, dst, bytes, cfg);
        const char *unit = "B";
        float disp = (float)bytes;
        if (bytes >= 1048576) { disp = bytes / 1048576.0f; unit = "MB"; }
        else if (bytes >= 1024) { disp = bytes / 1024.0f; unit = "KB"; }
        printf("%-4.0f%-6s  %6.1f GB/s  %6.1f GB/s  %5.1f%%   %s\n",
            disp, unit, r.raw_bw_gbps, r.engine_bw_gbps, r.overhead_pct,
            r.data_ok ? "OK" : "FAIL");
    }
}

static void bench_concurrent_fanout(PeerTransport* peer, Transport* xport,
    int src, int ndev, size_t bytes, int iters) {
    printf("\n--- fan-out %d -> all (%zuKB x %d iters) ---\n", src, bytes/1024, iters);
    float *s_buf;
    CCHK(cudaSetDevice(peer->devId(src))); CCHK(cudaMalloc(&s_buf, bytes));
    CCHK(cudaMemset(s_buf, 1, bytes));
    float *d_bufs[INFCCL_MAX_DEVS];
    for (int d = 0; d < ndev; d++) {
        if (d == src) { d_bufs[d] = nullptr; continue; }
        CCHK(cudaSetDevice(peer->devId(d))); CCHK(cudaMalloc(&d_bufs[d], bytes));
    }

    int ntasks = ndev - 1;
    auto bid = xport->allocateBatchID(ntasks);
    xport->submitTransfer(bid, [&]{
        std::vector<Transport::TransferRequest> reqs;
        for (int d = 0; d < ndev; d++) {
            if (d == src) continue;
            reqs.push_back({Transport::TransferRequest::WRITE,
                s_buf, d_bufs[d], bytes, src, d, 0, 0});
        }
        return reqs;
    }());
    xport->waitBatch(bid);
    xport->freeBatchID(bid);

    int64_t start = now_us();
    for (int it = 0; it < iters; it++) {
        auto bid2 = xport->allocateBatchID(ntasks);
        std::vector<Transport::TransferRequest> reqs;
        for (int d = 0; d < ndev; d++) {
            if (d == src) continue;
            reqs.push_back({Transport::TransferRequest::WRITE,
                s_buf, d_bufs[d], bytes, src, d, 0, 0});
        }
        xport->submitTransfer(bid2, reqs);
        xport->waitBatch(bid2);
        xport->freeBatchID(bid2);
    }
    int64_t elapsed = now_us() - start;
    float total_bytes = (float)bytes * ntasks * iters;
    float bw = total_bytes * 1e-3f / (float)elapsed;
    float per_dest = bw / ntasks;
    printf("  aggregate: %.1f GB/s  per-dest: %.1f GB/s  time: %.1f ms\n",
        bw, per_dest, elapsed / 1000.0f);

    CCHK(cudaSetDevice(peer->devId(src))); cudaFree(s_buf);
    for (int d = 0; d < ndev; d++)
        if (d_bufs[d]) { CCHK(cudaSetDevice(peer->devId(d))); cudaFree(d_bufs[d]); }
}

static void bench_bidirectional(PeerTransport* peer, Transport* xport,
    int a, int b, size_t bytes, int iters) {
    printf("\n--- bidirectional %d <-> %d (%zuKB x %d iters) ---\n", a, b, bytes/1024, iters);
    float *ba, *bb, *da, *db;
    CCHK(cudaSetDevice(peer->devId(a))); CCHK(cudaMalloc(&ba, bytes)); CCHK(cudaMalloc(&da, bytes));
    CCHK(cudaSetDevice(peer->devId(b))); CCHK(cudaMalloc(&bb, bytes)); CCHK(cudaMalloc(&db, bytes));
    CCHK(cudaMemset(ba, 1, bytes)); CCHK(cudaMemset(bb, 2, bytes));

    auto bid = xport->allocateBatchID(2);
    xport->submitTransfer(bid, {
        {Transport::TransferRequest::WRITE, ba, db, bytes, a, b, 0, 0},
        {Transport::TransferRequest::WRITE, bb, da, bytes, b, a, 0, 0}});
    xport->waitBatch(bid);
    xport->freeBatchID(bid);

    int64_t start = now_us();
    for (int it = 0; it < iters; it++) {
        auto bid2 = xport->allocateBatchID(2);
        xport->submitTransfer(bid2, {
            {Transport::TransferRequest::WRITE, ba, db, bytes, a, b, 0, 0},
            {Transport::TransferRequest::WRITE, bb, da, bytes, b, a, 0, 0}});
        xport->waitBatch(bid2);
        xport->freeBatchID(bid2);
    }
    int64_t elapsed = now_us() - start;
    float bw = bytes * 2.0f * iters * 1e-3f / (float)elapsed;
    printf("  total: %.1f GB/s  per-dir: %.1f GB/s\n", bw, bw / 2);

    CCHK(cudaSetDevice(peer->devId(a))); cudaFree(ba); cudaFree(da);
    CCHK(cudaSetDevice(peer->devId(b))); cudaFree(bb); cudaFree(db);
}

int main(int argc, char** argv) {
    int ng; cudaGetDeviceCount(&ng);
    if (ng < 2) { fprintf(stderr, "need 2+ GPUs\n"); return 1; }
    if (ng > 4) ng = 4;
    int devs[4]; for (int i = 0; i < ng; i++) devs[i] = i;

    BenchConfig cfg = BenchConfig::fromEnv();
    printf("=== infccl transfer benchmark — %d GPUs ===\n", ng);
    printf("iters=%d warmup=%d verify=%d\n\n", cfg.measure_iters, cfg.warmup_iters, cfg.verify_data);

    auto meta = std::make_shared<TransferMetadata>();
    TransferEngine engine(meta);
    engine.init("local", ng, devs);
    Transport* xport = engine.installOrGetTransport("peer", nullptr);
    PeerTransport* peer = (PeerTransport*)xport;
    peer->context().printTopology();

    print_latency_matrix(ng, devs);
    print_bw_matrix(peer, xport, ng, 16*1024*1024, cfg);
    print_overhead_sweep(peer, xport, 0, 1, cfg);
    bench_concurrent_fanout(peer, xport, 0, ng, 16*1024*1024, cfg.measure_iters);
    bench_bidirectional(peer, xport, 0, 1, 16*1024*1024, cfg.measure_iters);

    printf("\n--- worker stats ---\n");
    peer->worker().printStats();
    printf("\n=== benchmark complete ===\n");
    return 0;
}
