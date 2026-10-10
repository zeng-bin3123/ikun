#ifndef INFCCL_AUTO_TUNER_H_
#define INFCCL_AUTO_TUNER_H_

#include "common.h"
#include "bi_v100.h"
#include "peer_context.h"
#include <cuda_runtime.h>

namespace infccl {

struct TuneResult {
    size_t optimal_slice_bytes;
    int optimal_num_streams;
    int use_staging;
    size_t staging_size;
    int preferred_direction;
    float expected_bw_gbps;
    float measured_overhead_pct;
};

struct PairTuneProfile {
    int src;
    int dst;
    float bw_1call_gbps;
    float bw_4call_gbps;
    float bw_16call_gbps;
    float bw_64call_gbps;
    float latency_us;
    float launch_overhead_us;
    size_t crossover_bytes;
    TuneResult best;
};

class AutoTuner {
public:
    AutoTuner() : ndev_(0) {
        memset(profiles_, 0, sizeof(profiles_));
        memset(tuned_, 0, sizeof(tuned_));
    }

    int init(PeerContext& ctx) {
        ndev_ = ctx.ndev();
        for (int i = 0; i < ndev_; i++)
            devs_[i] = ctx.gpu(i).dev_id;
        return OK;
    }

    int tunePair(int src, int dst) {
        if (src < 0 || src >= ndev_ || dst < 0 || dst >= ndev_ || src == dst)
            return ERR_INVALID_ARG;

        PairTuneProfile& p = profiles_[src][dst];
        p.src = src; p.dst = dst;

        int saved; cudaGetDevice(&saved);

        size_t buf_bytes = 64 * 1024 * 1024;
        float *s_buf, *d_buf;
        cudaSetDevice(devs_[src]);
        if (cudaMalloc(&s_buf, buf_bytes) != cudaSuccess) { cudaSetDevice(saved); return ERR_CUDA; }
        cudaSetDevice(devs_[dst]);
        if (cudaMalloc(&d_buf, buf_bytes) != cudaSuccess) {
            cudaSetDevice(devs_[src]); cudaFree(s_buf); cudaSetDevice(saved); return ERR_CUDA;
        }

        cudaSetDevice(devs_[src]);
        cudaStream_t s; cudaStreamCreate(&s);
        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);

        cudaMemcpyPeerAsync(d_buf, devs_[dst], s_buf, devs_[src], buf_bytes, s);
        cudaStreamSynchronize(s);

        p.latency_us = measureLatency(s, d_buf, devs_[dst], s_buf, devs_[src], t0, t1);
        p.launch_overhead_us = p.latency_us;

        p.bw_1call_gbps = measureBw(s, d_buf, devs_[dst], s_buf, devs_[src],
            16*1024*1024, 1, 50, t0, t1);
        p.bw_4call_gbps = measureBw(s, d_buf, devs_[dst], s_buf, devs_[src],
            4*1024*1024, 4, 50, t0, t1);
        p.bw_16call_gbps = measureBw(s, d_buf, devs_[dst], s_buf, devs_[src],
            1*1024*1024, 16, 50, t0, t1);
        p.bw_64call_gbps = measureBw(s, d_buf, devs_[dst], s_buf, devs_[src],
            256*1024, 64, 20, t0, t1);

        p.crossover_bytes = findCrossover(s, d_buf, devs_[dst], s_buf, devs_[src], t0, t1);

        p.best.optimal_slice_bytes = computeOptimalSlice(p);
        p.best.optimal_num_streams = computeOptimalStreams(p);
        p.best.use_staging = 0;
        p.best.staging_size = 0;
        p.best.expected_bw_gbps = p.bw_1call_gbps;
        p.best.measured_overhead_pct = 100.0f * (1.0f - p.bw_4call_gbps / p.bw_1call_gbps);

        float multi_stream_bw = measureMultiStreamBw(
            d_buf, devs_[dst], s_buf, devs_[src], 16*1024*1024, p.best.optimal_num_streams);
        if (multi_stream_bw > p.bw_1call_gbps * 1.05f) {
            p.best.expected_bw_gbps = multi_stream_bw;
        }

        cudaEventDestroy(t0); cudaEventDestroy(t1);
        cudaStreamDestroy(s);
        cudaSetDevice(devs_[src]); cudaFree(s_buf);
        cudaSetDevice(devs_[dst]); cudaFree(d_buf);
        cudaSetDevice(saved);

        tuned_[src][dst] = 1;
        return OK;
    }

    int tuneAll() {
        for (int i = 0; i < ndev_; i++)
            for (int j = 0; j < ndev_; j++) {
                if (i == j) continue;
                int rc = tunePair(i, j);
                if (rc != OK) return rc;
            }
        return OK;
    }

    const PairTuneProfile& profile(int src, int dst) const { return profiles_[src][dst]; }
    bool isTuned(int src, int dst) const { return tuned_[src][dst] != 0; }

    size_t sliceBytesFor(int src, int dst, size_t total) const {
        if (!tuned_[src][dst]) return 4 * 1024 * 1024;
        const auto& p = profiles_[src][dst];
        if (total <= p.crossover_bytes) return total;
        return p.best.optimal_slice_bytes;
    }

    void printProfile(int src, int dst, FILE* out = stdout) const {
        const auto& p = profiles_[src][dst];
        fprintf(out, "Pair %d->%d:\n", src, dst);
        fprintf(out, "  latency:     %.1f us\n", p.latency_us);
        fprintf(out, "  BW 1-call:   %.1f GB/s (16MB)\n", p.bw_1call_gbps);
        fprintf(out, "  BW 4-call:   %.1f GB/s (4x4MB)\n", p.bw_4call_gbps);
        fprintf(out, "  BW 16-call:  %.1f GB/s (16x1MB)\n", p.bw_16call_gbps);
        fprintf(out, "  BW 64-call:  %.1f GB/s (64x256KB)\n", p.bw_64call_gbps);
        fprintf(out, "  crossover:   %zu KB\n", p.crossover_bytes / 1024);
        fprintf(out, "  opt slice:   %zu KB\n", p.best.optimal_slice_bytes / 1024);
        fprintf(out, "  opt streams: %d\n", p.best.optimal_num_streams);
        fprintf(out, "  overhead:    %.1f%%\n", p.best.measured_overhead_pct);
        fprintf(out, "  expected BW: %.1f GB/s\n", p.best.expected_bw_gbps);
    }

    void printAll(FILE* out = stdout) const {
        fprintf(out, "=== AutoTuner results (%d GPUs) ===\n", ndev_);
        for (int i = 0; i < ndev_; i++)
            for (int j = 0; j < ndev_; j++) {
                if (i == j || !tuned_[i][j]) continue;
                printProfile(i, j, out);
            }
    }

private:
    float measureLatency(cudaStream_t s, float* dst, int dd, float* src, int sd,
        cudaEvent_t t0, cudaEvent_t t1) {
        int iters = 500;
        for (int i = 0; i < 50; i++)
            cudaMemcpyPeerAsync(dst, dd, src, sd, 4, s);
        cudaStreamSynchronize(s);
        cudaEventRecord(t0, s);
        for (int i = 0; i < iters; i++)
            cudaMemcpyPeerAsync(dst, dd, src, sd, 4, s);
        cudaEventRecord(t1, s); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);
        return ms * 1000.0f / iters;
    }

    float measureBw(cudaStream_t s, float* dst, int dd, float* src, int sd,
        size_t slice, int nslice, int iters, cudaEvent_t t0, cudaEvent_t t1) {
        size_t total = slice * nslice;
        for (int i = 0; i < nslice; i++)
            cudaMemcpyPeerAsync((char*)dst + i*slice, dd, (char*)src + i*slice, sd, slice, s);
        cudaStreamSynchronize(s);
        cudaEventRecord(t0, s);
        for (int it = 0; it < iters; it++)
            for (int i = 0; i < nslice; i++)
                cudaMemcpyPeerAsync((char*)dst + i*slice, dd, (char*)src + i*slice, sd, slice, s);
        cudaEventRecord(t1, s); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);
        return total * iters * 1e-6 / ms;
    }

    float measureMultiStreamBw(float* dst, int dd, float* src, int sd,
        size_t total, int nstreams) {
        if (nstreams < 1) nstreams = 1;
        if (nstreams > 4) nstreams = 4;
        cudaStream_t ss[4];
        for (int i = 0; i < nstreams; i++) cudaStreamCreate(&ss[i]);
        size_t per = total / nstreams;

        for (int i = 0; i < nstreams; i++)
            cudaMemcpyPeerAsync((char*)dst + i*per, dd, (char*)src + i*per, sd, per, ss[i]);
        for (int i = 0; i < nstreams; i++) cudaStreamSynchronize(ss[i]);

        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        int iters = 50;
        cudaEventRecord(t0, ss[0]);
        for (int it = 0; it < iters; it++)
            for (int i = 0; i < nstreams; i++)
                cudaMemcpyPeerAsync((char*)dst + i*per, dd, (char*)src + i*per, sd, per, ss[i]);
        for (int i = 0; i < nstreams; i++) cudaStreamSynchronize(ss[i]);
        cudaEventRecord(t1, ss[0]); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);

        cudaEventDestroy(t0); cudaEventDestroy(t1);
        for (int i = 0; i < nstreams; i++) cudaStreamDestroy(ss[i]);
        return total * iters * 1e-6 / ms;
    }

    size_t findCrossover(cudaStream_t s, float* dst, int dd, float* src, int sd,
        cudaEvent_t t0, cudaEvent_t t1) {
        size_t sizes[] = {4096, 16384, 65536, 262144, 1048576, 4194304};
        float best_bw = 0;
        size_t crossover = 1048576;
        for (int si = 0; si < 6; si++) {
            size_t sz = sizes[si];
            int iters = (sz <= 65536) ? 200 : 50;
            for (int i = 0; i < 10; i++)
                cudaMemcpyPeerAsync(dst, dd, src, sd, sz, s);
            cudaStreamSynchronize(s);
            cudaEventRecord(t0, s);
            for (int i = 0; i < iters; i++)
                cudaMemcpyPeerAsync(dst, dd, src, sd, sz, s);
            cudaEventRecord(t1, s); cudaEventSynchronize(t1);
            float ms; cudaEventElapsedTime(&ms, t0, t1);
            float bw = sz * iters * 1e-6 / ms;
            if (bw > best_bw * 0.9f && sz < crossover)
                crossover = sz;
            if (bw > best_bw) best_bw = bw;
        }
        return crossover;
    }

    size_t computeOptimalSlice(const PairTuneProfile& p) const {
        if (p.bw_4call_gbps > p.bw_1call_gbps * 0.9f) return 4 * 1024 * 1024;
        if (p.bw_16call_gbps > p.bw_1call_gbps * 0.85f) return 1 * 1024 * 1024;
        float target_us = p.launch_overhead_us * 50;
        size_t bytes = (size_t)(target_us * p.bw_1call_gbps * 1e3);
        if (bytes < 256 * 1024) bytes = 256 * 1024;
        if (bytes > 16 * 1024 * 1024) bytes = 16 * 1024 * 1024;
        return (bytes + 4095) & ~4095ULL;
    }

    int computeOptimalStreams(const PairTuneProfile& p) const {
        (void)p;
        return 2;
    }

    int ndev_;
    int devs_[INFCCL_MAX_DEVS];
    PairTuneProfile profiles_[INFCCL_MAX_DEVS][INFCCL_MAX_DEVS];
    int tuned_[INFCCL_MAX_DEVS][INFCCL_MAX_DEVS];
};

}
#endif
