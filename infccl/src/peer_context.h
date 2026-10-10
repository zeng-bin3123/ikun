#ifndef INFCCL_PEER_CONTEXT_H_
#define INFCCL_PEER_CONTEXT_H_

#include "common.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

namespace infccl {

struct GpuDeviceInfo {
    int dev_id;
    char name[64];
    char pci_bus_id[16];
    int numa_node;
    int sm_count;
    size_t total_mem;
    size_t free_mem;
    int clock_rate_khz;
    int mem_clock_khz;
    int pci_domain;
    int pci_bus;
    int pci_device;
    int can_map_host;
    int unified_addressing;
    int managed_memory;
    int concurrent_kernels;
    int warp_size;
};

struct PairTopology {
    int can_p2p;
    int p2p_write_works;
    int p2p_read_works;
    float measured_bw_gbps;
    float latency_us;
    int same_numa;
    int hops;
    size_t optimal_slice;
};

class PeerContext {
public:
    PeerContext() : ndev_(0), probed_(false) {
        memset(info_, 0, sizeof(info_));
        memset(topo_, 0, sizeof(topo_));
    }

    int init(int ndev, const int* devlist) {
        ndev_ = ndev;
        int saved; cudaGetDevice(&saved);

        for (int i = 0; i < ndev; i++) {
            int d = devlist ? devlist[i] : i;
            GpuDeviceInfo& g = info_[i];
            g.dev_id = d;

            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, d) != cudaSuccess)
                return ERR_CUDA;

            strncpy(g.name, prop.name, 63);
            g.sm_count = prop.multiProcessorCount;
            g.total_mem = prop.totalGlobalMem;
            g.clock_rate_khz = prop.clockRate;
            g.mem_clock_khz = prop.memoryClockRate;
            g.pci_domain = prop.pciDomainID;
            g.pci_bus = prop.pciBusID;
            g.pci_device = prop.pciDeviceID;
            g.can_map_host = prop.canMapHostMemory;
            g.unified_addressing = prop.unifiedAddressing;
            g.managed_memory = prop.managedMemory;
            g.concurrent_kernels = prop.concurrentKernels;
            g.warp_size = prop.warpSize;

            cudaSetDevice(d);
            cudaMemGetInfo(&g.free_mem, &g.total_mem);

            if (cudaDeviceGetPCIBusId(g.pci_bus_id, 16, d) != cudaSuccess)
                snprintf(g.pci_bus_id, 16, "%04x:%02x:%02x", g.pci_domain, g.pci_bus, g.pci_device);

            char path[256];
            snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/numa_node", g.pci_bus_id);
            FILE* f = fopen(path, "r");
            if (f) { fscanf(f, "%d", &g.numa_node); fclose(f); }
            else g.numa_node = -1;
        }

        for (int i = 0; i < ndev; i++) {
            for (int j = 0; j < ndev; j++) {
                PairTopology& t = topo_[i][j];
                if (i == j) {
                    t.can_p2p = 1;
                    t.same_numa = 1;
                    t.hops = 0;
                    continue;
                }
                cudaDeviceCanAccessPeer(&t.can_p2p, info_[i].dev_id, info_[j].dev_id);
                t.same_numa = (info_[i].numa_node >= 0 && info_[i].numa_node == info_[j].numa_node);
                t.hops = t.same_numa ? 1 : 2;
                t.measured_bw_gbps = 0.0f;
            }
        }

        cudaSetDevice(saved);
        return OK;
    }

    int probeBandwidth() {
        if (probed_) return OK;
        int saved; cudaGetDevice(&saved);
        const size_t probe_bytes = 4 * 1024 * 1024;
        const int iters = 20;

        for (int i = 0; i < ndev_; i++) {
            for (int j = 0; j < ndev_; j++) {
                if (i == j) continue;
                if (!topo_[i][j].can_p2p) continue;

                cudaSetDevice(info_[i].dev_id);
                void* src = nullptr;
                if (cudaMalloc(&src, probe_bytes) != cudaSuccess) continue;

                cudaSetDevice(info_[j].dev_id);
                void* dst = nullptr;
                if (cudaMalloc(&dst, probe_bytes) != cudaSuccess) {
                    cudaSetDevice(info_[i].dev_id); cudaFree(src); continue;
                }

                cudaMemcpyPeer(dst, info_[j].dev_id, src, info_[i].dev_id, probe_bytes);

                cudaEvent_t t0, t1;
                cudaEventCreate(&t0); cudaEventCreate(&t1);
                cudaEventRecord(t0);
                for (int k = 0; k < iters; k++)
                    cudaMemcpyPeer(dst, info_[j].dev_id, src, info_[i].dev_id, probe_bytes);
                cudaEventRecord(t1);
                cudaEventSynchronize(t1);

                float ms = 0;
                cudaEventElapsedTime(&ms, t0, t1);
                topo_[i][j].measured_bw_gbps = (float)(probe_bytes * iters) / (ms * 1e6f);

                cudaEventDestroy(t0); cudaEventDestroy(t1);
                cudaSetDevice(info_[j].dev_id); cudaFree(dst);
                cudaSetDevice(info_[i].dev_id); cudaFree(src);
            }
        }

        cudaSetDevice(saved);
        probed_ = true;
        return OK;
    }

    int probeP2PCapabilities() {
        int saved; cudaGetDevice(&saved);
        for (int i = 0; i < ndev_; i++) {
            for (int j = 0; j < ndev_; j++) {
                if (i == j) continue;
                if (!topo_[i][j].can_p2p) continue;

                cudaSetDevice(info_[i].dev_id);
                cudaDeviceEnablePeerAccess(info_[j].dev_id, 0);
                cudaGetLastError();
                cudaSetDevice(info_[j].dev_id);
                cudaDeviceEnablePeerAccess(info_[i].dev_id, 0);
                cudaGetLastError();

                float* buf; int* errs;
                cudaSetDevice(info_[j].dev_id);
                cudaMalloc(&buf, 1024 * 4); cudaMemset(buf, 0, 1024 * 4);
                cudaMalloc(&errs, 4); cudaMemset(errs, 0, 4);

                cudaSetDevice(info_[i].dev_id);
                auto fill = [](float* b, int n, float v) {
                    for (int x = 0; x < n; x++) ((float*)b)[x] = v;
                };
                (void)fill;

                float* h_check = (float*)malloc(4 * 4);

                float val = 7.0f;
                cudaMemset(buf, 0, 1024 * 4);
                cudaSetDevice(info_[i].dev_id);
                cudaMemcpyPeer(buf, info_[j].dev_id, &val, info_[i].dev_id, 0);

                cudaSetDevice(info_[j].dev_id);
                cudaMemcpy(h_check, buf, 16, cudaMemcpyDeviceToHost);

                topo_[i][j].p2p_write_works = 0;
                topo_[i][j].p2p_read_works = 0;

                cudaFree(buf); cudaFree(errs);
                free(h_check);
            }
        }
        cudaSetDevice(saved);
        return OK;
    }

    int probeOptimalSlice() {
        int saved; cudaGetDevice(&saved);
        size_t test_sizes[] = {4096, 16384, 65536, 262144, 1048576, 4194304};
        int n_sizes = 6;

        for (int i = 0; i < ndev_; i++) {
            for (int j = 0; j < ndev_; j++) {
                if (i == j) continue;
                if (!topo_[i][j].can_p2p) { topo_[i][j].optimal_slice = 65536; continue; }

                size_t best_slice = 65536;
                float best_bw = 0;

                for (int si = 0; si < n_sizes; si++) {
                    size_t total = 4 * 1024 * 1024;
                    size_t slice = test_sizes[si];
                    if (slice > total) continue;
                    int nslice = total / slice;

                    cudaSetDevice(info_[i].dev_id);
                    void* src; cudaMalloc(&src, total);
                    cudaSetDevice(info_[j].dev_id);
                    void* dst; cudaMalloc(&dst, total);
                    cudaStream_t s; cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);

                    for (int k = 0; k < nslice; k++)
                        cudaMemcpyPeerAsync((char*)dst + k * slice, info_[j].dev_id,
                            (char*)src + k * slice, info_[i].dev_id, slice, s);
                    cudaStreamSynchronize(s);

                    cudaEvent_t t0, t1;
                    cudaEventCreate(&t0); cudaEventCreate(&t1);
                    int iters = 20;
                    cudaEventRecord(t0, s);
                    for (int it = 0; it < iters; it++)
                        for (int k = 0; k < nslice; k++)
                            cudaMemcpyPeerAsync((char*)dst + k * slice, info_[j].dev_id,
                                (char*)src + k * slice, info_[i].dev_id, slice, s);
                    cudaEventRecord(t1, s);
                    cudaEventSynchronize(t1);

                    float ms; cudaEventElapsedTime(&ms, t0, t1);
                    float bw = (float)(total * iters) / (ms * 1e6f);

                    if (bw > best_bw) { best_bw = bw; best_slice = slice; }

                    cudaEventDestroy(t0); cudaEventDestroy(t1);
                    cudaStreamDestroy(s);
                    cudaSetDevice(info_[i].dev_id); cudaFree(src);
                    cudaSetDevice(info_[j].dev_id); cudaFree(dst);
                }
                topo_[i][j].optimal_slice = best_slice;
            }
        }
        cudaSetDevice(saved);
        return OK;
    }

    int probeLatency() {
        int saved; cudaGetDevice(&saved);
        for (int i = 0; i < ndev_; i++) {
            for (int j = 0; j < ndev_; j++) {
                if (i == j) { topo_[i][j].latency_us = 0; continue; }
                if (!topo_[i][j].can_p2p) { topo_[i][j].latency_us = 999; continue; }

                cudaSetDevice(info_[i].dev_id);
                void* src; cudaMalloc(&src, 4);
                cudaSetDevice(info_[j].dev_id);
                void* dst; cudaMalloc(&dst, 4);

                for (int w = 0; w < 20; w++)
                    cudaMemcpyPeer(dst, info_[j].dev_id, src, info_[i].dev_id, 4);

                cudaEvent_t t0, t1;
                cudaEventCreate(&t0); cudaEventCreate(&t1);
                int iters = 200;
                cudaEventRecord(t0);
                for (int it = 0; it < iters; it++)
                    cudaMemcpyPeer(dst, info_[j].dev_id, src, info_[i].dev_id, 4);
                cudaEventRecord(t1); cudaEventSynchronize(t1);
                float ms; cudaEventElapsedTime(&ms, t0, t1);
                topo_[i][j].latency_us = ms * 1000.0f / iters;

                cudaEventDestroy(t0); cudaEventDestroy(t1);
                cudaSetDevice(info_[i].dev_id); cudaFree(src);
                cudaSetDevice(info_[j].dev_id); cudaFree(dst);
            }
        }
        cudaSetDevice(saved);
        return OK;
    }

    size_t optimalSlice(int src, int dst) const {
        if (src >= 0 && src < ndev_ && dst >= 0 && dst < ndev_)
            return topo_[src][dst].optimal_slice;
        return 65536;
    }

    int bestPeer(int gpu) const {
        int best = -1;
        float best_bw = 0;
        for (int j = 0; j < ndev_; j++) {
            if (j == gpu) continue;
            if (topo_[gpu][j].measured_bw_gbps > best_bw) {
                best_bw = topo_[gpu][j].measured_bw_gbps;
                best = j;
            }
        }
        return best;
    }

    void printTopology(FILE* out = stdout) const {
        fprintf(out, "GPU topology (%d devices):\n", ndev_);
        for (int i = 0; i < ndev_; i++) {
            const GpuDeviceInfo& g = info_[i];
            fprintf(out, "  GPU%d: %s dev=%d pci=%s numa=%d SMs=%d mem=%zuMB warp=%d\n",
                i, g.name, g.dev_id, g.pci_bus_id, g.numa_node,
                g.sm_count, g.total_mem >> 20, g.warp_size);
        }
        fprintf(out, "P2P bandwidth (GB/s):\n     ");
        for (int j = 0; j < ndev_; j++) fprintf(out, " GPU%-2d", j);
        fprintf(out, "\n");
        for (int i = 0; i < ndev_; i++) {
            fprintf(out, "GPU%d:", i);
            for (int j = 0; j < ndev_; j++) {
                if (i == j) fprintf(out, "   -- ");
                else if (!topo_[i][j].can_p2p) fprintf(out, "   N  ");
                else if (topo_[i][j].measured_bw_gbps > 0)
                    fprintf(out, " %4.1f ", topo_[i][j].measured_bw_gbps);
                else fprintf(out, "   Y  ");
            }
            fprintf(out, "\n");
        }
        bool has_latency = false;
        for (int i = 0; i < ndev_; i++)
            for (int j = 0; j < ndev_; j++)
                if (i != j && topo_[i][j].latency_us > 0 && topo_[i][j].latency_us < 999) has_latency = true;
        if (has_latency) {
            fprintf(out, "P2P latency (us):\n     ");
            for (int j = 0; j < ndev_; j++) fprintf(out, " GPU%-2d", j);
            fprintf(out, "\n");
            for (int i = 0; i < ndev_; i++) {
                fprintf(out, "GPU%d:", i);
                for (int j = 0; j < ndev_; j++) {
                    if (i == j) fprintf(out, "   -- ");
                    else fprintf(out, " %4.1f ", topo_[i][j].latency_us);
                }
                fprintf(out, "\n");
            }
        }
        bool has_slice = false;
        for (int i = 0; i < ndev_; i++)
            for (int j = 0; j < ndev_; j++)
                if (i != j && topo_[i][j].optimal_slice > 0) has_slice = true;
        if (has_slice) {
            fprintf(out, "Optimal slice (KB):\n     ");
            for (int j = 0; j < ndev_; j++) fprintf(out, " GPU%-2d", j);
            fprintf(out, "\n");
            for (int i = 0; i < ndev_; i++) {
                fprintf(out, "GPU%d:", i);
                for (int j = 0; j < ndev_; j++) {
                    if (i == j) fprintf(out, "   -- ");
                    else fprintf(out, " %4zu ", topo_[i][j].optimal_slice / 1024);
                }
                fprintf(out, "\n");
            }
        }
    }

    int ndev() const { return ndev_; }
    const GpuDeviceInfo& gpu(int i) const { return info_[i]; }
    const PairTopology& pair(int i, int j) const { return topo_[i][j]; }

private:
    int ndev_;
    GpuDeviceInfo info_[INFCCL_MAX_DEVS];
    PairTopology topo_[INFCCL_MAX_DEVS][INFCCL_MAX_DEVS];
    bool probed_;
};

}
#endif
