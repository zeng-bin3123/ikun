#ifndef INFCCL_BI_V100_H_
#define INFCCL_BI_V100_H_

#include <cuda_runtime.h>
#include <cstdio>

namespace infccl {

struct BiV100Caps {
    int warp_size;
    int fp64_works;
    int p2p_write_works;
    int p2p_read_works;
    int ipc_read_works;
    int ipc_free_safe;
    float hbm_bw_gbps;
    float peer_bw_gbps;
    float launch_overhead_us;
    int sm_count;
    int corex_major;
    int corex_minor;
};

__global__ void cap_warp_kernel(int* out) { if (threadIdx.x == 0) out[0] = warpSize; }
__global__ void cap_fp64_kernel(double* out) { out[0] = 1.0 + 2.0; }
__global__ void cap_fill(float* b, int n, float v) { int i = blockIdx.x*blockDim.x+threadIdx.x; if(i<n) b[i]=v; }
__global__ void cap_read_bw(float* dst, const float* src, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) dst[i]=src[i]; }

static inline int probeBiV100Caps(BiV100Caps* caps, int gpu = 0) {
    int saved; cudaGetDevice(&saved);
    cudaSetDevice(gpu);

    int* d_warp; cudaMalloc(&d_warp, 4);
    cap_warp_kernel<<<1,1>>>(d_warp);
    cudaDeviceSynchronize();
    cudaMemcpy(&caps->warp_size, d_warp, 4, cudaMemcpyDeviceToHost);
    cudaFree(d_warp);

    double* d_fp64; cudaMalloc(&d_fp64, 8);
    double h_fp64 = 0;
    cap_fp64_kernel<<<1,1>>>(d_fp64);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_fp64, d_fp64, 8, cudaMemcpyDeviceToHost);
    caps->fp64_works = (h_fp64 == 3.0) ? 1 : 0;
    cudaFree(d_fp64);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu);
    caps->sm_count = prop.multiProcessorCount;

    int N = 32 * 1024 * 1024;
    float *bs, *bd;
    cudaMalloc(&bs, N*4); cudaMalloc(&bd, N*4);
    cudaMemset(bs, 1, N*4);
    cap_read_bw<<<(N+255)/256,256>>>(bd, bs, N); cudaDeviceSynchronize();
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < 50; i++) cap_read_bw<<<(N+255)/256,256>>>(bd, bs, N);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    caps->hbm_bw_gbps = 2.0f * N * 4 * 50 / (ms / 1000.0f) / 1e9f;
    cudaFree(bs); cudaFree(bd);

    for (int i = 0; i < 100; i++) { cap_warp_kernel<<<1,1>>>(d_warp = nullptr); cudaMalloc(&d_warp, 4); cudaFree(d_warp); }
    cudaMalloc(&d_warp, 4);
    cudaEventRecord(t0);
    for (int i = 0; i < 1000; i++) cap_warp_kernel<<<1,1>>>(d_warp);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1);
    caps->launch_overhead_us = ms * 1000.0f / 1000;
    cudaFree(d_warp);
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    caps->p2p_write_works = 0;
    caps->p2p_read_works = 0;
    caps->ipc_read_works = 0;
    caps->ipc_free_safe = 0;
    caps->peer_bw_gbps = 0;
    caps->corex_major = 3;
    caps->corex_minor = 2;

    cudaSetDevice(saved);
    return 0;
}

static inline void printBiV100Caps(const BiV100Caps* c, FILE* out = stdout) {
    fprintf(out, "BI-V100 capabilities:\n");
    fprintf(out, "  warp_size=%d fp64=%s SMs=%d\n",
        c->warp_size, c->fp64_works?"ok":"BROKEN", c->sm_count);
    fprintf(out, "  HBM=%.0f GB/s launch=%.1f us\n", c->hbm_bw_gbps, c->launch_overhead_us);
    fprintf(out, "  P2P write=%s read=%s IPC read=%s IPC free=%s\n",
        c->p2p_write_works?"ok":"FAIL", c->p2p_read_works?"ok":"FAIL",
        c->ipc_read_works?"ok":"FAIL", c->ipc_free_safe?"ok":"TRAP");
    fprintf(out, "  peer BW=%.1f GB/s CoreX=%d.%d\n",
        c->peer_bw_gbps, c->corex_major, c->corex_minor);
}

}
#endif
