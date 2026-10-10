#ifndef INFCCL_BI_V100_H_
#define INFCCL_BI_V100_H_

#include <cuda_runtime.h>
#include <cstdio>

namespace infccl {

struct BiV100Caps {
    int warp_size;
    int fp64_works;
    int p2p_kernel_write_works;
    int p2p_kernel_read_works;
    int ipc_handle_works;
    int ipc_free_safe;
    int peer_memcpy_needs_src_device;
    int peer_memcpy_needs_blocking_stream;
    int peer_memcpy_stream_any_device;
    float hbm_bw_gbps;
    float peer_bw_gbps;
    float peer_latency_us;
    float memcpy_launch_overhead_us;
    int sm_count;
    int max_threads_per_sm;
    int shared_mem_per_sm;
    int l2_cache_bytes;
    int corex_major;
    int corex_minor;
    int driver_major;
    int driver_minor;
    int cuda_compat_major;
    int cuda_compat_minor;
    int num_copy_engines;
};

static const BiV100Caps BI_V100_KNOWN_CAPS = {
    64,
    0,
    0,
    0,
    1,
    0,
    1,
    1,
    1,
    0, 0, 0, 0,
    16,
    0, 0, 0,
    0, 0,
    0, 0,
    10, 2,
    0
};

struct BiV100Constraints {
    static bool peerMemcpyNeedsSrcDevice() { return true; }
    static bool peerMemcpyNeedsBlockingStream() { return true; }
    static bool peerMemcpyStreamCanBeAnyDevice() { return true; }
    static bool fp64Broken() { return true; }
    static bool p2pKernelWriteBroken() { return true; }
    static bool ipcFreeSafe() { return false; }
    static int warpSize() { return 64; }

    static void applyCudaSetDevice(int src_dev) {
        cudaSetDevice(src_dev);
    }

    static cudaError_t safePeerMemcpy(void* dst, int dst_dev,
        const void* src, int src_dev, size_t bytes, cudaStream_t stream) {
        cudaSetDevice(src_dev);
        return cudaMemcpyPeerAsync(dst, dst_dev, src, src_dev, bytes, stream);
    }

    static cudaError_t safePeerMemcpySync(void* dst, int dst_dev,
        const void* src, int src_dev, size_t bytes) {
        cudaSetDevice(src_dev);
        return cudaMemcpyPeer(dst, dst_dev, src, src_dev, bytes);
    }

    static size_t recommendedSliceBytes(float peer_bw_gbps, float launch_overhead_us) {
        if (launch_overhead_us <= 0 || peer_bw_gbps <= 0) return 4 * 1024 * 1024;
        float target_transfer_us = launch_overhead_us * 20;
        size_t bytes = (size_t)(target_transfer_us * peer_bw_gbps * 1e3);
        if (bytes < 64 * 1024) bytes = 64 * 1024;
        if (bytes > 16 * 1024 * 1024) bytes = 16 * 1024 * 1024;
        bytes = (bytes + 4095) & ~4095ULL;
        return bytes;
    }
};

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu);
    caps->sm_count = prop.multiProcessorCount;
    caps->max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
    caps->shared_mem_per_sm = prop.sharedMemPerMultiprocessor;
    caps->l2_cache_bytes = prop.l2CacheSize;

    int* d_warp; cudaMalloc(&d_warp, 4);
    cap_warp_kernel<<<1,64>>>(d_warp);
    cudaMemcpy(&caps->warp_size, d_warp, 4, cudaMemcpyDeviceToHost);
    cudaFree(d_warp);

    double* d_fp64; cudaMalloc(&d_fp64, 8);
    cudaMemset(d_fp64, 0, 8);
    cap_fp64_kernel<<<1,1>>>(d_fp64);
    double h_fp64 = 0;
    cudaMemcpy(&h_fp64, d_fp64, 8, cudaMemcpyDeviceToHost);
    caps->fp64_works = (h_fp64 == 3.0) ? 1 : 0;
    cudaFree(d_fp64);

    size_t bw_bytes = 64 * 1024 * 1024;
    float *bw_src, *bw_dst;
    cudaMalloc(&bw_src, bw_bytes);
    cudaMalloc(&bw_dst, bw_bytes);
    int bw_n = bw_bytes / 4;
    cap_fill<<<(bw_n+255)/256,256>>>(bw_src, bw_n, 1.0f);
    cudaDeviceSynchronize();
    cap_read_bw<<<(bw_n+255)/256,256>>>(bw_dst, bw_src, bw_n);
    cudaDeviceSynchronize();
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < 20; i++)
        cap_read_bw<<<(bw_n+255)/256,256>>>(bw_dst, bw_src, bw_n);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    caps->hbm_bw_gbps = bw_bytes * 20.0 * 1e-6 / ms;
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(bw_src); cudaFree(bw_dst);

    int ngpu; cudaGetDeviceCount(&ngpu);
    if (ngpu >= 2) {
        int peer = (gpu == 0) ? 1 : 0;
        cudaDeviceEnablePeerAccess(peer, 0);
        cudaGetLastError();
        size_t peer_bytes = 16 * 1024 * 1024;
        float *p_src, *p_dst;
        cudaSetDevice(gpu); cudaMalloc(&p_src, peer_bytes);
        cudaSetDevice(peer); cudaMalloc(&p_dst, peer_bytes);
        cudaSetDevice(gpu);
        cudaStream_t ps; cudaStreamCreate(&ps);
        cudaMemcpyPeerAsync(p_dst, peer, p_src, gpu, peer_bytes, ps);
        cudaStreamSynchronize(ps);
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0, ps);
        for (int i = 0; i < 50; i++)
            cudaMemcpyPeerAsync(p_dst, peer, p_src, gpu, peer_bytes, ps);
        cudaEventRecord(t1, ps); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms, t0, t1);
        caps->peer_bw_gbps = peer_bytes * 50.0 * 1e-6 / ms;

        cudaEventRecord(t0, ps);
        for (int i = 0; i < 200; i++)
            cudaMemcpyPeerAsync(p_dst, peer, p_src, gpu, 4, ps);
        cudaEventRecord(t1, ps); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms, t0, t1);
        caps->peer_latency_us = ms * 1000.0f / 200;
        caps->memcpy_launch_overhead_us = caps->peer_latency_us;

        cudaEventDestroy(t0); cudaEventDestroy(t1);
        cudaStreamDestroy(ps);
        cudaSetDevice(gpu); cudaFree(p_src);
        cudaSetDevice(peer); cudaFree(p_dst);
    }

    cudaSetDevice(saved);
    return 0;
}

}
#endif
