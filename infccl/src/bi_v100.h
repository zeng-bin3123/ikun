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
    64, 0, 0, 0, 1, 0, 1, 1, 1,
    0, 0, 0, 0,
    16, 0, 0, 0,
    0, 0, 0, 0, 10, 2, 0
};

struct BiV100Constraints {
    static bool peerMemcpyNeedsSrcDevice() { return true; }
    static bool peerMemcpyNeedsBlockingStream() { return true; }
    static bool peerMemcpyStreamCanBeAnyDevice() { return true; }
    static bool fp64Broken() { return true; }
    static bool p2pKernelWriteBroken() { return true; }
    static bool ipcFreeSafe() { return false; }
    static int warpSize() { return 64; }

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
};

}
#endif
