#!/usr/bin/env bash
# probe_mooncake_path.sh — BI-V100 上走 Mooncake 路线需要验证的全部原语
# 在 cc-adc62d1c 四卡机上直接跑: bash probe_mooncake_path.sh
# 用 corex clang-16 编译, --cuda-gpu-arch=ivcore10
set -e

COREX="/usr/local/corex"
CXX="${COREX}/bin/clang++"
CUDA_PATH=$(find /usr/local -maxdepth 1 -name "corex-*" -type d 2>/dev/null | head -1)
[ -z "$CUDA_PATH" ] && CUDA_PATH="${COREX}"

WORKDIR="/tmp/probe_mooncake_$$"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "========== Mooncake-path Probe Suite for BI-V100 =========="
echo "Machine: $(hostname)"
echo "Date:    $(date)"
echo "GPUs:    $(ixsmi -L 2>/dev/null | head -4 || echo 'ixsmi not available')"
echo "WorkDir: $WORKDIR"
echo ""

# ============================================================
# TEST 1: IPC kernel READ (the critical missing test)
# Mooncake's rdma_transport reads remote memory. NCCL LL protocol
# polls remote memory with volatile reads. Can BI-V100 do this?
# ============================================================
cat > t1_ipc_read.cu << 'CUDA'
#include <cstdio>
#include <cuda_runtime.h>

#define CHK(x) do{cudaError_t e=(x);if(e){printf("CUDA ERR %s @%d\n",cudaGetErrorString(e),__LINE__);return 1;}}while(0)

// kernel on GPU1 reads from IPC-mapped pointer to GPU0's memory
__global__ void k_read_remote(const float* __restrict__ remote, float* __restrict__ local, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) local[i] = remote[i];
}

// kernel on GPU1 reads with volatile (NCCL LL-style polling)
__global__ void k_volatile_read(const volatile float* remote, float* local, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) local[i] = remote[i];
}

// kernel on GPU0 writes, GPU1 polls until it sees the value
__global__ void k_flag_write(volatile int* flag) { *flag = 42; }
__global__ void k_flag_poll(const volatile int* flag, int* result, int* iters) {
    int cnt = 0;
    while (*flag != 42 && cnt < 10000000) { cnt++; }
    *result = *flag;
    *iters = cnt;
}

int main() {
    int ng; cudaGetDeviceCount(&ng);
    if (ng < 2) { printf("need 2+ GPUs\n"); return 1; }
    printf("=== TEST 1: IPC kernel READ ===\n");

    int N = 4096;
    // GPU0: allocate and fill
    CHK(cudaSetDevice(0));
    float *d0; CHK(cudaMalloc(&d0, N*4));
    float h_src[4096]; for(int i=0;i<N;i++) h_src[i]=(float)i;
    CHK(cudaMemcpy(d0, h_src, N*4, cudaMemcpyHostToDevice));

    // get IPC handle
    cudaIpcMemHandle_t handle;
    CHK(cudaIpcGetMemHandle(&handle, d0));

    // GPU1: open IPC handle and try kernel read
    CHK(cudaSetDevice(1));
    float *mapped = NULL;
    cudaError_t e = cudaIpcOpenMemHandle((void**)&mapped, handle, cudaIpcMemLazyEnablePeerAccess);
    if (e != cudaSuccess) {
        printf("  IpcOpenMemHandle FAILED: %s\n", cudaGetErrorString(e));
        printf("  verdict: IPC not available\n");
        cudaSetDevice(0); cudaFree(d0);
        return 0;
    }

    float *d1; CHK(cudaMalloc(&d1, N*4));
    CHK(cudaMemset(d1, 0, N*4));

    // test 1a: normal kernel read
    k_read_remote<<<(N+255)/256, 256>>>(mapped, d1, N);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        printf("  kernel read CRASHED: %s\n", cudaGetErrorString(e));
    } else {
        float h[8]; CHK(cudaMemcpy(h, d1, 32, cudaMemcpyDeviceToHost));
        int ok = (h[0]==0.0f && h[1]==1.0f && h[7]==7.0f);
        printf("  kernel read: [%.0f,%.0f,%.0f,...,%.0f] %s\n", h[0],h[1],h[2],h[7], ok?"WORKS":"FAIL");
    }

    // test 1b: volatile kernel read
    CHK(cudaMemset(d1, 0, N*4));
    k_volatile_read<<<(N+255)/256, 256>>>(mapped, d1, N);
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        printf("  volatile read CRASHED: %s\n", cudaGetErrorString(e));
    } else {
        float h[8]; CHK(cudaMemcpy(h, d1, 32, cudaMemcpyDeviceToHost));
        int ok = (h[0]==0.0f && h[1]==1.0f && h[7]==7.0f);
        printf("  volatile read: [%.0f,%.0f,%.0f,...,%.0f] %s\n", h[0],h[1],h[2],h[7], ok?"WORKS":"FAIL");
    }

    // test 1c: cross-GPU flag polling (NCCL LL protocol core)
    CHK(cudaSetDevice(0));
    int *flag0; CHK(cudaMalloc(&flag0, 4)); CHK(cudaMemset(flag0, 0, 4));
    cudaIpcMemHandle_t fh; CHK(cudaIpcGetMemHandle(&fh, flag0));
    CHK(cudaSetDevice(1));
    int *flag_mapped; cudaIpcOpenMemHandle((void**)&flag_mapped, fh, cudaIpcMemLazyEnablePeerAccess);
    int *result1, *iters1;
    CHK(cudaMalloc(&result1, 4)); CHK(cudaMemset(result1, 0, 4));
    CHK(cudaMalloc(&iters1, 4)); CHK(cudaMemset(iters1, 0, 4));

    // launch poller on GPU1 first, then writer on GPU0
    cudaStream_t s1; CHK(cudaStreamCreate(&s1));
    k_flag_poll<<<1,1,0,s1>>>((const volatile int*)flag_mapped, result1, iters1);

    CHK(cudaSetDevice(0));
    cudaStream_t s0; CHK(cudaStreamCreate(&s0));
    // small delay so poll kernel is running
    for(int i=0;i<100;i++){} // host spin
    k_flag_write<<<1,1,0,s0>>>((volatile int*)flag0);
    CHK(cudaStreamSynchronize(s0));

    CHK(cudaSetDevice(1));
    e = cudaStreamSynchronize(s1);
    if (e != cudaSuccess) {
        printf("  flag poll CRASHED: %s\n", cudaGetErrorString(e));
    } else {
        int hr, hi;
        CHK(cudaMemcpy(&hr, result1, 4, cudaMemcpyDeviceToHost));
        CHK(cudaMemcpy(&hi, iters1, 4, cudaMemcpyDeviceToHost));
        printf("  flag poll: GPU0 wrote 42, GPU1 read %d after %d iters %s\n",
               hr, hi, hr==42?"WORKS":"TIMEOUT");
    }

    // cleanup
    cudaIpcCloseMemHandle(mapped);
    cudaIpcCloseMemHandle(flag_mapped);
    cudaStreamDestroy(s1); cudaSetDevice(0); cudaStreamDestroy(s0);
    cudaFree(d0); cudaFree(flag0);
    cudaSetDevice(1); cudaFree(d1); cudaFree(result1); cudaFree(iters1);
    return 0;
}
CUDA

# ============================================================
# TEST 2: IPC read bandwidth (kernel read vs cudaMemcpyAsync vs cudaMemcpyPeer)
# This decides whether fused reduce+transfer is possible
# ============================================================
cat > t2_ipc_bw.cu << 'CUDA'
#include <cstdio>
#include <cuda_runtime.h>
#define CHK(x) do{cudaError_t e=(x);if(e){printf("ERR %s @%d\n",cudaGetErrorString(e),__LINE__);return 1;}}while(0)

__global__ void k_copy(float* dst, const float* src, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// fused read+reduce: the holy grail
__global__ void k_read_reduce(float* local, const float* remote, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n) local[i] += remote[i];
}

int main() {
    printf("\n=== TEST 2: IPC read bandwidth (3 methods) ===\n");
    int ng; cudaGetDeviceCount(&ng);
    if (ng < 2) return 1;

    int sizes[] = {1024, 16384, 262144, 1048576, 4194304, 16777216};
    int nsizes = 6;

    printf("%-12s  %-20s %-20s %-20s\n", "N(floats)", "cudaMemcpyPeer", "IPC+MemcpyAsync", "IPC+KernelRead");

    for (int si = 0; si < nsizes; si++) {
        int N = sizes[si]; size_t bytes = (size_t)N * 4;

        CHK(cudaSetDevice(0));
        float *d0; CHK(cudaMalloc(&d0, bytes));
        CHK(cudaMemset(d0, 1, bytes));
        cudaIpcMemHandle_t handle; CHK(cudaIpcGetMemHandle(&handle, d0));

        CHK(cudaSetDevice(1));
        float *d1; CHK(cudaMalloc(&d1, bytes));
        float *mapped; CHK(cudaIpcOpenMemHandle((void**)&mapped, handle, cudaIpcMemLazyEnablePeerAccess));

        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaStream_t s; cudaStreamCreate(&s);
        int iters = (N <= 262144) ? 200 : 50;
        float ms;

        // warmup
        cudaMemcpyPeer(d1,1,d0,0,bytes);
        cudaMemcpyAsync(d1,mapped,bytes,cudaMemcpyDeviceToDevice,s); cudaStreamSynchronize(s);
        k_copy<<<(N+255)/256,256,0,s>>>(d1,mapped,N); cudaStreamSynchronize(s);

        // method 1: cudaMemcpyPeer
        cudaEventRecord(t0,s);
        for(int i=0;i<iters;i++) cudaMemcpyPeer(d1,1,d0,0,bytes);
        cudaEventRecord(t1,s); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms,t0,t1);
        float bw1 = bytes*1e-6/(ms/iters);

        // method 2: IPC + cudaMemcpyAsync
        cudaEventRecord(t0,s);
        for(int i=0;i<iters;i++) cudaMemcpyAsync(d1,mapped,bytes,cudaMemcpyDeviceToDevice,s);
        cudaEventRecord(t1,s); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms,t0,t1);
        float bw2 = bytes*1e-6/(ms/iters);

        // method 3: IPC + kernel read
        float bw3 = 0;
        k_copy<<<(N+255)/256,256,0,s>>>(d1,mapped,N);
        cudaError_t e = cudaStreamSynchronize(s);
        if (e == cudaSuccess) {
            cudaEventRecord(t0,s);
            for(int i=0;i<iters;i++) k_copy<<<(N+255)/256,256,0,s>>>(d1,mapped,N);
            cudaEventRecord(t1,s); cudaEventSynchronize(t1);
            cudaEventElapsedTime(&ms,t0,t1);
            bw3 = bytes*1e-6/(ms/iters);
        }

        printf("%-12d  %6.1fus %6.1fGB/s   %6.1fus %6.1fGB/s   ",
            N,
            (float)bytes*iters/(bw1*1e3*iters), bw1,
            (float)bytes*iters/(bw2*1e3*iters), bw2);
        if (bw3 > 0) printf("%6.1fus %6.1fGB/s\n", (float)bytes*iters/(bw3*1e3*iters), bw3);
        else printf("CRASH\n");

        cudaIpcCloseMemHandle(mapped);
        cudaEventDestroy(t0); cudaEventDestroy(t1); cudaStreamDestroy(s);
        cudaSetDevice(0); cudaFree(d0); cudaSetDevice(1); cudaFree(d1);
    }

    // bonus: fused read+reduce test
    printf("\n--- fused read+reduce (IPC kernel) ---\n");
    int N=4194304; size_t bytes=(size_t)N*4;
    CHK(cudaSetDevice(0));
    float *d0; CHK(cudaMalloc(&d0,bytes)); CHK(cudaMemset(d0,0,bytes));
    cudaIpcMemHandle_t h; CHK(cudaIpcGetMemHandle(&h,d0));
    CHK(cudaSetDevice(1));
    float *d1; CHK(cudaMalloc(&d1,bytes)); CHK(cudaMemset(d1,0,bytes));
    float *m; CHK(cudaIpcOpenMemHandle((void**)&m,h,cudaIpcMemLazyEnablePeerAccess));
    k_read_reduce<<<(N+255)/256,256>>>(d1,m,N);
    cudaError_t e = cudaDeviceSynchronize();
    printf("  fused read+reduce 16M floats: %s\n", e==cudaSuccess?"WORKS":"CRASH");
    if (e==cudaSuccess) {
        cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0);
        for(int i=0;i<50;i++) k_read_reduce<<<(N+255)/256,256>>>(d1,m,N);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms,t0,t1);
        printf("  bandwidth: %.1f GB/s\n", bytes*1e-6*50/ms);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }
    cudaIpcCloseMemHandle(m);
    cudaSetDevice(0); cudaFree(d0); cudaSetDevice(1); cudaFree(d1);
    return 0;
}
CUDA

# ============================================================
# TEST 3: concurrent multi-pair transfer (can we saturate PCIe switch?)
# Mooncake does fan-out to multiple targets simultaneously
# ============================================================
cat > t3_concurrent.cu << 'CUDA'
#include <cstdio>
#include <cuda_runtime.h>
#define CHK(x) do{cudaError_t e=(x);if(e){printf("ERR %s @%d\n",cudaGetErrorString(e),__LINE__);return 1;}}while(0)

int main() {
    printf("\n=== TEST 3: concurrent multi-pair transfers ===\n");
    int ng; cudaGetDeviceCount(&ng);
    if (ng < 3) { printf("need 3+ GPUs\n"); return 1; }

    size_t bytes = 16*1024*1024;
    float *bufs[8] = {};
    cudaStream_t streams[8] = {};
    cudaEvent_t starts[8]={}, stops[8]={};

    for (int g=0; g<ng; g++) {
        CHK(cudaSetDevice(g));
        CHK(cudaMalloc(&bufs[g], bytes));
        CHK(cudaMemset(bufs[g], g+1, bytes));
        CHK(cudaStreamCreateWithFlags(&streams[g], cudaStreamNonBlocking));
        CHK(cudaEventCreate(&starts[g]));
        CHK(cudaEventCreate(&stops[g]));
        for (int o=0; o<ng; o++) if(o!=g) { cudaDeviceEnablePeerAccess(o,0); cudaGetLastError(); }
    }

    // sequential: 0->1, then 0->2, then 0->3
    CHK(cudaSetDevice(0));
    cudaDeviceSynchronize();
    cudaEvent_t seq0,seq1; cudaEventCreate(&seq0); cudaEventCreate(&seq1);
    cudaEventRecord(seq0, streams[0]);
    for(int dst=1; dst<ng; dst++)
        for(int i=0;i<10;i++) cudaMemcpyPeerAsync(bufs[dst],dst,bufs[0],0,bytes,streams[0]);
    cudaEventRecord(seq1, streams[0]); cudaEventSynchronize(seq1);
    float seq_ms; cudaEventElapsedTime(&seq_ms, seq0, seq1);
    printf("  sequential 0->1,2,3 (10x16MB each): %.1f ms  %.1f GB/s aggregate\n",
        seq_ms, (double)bytes*10*(ng-1)*1e-6/seq_ms);

    // concurrent: 0->1 and 0->2 and 0->3 on separate streams
    for(int g=0;g<ng;g++){CHK(cudaSetDevice(g));cudaDeviceSynchronize();}
    CHK(cudaSetDevice(0));
    cudaEventRecord(seq0, streams[0]);
    for(int dst=1; dst<ng; dst++)
        for(int i=0;i<10;i++) cudaMemcpyPeerAsync(bufs[dst],dst,bufs[0],0,bytes,streams[dst]);
    for(int dst=1;dst<ng;dst++) cudaEventRecord(stops[dst],streams[dst]);
    cudaEventRecord(seq1, streams[0]);
    for(int dst=1;dst<ng;dst++) cudaEventSynchronize(stops[dst]);
    cudaEventSynchronize(seq1);
    float conc_ms; cudaEventElapsedTime(&conc_ms, seq0, seq1);
    // also check individual stream times
    float worst=0;
    for(int dst=1;dst<ng;dst++){
        float ms; cudaEventElapsedTime(&ms,seq0,stops[dst]);
        if(ms>worst)worst=ms;
    }
    printf("  concurrent 0->{1,2,3} (10x16MB each): %.1f ms  %.1f GB/s aggregate\n",
        worst, (double)bytes*10*(ng-1)*1e-6/worst);
    printf("  speedup: %.2fx\n", seq_ms/worst);

    // bidirectional: 0->1 and 1->0 simultaneously
    CHK(cudaSetDevice(0)); cudaDeviceSynchronize();
    CHK(cudaSetDevice(1)); cudaDeviceSynchronize();
    cudaEvent_t bi0,bi1;
    CHK(cudaSetDevice(0)); cudaEventCreate(&bi0); cudaEventCreate(&bi1);
    cudaEventRecord(bi0, streams[0]);
    for(int i=0;i<20;i++) cudaMemcpyPeerAsync(bufs[1],1,bufs[0],0,bytes,streams[0]);
    for(int i=0;i<20;i++) cudaMemcpyPeerAsync(bufs[0],0,bufs[1],1,bytes,streams[1]);
    cudaEventRecord(stops[0],streams[0]); cudaEventRecord(stops[1],streams[1]);
    cudaEventSynchronize(stops[0]); cudaEventSynchronize(stops[1]);
    float bidi0,bidi1;
    cudaEventElapsedTime(&bidi0,bi0,stops[0]);
    cudaEventElapsedTime(&bidi1,bi0,stops[1]);
    float bidi_worst = bidi0>bidi1?bidi0:bidi1;
    printf("  bidirectional 0<->1 (20x16MB each dir): %.1f ms  %.1f GB/s total\n",
        bidi_worst, (double)bytes*20*2*1e-6/bidi_worst);
    printf("  (unidirectional ref: %.1f GB/s)\n", (double)bytes*20*1e-6/bidi0);

    for(int g=0;g<ng;g++){
        CHK(cudaSetDevice(g));
        cudaFree(bufs[g]); cudaStreamDestroy(streams[g]);
        cudaEventDestroy(starts[g]); cudaEventDestroy(stops[g]);
    }
    cudaEventDestroy(seq0); cudaEventDestroy(seq1);
    cudaEventDestroy(bi0); cudaEventDestroy(bi1);
    return 0;
}
CUDA

# ============================================================
# TEST 4: transfer+compute overlap (can we hide transfer behind kernel?)
# Mooncake/NCCL pipeline: transfer chunk N while computing chunk N-1
# ============================================================
cat > t4_overlap.cu << 'CUDA'
#include <cstdio>
#include <cuda_runtime.h>
#define CHK(x) do{cudaError_t e=(x);if(e){printf("ERR %s @%d\n",cudaGetErrorString(e),__LINE__);return 1;}}while(0)

__global__ void k_burn(float* buf, int n, int reps) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n) { float v=buf[i]; for(int r=0;r<reps;r++) v=v*1.0001f+0.0001f; buf[i]=v; }
}

int main() {
    printf("\n=== TEST 4: transfer + compute overlap ===\n");
    int ng; cudaGetDeviceCount(&ng);
    if (ng < 2) return 1;

    size_t bytes = 16*1024*1024; int N=bytes/4;
    CHK(cudaSetDevice(0));
    float *d0a,*d0b; CHK(cudaMalloc(&d0a,bytes)); CHK(cudaMalloc(&d0b,bytes));
    CHK(cudaMemset(d0a,1,bytes)); CHK(cudaMemset(d0b,1,bytes));
    CHK(cudaSetDevice(1));
    float *d1; CHK(cudaMalloc(&d1,bytes));
    for(int o=0;o<ng;o++){CHK(cudaSetDevice(o));for(int p=0;p<ng;p++)if(p!=o){cudaDeviceEnablePeerAccess(p,0);cudaGetLastError();}}

    CHK(cudaSetDevice(0));
    cudaStream_t s_xfer, s_comp;
    CHK(cudaStreamCreateWithFlags(&s_xfer,cudaStreamNonBlocking));
    CHK(cudaStreamCreateWithFlags(&s_comp,cudaStreamNonBlocking));
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);

    // baseline: transfer only
    cudaDeviceSynchronize();
    cudaEventRecord(t0,s_xfer);
    for(int i=0;i<20;i++) cudaMemcpyPeerAsync(d1,1,d0a,0,bytes,s_xfer);
    cudaEventRecord(t1,s_xfer); cudaEventSynchronize(t1);
    float xfer_ms; cudaEventElapsedTime(&xfer_ms,t0,t1);

    // baseline: compute only
    cudaDeviceSynchronize();
    cudaEventRecord(t0,s_comp);
    for(int i=0;i<20;i++) k_burn<<<(N+255)/256,256,0,s_comp>>>(d0b,N,100);
    cudaEventRecord(t1,s_comp); cudaEventSynchronize(t1);
    float comp_ms; cudaEventElapsedTime(&comp_ms,t0,t1);

    // overlapped: transfer on s_xfer, compute on s_comp
    cudaDeviceSynchronize();
    cudaEvent_t ov0,ov1; cudaEventCreate(&ov0); cudaEventCreate(&ov1);
    cudaEventRecord(ov0,s_xfer);
    for(int i=0;i<20;i++){
        cudaMemcpyPeerAsync(d1,1,d0a,0,bytes,s_xfer);
        k_burn<<<(N+255)/256,256,0,s_comp>>>(d0b,N,100);
    }
    cudaEventRecord(t0,s_xfer); cudaEventRecord(t1,s_comp);
    cudaEventSynchronize(t0); cudaEventSynchronize(t1);
    float ov_xfer,ov_comp;
    cudaEventElapsedTime(&ov_xfer,ov0,t0);
    cudaEventElapsedTime(&ov_comp,ov0,t1);
    float overlap_ms = ov_xfer>ov_comp?ov_xfer:ov_comp;

    printf("  transfer only:  %.1f ms\n", xfer_ms);
    printf("  compute only:   %.1f ms\n", comp_ms);
    printf("  overlapped:     %.1f ms (sum=%.1f, saved=%.1f ms)\n",
        overlap_ms, xfer_ms+comp_ms, xfer_ms+comp_ms-overlap_ms);
    printf("  overlap ratio:  %.0f%%\n", 100.0*(xfer_ms+comp_ms-overlap_ms)/(xfer_ms+comp_ms));

    cudaStreamDestroy(s_xfer); cudaStreamDestroy(s_comp);
    cudaEventDestroy(t0); cudaEventDestroy(t1); cudaEventDestroy(ov0); cudaEventDestroy(ov1);
    cudaSetDevice(0); cudaFree(d0a); cudaFree(d0b);
    cudaSetDevice(1); cudaFree(d1);
    return 0;
}
CUDA

# ============================================================
# TEST 5: full P2P bandwidth matrix (all pairs, both directions)
# ============================================================
cat > t5_matrix.cu << 'CUDA'
#include <cstdio>
#include <cuda_runtime.h>
#define CHK(x) do{cudaError_t e=(x);if(e){printf("ERR %s @%d\n",cudaGetErrorString(e),__LINE__);return 1;}}while(0)

int main() {
    printf("\n=== TEST 5: P2P bandwidth matrix (16MB, GB/s) ===\n");
    int ng; cudaGetDeviceCount(&ng);
    size_t bytes=16*1024*1024;
    float *bufs[8]={};
    for(int g=0;g<ng;g++){
        CHK(cudaSetDevice(g)); CHK(cudaMalloc(&bufs[g],bytes)); CHK(cudaMemset(bufs[g],g+1,bytes));
        for(int o=0;o<ng;o++)if(o!=g){cudaDeviceEnablePeerAccess(o,0);cudaGetLastError();}
    }
    printf("        "); for(int j=0;j<ng;j++) printf("  GPU%-2d  ",j); printf("\n");
    for(int src=0;src<ng;src++){
        printf("GPU%d -> ",src);
        for(int dst=0;dst<ng;dst++){
            if(src==dst){printf("   --    ");continue;}
            CHK(cudaSetDevice(src));
            cudaStream_t s; cudaStreamCreate(&s);
            cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
            cudaMemcpyPeerAsync(bufs[dst],dst,bufs[src],src,bytes,s); cudaStreamSynchronize(s);
            cudaEventRecord(t0,s);
            for(int i=0;i<50;i++) cudaMemcpyPeerAsync(bufs[dst],dst,bufs[src],src,bytes,s);
            cudaEventRecord(t1,s); cudaEventSynchronize(t1);
            float ms; cudaEventElapsedTime(&ms,t0,t1);
            printf(" %5.1f   ", bytes*50.0*1e-6/ms);
            cudaStreamDestroy(s); cudaEventDestroy(t0); cudaEventDestroy(t1);
        }
        printf("\n");
    }
    for(int g=0;g<ng;g++){CHK(cudaSetDevice(g));cudaFree(bufs[g]);}
    return 0;
}
CUDA

# ============================================================
# COMPILE AND RUN
# ============================================================
echo ""
echo "===== COMPILING (corex clang-16, ivcore10) ====="

compile_and_run() {
    local src=$1 bin=$2
    echo "[build] $src -> $bin"
    "$CXX" -x cuda "$src" -o "$bin" \
        --cuda-gpu-arch=ivcore10 \
        --cuda-path="$CUDA_PATH" \
        -L"${COREX}/lib64" -lcudart \
        -I"${COREX}/include" \
        -lstdc++ -lpthread \
        -O2 2>&1 | head -5

    if [ -f "$bin" ]; then
        echo "[run]  $bin"
        timeout 120 "./$bin" 2>&1 || echo "[TIMEOUT or CRASH]"
    else
        echo "[FAIL] compilation failed"
    fi
    echo ""
}

compile_and_run t1_ipc_read.cu t1
compile_and_run t2_ipc_bw.cu t2
compile_and_run t3_concurrent.cu t3
compile_and_run t4_overlap.cu t4
compile_and_run t5_matrix.cu t5

echo "========== PROBE COMPLETE =========="
echo "Results in: $WORKDIR"
echo ""
echo "KEY QUESTIONS ANSWERED:"
echo "  T1: Can GPU kernel READ IPC-mapped remote memory?"
echo "      -> if YES: fused reduce+transfer kernel possible (NCCL LL path)"
echo "      -> if NO:  stuck on cudaMemcpyPeerAsync (current ceiling)"
echo "  T2: IPC kernel read bandwidth vs memcpy?"
echo "      -> if comparable: kernel path is viable"
echo "  T3: Does multi-stream concurrent transfer scale?"
echo "      -> tells you if PCIe switch supports concurrent DMA"
echo "  T4: Can transfer overlap with compute?"
echo "      -> if YES: pipeline is possible, Mooncake slice pattern works"
echo "  T5: Bandwidth matrix symmetry?"
echo "      -> reveals PCIe topology bottlenecks"
