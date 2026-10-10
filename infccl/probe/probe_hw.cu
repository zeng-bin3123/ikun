#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(cmd) do { cudaError_t e = (cmd); if (e != cudaSuccess) { printf("  CUDA ERROR %s @ %d\n", cudaGetErrorString(e), __LINE__); return; } } while(0)

__global__ void k_warpsize(int *out) { if (threadIdx.x == 0) out[0] = warpSize; }

__global__ void k_shfl64(float *out) {
    float v = 1.0f;
    v += __shfl_down_sync(0xffffffff, v, 32);
    v += __shfl_down_sync(0xffffffff, v, 16);
    v += __shfl_down_sync(0xffffffff, v, 8);
    v += __shfl_down_sync(0xffffffff, v, 4);
    v += __shfl_down_sync(0xffffffff, v, 2);
    v += __shfl_down_sync(0xffffffff, v, 1);
    if (threadIdx.x == 0) out[0] = v;
}

__global__ void k_smem_novolatile(float *out) {
    __shared__ float s[64];
    s[threadIdx.x] = 2.0f;
    __syncwarp();
    if (threadIdx.x < 32) s[threadIdx.x] += s[threadIdx.x + 32]; __syncwarp();
    if (threadIdx.x < 16) s[threadIdx.x] += s[threadIdx.x + 16]; __syncwarp();
    if (threadIdx.x <  8) s[threadIdx.x] += s[threadIdx.x +  8]; __syncwarp();
    if (threadIdx.x <  4) s[threadIdx.x] += s[threadIdx.x +  4]; __syncwarp();
    if (threadIdx.x <  2) s[threadIdx.x] += s[threadIdx.x +  2]; __syncwarp();
    if (threadIdx.x <  1) s[threadIdx.x] += s[threadIdx.x +  1]; __syncwarp();
    if (threadIdx.x == 0) out[0] = s[0];
}

__global__ void k_smem_volatile(float *out) {
    volatile __shared__ float s[64];
    s[threadIdx.x] = 2.0f;
    __syncwarp();
    if (threadIdx.x < 32) s[threadIdx.x] += s[threadIdx.x + 32]; __syncwarp();
    if (threadIdx.x < 16) s[threadIdx.x] += s[threadIdx.x + 16]; __syncwarp();
    if (threadIdx.x <  8) s[threadIdx.x] += s[threadIdx.x +  8]; __syncwarp();
    if (threadIdx.x <  4) s[threadIdx.x] += s[threadIdx.x +  4]; __syncwarp();
    if (threadIdx.x <  2) s[threadIdx.x] += s[threadIdx.x +  2]; __syncwarp();
    if (threadIdx.x <  1) s[threadIdx.x] += s[threadIdx.x +  1]; __syncwarp();
    if (threadIdx.x == 0) out[0] = (float)s[0];
}

__global__ void k_copy(float *d, const float *s, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) d[i]=s[i]; }
__global__ void k_empty() {}

__global__ void k_fill(float *b, int n, float v) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) b[i]=v; }
__global__ void k_check(const float *b, int n, float exp, int *errs) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n && b[i]!=exp) atomicAdd(errs,1); }

__global__ void k_double_add(double *dst, const double *src, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) dst[i]=dst[i]+src[i]; }

void test_warp(int g) {
    printf("[1] GPU %d: warp size\n", g);
    CHECK(cudaSetDevice(g));
    int *d, h=0; CHECK(cudaMalloc(&d,4)); k_warpsize<<<1,1>>>(d); CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(&h,d,4,cudaMemcpyDeviceToHost)); cudaFree(d);
    printf("  warpSize=%d\n", h);
}

void test_shfl(int g) {
    printf("[2] GPU %d: shfl_down 64-thread\n", g);
    CHECK(cudaSetDevice(g));
    float *d, h=0; CHECK(cudaMalloc(&d,4)); k_shfl64<<<1,64>>>(d); CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(&h,d,4,cudaMemcpyDeviceToHost)); cudaFree(d);
    printf("  expect=64.0 got=%.1f %s\n", h, h==64.0f?"OK":"FAIL");
}

void test_smem(int g) {
    printf("[3] GPU %d: smem volatile behavior\n", g);
    CHECK(cudaSetDevice(g));
    float *d, h; CHECK(cudaMalloc(&d,4));
    k_smem_novolatile<<<1,64>>>(d); CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(&h,d,4,cudaMemcpyDeviceToHost));
    printf("  no volatile: expect=128.0 got=%.1f %s\n", h, h==128.0f?"OK(no opt)":"COMPILER_HOISTED");
    k_smem_volatile<<<1,64>>>(d); CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(&h,d,4,cudaMemcpyDeviceToHost));
    printf("  volatile:    expect=128.0 got=%.1f %s\n", h, h==128.0f?"OK":"FAIL");
    cudaFree(d);
}

void test_bandwidth(int g) {
    printf("[4] GPU %d: HBM bandwidth\n", g);
    CHECK(cudaSetDevice(g));
    int N=32*1024*1024; size_t b=(size_t)N*4;
    float *s,*d; CHECK(cudaMalloc(&s,b)); CHECK(cudaMalloc(&d,b)); CHECK(cudaMemset(s,1,b));
    k_copy<<<(N+255)/256,256>>>(d,s,N); cudaDeviceSynchronize();
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0); for(int i=0;i<50;i++) k_copy<<<(N+255)/256,256>>>(d,s,N);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms,t0,t1);
    printf("  %.0f GB/s\n", 2.0*b*50/(ms/1000.0)/1e9);
    cudaFree(s); cudaFree(d); cudaEventDestroy(t0); cudaEventDestroy(t1);
}

void test_launch(int g) {
    printf("[5] GPU %d: kernel launch overhead\n", g);
    CHECK(cudaSetDevice(g));
    for(int i=0;i<100;i++) k_empty<<<1,1>>>(); cudaDeviceSynchronize();
    cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0); for(int i=0;i<1000;i++) k_empty<<<1,1>>>();
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms,t0,t1);
    printf("  %.1f us/launch\n", ms*1000/1000);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

void test_fp64(int g) {
    printf("[6] GPU %d: FP64 kernel arithmetic\n", g);
    CHECK(cudaSetDevice(g));
    int N=16; double *d0,*d1,*h=(double*)malloc(N*8);
    CHECK(cudaMalloc(&d0,N*8)); CHECK(cudaMalloc(&d1,N*8));
    double ones[16], twos[16];
    for(int i=0;i<16;i++){ones[i]=1.0;twos[i]=2.0;}
    CHECK(cudaMemcpy(d0,ones,N*8,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d1,twos,N*8,cudaMemcpyHostToDevice));
    k_double_add<<<1,256>>>(d0,d1,N); CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h,d0,N*8,cudaMemcpyDeviceToHost));
    int ok = (h[0]==3.0 && h[1]==3.0);
    printf("  1.0+2.0=%.1f %s\n", h[0], ok?"OK":"BROKEN(expect 3.0)");
    free(h); cudaFree(d0); cudaFree(d1);
}

void test_p2p(int ngpu) {
    printf("[7] P2P matrix (%d GPUs)\n", ngpu);
    printf("       "); for(int j=0;j<ngpu;j++) printf("GPU%-2d ",j); printf("\n");
    for(int i=0;i<ngpu;i++){
        printf("  GPU%d:",i);
        for(int j=0;j<ngpu;j++){
            if(i==j){printf("  --  ");continue;}
            int can=0; cudaDeviceCanAccessPeer(&can,i,j);
            printf("  %s   ", can?"Y":"N");
        }
        printf("\n");
    }
}

void test_p2p_kernel_direct(int ngpu) {
    printf("[8] P2P kernel direct write test\n");
    if(ngpu<2){printf("  need 2+ GPUs\n");return;}
    cudaSetDevice(0); cudaDeviceEnablePeerAccess(1,0);
    cudaSetDevice(1); cudaDeviceEnablePeerAccess(0,0);
    const int N=1024;
    float *buf1; int *errs;
    cudaSetDevice(1); cudaMalloc(&buf1,N*4); cudaMemset(buf1,0,N*4);
    cudaMalloc(&errs,4); cudaMemset(errs,0,4);
    cudaSetDevice(0);
    k_fill<<<4,256>>>(buf1,N,7.0f); cudaDeviceSynchronize();
    cudaSetDevice(1);
    k_check<<<4,256>>>(buf1,N,7.0f,errs); cudaDeviceSynchronize();
    int he=0; cudaMemcpy(&he,errs,4,cudaMemcpyDeviceToHost);
    printf("  GPU0 kernel -> GPU1 buf: %d/%d errors %s\n", he, N, he==0?"WORKS":"FAIL(use cudaMemcpyPeer)");
    cudaFree(buf1); cudaFree(errs);
}

void test_ipc(int ngpu) {
    printf("[9] IPC handle + cudaMemcpyAsync test\n");
    if(ngpu<2){printf("  need 2+ GPUs\n");return;}
    int N=4096;
    cudaSetDevice(0);
    float *d0; cudaMalloc(&d0,N*4);
    k_fill<<<16,256>>>(d0,N,42.0f); cudaDeviceSynchronize();
    cudaIpcMemHandle_t handle;
    cudaError_t e = cudaIpcGetMemHandle(&handle,d0);
    if(e!=cudaSuccess){printf("  IpcGetMemHandle: %s\n",cudaGetErrorString(e));cudaFree(d0);return;}
    cudaSetDevice(1);
    float *mapped=NULL;
    e = cudaIpcOpenMemHandle((void**)&mapped,handle,cudaIpcMemLazyEnablePeerAccess);
    if(e!=cudaSuccess){printf("  IpcOpenMemHandle: %s\n",cudaGetErrorString(e));cudaSetDevice(0);cudaFree(d0);return;}
    float *d1; cudaMalloc(&d1,N*4); cudaMemset(d1,0,N*4);
    cudaMemcpyAsync(d1,mapped,N*4,cudaMemcpyDeviceToDevice,0); cudaDeviceSynchronize();
    float h[4]={0}; cudaMemcpy(h,d1,16,cudaMemcpyDeviceToHost);
    printf("  values: %.1f %.1f %.1f %.1f %s\n", h[0],h[1],h[2],h[3], h[0]==42.0f?"WORKS":"FAIL");
    cudaIpcCloseMemHandle(mapped);
    cudaSetDevice(0); cudaFree(d0);
    cudaSetDevice(1); cudaFree(d1);
}

void test_p2p_bandwidth(int ngpu) {
    printf("[10] P2P bandwidth (cudaMemcpyPeer vs IPC+Async)\n");
    if(ngpu<2){printf("  need 2+ GPUs\n");return;}
    int sizes[]={2048,4096,65536,262144,1048576};
    printf("  %-10s %-18s %-18s\n","N","cudaMemcpyPeer","IPC+MemcpyAsync");
    cudaSetDevice(0); cudaDeviceEnablePeerAccess(1,0);
    cudaSetDevice(1); cudaDeviceEnablePeerAccess(0,0);
    for(int si=0;si<5;si++){
        int N=sizes[si]; size_t bytes=N*4;
        cudaSetDevice(0); float *d0; cudaMalloc(&d0,bytes);
        k_fill<<<(N+255)/256,256>>>(d0,N,1.0f); cudaDeviceSynchronize();
        cudaIpcMemHandle_t handle; cudaIpcGetMemHandle(&handle,d0);
        cudaSetDevice(1); float *d1; cudaMalloc(&d1,bytes);
        float *mapped; cudaIpcOpenMemHandle((void**)&mapped,handle,cudaIpcMemLazyEnablePeerAccess);
        cudaStream_t s; cudaStreamCreate(&s);
        cudaEvent_t t0,t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        int iters=(N<=65536)?500:100;
        cudaMemcpyPeer(d1,1,d0,0,bytes);
        cudaSetDevice(1); cudaEventRecord(t0);
        for(int i=0;i<iters;i++) cudaMemcpyPeer(d1,1,d0,0,bytes);
        cudaEventRecord(t1); cudaEventSynchronize(t1);
        float ms1; cudaEventElapsedTime(&ms1,t0,t1);
        cudaMemcpyAsync(d1,mapped,bytes,cudaMemcpyDeviceToDevice,s); cudaStreamSynchronize(s);
        cudaEventRecord(t0,s);
        for(int i=0;i<iters;i++) cudaMemcpyAsync(d1,mapped,bytes,cudaMemcpyDeviceToDevice,s);
        cudaEventRecord(t1,s); cudaEventSynchronize(t1);
        float ms2; cudaEventElapsedTime(&ms2,t0,t1);
        printf("  %-10d %6.1fus %5.1fGB/s  %6.1fus %5.1fGB/s\n",
            N, ms1*1000/iters, bytes*1e-6/(ms1/iters), ms2*1000/iters, bytes*1e-6/(ms2/iters));
        cudaIpcCloseMemHandle(mapped);
        cudaStreamDestroy(s); cudaEventDestroy(t0); cudaEventDestroy(t1);
        cudaSetDevice(0); cudaFree(d0); cudaSetDevice(1); cudaFree(d1);
    }
}

int main() {
    int ngpu; cudaGetDeviceCount(&ngpu);
    printf("========== BI-V100 Hardware Probe ==========\n");
    printf("GPUs: %d\n", ngpu);
    for(int g=0;g<ngpu;g++){
        cudaDeviceProp p; cudaGetDeviceProperties(&p,g);
        printf("  GPU%d: %s %dMB %dSMs\n", g, p.name, (int)(p.totalGlobalMem>>20), p.multiProcessorCount);
    }
    printf("\n");
    test_warp(0); printf("\n");
    test_shfl(0); printf("\n");
    test_smem(0); printf("\n");
    test_bandwidth(0); printf("\n");
    test_launch(0); printf("\n");
    test_fp64(0); printf("\n");
    test_p2p(ngpu); printf("\n");
    test_p2p_kernel_direct(ngpu); printf("\n");
    test_ipc(ngpu); printf("\n");
    test_p2p_bandwidth(ngpu); printf("\n");
    if(ngpu>1){printf("cross-GPU warp consistency:\n");for(int g=1;g<ngpu;g++)test_warp(g);}
    printf("\n========== done ==========\n");
    return 0;
}
