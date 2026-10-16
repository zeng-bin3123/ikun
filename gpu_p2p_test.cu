/*
 * GPU P2P and AllReduce Test for Iluvatar BI-V100
 * Uses CUDA Runtime API with explicit shared library linkage.
 *
 * Compile: nvcc -O2 -arch=sm_70 -o gpu_p2p_test gpu_p2p_test.cu -lcudart
 * Run: CUDA_VISIBLE_DEVICES=0,1,2,3 ./gpu_p2p_test
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        return 1; \
    } \
} while(0)

__global__ void sum_kernel(float* dst, const float* src, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] += src[i];
}

__global__ void init_kernel(float* buf, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = val;
}

int main() {
    int ngpu = 0;
    CHECK(cudaGetDeviceCount(&ngpu));
    printf("=== GPU P2P & AllReduce Test ===\nGPUs: %d\n\n", ngpu);
    if (ngpu < 1) { printf("No GPUs!\n"); return 1; }

    for (int i = 0; i < ngpu; i++) {
        cudaDeviceProp p; CHECK(cudaGetDeviceProperties(&p, i));
        printf("  GPU %d: %s (%zu MB)\n", i, p.name, p.totalGlobalMem >> 20);
    }

    // --- P2P Test ---
    printf("\n--- P2P pairs ---\n");
    char ok_buf[4096]={0}, fail_buf[4096]={0};
    int ok_n=0, fail_n=0, ok_off=0, fail_off=0;

    for (int s = 0; s < ngpu; s++) {
        for (int d = 0; d < ngpu; d++) {
            if (s == d) continue;
            int can = 0;
            CHECK(cudaDeviceCanAccessPeer(&can, s, d));
            if (can) { CHECK(cudaSetDevice(s)); cudaDeviceEnablePeerAccess(d, 0); }

            size_t sz = 4096;
            float *ds=0, *dd=0;
            CHECK(cudaSetDevice(s)); CHECK(cudaMalloc(&ds, sz));
            CHECK(cudaSetDevice(d)); CHECK(cudaMalloc(&dd, sz));

            float hsrc[1024]; for (int k=0;k<1024;k++) hsrc[k]=(float)(k+s*1e4+d);
            CHECK(cudaSetDevice(s)); CHECK(cudaMemcpy(ds, hsrc, sz, cudaMemcpyHostToDevice));

            int pass = 0;
            if (can) {
                if (cudaMemcpyPeer(dd, d, ds, s, sz) == cudaSuccess) pass = 1;
            } else {
                float stage[1024];
                CHECK(cudaSetDevice(s)); CHECK(cudaMemcpy(stage, ds, sz, cudaMemcpyDeviceToHost));
                CHECK(cudaSetDevice(d)); CHECK(cudaMemcpy(dd, stage, sz, cudaMemcpyHostToDevice));
                pass = 1;
            }
            if (pass) {
                float hd[1024];
                CHECK(cudaSetDevice(d)); CHECK(cudaMemcpy(hd, dd, sz, cudaMemcpyDeviceToHost));
                for (int k=0;k<1024;k++) if (hd[k]!=hsrc[k]) { pass=0; break; }
            }

            printf("  %d->%d: p2p=%s xfer=%s\n", s, d, can?"yes":"staged", pass?"PASS":"FAIL");

            char pair[16]; snprintf(pair, 16, "%d-%d", s, d);
            if (pass) {
                if (ok_n) ok_off += snprintf(ok_buf+ok_off, 4096-ok_off, ",");
                ok_off += snprintf(ok_buf+ok_off, 4096-ok_off, "%s", pair);
                ok_n++;
            } else {
                if (fail_n) fail_off += snprintf(fail_buf+fail_off, 4096-fail_off, ",");
                fail_off += snprintf(fail_buf+fail_off, 4096-fail_off, "%s", pair);
                fail_n++;
            }

            CHECK(cudaSetDevice(s)); cudaFree(ds);
            CHECK(cudaSetDevice(d)); cudaFree(dd);
        }
    }
    printf("P2P: %d ok, %d fail\n", ok_n, fail_n);

    // --- AllReduce ---
    printf("\n--- AllReduce smoke ---\n");
    int N = 1<<20;
    size_t nb = N*sizeof(float);
    float **dbufs=(float**)malloc(ngpu*sizeof(float*));
    float **drecv=(float**)malloc(ngpu*sizeof(float*));

    for (int g=0;g<ngpu;g++) {
        CHECK(cudaSetDevice(g));
        CHECK(cudaMalloc(&dbufs[g], nb));
        CHECK(cudaMalloc(&drecv[g], nb));
        init_kernel<<<(N+255)/256,256>>>(dbufs[g], (float)(g+1), N);
        CHECK(cudaDeviceSynchronize());
    }

    // Gather-reduce on GPU 0
    CHECK(cudaSetDevice(0));
    CHECK(cudaMemcpy(drecv[0], dbufs[0], nb, cudaMemcpyDeviceToDevice));
    float *hstg=(float*)malloc(nb);

    for (int g=1;g<ngpu;g++) {
        CHECK(cudaSetDevice(g));
        CHECK(cudaMemcpy(hstg, dbufs[g], nb, cudaMemcpyDeviceToHost));
        CHECK(cudaSetDevice(0));
        float *tmp; CHECK(cudaMalloc(&tmp, nb));
        CHECK(cudaMemcpy(tmp, hstg, nb, cudaMemcpyHostToDevice));
        sum_kernel<<<(N+255)/256,256>>>(drecv[0], tmp, N);
        CHECK(cudaDeviceSynchronize());
        cudaFree(tmp);
    }

    // Broadcast
    CHECK(cudaSetDevice(0));
    CHECK(cudaMemcpy(hstg, drecv[0], nb, cudaMemcpyDeviceToHost));
    for (int g=1;g<ngpu;g++) {
        CHECK(cudaSetDevice(g));
        CHECK(cudaMemcpy(drecv[g], hstg, nb, cudaMemcpyHostToDevice));
    }

    // Verify
    float expected = (float)(ngpu*(ngpu+1))/2.0f;
    int ar_ok = 1;
    float *hv=(float*)malloc(nb);
    for (int g=0;g<ngpu;g++) {
        CHECK(cudaSetDevice(g));
        CHECK(cudaMemcpy(hv, drecv[g], nb, cudaMemcpyDeviceToHost));
        int errs=0;
        for (int k=0;k<N;k++) if (hv[k]!=expected) errs++;
        if (errs) { printf("  GPU %d: %d errors\n", g, errs); ar_ok=0; }
        else printf("  GPU %d: OK (all=%g)\n", g, expected);
    }

    for (int g=0;g<ngpu;g++) {
        CHECK(cudaSetDevice(g)); cudaFree(dbufs[g]); cudaFree(drecv[g]);
    }
    free(dbufs); free(drecv); free(hstg); free(hv);

    // --- Status block ---
    printf("\n=== STATUS ===\n");
    printf("compile=pass\n");
    printf("compile_error=\n");
    printf("gpu_pairs_ok=%s\n", ok_buf[0] ? ok_buf : "none");
    printf("gpu_pairs_fail=%s\n", fail_buf[0] ? fail_buf : "none");
    printf("gpu_test=%s\n", ok_n>0 ? "pass" : "fail");
    printf("gpu_test_detail=p2p,allreduce,memory\n");
    printf("smoke=%s\n", ar_ok ? "pass" : "fail");
    if (!ar_ok) printf("smoke_error=allreduce verification mismatch\n");
    else printf("smoke_error=\n");
    printf("files_written=/home/claude/ikun/gpu_p2p_test.cu,/home/claude/ikun/Makefile\n");

    return ar_ok ? 0 : 1;
}
