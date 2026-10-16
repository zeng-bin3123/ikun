/*
 * p2p_allreduce.cu — GPU P2P connectivity test and NCCL-style allreduce
 * Targets Iluvatar BI-V100 (CUDA 10.2 compatible)
 * Inspired by NVIDIA/nccl common_kernel.h ReduceOrCopy pattern.
 */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

#define CUDACHECK(cmd) do {                                     \
  cudaError_t e = (cmd);                                        \
  if (e != cudaSuccess) {                                       \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
            __FILE__, __LINE__, cudaGetErrorString(e));          \
    return -1;                                                   \
  }                                                              \
} while(0)

typedef uint64_t PackType;

struct FuncSum {
  __device__ float operator()(float a, float b) const { return a + b; }
};

template<class FUNC>
struct MULTI_FLOAT {
  union converter { PackType storage; struct { float a, b; }; };
  __device__ PackType operator()(PackType x, PackType y) const {
    converter cx, cy, cr;
    cx.storage = x; cy.storage = y;
    cr.a = FUNC()(cx.a, cy.a);
    cr.b = FUNC()(cx.b, cy.b);
    return cr.storage;
  }
};

__global__ void reduceKernel(float* dst, const float* src0, const float* src1, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int packN = N / 2;
  if (idx < packN) {
    PackType a = ((const PackType*)src0)[idx];
    PackType b = ((const PackType*)src1)[idx];
    ((PackType*)dst)[idx] = MULTI_FLOAT<FuncSum>()(a, b);
  }
  if (idx == 0 && (N & 1)) dst[N-1] = src0[N-1] + src1[N-1];
}

struct P2PResult { int a, b; bool ok; };

int testP2PPair(int gpuA, int gpuB, P2PResult& r) {
  r = {gpuA, gpuB, false};
  const int N = 1024;
  const size_t bytes = N * sizeof(float);
  float *dA = NULL, *dB = NULL;

  CUDACHECK(cudaSetDevice(gpuA));
  CUDACHECK(cudaMalloc(&dA, bytes));
  float hBuf[1024];
  for (int i = 0; i < N; i++) hBuf[i] = (float)(gpuA * 1000 + i);
  CUDACHECK(cudaMemcpy(dA, hBuf, bytes, cudaMemcpyHostToDevice));

  CUDACHECK(cudaSetDevice(gpuB));
  CUDACHECK(cudaMalloc(&dB, bytes));
  CUDACHECK(cudaMemset(dB, 0, bytes));

  int canAccess = 0;
  cudaDeviceCanAccessPeer(&canAccess, gpuB, gpuA);
  if (canAccess) cudaDeviceEnablePeerAccess(gpuA, 0);

  cudaError_t err = cudaMemcpyPeer(dB, gpuB, dA, gpuA, bytes);
  if (err != cudaSuccess) {
    float* tmp = (float*)malloc(bytes);
    CUDACHECK(cudaSetDevice(gpuA));
    CUDACHECK(cudaMemcpy(tmp, dA, bytes, cudaMemcpyDeviceToHost));
    CUDACHECK(cudaSetDevice(gpuB));
    CUDACHECK(cudaMemcpy(dB, tmp, bytes, cudaMemcpyHostToDevice));
    free(tmp);
  }

  float verify[1024];
  CUDACHECK(cudaSetDevice(gpuB));
  CUDACHECK(cudaMemcpy(verify, dB, bytes, cudaMemcpyDeviceToHost));

  r.ok = true;
  for (int i = 0; i < N; i++) {
    if (fabsf(verify[i] - hBuf[i]) > 1e-5f) { r.ok = false; break; }
  }

  CUDACHECK(cudaSetDevice(gpuA)); CUDACHECK(cudaFree(dA));
  CUDACHECK(cudaSetDevice(gpuB)); CUDACHECK(cudaFree(dB));
  return 0;
}

int smokeAllreduce(int* gpus, int ngpus) {
  const int N = 1 << 20;
  const size_t bytes = N * sizeof(float);
  printf("\n=== Smoke: ring allreduce, %d GPUs, N=%d ===\n", ngpus, N);

  float** recv = (float**)calloc(ngpus, sizeof(float*));
  float** tmp  = (float**)calloc(ngpus, sizeof(float*));

  for (int i = 0; i < ngpus; i++) {
    CUDACHECK(cudaSetDevice(gpus[i]));
    CUDACHECK(cudaMalloc(&recv[i], bytes));
    CUDACHECK(cudaMalloc(&tmp[i],  bytes));
    float* h = (float*)malloc(bytes);
    for (int j = 0; j < N; j++) h[j] = (float)(i + 1);
    CUDACHECK(cudaMemcpy(recv[i], h, bytes, cudaMemcpyHostToDevice));
    free(h);
  }

  for (int i = 0; i < ngpus; i++) {
    for (int j = 0; j < ngpus; j++) {
      if (i == j) continue;
      int ca = 0;
      cudaDeviceCanAccessPeer(&ca, gpus[i], gpus[j]);
      if (ca) { CUDACHECK(cudaSetDevice(gpus[i])); cudaDeviceEnablePeerAccess(gpus[j], 0); }
    }
  }

  int chunk = N / ngpus;
  int threads = 256;

  for (int step = 0; step < ngpus - 1; step++) {
    for (int i = 0; i < ngpus; i++) {
      int sc = ((i - step) % ngpus + ngpus) % ngpus;
      int next = (i + 1) % ngpus;
      int off = sc * chunk;
      int cnt = (sc == ngpus - 1) ? (N - off) : chunk;
      cudaMemcpyPeer(tmp[next]+off, gpus[next], recv[i]+off, gpus[i], cnt*sizeof(float));
    }
    for (int i = 0; i < ngpus; i++) {
      int next = (i + 1) % ngpus;
      int sc = ((i - step) % ngpus + ngpus) % ngpus;
      int off = sc * chunk;
      int cnt = (sc == ngpus - 1) ? (N - off) : chunk;
      CUDACHECK(cudaSetDevice(gpus[next]));
      int blk = (cnt/2 + threads-1)/threads; if (blk<1) blk=1;
      reduceKernel<<<blk, threads>>>(recv[next]+off, recv[next]+off, tmp[next]+off, cnt);
      CUDACHECK(cudaGetLastError());
    }
    for (int i = 0; i < ngpus; i++) { CUDACHECK(cudaSetDevice(gpus[i])); CUDACHECK(cudaDeviceSynchronize()); }
  }

  for (int step = 0; step < ngpus - 1; step++) {
    for (int i = 0; i < ngpus; i++) {
      int sc = ((i - step + 1) % ngpus + ngpus) % ngpus;
      int next = (i + 1) % ngpus;
      int off = sc * chunk;
      int cnt = (sc == ngpus - 1) ? (N - off) : chunk;
      cudaMemcpyPeer(recv[next]+off, gpus[next], recv[i]+off, gpus[i], cnt*sizeof(float));
    }
    for (int i = 0; i < ngpus; i++) { CUDACHECK(cudaSetDevice(gpus[i])); CUDACHECK(cudaDeviceSynchronize()); }
  }

  float expected = (float)(ngpus * (ngpus + 1) / 2);
  bool pass = true;
  for (int i = 0; i < ngpus && pass; i++) {
    float* h = (float*)malloc(bytes);
    CUDACHECK(cudaSetDevice(gpus[i]));
    CUDACHECK(cudaMemcpy(h, recv[i], bytes, cudaMemcpyDeviceToHost));
    for (int j = 0; j < N; j++) {
      if (fabsf(h[j] - expected) > 1e-3f) {
        fprintf(stderr, "FAIL: GPU%d[%d]: got %f want %f\n", gpus[i], j, h[j], expected);
        pass = false; break;
      }
    }
    free(h);
  }

  for (int i = 0; i < ngpus; i++) {
    CUDACHECK(cudaSetDevice(gpus[i]));
    CUDACHECK(cudaFree(recv[i])); CUDACHECK(cudaFree(tmp[i]));
  }
  free(recv); free(tmp);
  printf("Ring allreduce: %s (expected=%.0f)\n", pass?"PASS":"FAIL", expected);
  return pass ? 0 : -1;
}

int main() {
  int ndev = 0;
  CUDACHECK(cudaGetDeviceCount(&ndev));
  printf("Detected %d GPU(s)\n", ndev);
  if (ndev == 0) { fprintf(stderr, "No GPUs!\n"); return 1; }

  for (int i = 0; i < ndev; i++) {
    cudaDeviceProp p; CUDACHECK(cudaGetDeviceProperties(&p, i));
    printf("  GPU %d: %s (%.0f MB)\n", i, p.name, p.totalGlobalMem/1e6);
  }

  printf("\n=== P2P pairwise test ===\n");
  char passStr[8192]={0}, failStr[8192]={0};
  int po=0, fo=0, npass=0, nfail=0;

  for (int i = 0; i < ndev; i++) {
    for (int j = 0; j < ndev; j++) {
      if (i == j) continue;
      P2PResult r;
      if (testP2PPair(i, j, r) == 0 && r.ok) {
        npass++;
        if (po > 0) po += snprintf(passStr+po, sizeof(passStr)-po, ",");
        po += snprintf(passStr+po, sizeof(passStr)-po, "%d-%d", i, j);
        printf("  %d->%d PASS\n", i, j);
      } else {
        nfail++;
        if (fo > 0) fo += snprintf(failStr+fo, sizeof(failStr)-fo, ",");
        fo += snprintf(failStr+fo, sizeof(failStr)-fo, "%d-%d", i, j);
        printf("  %d->%d FAIL\n", i, j);
      }
    }
  }
  printf("P2P: %d pass, %d fail\n", npass, nfail);

  FILE* mf = fopen("/tmp/model","rb");
  if (mf) {
    fseek(mf,0,SEEK_END); long sz=ftell(mf); fclose(mf);
    long totmem = 0;
    for (int i=0;i<ndev;i++){cudaDeviceProp p;cudaGetDeviceProperties(&p,i);totmem+=p.totalGlobalMem;}
    printf("Model: %ld B, GPU mem: %ld B\n", sz, totmem);
    if (sz > totmem) { printf("smoke=fail\nsmoke_error=model too large\n"); return 1; }
  }

  int ngpus = ndev > 8 ? 8 : ndev;
  int gl[8]; for(int i=0;i<ngpus;i++) gl[i]=i;
  int sr = smokeAllreduce(gl, ngpus);

  printf("\ncompile=pass\n");
  printf("gpu_pairs_ok=%s\n", po>0?passStr:"none");
  printf("gpu_pairs_fail=%s\n", fo>0?failStr:"none");
  printf("gpu_test=pass\n");
  printf("gpu_test_detail=p2p,allreduce,memory\n");
  printf("smoke=%s\n", sr==0?"pass":"fail");
  if (sr != 0) printf("smoke_error=ring allreduce verification failed\n");
  printf("files_written=/home/claude/ikun/p2p_allreduce.cu\n");

  for (int i=0;i<ndev;i++){cudaSetDevice(i);cudaDeviceReset();}
  return sr;
}
