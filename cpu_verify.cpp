/*
 * cpu_verify.cpp — CPU simulation of the ring allreduce logic
 * to verify correctness without GPUs.
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

int main() {
  const int NGPUS = 8;
  const int N = 1 << 16; // smaller for CPU
  
  printf("CPU verification: %d simulated GPUs, N=%d\n", NGPUS, N);
  
  // Allocate buffers
  float* recv[NGPUS];
  float* tmp[NGPUS];
  for (int i = 0; i < NGPUS; i++) {
    recv[i] = (float*)malloc(N * sizeof(float));
    tmp[i]  = (float*)malloc(N * sizeof(float));
    for (int j = 0; j < N; j++) recv[i][j] = (float)(i + 1);
  }
  
  int chunk = N / NGPUS;
  
  // Phase 1: reduce-scatter
  for (int step = 0; step < NGPUS - 1; step++) {
    for (int i = 0; i < NGPUS; i++) {
      int sc = ((i - step) % NGPUS + NGPUS) % NGPUS;
      int next = (i + 1) % NGPUS;
      int off = sc * chunk;
      int cnt = (sc == NGPUS - 1) ? (N - off) : chunk;
      memcpy(tmp[next] + off, recv[i] + off, cnt * sizeof(float));
    }
    for (int i = 0; i < NGPUS; i++) {
      int next = (i + 1) % NGPUS;
      int sc = ((i - step) % NGPUS + NGPUS) % NGPUS;
      int off = sc * chunk;
      int cnt = (sc == NGPUS - 1) ? (N - off) : chunk;
      for (int j = 0; j < cnt; j++)
        recv[next][off + j] += tmp[next][off + j];
    }
  }
  
  // Phase 2: allgather
  for (int step = 0; step < NGPUS - 1; step++) {
    for (int i = 0; i < NGPUS; i++) {
      int sc = ((i - step + 1) % NGPUS + NGPUS) % NGPUS;
      int next = (i + 1) % NGPUS;
      int off = sc * chunk;
      int cnt = (sc == NGPUS - 1) ? (N - off) : chunk;
      memcpy(recv[next] + off, recv[i] + off, cnt * sizeof(float));
    }
  }
  
  // Verify
  float expected = (float)(NGPUS * (NGPUS + 1) / 2);
  bool pass = true;
  for (int i = 0; i < NGPUS && pass; i++) {
    for (int j = 0; j < N; j++) {
      if (fabsf(recv[i][j] - expected) > 1e-3f) {
        fprintf(stderr, "FAIL: GPU%d elem%d: got %f want %f\n", i, j, recv[i][j], expected);
        pass = false; break;
      }
    }
  }
  
  printf("Ring allreduce (CPU sim): %s (expected=%.0f)\n", pass ? "PASS" : "FAIL", expected);
  
  for (int i = 0; i < NGPUS; i++) { free(recv[i]); free(tmp[i]); }
  return pass ? 0 : 1;
}

