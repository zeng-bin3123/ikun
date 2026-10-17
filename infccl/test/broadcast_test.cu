/*
 * broadcast_test.cu — Broadcast correctness test
 */

#include "../src/infccl.h"
#include "../src/core.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern infcclResult_t infcclCommInitAll(infcclComm_t* comms, int nranks);

#define CHECK(x) do {                                           \
    infcclResult_t _r = (x);                                    \
    if (_r != infcclSuccess) {                                  \
        fprintf(stderr, "FAIL code=%d\n", (int)_r); exit(1);   \
    }                                                           \
} while(0)

#define CUDA_CHECK(x) do {                                      \
    cudaError_t _e = (x);                                       \
    if (_e != cudaSuccess) {                                    \
        fprintf(stderr, "CUDA FAIL: %s\n",                       \
                cudaGetErrorString(_e)); exit(1);                \
    }                                                           \
} while(0)

int main() {
    int nGPUs;
    CUDA_CHECK(cudaGetDeviceCount(&nGPUs));
    if (nGPUs < 2) { fprintf(stderr, "Need >= 2 GPUs\n"); return 1; }
    printf("[infccl] Broadcast test with %d GPUs\n", nGPUs);

    infcclComm_t* comms = (infcclComm_t*)malloc(nGPUs * sizeof(infcclComm_t));
    CHECK(infcclCommInitAll(comms, nGPUs));

    const int N = 1024;
    int root = 0;  /* test broadcast from rank 0 */

    float** d_in  = (float**)malloc(nGPUs * sizeof(float*));
    float** d_out = (float**)malloc(nGPUs * sizeof(float*));
    float*  h_data = (float*)malloc(N * sizeof(float));

    for (int r = 0; r < nGPUs; r++) {
        cudaSetDevice(r);
        CUDA_CHECK(cudaMalloc(&d_in[r],  N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out[r], N * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_out[r], 0, N * sizeof(float)));
    }

    /* root fills data */
    cudaSetDevice(root);
    for (int i = 0; i < N; i++) h_data[i] = (float)(i * 3 + 7);
    CUDA_CHECK(cudaMemcpy(d_in[root], h_data, N * sizeof(float),
                          cudaMemcpyHostToDevice));

    for (int r = 0; r < nGPUs; r++) {
        cudaSetDevice(r);
        CHECK(infcclBroadcast(d_in[r], d_out[r], N, infcclFloat32,
                              root, comms[r], 0));
    }

    int errors = 0;
    for (int r = 0; r < nGPUs; r++) {
        cudaSetDevice(r);
        float* h_out = (float*)malloc(N * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_out, d_out[r], N * sizeof(float),
                              cudaMemcpyDeviceToHost));
        for (int i = 0; i < N; i++) {
            if (fabsf(h_out[i] - h_data[i]) > 1e-6f) {
                if (errors < 5)
                    fprintf(stderr, "  rank %d elem %d: got %f expected %f\n",
                            r, i, h_out[i], h_data[i]);
                errors++;
            }
        }
        free(h_out);
    }

    printf("  broadcast_fp32: %s (%d errors)\n",
           errors == 0 ? "PASS" : "FAIL", errors);

    for (int r = 0; r < nGPUs; r++) {
        cudaSetDevice(r); cudaFree(d_in[r]); cudaFree(d_out[r]);
    }
    free(h_data); free(d_in); free(d_out);
    for (int r = 0; r < nGPUs; r++) infcclCommDestroy(comms[r]);
    free(comms);
    return errors == 0 ? 0 : 1;
}
