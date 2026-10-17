/*
 * reduce_test.cu — Reduce correctness test (sum to root)
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
    printf("[infccl] Reduce test with %d GPUs\n", nGPUs);

    infcclComm_t* comms = (infcclComm_t*)malloc(nGPUs * sizeof(infcclComm_t));
    CHECK(infcclCommInitAll(comms, nGPUs));

    const int N = 1024;
    int root = 0;

    float** d_in  = (float**)malloc(nGPUs * sizeof(float*));
    float** d_out = (float**)malloc(nGPUs * sizeof(float*));
    float*  h_in  = (float*)malloc(N * sizeof(float));

    for (int r = 0; r < nGPUs; r++) {
        cudaSetDevice(r);
        CUDA_CHECK(cudaMalloc(&d_in[r],  N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out[r], N * sizeof(float)));
        for (int i = 0; i < N; i++) h_in[i] = (float)(i + 1) * (r + 1);
        CUDA_CHECK(cudaMemcpy(d_in[r], h_in, N * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    for (int r = 0; r < nGPUs; r++) {
        cudaSetDevice(r);
        CHECK(infcclReduce(d_in[r], d_out[r], N, infcclFloat32,
                           infcclSum, root, comms[r], 0));
    }

    /* only root should have the correct result */
    float scale = (float)(nGPUs * (nGPUs + 1)) / 2.0f;
    int errors = 0;
    float* h_out = (float*)malloc(N * sizeof(float));
    cudaSetDevice(root);
    CUDA_CHECK(cudaMemcpy(h_out, d_out[root], N * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; i++) {
        float expected = (float)(i + 1) * scale;
        if (fabsf(h_out[i] - expected) > 1e-3f * fabsf(expected)) {
            if (errors < 5)
                fprintf(stderr, "  root elem %d: got %f expected %f\n",
                        i, h_out[i], expected);
            errors++;
        }
    }

    printf("  reduce_fp32: %s (%d errors)\n",
           errors == 0 ? "PASS" : "FAIL", errors);

    free(h_out);
    for (int r = 0; r < nGPUs; r++) {
        cudaSetDevice(r); cudaFree(d_in[r]); cudaFree(d_out[r]);
    }
    free(h_in); free(d_in); free(d_out);
    for (int r = 0; r < nGPUs; r++) infcclCommDestroy(comms[r]);
    free(comms);
    return errors == 0 ? 0 : 1;
}
