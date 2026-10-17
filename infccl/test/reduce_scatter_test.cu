/*
 * reduce_scatter_test.cu — ReduceScatter correctness test
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
        fprintf(stderr, "FAIL code=%d at %s:%d\n",              \
                (int)_r, __FILE__, __LINE__); exit(1);          \
    }                                                           \
} while(0)

#define CUDA_CHECK(x) do {                                      \
    cudaError_t _e = (x);                                       \
    if (_e != cudaSuccess) {                                    \
        fprintf(stderr, "CUDA FAIL: %s at %s:%d\n",             \
                cudaGetErrorString(_e), __FILE__, __LINE__);     \
        exit(1);                                                \
    }                                                           \
} while(0)

int main() {
    int nGPUs;
    CUDA_CHECK(cudaGetDeviceCount(&nGPUs));
    if (nGPUs < 2) { fprintf(stderr, "Need >= 2 GPUs\n"); return 1; }
    printf("[infccl] ReduceScatter test with %d GPUs\n", nGPUs);

    infcclComm_t* comms = (infcclComm_t*)malloc(nGPUs * sizeof(infcclComm_t));
    CHECK(infcclCommInitAll(comms, nGPUs));

    /* Test: each rank sends [1,2,...,N*nGPUs] scaled by (rank+1).
       After reduce-scatter, rank r gets shard r of the sum. */
    const int shard = 256;
    const int N = shard * nGPUs;

    float** d_in  = (float**)malloc(nGPUs * sizeof(float*));
    float** d_out = (float**)malloc(nGPUs * sizeof(float*));
    float*  h_in  = (float*)malloc(N * sizeof(float));
    float*  h_out = (float*)malloc(shard * sizeof(float));

    for (int r = 0; r < nGPUs; r++) {
        cudaSetDevice(r);
        CUDA_CHECK(cudaMalloc(&d_in[r],  N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out[r], shard * sizeof(float)));
        for (int i = 0; i < N; i++) h_in[i] = (float)(i + 1) * (r + 1);
        CUDA_CHECK(cudaMemcpy(d_in[r], h_in, N * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    for (int r = 0; r < nGPUs; r++) {
        cudaSetDevice(r);
        CHECK(infcclReduceScatter(d_in[r], d_out[r], shard,
                                  infcclFloat32, infcclSum, comms[r], 0));
    }

    float scale = (float)(nGPUs * (nGPUs + 1)) / 2.0f;
    int errors = 0;
    for (int r = 0; r < nGPUs; r++) {
        cudaSetDevice(r);
        CUDA_CHECK(cudaMemcpy(h_out, d_out[r], shard * sizeof(float),
                              cudaMemcpyDeviceToHost));
        for (int i = 0; i < shard; i++) {
            int global_idx = r * shard + i;
            float expected = (float)(global_idx + 1) * scale;
            if (fabsf(h_out[i] - expected) > 1e-3f * fabsf(expected)) {
                if (errors < 5)
                    fprintf(stderr, "  rank %d shard_elem %d: got %f expected %f\n",
                            r, i, h_out[i], expected);
                errors++;
            }
        }
    }

    printf("  reduce_scatter_fp32: %s (%d errors)\n",
           errors == 0 ? "PASS" : "FAIL", errors);

    for (int r = 0; r < nGPUs; r++) {
        cudaSetDevice(r); cudaFree(d_in[r]); cudaFree(d_out[r]);
    }
    free(h_in); free(h_out); free(d_in); free(d_out);
    for (int r = 0; r < nGPUs; r++) infcclCommDestroy(comms[r]);
    free(comms);
    return errors == 0 ? 0 : 1;
}
