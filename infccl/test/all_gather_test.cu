/*
 * all_gather_test.cu — AllGather correctness test
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
    printf("[infccl] AllGather test with %d GPUs\n", nGPUs);

    infcclComm_t* comms = (infcclComm_t*)malloc(nGPUs * sizeof(infcclComm_t));
    CHECK(infcclCommInitAll(comms, nGPUs));

    /* Each rank sends shard elements: rank r sends [r*100+1, ..., r*100+shard] */
    const int shard = 512;
    const int total = shard * nGPUs;

    float** d_in  = (float**)malloc(nGPUs * sizeof(float*));
    float** d_out = (float**)malloc(nGPUs * sizeof(float*));
    float*  h_in  = (float*)malloc(shard * sizeof(float));
    float*  h_out = (float*)malloc(total * sizeof(float));

    for (int r = 0; r < nGPUs; r++) {
        cudaSetDevice(r);
        CUDA_CHECK(cudaMalloc(&d_in[r],  shard * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out[r], total * sizeof(float)));
        for (int i = 0; i < shard; i++) h_in[i] = (float)(r * 100 + i + 1);
        CUDA_CHECK(cudaMemcpy(d_in[r], h_in, shard * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    for (int r = 0; r < nGPUs; r++) {
        cudaSetDevice(r);
        CHECK(infcclAllGather(d_in[r], d_out[r], shard,
                              infcclFloat32, comms[r], 0));
    }

    int errors = 0;
    for (int r = 0; r < nGPUs; r++) {
        cudaSetDevice(r);
        CUDA_CHECK(cudaMemcpy(h_out, d_out[r], total * sizeof(float),
                              cudaMemcpyDeviceToHost));
        for (int p = 0; p < nGPUs; p++) {
            for (int i = 0; i < shard; i++) {
                float expected = (float)(p * 100 + i + 1);
                float got = h_out[p * shard + i];
                if (fabsf(got - expected) > 1e-3f) {
                    if (errors < 5)
                        fprintf(stderr, "  rank %d pos %d: got %f expected %f\n",
                                r, p * shard + i, got, expected);
                    errors++;
                }
            }
        }
    }

    printf("  all_gather_fp32: %s (%d errors)\n",
           errors == 0 ? "PASS" : "FAIL", errors);

    for (int r = 0; r < nGPUs; r++) {
        cudaSetDevice(r); cudaFree(d_in[r]); cudaFree(d_out[r]);
    }
    free(h_in); free(h_out); free(d_in); free(d_out);
    for (int r = 0; r < nGPUs; r++) infcclCommDestroy(comms[r]);
    free(comms);
    return errors == 0 ? 0 : 1;
}
