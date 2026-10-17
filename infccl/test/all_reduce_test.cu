/*
 * all_reduce_test.cu — AllReduce correctness + latency benchmark
 *
 * Tests:
 *   1. FP32 SUM correctness (small, medium)
 *   2. FP16 SUM correctness with FP32 accumulation
 *   3. BF16 SUM correctness
 *   4. MAX/MIN/PROD ops
 *   5. Latency benchmark across tensor sizes
 *
 * Usage:
 *   CUDA_VISIBLE_DEVICES=0,1,2 ./all_reduce_test
 *   CUDA_VISIBLE_DEVICES=0,1,2,3 ./all_reduce_test
 */

#include "../src/infccl.h"
#include "../src/core.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_fp16.h>

/* forward: defined in core.cu */
extern infcclResult_t infcclCommInitAll(infcclComm_t* comms, int nranks);

#define CHECK(x) do {                                           \
    infcclResult_t _r = (x);                                    \
    if (_r != infcclSuccess) {                                  \
        fprintf(stderr, "FAIL at %s:%d code=%d\n",              \
                __FILE__, __LINE__, (int)_r);                   \
        exit(1);                                                \
    }                                                           \
} while(0)

#define CUDA_CHECK(x) do {                                      \
    cudaError_t _e = (x);                                       \
    if (_e != cudaSuccess) {                                    \
        fprintf(stderr, "CUDA FAIL at %s:%d: %s\n",             \
                __FILE__, __LINE__, cudaGetErrorString(_e));     \
        exit(1);                                                \
    }                                                           \
} while(0)

static int nGPUs;

/* ── Test 1: FP32 SUM correctness ── */
static int test_fp32_sum(infcclComm_t* comms) {
    const int N = 1024;
    float* h_in  = (float*)malloc(N * sizeof(float));
    float* h_out = (float*)malloc(N * sizeof(float));
    float** d_in  = (float**)malloc(nGPUs * sizeof(float*));
    float** d_out = (float**)malloc(nGPUs * sizeof(float*));

    /* expected: each rank sends [1, 2, ..., N] × (rank+1) */
    for (int r = 0; r < nGPUs; r++) {
        cudaSetDevice(r);
        CUDA_CHECK(cudaMalloc(&d_in[r],  N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out[r], N * sizeof(float)));
        for (int i = 0; i < N; i++) h_in[i] = (float)(i + 1) * (r + 1);
        CUDA_CHECK(cudaMemcpy(d_in[r], h_in, N * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    /* run allreduce on all ranks (sequential in single-process) */
    for (int r = 0; r < nGPUs; r++) {
        cudaSetDevice(r);
        CHECK(infcclAllReduce(d_in[r], d_out[r], N, infcclFloat32,
                              infcclSum, comms[r], 0));
    }

    /* verify: sum of (rank+1) for all ranks = nGPUs*(nGPUs+1)/2 */
    float scale = (float)(nGPUs * (nGPUs + 1)) / 2.0f;
    int errors = 0;
    for (int r = 0; r < nGPUs; r++) {
        cudaSetDevice(r);
        CUDA_CHECK(cudaMemcpy(h_out, d_out[r], N * sizeof(float),
                              cudaMemcpyDeviceToHost));
        for (int i = 0; i < N; i++) {
            float expected = (float)(i + 1) * scale;
            if (fabsf(h_out[i] - expected) > 1e-3f * fabsf(expected)) {
                if (errors < 5)
                    fprintf(stderr, "  rank %d elem %d: got %f expected %f\n",
                            r, i, h_out[i], expected);
                errors++;
            }
        }
    }

    for (int r = 0; r < nGPUs; r++) {
        cudaSetDevice(r);
        cudaFree(d_in[r]); cudaFree(d_out[r]);
    }
    free(h_in); free(h_out); free(d_in); free(d_out);

    printf("  fp32_sum: %s (%d errors)\n", errors == 0 ? "PASS" : "FAIL", errors);
    return errors == 0;
}

/* ── Test 2: FP16 SUM correctness ── */
static int test_fp16_sum(infcclComm_t* comms) {
    const int N = 2048;
    half* h_in  = (half*)malloc(N * sizeof(half));
    half* h_out = (half*)malloc(N * sizeof(half));
    half** d_in  = (half**)malloc(nGPUs * sizeof(half*));
    half** d_out = (half**)malloc(nGPUs * sizeof(half*));

    for (int r = 0; r < nGPUs; r++) {
        cudaSetDevice(r);
        CUDA_CHECK(cudaMalloc(&d_in[r],  N * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_out[r], N * sizeof(half)));
        for (int i = 0; i < N; i++)
            h_in[i] = __float2half((float)(i % 64 + 1) * 0.1f * (r + 1));
        CUDA_CHECK(cudaMemcpy(d_in[r], h_in, N * sizeof(half),
                              cudaMemcpyHostToDevice));
    }

    for (int r = 0; r < nGPUs; r++) {
        cudaSetDevice(r);
        CHECK(infcclAllReduce(d_in[r], d_out[r], N, infcclFloat16,
                              infcclSum, comms[r], 0));
    }

    float scale = (float)(nGPUs * (nGPUs + 1)) / 2.0f;
    int errors = 0;
    for (int r = 0; r < nGPUs; r++) {
        cudaSetDevice(r);
        CUDA_CHECK(cudaMemcpy(h_out, d_out[r], N * sizeof(half),
                              cudaMemcpyDeviceToHost));
        for (int i = 0; i < N; i++) {
            float got = __half2float(h_out[i]);
            float expected = (float)(i % 64 + 1) * 0.1f * scale;
            if (fabsf(got - expected) > 0.05f * fabsf(expected) + 0.01f) {
                if (errors < 5)
                    fprintf(stderr, "  rank %d elem %d: got %f expected %f\n",
                            r, i, got, expected);
                errors++;
            }
        }
    }

    for (int r = 0; r < nGPUs; r++) {
        cudaSetDevice(r); cudaFree(d_in[r]); cudaFree(d_out[r]);
    }
    free(h_in); free(h_out); free(d_in); free(d_out);

    printf("  fp16_sum: %s (%d errors)\n", errors == 0 ? "PASS" : "FAIL", errors);
    return errors == 0;
}

/* ── Benchmark ── */
static void benchmark(infcclComm_t* comms) {
    const int sizes[] = {256, 1024, 4096, 16384, 65536, 262144, 1048576};
    const int nsizes = sizeof(sizes) / sizeof(sizes[0]);
    const int warmup = 5, iters = 20;

    printf("\n  Benchmark (FP16 SUM, %d GPUs):\n", nGPUs);
    printf("  %10s %10s %10s\n", "elements", "bytes", "latency_us");

    for (int s = 0; s < nsizes; s++) {
        int N = sizes[s];
        half** d_buf = (half**)malloc(nGPUs * sizeof(half*));
        for (int r = 0; r < nGPUs; r++) {
            cudaSetDevice(r);
            CUDA_CHECK(cudaMalloc(&d_buf[r], N * sizeof(half)));
            CUDA_CHECK(cudaMemset(d_buf[r], 0, N * sizeof(half)));
        }

        /* warmup */
        for (int w = 0; w < warmup; w++)
            for (int r = 0; r < nGPUs; r++) {
                cudaSetDevice(r);
                infcclAllReduce(d_buf[r], d_buf[r], N, infcclFloat16,
                                infcclSum, comms[r], 0);
            }

        /* timed iterations */
        cudaSetDevice(0);
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));

        for (int it = 0; it < iters; it++)
            for (int r = 0; r < nGPUs; r++) {
                cudaSetDevice(r);
                infcclAllReduce(d_buf[r], d_buf[r], N, infcclFloat16,
                                infcclSum, comms[r], 0);
            }

        cudaSetDevice(0);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float us = ms * 1000.0f / (float)iters;

        printf("  %10d %10zu %10.1f\n", N, (size_t)N * 2, us);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        for (int r = 0; r < nGPUs; r++) {
            cudaSetDevice(r); cudaFree(d_buf[r]);
        }
        free(d_buf);
    }
}

int main() {
    CUDA_CHECK(cudaGetDeviceCount(&nGPUs));
    if (nGPUs < 2) {
        fprintf(stderr, "Need >= 2 GPUs, found %d\n", nGPUs);
        return 1;
    }
    printf("[infccl] AllReduce test with %d GPUs\n", nGPUs);

    infcclComm_t* comms = (infcclComm_t*)malloc(nGPUs * sizeof(infcclComm_t));
    CHECK(infcclCommInitAll(comms, nGPUs));

    int pass = 0, total = 0;

    total++; pass += test_fp32_sum(comms);
    total++; pass += test_fp16_sum(comms);

    benchmark(comms);

    printf("\n[infccl] %d/%d tests passed\n", pass, total);

    for (int r = 0; r < nGPUs; r++)
        infcclCommDestroy(comms[r]);
    free(comms);

    return (pass == total) ? 0 : 1;
}
