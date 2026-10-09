#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void k_complex(float *out, int n) {
    __shared__ float smem[128];
    half2 a = __float2half2_rn(1.0f);
    half2 b = __float2half2_rn(0.5f);
    half2 c = __float2half2_rn(0.0f);
    #pragma unroll
    for (int i = 0; i < 100; i++) c = __hfma2(a, b, c);
    float val = __half2float(__low2half(c)) + __half2float(__high2half(c));
    smem[threadIdx.x] = val;
    __syncthreads();
    val = smem[threadIdx.x];
    val += __shfl_down_sync(0xffffffff, val, 32);
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    if (threadIdx.x == 0) {
        out[0] = val;
        out[1] = (float)n;
        out[2] = (float)warpSize;
    }
}

int main() {
    float *d, h[3] = {0};
    cudaMalloc(&d, 12);
    k_complex<<<1, 64>>>(d, 12345);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("LAUNCH FAILED: %s\n", cudaGetErrorString(err));
        cudaFree(d);
        return 1;
    }
    cudaMemcpy(h, d, 12, cudaMemcpyDeviceToHost);
    cudaFree(d);
    printf("reduce=%.1f param=%.0f warp=%.0f\n", h[0], h[1], h[2]);
    if (h[0] > 6300.0f && h[0] < 6500.0f && h[1] == 12345.0f && h[2] == 64.0f)
        printf("CORRECT\n");
    else if (h[0] == 0.0f && h[1] == 0.0f && h[2] == 0.0f)
        printf("ALL ZERO\n");
    else
        printf("WRONG\n");
    return 0;
}
