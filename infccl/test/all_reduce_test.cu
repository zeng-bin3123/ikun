#include "infccl.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_fp16.h>

#define CHK(c) do{infcclResult_t r=(c);if(r!=infcclSuccess){printf("FAIL %d: %s\n",__LINE__,infcclGetErrorString(r));return 1;}}while(0)
#define CUCHK(c) do{cudaError_t e=(c);if(e!=cudaSuccess){printf("CUDA %d: %s\n",__LINE__,cudaGetErrorString(e));return 1;}}while(0)

__global__ void fill_f(float*b,int n,float v){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)b[i]=v;}
__global__ void fill_h(half*b,int n,float v){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)b[i]=__float2half(v);}

static int ngpu;
static infcclComm_t comms[INFCCL_MAX_DEVS];

int vf(float*d,int n,float exp,int gpu){
    float*h=(float*)malloc(n*4);cudaSetDevice(gpu);cudaMemcpy(h,d,n*4,cudaMemcpyDeviceToHost);
    int e=0;for(int i=0;i<n;i++)if(fabsf(h[i]-exp)>1e-3f){if(e<2)printf("  [%d]=%.4f want %.4f\n",i,h[i],exp);e++;}
    free(h);return e;
}

int test_float(){
    printf("[float]\n");
    int sizes[]={1,7,256,4096,100003,262144};
    const char*opn[]={"Sum","Prod","Max","Min"};
    infcclRedOp_t ops[]={infcclSum,infcclProd,infcclMax,infcclMin};
    int total=0;
    for(int si=0;si<6;si++){
        int N=sizes[si];
        float*bufs[INFCCL_MAX_DEVS];
        for(int g=0;g<ngpu;g++){CUCHK(cudaSetDevice(g));CUCHK(cudaMalloc(&bufs[g],N*4));}
        for(int oi=0;oi<4;oi++){
            for(int g=0;g<ngpu;g++){CUCHK(cudaSetDevice(g));fill_f<<<(N+255)/256,256>>>(bufs[g],N,(float)(g+1));CUCHK(cudaDeviceSynchronize());}
            CHK(infcclAllReduce((void**)bufs,N,infcclFloat,ops[oi],comms[0],0));
            float exp;
            if(oi==0){exp=0;for(int g=0;g<ngpu;g++)exp+=(g+1);}
            else if(oi==1){exp=1;for(int g=0;g<ngpu;g++)exp*=(g+1);}
            else if(oi==2)exp=(float)ngpu;
            else exp=1.0f;
            int e=0;for(int g=0;g<ngpu;g++)e+=vf(bufs[g],N,exp,g);
            printf("  N=%-8d %-4s err=%d %s\n",N,opn[oi],e,e==0?"OK":"FAIL");
            total+=e;
        }
        for(int g=0;g<ngpu;g++){CUCHK(cudaSetDevice(g));cudaFree(bufs[g]);}
    }
    return total;
}

int test_half(){
    printf("[half]\n");
    int N=4096;
    half*bufs[INFCCL_MAX_DEVS];
    for(int g=0;g<ngpu;g++){CUCHK(cudaSetDevice(g));CUCHK(cudaMalloc(&bufs[g],N*2));}
    for(int g=0;g<ngpu;g++){CUCHK(cudaSetDevice(g));fill_h<<<(N+255)/256,256>>>(bufs[g],N,(float)(g+1));CUCHK(cudaDeviceSynchronize());}
    CHK(infcclAllReduce((void**)bufs,N,infcclHalf,infcclSum,comms[0],0));
    float exp=0;for(int g=0;g<ngpu;g++)exp+=(g+1);
    half*h=(half*)malloc(N*2);cudaSetDevice(0);cudaMemcpy(h,bufs[0],N*2,cudaMemcpyDeviceToHost);
    int e=0;for(int i=0;i<N;i++)if(fabsf(__half2float(h[i])-exp)>0.5f)e++;
    printf("  N=%-8d Sum  err=%d %s\n",N,e,e==0?"OK":"FAIL");
    free(h);for(int g=0;g<ngpu;g++){cudaSetDevice(g);cudaFree(bufs[g]);}
    return e;
}

void bench(){
    printf("[bench]\n");
    int sizes[]={2048,4096,65536,262144,1048576};
    for(int si=0;si<5;si++){
        int N=sizes[si];int iters=(N<=65536)?500:100;
        float*bufs[INFCCL_MAX_DEVS];
        for(int g=0;g<ngpu;g++){cudaSetDevice(g);cudaMalloc(&bufs[g],N*4);fill_f<<<(N+255)/256,256>>>(bufs[g],N,1.0f);cudaDeviceSynchronize();}
        for(int i=0;i<10;i++)infcclAllReduce((void**)bufs,N,infcclFloat,infcclSum,comms[0],0);
        cudaSetDevice(0);cudaDeviceSynchronize();
        cudaEvent_t t0,t1;cudaEventCreate(&t0);cudaEventCreate(&t1);
        cudaEventRecord(t0);
        for(int i=0;i<iters;i++)infcclAllReduce((void**)bufs,N,infcclFloat,infcclSum,comms[0],0);
        cudaEventRecord(t1);cudaEventSynchronize(t1);
        float ms;cudaEventElapsedTime(&ms,t0,t1);
        printf("  N=%-8d %6.1fus  %.2fGB/s\n",N,ms*1000/iters,2.0*N*4e-3/(ms/iters));
        cudaEventDestroy(t0);cudaEventDestroy(t1);
        for(int g=0;g<ngpu;g++){cudaSetDevice(g);cudaFree(bufs[g]);}
    }
}

int main(){
    cudaGetDeviceCount(&ngpu);
    printf("=== AllReduce test (%d GPUs) ===\n",ngpu);
    if(ngpu<2){printf("need 2+\n");return 1;}
    CHK(infcclCommInitAll(comms,ngpu,NULL));
    int err=0;
    err+=test_float();
    err+=test_half();
    bench();
    for(int r=0;r<ngpu;r++)infcclCommDestroy(comms[r]);
    printf("=== errors: %d ===\n",err);
    return err>0?1:0;
}
