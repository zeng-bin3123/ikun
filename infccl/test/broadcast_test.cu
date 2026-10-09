#include "../src/infccl.h"
#include "../src/core.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CHECK(cmd) do{infcclResult_t r=(cmd);if(r!=infcclSuccess){printf("FAIL %d: %s\n",__LINE__,infcclGetErrorString(r));return 1;}}while(0)
#define CUCHK(cmd) do{cudaError_t e=(cmd);if(e!=cudaSuccess){printf("CUDA %d: %s\n",__LINE__,cudaGetErrorString(e));return 1;}}while(0)

__global__ void fill_f(float* b,int n,float v){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)b[i]=v;}

static int ngpu;
static infcclComm_t comms[INFCCL_MAX_DEVS];

int vf(float* d,int n,float exp,int gpu){
    float*h=(float*)malloc(n*4);cudaSetDevice(gpu);cudaMemcpy(h,d,n*4,cudaMemcpyDeviceToHost);
    int e=0;for(int i=0;i<n;i++)if(fabsf(h[i]-exp)>1e-3f){if(e<2)printf("    gpu%d[%d]=%.4f want %.4f\n",gpu,i,h[i],exp);e++;}
    free(h);return e;
}

int main(){
    cudaGetDeviceCount(&ngpu);
    printf("=== Broadcast test (%d GPUs) ===\n",ngpu);
    if(ngpu<2){printf("need 2+\n");return 1;}
    CHECK(infcclCommInitAll(comms,ngpu,NULL));
    int err=0;
    int sizes[]={1,7,256,4096,100003,262144};

    for(int root=0;root<ngpu;root++){
        printf("  root=%d\n",root);
        for(int si=0;si<6;si++){
            int N=sizes[si];
            float*bufs[INFCCL_MAX_DEVS];
            for(int g=0;g<ngpu;g++){CUCHK(cudaSetDevice(g));CUCHK(cudaMalloc(&bufs[g],N*4));}
            float val = (float)(root+1)*77.0f;
            CUCHK(cudaSetDevice(root));fill_f<<<(N+255)/256,256>>>(bufs[root],N,val);CUCHK(cudaDeviceSynchronize());
            for(int g=0;g<ngpu;g++){if(g==root)continue;CUCHK(cudaSetDevice(g));CUCHK(cudaMemset(bufs[g],0,N*4));}
            CHECK(infcclBcast((void**)bufs,N,infcclFloat,root,comms[0],0));
            int e=0;for(int g=0;g<ngpu;g++)e+=vf(bufs[g],N,val,g);
            printf("    N=%-8d err=%d %s\n",N,e,e==0?"OK":"FAIL");
            err+=e;
            for(int g=0;g<ngpu;g++){CUCHK(cudaSetDevice(g));cudaFree(bufs[g]);}
        }
    }

    printf("  [bench]\n");
    {
        int sizes[]={2048,4096,65536,262144};
        for(int si=0;si<4;si++){
            int N=sizes[si];int iters=(N<=65536)?500:100;
            float*bufs[INFCCL_MAX_DEVS];
            for(int g=0;g<ngpu;g++){cudaSetDevice(g);cudaMalloc(&bufs[g],N*4);fill_f<<<(N+255)/256,256>>>(bufs[g],N,1.0f);cudaDeviceSynchronize();}
            for(int i=0;i<10;i++)infcclBcast((void**)bufs,N,infcclFloat,0,comms[0],0);
            cudaSetDevice(0);cudaDeviceSynchronize();
            cudaEvent_t t0,t1;cudaEventCreate(&t0);cudaEventCreate(&t1);
            cudaEventRecord(t0);
            for(int i=0;i<iters;i++)infcclBcast((void**)bufs,N,infcclFloat,0,comms[0],0);
            cudaEventRecord(t1);cudaEventSynchronize(t1);
            float ms;cudaEventElapsedTime(&ms,t0,t1);
            printf("    N=%-8d %6.1f us\n",N,ms*1000/iters);
            cudaEventDestroy(t0);cudaEventDestroy(t1);
            for(int g=0;g<ngpu;g++){cudaSetDevice(g);cudaFree(bufs[g]);}
        }
    }

    for(int r=0;r<ngpu;r++)infcclCommDestroy(comms[r]);
    printf("=== Broadcast errors: %d ===\n",err);
    return err>0?1:0;
}
