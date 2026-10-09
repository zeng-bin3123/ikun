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
    printf("=== ReduceScatter test (%d GPUs) ===\n",ngpu);
    if(ngpu<2){printf("need 2+\n");return 1;}
    CHECK(infcclCommInitAll(comms,ngpu,NULL));
    int err=0;
    int sizes[]={1,7,256,1024,4096,65536};

    for(int si=0;si<6;si++){
        int rc=sizes[si];
        int totalN=rc*ngpu;
        float*sbufs[INFCCL_MAX_DEVS];
        float*rbufs[INFCCL_MAX_DEVS];
        for(int g=0;g<ngpu;g++){
            CUCHK(cudaSetDevice(g));
            CUCHK(cudaMalloc(&sbufs[g],totalN*4));
            CUCHK(cudaMalloc(&rbufs[g],rc*4));
            fill_f<<<(totalN+255)/256,256>>>(sbufs[g],totalN,(float)(g+1));
            CUCHK(cudaDeviceSynchronize());
        }
        CHECK(infcclReduceScatter((void**)sbufs,(void**)rbufs,rc,infcclFloat,infcclSum,comms[0],0));
        float sum_exp=0;for(int g=0;g<ngpu;g++)sum_exp+=(g+1);
        int e=0;for(int g=0;g<ngpu;g++)e+=vf(rbufs[g],rc,sum_exp,g);
        printf("  rc=%-8d err=%d %s\n",rc,e,e==0?"OK":"FAIL");
        err+=e;
        for(int g=0;g<ngpu;g++){CUCHK(cudaSetDevice(g));cudaFree(sbufs[g]);cudaFree(rbufs[g]);}
    }

    printf("  [bench]\n");
    {
        int sizes[]={256,1024,4096,65536};
        for(int si=0;si<4;si++){
            int rc=sizes[si];int totalN=rc*ngpu;int iters=(rc<=4096)?500:100;
            float*sbufs[INFCCL_MAX_DEVS];float*rbufs[INFCCL_MAX_DEVS];
            for(int g=0;g<ngpu;g++){cudaSetDevice(g);cudaMalloc(&sbufs[g],totalN*4);cudaMalloc(&rbufs[g],rc*4);fill_f<<<(totalN+255)/256,256>>>(sbufs[g],totalN,1.0f);cudaDeviceSynchronize();}
            for(int i=0;i<10;i++)infcclReduceScatter((void**)sbufs,(void**)rbufs,rc,infcclFloat,infcclSum,comms[0],0);
            cudaSetDevice(0);cudaDeviceSynchronize();
            cudaEvent_t t0,t1;cudaEventCreate(&t0);cudaEventCreate(&t1);
            cudaEventRecord(t0);
            for(int i=0;i<iters;i++)infcclReduceScatter((void**)sbufs,(void**)rbufs,rc,infcclFloat,infcclSum,comms[0],0);
            cudaEventRecord(t1);cudaEventSynchronize(t1);
            float ms;cudaEventElapsedTime(&ms,t0,t1);
            printf("    rc=%-8d %6.1f us\n",rc,ms*1000/iters);
            cudaEventDestroy(t0);cudaEventDestroy(t1);
            for(int g=0;g<ngpu;g++){cudaSetDevice(g);cudaFree(sbufs[g]);cudaFree(rbufs[g]);}
        }
    }

    for(int r=0;r<ngpu;r++)infcclCommDestroy(comms[r]);
    printf("=== ReduceScatter errors: %d ===\n",err);
    return err>0?1:0;
}
