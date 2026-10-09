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

int main(){
    cudaGetDeviceCount(&ngpu);
    printf("=== AllGather test (%d GPUs) ===\n",ngpu);
    if(ngpu<2){printf("need 2+\n");return 1;}
    CHECK(infcclCommInitAll(comms,ngpu,NULL));
    int err=0;
    int sizes[]={1,7,256,1024,4096,65536};

    for(int si=0;si<6;si++){
        int sc=sizes[si];
        int totalN=sc*ngpu;
        float*sbufs[INFCCL_MAX_DEVS];
        float*rbufs[INFCCL_MAX_DEVS];
        for(int g=0;g<ngpu;g++){
            CUCHK(cudaSetDevice(g));
            CUCHK(cudaMalloc(&sbufs[g],sc*4));
            CUCHK(cudaMalloc(&rbufs[g],totalN*4));
            fill_f<<<(sc+255)/256,256>>>(sbufs[g],sc,(float)(g+1)*10.0f);
            CUCHK(cudaDeviceSynchronize());
            CUCHK(cudaMemset(rbufs[g],0,totalN*4));
        }
        CHECK(infcclAllGather((void**)sbufs,(void**)rbufs,sc,infcclFloat,comms[0],0));
        int e=0;
        for(int g=0;g<ngpu;g++){
            float*h=(float*)malloc(totalN*4);
            CUCHK(cudaSetDevice(g));
            CUCHK(cudaMemcpy(h,rbufs[g],totalN*4,cudaMemcpyDeviceToHost));
            for(int s=0;s<ngpu;s++){
                float exp=(float)(s+1)*10.0f;
                for(int i=0;i<sc;i++){
                    if(fabsf(h[s*sc+i]-exp)>1e-3f){
                        if(e<3)printf("    gpu%d chunk%d[%d]=%.1f want %.1f\n",g,s,i,h[s*sc+i],exp);
                        e++;
                    }
                }
            }
            free(h);
        }
        printf("  sc=%-8d err=%d %s\n",sc,e,e==0?"OK":"FAIL");
        err+=e;
        for(int g=0;g<ngpu;g++){CUCHK(cudaSetDevice(g));cudaFree(sbufs[g]);cudaFree(rbufs[g]);}
    }

    printf("  [bench]\n");
    {
        int sizes[]={256,1024,4096,65536};
        for(int si=0;si<4;si++){
            int sc=sizes[si];int totalN=sc*ngpu;int iters=(sc<=4096)?500:100;
            float*sbufs[INFCCL_MAX_DEVS];float*rbufs[INFCCL_MAX_DEVS];
            for(int g=0;g<ngpu;g++){cudaSetDevice(g);cudaMalloc(&sbufs[g],sc*4);cudaMalloc(&rbufs[g],totalN*4);fill_f<<<(sc+255)/256,256>>>(sbufs[g],sc,1.0f);cudaDeviceSynchronize();}
            for(int i=0;i<10;i++)infcclAllGather((void**)sbufs,(void**)rbufs,sc,infcclFloat,comms[0],0);
            cudaSetDevice(0);cudaDeviceSynchronize();
            cudaEvent_t t0,t1;cudaEventCreate(&t0);cudaEventCreate(&t1);
            cudaEventRecord(t0);
            for(int i=0;i<iters;i++)infcclAllGather((void**)sbufs,(void**)rbufs,sc,infcclFloat,comms[0],0);
            cudaEventRecord(t1);cudaEventSynchronize(t1);
            float ms;cudaEventElapsedTime(&ms,t0,t1);
            printf("    sc=%-8d %6.1f us\n",sc,ms*1000/iters);
            cudaEventDestroy(t0);cudaEventDestroy(t1);
            for(int g=0;g<ngpu;g++){cudaSetDevice(g);cudaFree(sbufs[g]);cudaFree(rbufs[g]);}
        }
    }

    for(int r=0;r<ngpu;r++)infcclCommDestroy(comms[r]);
    printf("=== AllGather errors: %d ===\n",err);
    return err>0?1:0;
}
