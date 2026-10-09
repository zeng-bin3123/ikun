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
    int e=0;for(int i=0;i<n;i++)if(fabsf(h[i]-exp)>1e-3f){if(e<2)printf("    [%d]=%.4f want %.4f\n",i,h[i],exp);e++;}
    free(h);return e;
}

int main(){
    cudaGetDeviceCount(&ngpu);
    printf("=== Reduce test (%d GPUs) ===\n",ngpu);
    if(ngpu<2){printf("need 2+\n");return 1;}
    CHECK(infcclCommInitAll(comms,ngpu,NULL));
    int err=0;
    int sizes[]={1,256,4096,100003};
    const char* opnames[]={"Sum","Prod","Max","Min"};
    infcclRedOp_t ops[]={infcclSum,infcclProd,infcclMax,infcclMin};

    for(int root=0;root<ngpu;root++){
        printf("  root=%d\n",root);
        for(int si=0;si<4;si++){
            int N=sizes[si];
            float*bufs[INFCCL_MAX_DEVS];
            for(int g=0;g<ngpu;g++){CUCHK(cudaSetDevice(g));CUCHK(cudaMalloc(&bufs[g],N*4));}
            for(int oi=0;oi<4;oi++){
                for(int g=0;g<ngpu;g++){CUCHK(cudaSetDevice(g));fill_f<<<(N+255)/256,256>>>(bufs[g],N,(float)(g+1));CUCHK(cudaDeviceSynchronize());}
                CHECK(infcclReduce((void**)bufs,N,infcclFloat,ops[oi],root,comms[0],0));
                float exp;
                if(oi==0){exp=0;for(int g=0;g<ngpu;g++)exp+=(g+1);}
                else if(oi==1){exp=1;for(int g=0;g<ngpu;g++)exp*=(g+1);}
                else if(oi==2){exp=(float)ngpu;}
                else{exp=1.0f;}
                int e=vf(bufs[root],N,exp,root);
                printf("    N=%-8d %-4s err=%d %s\n",N,opnames[oi],e,e==0?"OK":"FAIL");
                err+=e;
            }
            for(int g=0;g<ngpu;g++){CUCHK(cudaSetDevice(g));cudaFree(bufs[g]);}
        }
    }
    for(int r=0;r<ngpu;r++)infcclCommDestroy(comms[r]);
    printf("=== Reduce errors: %d ===\n",err);
    return err>0?1:0;
}
