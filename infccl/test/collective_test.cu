#include "infccl.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_fp16.h>

#define CHK(c) do{infcclResult_t r=(c);if(r!=infcclSuccess){printf("FAIL %d: %s\n",__LINE__,infcclGetErrorString(r));return 1;}}while(0)
#define CUCHK(c) do{cudaError_t e=(c);if(e!=cudaSuccess){printf("CUDA %d: %s\n",__LINE__,cudaGetErrorString(e));return 1;}}while(0)

__global__ void fill_f(float*b,int n,float v){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)b[i]=v;}

static int ngpu;
static infcclComm_t comms[INFCCL_MAX_DEVS];

int vf(float*d,int n,float exp,int gpu){
    float*h=(float*)malloc(n*4);cudaSetDevice(gpu);cudaMemcpy(h,d,n*4,cudaMemcpyDeviceToHost);
    int e=0;for(int i=0;i<n;i++)if(fabsf(h[i]-exp)>1e-3f){if(e<2)printf("  [%d]=%.4f want %.4f\n",i,h[i],exp);e++;}
    free(h);return e;
}

int test_bcast(){
    printf("[Broadcast]\n");
    int N=4096;int total=0;
    for(int root=0;root<ngpu;root++){
        float*bufs[INFCCL_MAX_DEVS];
        for(int g=0;g<ngpu;g++){CUCHK(cudaSetDevice(g));CUCHK(cudaMalloc(&bufs[g],N*4));}
        for(int g=0;g<ngpu;g++){CUCHK(cudaSetDevice(g));fill_f<<<(N+255)/256,256>>>(bufs[g],N,(float)(g+1));CUCHK(cudaDeviceSynchronize());}
        CHK(infcclBcast((void**)bufs,N,infcclFloat,root,comms[0],0));
        int e=0;
        for(int g=0;g<ngpu;g++)e+=vf(bufs[g],N,(float)(root+1),g);
        printf("  root=%d err=%d %s\n",root,e,e==0?"OK":"FAIL");
        total+=e;
        for(int g=0;g<ngpu;g++){CUCHK(cudaSetDevice(g));cudaFree(bufs[g]);}
    }
    return total;
}

int test_reduce(){
    printf("[Reduce]\n");
    int N=4096;int total=0;
    for(int root=0;root<ngpu;root++){
        float*bufs[INFCCL_MAX_DEVS];
        for(int g=0;g<ngpu;g++){CUCHK(cudaSetDevice(g));CUCHK(cudaMalloc(&bufs[g],N*4));}
        for(int g=0;g<ngpu;g++){CUCHK(cudaSetDevice(g));fill_f<<<(N+255)/256,256>>>(bufs[g],N,(float)(g+1));CUCHK(cudaDeviceSynchronize());}
        CHK(infcclReduce((void**)bufs,N,infcclFloat,infcclSum,root,comms[0],0));
        float exp=0;for(int g=0;g<ngpu;g++)exp+=(g+1);
        int e=vf(bufs[root],N,exp,root);
        printf("  root=%d err=%d %s\n",root,e,e==0?"OK":"FAIL");
        total+=e;
        for(int g=0;g<ngpu;g++){CUCHK(cudaSetDevice(g));cudaFree(bufs[g]);}
    }
    return total;
}

int test_allgather(){
    printf("[AllGather]\n");
    int sc=1024;int N=sc*ngpu;int total=0;
    float*sbufs[INFCCL_MAX_DEVS];
    float*rbufs[INFCCL_MAX_DEVS];
    for(int g=0;g<ngpu;g++){
        CUCHK(cudaSetDevice(g));
        CUCHK(cudaMalloc(&sbufs[g],sc*4));
        CUCHK(cudaMalloc(&rbufs[g],N*4));
        fill_f<<<(sc+255)/256,256>>>(sbufs[g],sc,(float)(g+1)*100.0f);
        CUCHK(cudaDeviceSynchronize());
    }
    CHK(infcclAllGather((void**)sbufs,(void**)rbufs,sc,infcclFloat,comms[0],0));
    for(int g=0;g<ngpu;g++){
        float*h=(float*)malloc(N*4);
        cudaSetDevice(g);cudaMemcpy(h,rbufs[g],N*4,cudaMemcpyDeviceToHost);
        int e=0;
        for(int src=0;src<ngpu;src++){
            float exp=(src+1)*100.0f;
            for(int i=0;i<sc;i++)
                if(fabsf(h[src*sc+i]-exp)>1e-3f)e++;
        }
        printf("  GPU%d err=%d %s\n",g,e,e==0?"OK":"FAIL");
        total+=e;free(h);
    }
    for(int g=0;g<ngpu;g++){CUCHK(cudaSetDevice(g));cudaFree(sbufs[g]);cudaFree(rbufs[g]);}
    return total;
}

int test_reduce_scatter(){
    printf("[ReduceScatter]\n");
    int rc=1024;int N=rc*ngpu;int total=0;
    float*sbufs[INFCCL_MAX_DEVS];
    float*rbufs[INFCCL_MAX_DEVS];
    for(int g=0;g<ngpu;g++){
        CUCHK(cudaSetDevice(g));
        CUCHK(cudaMalloc(&sbufs[g],N*4));
        CUCHK(cudaMalloc(&rbufs[g],rc*4));
        fill_f<<<(N+255)/256,256>>>(sbufs[g],N,1.0f);
        CUCHK(cudaDeviceSynchronize());
    }
    CHK(infcclReduceScatter((void**)sbufs,(void**)rbufs,rc,infcclFloat,infcclSum,comms[0],0));
    for(int g=0;g<ngpu;g++){
        float exp=(float)ngpu;
        int e=vf(rbufs[g],rc,exp,g);
        printf("  GPU%d err=%d %s\n",g,e,e==0?"OK":"FAIL");
        total+=e;
    }
    for(int g=0;g<ngpu;g++){CUCHK(cudaSetDevice(g));cudaFree(sbufs[g]);cudaFree(rbufs[g]);}
    return total;
}

int main(){
    cudaGetDeviceCount(&ngpu);
    printf("=== Collective ops test (%d GPUs) ===\n",ngpu);
    if(ngpu<2){printf("need 2+\n");return 1;}
    CHK(infcclCommInitAll(comms,ngpu,NULL));
    int err=0;
    err+=test_bcast();
    err+=test_reduce();
    err+=test_allgather();
    err+=test_reduce_scatter();
    for(int r=0;r<ngpu;r++)infcclCommDestroy(comms[r]);
    printf("=== errors: %d ===\n",err);
    return err>0?1:0;
}
