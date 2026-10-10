#include "transfer_engine.h"
#include "peer_transport.h"
#include <cstdio>
#include <cmath>

using namespace infccl;

__global__ void fill(float*b,int n,float v){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)b[i]=v;}

int main(){
    int ngpu;cudaGetDeviceCount(&ngpu);
    printf("=== Transfer Engine test (%d GPUs) ===\n",ngpu);
    if(ngpu<2){printf("need 2+\n");return 1;}

    auto meta=std::make_shared<TransferMetadata>();
    TransferEngine engine(meta);
    int devs[8];for(int i=0;i<ngpu;i++)devs[i]=i;
    engine.init("local",ngpu,devs);
    Transport*xport=engine.installOrGetTransport("peer",nullptr);
    if(!xport){printf("install FAIL\n");return 1;}

    int N=1<<20;size_t bytes=N*4;
    float*bufs[8];
    for(int g=0;g<ngpu;g++){
        cudaSetDevice(g);cudaMalloc(&bufs[g],bytes);
        fill<<<(N+255)/256,256>>>(bufs[g],N,(float)(g+1)*10.0f);cudaDeviceSynchronize();
        char loc[16];snprintf(loc,16,"gpu%d",g);
        engine.registerLocalMemory(bufs[g],bytes,loc);
    }

    SegmentID seg_id=engine.openSegment("local");
    printf("seg_id=%lu\n",(unsigned long)seg_id);

    printf("[1] READ via Transfer Engine\n");
    float*local;cudaSetDevice(1);cudaMalloc(&local,bytes);cudaMemset(local,0,bytes);
    char loc1[16];snprintf(loc1,16,"gpu1");
    engine.registerLocalMemory(local,bytes,loc1);

    Transport::BatchID batch=xport->allocateBatchID(1);
    std::vector<Transport::TransferRequest> reqs(1);
    reqs[0].opcode=Transport::TransferRequest::READ;
    reqs[0].source=local;
    reqs[0].target_id=seg_id;
    reqs[0].target_offset=0;
    reqs[0].length=bytes;
    int rc=xport->submitTransfer(batch,reqs);
    printf("  submit=%d\n",rc);

    cudaSetDevice(1);cudaDeviceSynchronize();
    Transport::TransferStatus st;
    while(xport->getTransferStatus(batch,0,st)==0){}
    float h[4];cudaMemcpy(h,local,16,cudaMemcpyDeviceToHost);
    int errs=(h[0]!=10.0f)?1:0;
    printf("  %.0f %.0f %.0f %.0f %s\n",h[0],h[1],h[2],h[3],errs==0?"OK":"FAIL");

    printf("[2] BW: READ 4MB x100\n");
    cudaEvent_t t0,t1;cudaEventCreate(&t0);cudaEventCreate(&t1);
    cudaSetDevice(1);cudaEventRecord(t0);
    for(int i=0;i<100;i++){
        Transport::BatchID b=xport->allocateBatchID(1);
        xport->submitTransfer(b,reqs);
        cudaDeviceSynchronize();
        xport->freeBatchID(b);
    }
    cudaEventRecord(t1);cudaEventSynchronize(t1);
    float ms;cudaEventElapsedTime(&ms,t0,t1);
    printf("  %.1f GB/s\n",bytes*100*1e-6/ms);

    printf("[3] All-pairs READ\n");
    int pair_err=0;
    for(int src=0;src<ngpu;src++)for(int dst=0;dst<ngpu;dst++){
        if(src==dst)continue;
        float*lb;cudaSetDevice(dst);cudaMalloc(&lb,1024*4);cudaMemset(lb,0,1024*4);
        char ll[16];snprintf(ll,16,"tmp%d%d",src,dst);
        engine.registerLocalMemory(lb,1024*4,ll);

        Transport::BatchID b2=xport->allocateBatchID(1);
        std::vector<Transport::TransferRequest> r2(1);
        r2[0].opcode=Transport::TransferRequest::READ;
        r2[0].source=lb;
        r2[0].target_id=seg_id;
        r2[0].target_offset=src*bytes;
        r2[0].length=1024*4;
        xport->submitTransfer(b2,r2);
        cudaDeviceSynchronize();
        float v;cudaMemcpy(&v,lb,4,cudaMemcpyDeviceToHost);
        float exp=(src+1)*10.0f;
        if(fabsf(v-exp)>0.1f){printf("  %d->%d FAIL: %.0f!=%.0f\n",src,dst,v,exp);pair_err++;}
        engine.unregisterLocalMemory(lb);
        cudaFree(lb);
        xport->freeBatchID(b2);
    }
    printf("  %d pairs, %d errors %s\n",ngpu*(ngpu-1),pair_err,pair_err==0?"OK":"FAIL");
    errs+=pair_err;

    xport->freeBatchID(batch);
    engine.unregisterLocalMemory(local);cudaFree(local);
    for(int g=0;g<ngpu;g++){engine.unregisterLocalMemory(bufs[g]);cudaSetDevice(g);cudaFree(bufs[g]);}
    cudaEventDestroy(t0);cudaEventDestroy(t1);
    printf("=== errors: %d ===\n",errs);
    return errs>0?1:0;
}
