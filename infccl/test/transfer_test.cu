#include "transfer_engine.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
using namespace infccl;
#define CCHK(x) do{cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
static void fill(float* h, int n, float seed) { for(int i=0;i<n;i++) h[i]=seed+(float)i*0.001f; }
static int verify(float* h, int n, float seed) {
    int errs=0;
    for(int i=0;i<n;i++) if(fabsf(h[i]-(seed+(float)i*0.001f))>0.01f) errs++;
    return errs;
}
int main() {
    int ng; cudaGetDeviceCount(&ng);
    if(ng<2){fprintf(stderr,"need 2+ GPUs\n");return 1;}
    if(ng>4)ng=4;
    int devs[4]; for(int i=0;i<ng;i++)devs[i]=i;
    printf("=== transfer_test — %d GPUs ===\n\n",ng);
    auto meta=std::make_shared<TransferMetadata>();
    TransferEngine engine(meta);
    engine.init("local",ng,devs);
    Transport* xport=engine.installOrGetTransport("peer",nullptr);
    if(!xport){fprintf(stderr,"install failed\n");return 1;}
    PeerTransport* peer=(PeerTransport*)xport;
    peer->context().printTopology();
    printf("\n");

    auto run=[&](const char* label, int N, int src, int dst) -> int {
        size_t bytes=N*sizeof(float);
        float* h=(float*)malloc(bytes); fill(h,N,(float)(src*10+dst));
        CCHK(cudaSetDevice(src));
        float* ds; CCHK(cudaMalloc(&ds,bytes)); CCHK(cudaMemcpy(ds,h,bytes,cudaMemcpyHostToDevice));
        CCHK(cudaSetDevice(dst));
        float* dd; CCHK(cudaMalloc(&dd,bytes)); CCHK(cudaMemset(dd,0,bytes));
        auto bid=xport->allocateBatchID(1);
        std::vector<Transport::TransferRequest> reqs={
            {Transport::TransferRequest::WRITE, ds, dd, bytes, src, dst, 0, 0}};
        xport->submitTransfer(bid,reqs);
        int rc=xport->waitBatch(bid);
        float* h2=(float*)malloc(bytes);
        CCHK(cudaSetDevice(dst));
        CCHK(cudaMemcpy(h2,dd,bytes,cudaMemcpyDeviceToHost));
        int errs=verify(h2,N,(float)(src*10+dst));
        int nslice=xport->getBatch(bid)->tasks[0].total;
        printf("[%s] %d->%d %zuB slices=%d errs=%d %s\n",
            label,src,dst,bytes,nslice,errs,(rc==OK&&errs==0)?"PASS":"FAIL");
        xport->freeBatchID(bid);
        CCHK(cudaSetDevice(src));cudaFree(ds);
        CCHK(cudaSetDevice(dst));cudaFree(dd);
        free(h);free(h2);
        return (rc==OK&&errs==0)?0:1;
    };

    int fails=0;
    fails+=run("4KB-noslice",1024,0,1);
    fails+=run("256KB-sliced",65536,0,1);
    fails+=run("16MB-sliced",4*1024*1024,0,1);
    if(ng>=3) fails+=run("16MB-0to2",4*1024*1024,0,2);
    if(ng>=4) fails+=run("16MB-0to3",4*1024*1024,0,3);
    if(ng>=4) fails+=run("16MB-3to0",4*1024*1024,3,0);

    printf("\n[all-pairs] %d transfers\n",ng*(ng-1));
    {
        int N=1024*1024; size_t bytes=N*sizeof(float);
        float* bufs_d[4];
        for(int g=0;g<ng;g++){
            CCHK(cudaSetDevice(g));CCHK(cudaMalloc(&bufs_d[g],bytes));
            float*h=(float*)malloc(bytes);fill(h,N,100.0f+g);
            CCHK(cudaMemcpy(bufs_d[g],h,bytes,cudaMemcpyHostToDevice));free(h);
        }
        float* recv_d[4][4]={};
        for(int s=0;s<ng;s++)for(int d=0;d<ng;d++){
            if(s==d)continue;
            CCHK(cudaSetDevice(d));CCHK(cudaMalloc(&recv_d[s][d],bytes));CCHK(cudaMemset(recv_d[s][d],0,bytes));
        }
        int nt=ng*(ng-1);
        auto bid=xport->allocateBatchID(nt);
        std::vector<Transport::TransferRequest> reqs;
        for(int s=0;s<ng;s++)for(int d=0;d<ng;d++){
            if(s==d)continue;
            reqs.push_back({Transport::TransferRequest::WRITE,bufs_d[s],recv_d[s][d],bytes,s,d,0,0});
        }
        xport->submitTransfer(bid,reqs);
        int rc=xport->waitBatch(bid);
        int total_errs=0;
        int ti=0;
        for(int s=0;s<ng;s++)for(int d=0;d<ng;d++){
            if(s==d)continue;
            float*h2=(float*)malloc(bytes);CCHK(cudaSetDevice(d));
            CCHK(cudaMemcpy(h2,recv_d[s][d],bytes,cudaMemcpyDeviceToHost));
            int e=verify(h2,N,100.0f+s);total_errs+=e;free(h2);ti++;
        }
        printf("  %d transfers errs=%d %s\n",nt,total_errs,(rc==OK&&total_errs==0)?"PASS":"FAIL");
        if(rc!=OK||total_errs>0) fails++;
        xport->freeBatchID(bid);
        for(int g=0;g<ng;g++){CCHK(cudaSetDevice(g));cudaFree(bufs_d[g]);
            for(int o=0;o<ng;o++)if(recv_d[g][o])cudaFree(recv_d[g][o]);}
    }

    printf("\n[concurrent] 20 batches x 16MB\n");
    {
        int N=4*1024*1024;size_t bytes=N*sizeof(float);int NB=20;
        CCHK(cudaSetDevice(0));
        float*sd;CCHK(cudaMalloc(&sd,bytes));float*h=(float*)malloc(bytes);fill(h,N,7.0f);
        CCHK(cudaMemcpy(sd,h,bytes,cudaMemcpyHostToDevice));
        CCHK(cudaSetDevice(1));
        float*dd[20];for(int i=0;i<NB;i++){CCHK(cudaMalloc(&dd[i],bytes));CCHK(cudaMemset(dd[i],0,bytes));}
        Transport::BatchID bids[20];
        for(int i=0;i<NB;i++){
            bids[i]=xport->allocateBatchID(1);
            xport->submitTransfer(bids[i],{{Transport::TransferRequest::WRITE,sd,dd[i],bytes,0,1,0,0}});
        }
        int ok=0;
        for(int i=0;i<NB;i++){
            int rc=xport->waitBatch(bids[i]);
            float*h2=(float*)malloc(bytes);CCHK(cudaSetDevice(1));
            CCHK(cudaMemcpy(h2,dd[i],bytes,cudaMemcpyDeviceToHost));
            if(rc==OK&&verify(h2,N,7.0f)==0)ok++;
            free(h2);xport->freeBatchID(bids[i]);
        }
        printf("  %d/%d correct %s\n",ok,NB,ok==NB?"PASS":"FAIL");
        if(ok!=NB)fails++;
        CCHK(cudaSetDevice(0));cudaFree(sd);
        CCHK(cudaSetDevice(1));for(int i=0;i<NB;i++)cudaFree(dd[i]);
        free(h);
    }

    printf("\n[bandwidth] 16MB 0->1\n");
    {
        size_t bytes=16*1024*1024;
        CCHK(cudaSetDevice(0));float*d0;CCHK(cudaMalloc(&d0,bytes));CCHK(cudaMemset(d0,1,bytes));
        CCHK(cudaSetDevice(1));float*d1;CCHK(cudaMalloc(&d1,bytes));
        CCHK(cudaSetDevice(0));
        cudaEvent_t t0,t1;cudaEventCreate(&t0);cudaEventCreate(&t1);
        cudaStream_t rs;cudaStreamCreate(&rs);
        cudaMemcpyPeerAsync(d1,1,d0,0,bytes,rs);cudaStreamSynchronize(rs);
        cudaEventRecord(t0,rs);
        for(int i=0;i<50;i++)cudaMemcpyPeerAsync(d1,1,d0,0,bytes,rs);
        cudaEventRecord(t1,rs);cudaEventSynchronize(t1);
        float raw_ms;cudaEventElapsedTime(&raw_ms,t0,t1);
        float raw_bw=bytes*50.0*1e-6/raw_ms;
        int64_t es=now_us();
        for(int i=0;i<50;i++){
            auto bid=xport->allocateBatchID(1);
            xport->submitTransfer(bid,{{Transport::TransferRequest::WRITE,d0,d1,bytes,0,1,0,0}});
            xport->waitBatch(bid);
            xport->freeBatchID(bid);
        }
        float eng_bw=bytes*50.0*1e-3/(float)(now_us()-es);
        printf("  raw:    %.1f GB/s\n  engine: %.1f GB/s\n  overhead: %.0f%%\n",
            raw_bw,eng_bw,100.0*(1.0-eng_bw/raw_bw));
        cudaStreamDestroy(rs);cudaEventDestroy(t0);cudaEventDestroy(t1);
        CCHK(cudaSetDevice(0));cudaFree(d0);CCHK(cudaSetDevice(1));cudaFree(d1);
    }

    auto& ws=peer->worker().stats();
    printf("\nworker stats: posted=%lu completed=%lu failed=%lu retried=%lu bytes=%lu polls=%lu\n",
        ws.slices_submitted.load(),ws.slices_completed.load(),ws.slices_failed.load(),
        ws.slices_retried.load(),ws.bytes_transferred.load(),ws.poll_rounds.load());

    printf("\n=== %s ===\n", fails==0?"ALL PASS":"FAILURES DETECTED");
    return fails;
}
