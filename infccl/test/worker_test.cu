#include "peer_transport.h"
#include "transfer_engine.h"
#include <cstdio>
#include <cmath>
#include <atomic>
#include <cuda_runtime.h>
using namespace infccl;
#define CCHK(x) do{cudaError_t e=(x);if(e){fprintf(stderr,"CUDA %s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
static void fill(float* h,int n,float s){for(int i=0;i<n;i++)h[i]=s+(float)i*0.001f;}
static int verify(float* h,int n,float s){int e=0;for(int i=0;i<n;i++)if(fabsf(h[i]-(s+(float)i*0.001f))>0.01f)e++;return e;}

int main(){
    int ng;cudaGetDeviceCount(&ng);
    if(ng<2){fprintf(stderr,"need 2+ GPUs\n");return 1;}
    if(ng>4)ng=4;
    int devs[4];for(int i=0;i<ng;i++)devs[i]=i;
    printf("=== worker_test — %d GPUs ===\n\n",ng);
    auto meta=std::make_shared<TransferMetadata>();
    TransferEngine engine(meta);
    engine.init("local",ng,devs);
    Transport* xport=engine.installOrGetTransport("peer",nullptr);
    PeerTransport* peer=(PeerTransport*)xport;
    int fails=0;

    printf("[1] completion callback\n");
    {
        int N=1024*1024;size_t bytes=N*sizeof(float);
        float*h=(float*)malloc(bytes);fill(h,N,11.0f);
        CCHK(cudaSetDevice(0));float*d0;CCHK(cudaMalloc(&d0,bytes));
        CCHK(cudaMemcpy(d0,h,bytes,cudaMemcpyHostToDevice));
        CCHK(cudaSetDevice(1));float*d1;CCHK(cudaMalloc(&d1,bytes));
        CCHK(cudaMemset(d1,0,bytes));
        auto bid=xport->allocateBatchID(1);
        std::atomic<int> cb_called{0};
        std::atomic<int> cb_success{0};
        std::vector<Transport::TransferRequest> reqs={
            {Transport::TransferRequest::WRITE,d0,d1,bytes,0,1,0,0}};
        auto& bd=Transport::toBatch(bid);
        bd.tasks[0].orig=reqs[0];
        size_t slice_bytes=peer->config().slice_bytes;
        int nslice=(bytes<=slice_bytes)?1:(int)((bytes+slice_bytes-1)/slice_bytes);
        bd.tasks[0].slices.resize(nslice);
        for(int si=0;si<nslice;si++){
            size_t off=(size_t)si*slice_bytes;
            size_t len=slice_bytes;if(off+len>bytes)len=bytes-off;
            auto& s=bd.tasks[0].slices[si];
            s={};s.src_addr=(char*)d0+off;s.dst_addr=(char*)d1+off;
            s.length=len;s.opcode=Transport::TransferRequest::WRITE;
            s.src_gpu=0;s.dst_gpu=1;s.status=Transport::Slice::S_PENDING;
            s.task=&bd.tasks[0];
        }
        bd.tasks[0].init(nslice);
        int saved;cudaGetDevice(&saved);
        for(int si=0;si<nslice;si++){
            auto& sl=bd.tasks[0].slices[si];
            cudaSetDevice(peer->devId(sl.src_gpu));
            cudaStream_t st=peer->selectStream(sl.src_gpu);
            cudaMemcpyPeerAsync(sl.dst_addr,peer->devId(sl.dst_gpu),
                sl.src_addr,peer->devId(sl.src_gpu),sl.length,st);
            int ev=si%256;
            cudaEventRecord(peer->context().eventPool(sl.src_gpu,ev),st);
            sl.markPosted(now_us(),0,ev);
        }
        cudaSetDevice(saved);
        peer->worker().submitForPollingWithCallback(
            bd.tasks[0].slices.data(),nslice,bid,0,
            [&](Transport::BatchID,int,bool ok){
                cb_called.fetch_add(1);
                if(ok)cb_success.fetch_add(1);
            });
        peer->worker().drain();
        float*h2=(float*)malloc(bytes);
        CCHK(cudaSetDevice(1));CCHK(cudaMemcpy(h2,d1,bytes,cudaMemcpyDeviceToHost));
        int errs=verify(h2,N,11.0f);
        printf("  cb_called=%d cb_success=%d errs=%d %s\n",
            cb_called.load(),cb_success.load(),errs,
            (errs==0&&cb_called.load()==nslice&&cb_success.load()==nslice)?"PASS":"FAIL");
        if(errs>0||cb_called.load()!=nslice)fails++;
        xport->freeBatchID(bid);
        CCHK(cudaSetDevice(0));cudaFree(d0);CCHK(cudaSetDevice(1));cudaFree(d1);
        free(h);free(h2);
    }

    printf("[2] latency tracking\n");
    {
        auto& ws=peer->worker().stats();
        uint64_t prev_completed=ws.slices_completed.load();
        int N=256*1024;size_t bytes=N*sizeof(float);
        CCHK(cudaSetDevice(0));float*d0;CCHK(cudaMalloc(&d0,bytes));CCHK(cudaMemset(d0,1,bytes));
        CCHK(cudaSetDevice(1));float*d1;CCHK(cudaMalloc(&d1,bytes));
        uint8_t src_used=0;
        auto bid=xport->allocateBatchID(1);
        peer->submitTransferAsync(bid,
            {{Transport::TransferRequest::WRITE,d0,d1,bytes,0,1,0,0}},&src_used);
        xport->waitBatch(bid);
        uint64_t new_completed=ws.slices_completed.load();
        int64_t max_lat=ws.max_completion_latency_us.load();
        int64_t total_lat=ws.total_completion_latency_us.load();
        uint64_t delta=new_completed-prev_completed;
        float avg_lat=delta>0?(float)total_lat/(float)new_completed:0;
        printf("  completed=%lu max_lat=%ldus avg_lat=%.1fus %s\n",
            delta,max_lat,avg_lat,
            (delta>0&&max_lat>0)?"PASS":"FAIL");
        if(delta==0||max_lat<=0)fails++;
        xport->freeBatchID(bid);
        CCHK(cudaSetDevice(0));cudaFree(d0);CCHK(cudaSetDevice(1));cudaFree(d1);
    }

    printf("[3] drain with timeout\n");
    {
        int N=1024*1024;size_t bytes=N*sizeof(float);
        CCHK(cudaSetDevice(0));float*d0;CCHK(cudaMalloc(&d0,bytes));CCHK(cudaMemset(d0,1,bytes));
        CCHK(cudaSetDevice(1));float*d1;CCHK(cudaMalloc(&d1,bytes));
        uint8_t src_used=0;
        auto bid=xport->allocateBatchID(1);
        peer->submitTransferAsync(bid,
            {{Transport::TransferRequest::WRITE,d0,d1,bytes,0,1,0,0}},&src_used);
        bool drained=peer->worker().drainWithTimeout(5000000);
        printf("  drained=%d inflight=%d %s\n",
            drained,peer->worker().inflightCount(),
            (drained&&peer->worker().inflightCount()==0)?"PASS":"FAIL");
        if(!drained)fails++;
        xport->freeBatchID(bid);
        CCHK(cudaSetDevice(0));cudaFree(d0);CCHK(cudaSetDevice(1));cudaFree(d1);
    }

    printf("[4] WorkerConfig from env\n");
    {
        setenv("INFCCL_WORKER_TIMEOUT_US","2000000",1);
        setenv("INFCCL_WORKER_MAX_RETRY","7",1);
        setenv("INFCCL_WORKER_DRAIN_SIZE","128",1);
        setenv("INFCCL_WORKER_SPIN","50",1);
        WorkerConfig c=WorkerConfig::fromEnv();
        int ok=(c.timeout_us==2000000&&c.max_retry==7&&c.batch_drain_size==128&&c.spin_count==50);
        printf("  timeout=%ld retry=%d drain=%d spin=%d %s\n",
            c.timeout_us,c.max_retry,c.batch_drain_size,c.spin_count,
            ok?"PASS":"FAIL");
        if(!ok)fails++;
        unsetenv("INFCCL_WORKER_TIMEOUT_US");
        unsetenv("INFCCL_WORKER_MAX_RETRY");
        unsetenv("INFCCL_WORKER_DRAIN_SIZE");
        unsetenv("INFCCL_WORKER_SPIN");
    }

    printf("[5] WorkerConfig presets\n");
    {
        WorkerConfig f=WorkerConfig::fast();
        WorkerConfig c=WorkerConfig::conservative();
        int ok=(f.timeout_us<c.timeout_us&&f.batch_drain_size>c.batch_drain_size&&f.spin_count<c.spin_count);
        printf("  fast: timeout=%ld drain=%d spin=%d\n",f.timeout_us,f.batch_drain_size,f.spin_count);
        printf("  conservative: timeout=%ld drain=%d spin=%d\n",c.timeout_us,c.batch_drain_size,c.spin_count);
        printf("  %s\n",ok?"PASS":"FAIL");
        if(!ok)fails++;
    }

    printf("[6] inflight count tracking\n");
    {
        int N=4*1024*1024;size_t bytes=N*sizeof(float);
        CCHK(cudaSetDevice(0));float*d0;CCHK(cudaMalloc(&d0,bytes));CCHK(cudaMemset(d0,1,bytes));
        CCHK(cudaSetDevice(1));float*d1;CCHK(cudaMalloc(&d1,bytes));
        uint8_t src_used=0;
        auto bid=xport->allocateBatchID(1);
        int pre=peer->worker().inflightCount();
        peer->submitTransferAsync(bid,
            {{Transport::TransferRequest::WRITE,d0,d1,bytes,0,1,0,0}},&src_used);
        int mid=peer->worker().hasInflight()?1:0;
        xport->waitBatch(bid);
        int post=peer->worker().inflightCount();
        printf("  pre=%d during=%d post=%d %s\n",pre,mid,post,
            (pre==0&&post==0)?"PASS":"FAIL");
        if(pre!=0||post!=0)fails++;
        xport->freeBatchID(bid);
        CCHK(cudaSetDevice(0));cudaFree(d0);CCHK(cudaSetDevice(1));cudaFree(d1);
    }

    printf("[7] multi-batch concurrent with callbacks\n");
    {
        int NB=10;int N=1024*1024;size_t bytes=N*sizeof(float);
        CCHK(cudaSetDevice(0));float*d0;CCHK(cudaMalloc(&d0,bytes));CCHK(cudaMemset(d0,1,bytes));
        CCHK(cudaSetDevice(1));float*dd[10];
        for(int i=0;i<NB;i++){CCHK(cudaMalloc(&dd[i],bytes));}
        std::atomic<int> total_cb{0};
        for(int i=0;i<NB;i++){
            auto bid=xport->allocateBatchID(1);
            auto& bd=Transport::toBatch(bid);
            bd.tasks[0].orig={Transport::TransferRequest::WRITE,d0,dd[i],bytes,0,1,0,0};
            bd.tasks[0].slices.resize(1);
            auto& s=bd.tasks[0].slices[0];
            s={};s.src_addr=d0;s.dst_addr=dd[i];s.length=bytes;
            s.opcode=Transport::TransferRequest::WRITE;
            s.src_gpu=0;s.dst_gpu=1;s.status=Transport::Slice::S_PENDING;
            s.task=&bd.tasks[0];
            bd.tasks[0].init(1);
            int saved;cudaGetDevice(&saved);
            cudaSetDevice(peer->devId(0));
            cudaStream_t st=peer->selectStream(0);
            cudaMemcpyPeerAsync(dd[i],peer->devId(1),d0,peer->devId(0),bytes,st);
            int ev=i%256;
            cudaEventRecord(peer->context().eventPool(0,ev),st);
            s.markPosted(now_us(),0,ev);
            cudaSetDevice(saved);
            peer->worker().submitForPollingWithCallback(
                bd.tasks[0].slices.data(),1,bid,0,
                [&](Transport::BatchID,int,bool){total_cb.fetch_add(1);});
        }
        peer->worker().drain();
        printf("  batches=%d callbacks=%d %s\n",NB,total_cb.load(),
            total_cb.load()==NB?"PASS":"FAIL");
        if(total_cb.load()!=NB)fails++;
        CCHK(cudaSetDevice(0));cudaFree(d0);
        CCHK(cudaSetDevice(1));for(int i=0;i<NB;i++)cudaFree(dd[i]);
    }

    printf("[8] printStats\n");
    peer->worker().printStats();

    printf("\n=== %s ===\n",fails==0?"ALL PASS":"FAILURES DETECTED");
    return fails;
}
