#include "peer_transport.h"
#include "transfer_engine.h"
#include "bi_v100.h"
#include <cstdio>
#include <cuda_runtime.h>
using namespace infccl;
#define CCHK(x) do{cudaError_t e=(x);if(e){fprintf(stderr,"%s @%d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)

int main() {
    int ng; cudaGetDeviceCount(&ng);
    if (ng < 2) { fprintf(stderr, "need 2+ GPUs\n"); return 1; }
    if (ng > 4) ng = 4;
    int devs[4]; for (int i = 0; i < ng; i++) devs[i] = i;
    printf("=== memory pool example — %d GPUs ===\n\n", ng);

    auto meta = std::make_shared<TransferMetadata>();
    TransferEngine engine(meta);
    engine.init("local", ng, devs);
    Transport* xport = engine.installOrGetTransport("peer", nullptr);

    auto seg_desc = std::make_shared<SegmentDesc>();
    seg_desc->name = "gpu_pool";
    seg_desc->protocol = "peer";
    SegmentID seg = meta->addLocalSegment("gpu_pool", seg_desc);
    printf("segment: id=%lu name=%s\n\n", seg, seg_desc->name.c_str());

    size_t pool_size = 64 * 1024 * 1024;
    float* pools[4] = {};
    for (int g = 0; g < ng; g++) {
        CCHK(cudaSetDevice(g));
        CCHK(cudaMalloc(&pools[g], pool_size));
        BufferDesc bd;
        bd.addr = pools[g]; bd.length = pool_size; bd.gpu = g;
        bd.location = "gpu"; bd.access_flags = 0;
        meta->addLocalMemoryBuffer(seg, bd);
        printf("registered: GPU%d addr=%p size=%zuMB\n", g, pools[g], pool_size/(1024*1024));
    }

    printf("\nsegment stats: %d buffers, %zuMB total\n",
        meta->totalBufferCount(),
        meta->totalRegisteredBytes() / (1024*1024));

    for (int g = 0; g < ng; g++) {
        int found = meta->findGpuForAddr(seg, pools[g]);
        printf("lookup GPU%d pool -> gpu=%d %s\n", g, found, found==g ? "OK" : "FAIL");
    }

    printf("\ntransfer via pool: GPU0[0:4MB] -> GPU1[0:4MB]\n");
    size_t xfer = 4 * 1024 * 1024;
    CCHK(cudaSetDevice(0)); CCHK(cudaMemset(pools[0], 0x42, xfer));
    auto bid = xport->allocateBatchID(1);
    xport->submitTransfer(bid, {{Transport::TransferRequest::WRITE,
        pools[0], pools[1], xfer, 0, 1, 0, 0}});
    xport->waitBatch(bid);
    xport->freeBatchID(bid);

    int val = 0;
    CCHK(cudaSetDevice(1));
    CCHK(cudaMemcpy(&val, pools[1], 4, cudaMemcpyDeviceToHost));
    printf("GPU1[0] = 0x%08x %s\n", val, val == 0x42424242 ? "OK" : "FAIL");

    meta->dump();

    for (int g = 0; g < ng; g++) {
        meta->removeLocalMemoryBuffer(seg, pools[g]);
        CCHK(cudaSetDevice(g)); cudaFree(pools[g]);
    }
    meta->removeSegment("gpu_pool");
    printf("\ncleanup done. segments=%d buffers=%d\n",
        meta->segmentCount(), meta->totalBufferCount());
    return 0;
}
