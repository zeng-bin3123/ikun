#include "transfer_engine_c.h"
#include "transfer_engine.h"

using namespace infccl;

infccl_engine_t infccl_create_engine(int ndev, const int* devlist) {
    auto meta = std::make_shared<TransferMetadata>();
    auto* engine = new TransferEngine(meta);
    engine->init("local", ndev, devlist);
    return (infccl_engine_t)engine;
}

int infccl_init_engine(infccl_engine_t engine, const char* server_name) {
    (void)server_name;
    return OK;
}

infccl_transport_t infccl_install_transport(infccl_engine_t engine, const char* proto) {
    return (infccl_transport_t)((TransferEngine*)engine)->installOrGetTransport(proto, nullptr);
}

void infccl_destroy_engine(infccl_engine_t engine) {
    delete (TransferEngine*)engine;
}

infccl_segment_id_t infccl_open_segment(infccl_engine_t engine, const char* name) {
    return ((TransferEngine*)engine)->openSegment(name);
}

int infccl_register_memory(infccl_engine_t engine, void* addr, size_t length, const char* location) {
    return ((TransferEngine*)engine)->registerLocalMemory(addr, length, location ? location : "gpu", true);
}

int infccl_unregister_memory(infccl_engine_t engine, void* addr) {
    return ((TransferEngine*)engine)->unregisterLocalMemory(addr);
}

infccl_batch_id_t infccl_alloc_batch(infccl_transport_t xport, size_t batch_size) {
    return ((Transport*)xport)->allocateBatchID(batch_size);
}

int infccl_submit_transfer(infccl_transport_t xport, infccl_batch_id_t batch_id,
    struct infccl_transfer_request* entries, size_t count) {
    std::vector<Transport::TransferRequest> reqs(count);
    for (size_t i = 0; i < count; i++) {
        reqs[i].opcode = (Transport::TransferRequest::OpCode)entries[i].opcode;
        reqs[i].source = entries[i].source;
        reqs[i].dest = nullptr;
        reqs[i].target_id = entries[i].target_id;
        reqs[i].target_offset = entries[i].target_offset;
        reqs[i].length = entries[i].length;
        reqs[i].src_gpu = 0;
        reqs[i].dst_gpu = 0;
    }
    return ((Transport*)xport)->submitTransfer(batch_id, reqs);
}

int infccl_submit_peer_transfer(infccl_transport_t xport, infccl_batch_id_t batch_id,
    void* src, int src_gpu, void* dst, int dst_gpu, size_t length) {
    std::vector<Transport::TransferRequest> reqs(1);
    reqs[0].opcode = Transport::TransferRequest::WRITE;
    reqs[0].source = src;
    reqs[0].dest = dst;
    reqs[0].length = length;
    reqs[0].src_gpu = src_gpu;
    reqs[0].dst_gpu = dst_gpu;
    reqs[0].target_id = 0;
    reqs[0].target_offset = 0;
    return ((Transport*)xport)->submitTransfer(batch_id, reqs);
}

int infccl_wait_batch(infccl_transport_t xport, infccl_batch_id_t batch_id, int timeout_ms) {
    return ((Transport*)xport)->waitBatch(batch_id, timeout_ms);
}

int infccl_get_transfer_status(infccl_transport_t xport, infccl_batch_id_t batch_id,
    size_t task_id, struct infccl_transfer_status* status) {
    Transport::TransferStatus native;
    int rc = ((Transport*)xport)->getTransferStatus(batch_id, task_id, native);
    status->status = (int)native.s;
    status->transferred_bytes = native.transferred_bytes;
    return rc;
}

int infccl_free_batch(infccl_transport_t xport, infccl_batch_id_t batch_id) {
    return ((Transport*)xport)->freeBatchID(batch_id);
}

int infccl_probe_caps(int gpu, struct infccl_bi_v100_caps* out) {
    BiV100Caps caps;
    int rc = probeBiV100Caps(&caps, gpu);
    if (rc != 0) return rc;
    out->warp_size = caps.warp_size;
    out->sm_count = caps.sm_count;
    out->fp64_works = caps.fp64_works;
    out->hbm_bw_gbps = caps.hbm_bw_gbps;
    out->peer_bw_gbps = caps.peer_bw_gbps;
    out->peer_latency_us = caps.peer_latency_us;
    out->peer_memcpy_needs_src_device = caps.peer_memcpy_needs_src_device;
    out->peer_memcpy_needs_blocking_stream = caps.peer_memcpy_needs_blocking_stream;
    return 0;
}

void infccl_print_caps(int gpu) {
    BiV100Caps caps;
    probeBiV100Caps(&caps, gpu);
    printBiV100Caps(&caps);
}
