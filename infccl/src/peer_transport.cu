#include "peer_transport.h"
#include <cstring>
#include <cstdlib>
namespace infccl {
PeerTransport::PeerTransport() : installed_(false) {
    memset(streams_, 0, sizeof(streams_));
    memset(stream_rr_, 0, sizeof(stream_rr_));
    memset(nstreams_, 0, sizeof(nstreams_));
    memset(staging_, 0, sizeof(staging_));
    memset(staging_bytes_, 0, sizeof(staging_bytes_));
    memset(evpool_, 0, sizeof(evpool_));
    memset(evnext_, 0, sizeof(evnext_));
}
PeerTransport::PeerTransport(const PeerTransportConfig& cfg) : PeerTransport() {
    cfg_ = cfg;
}
PeerTransport::~PeerTransport() {
    worker_.stop();
    if (!installed_) return;
    int saved; cudaGetDevice(&saved);
    for (int g = 0; g < ctx_.ndev(); g++) {
        cudaSetDevice(ctx_.gpu(g).dev_id);
        if (staging_[g]) { cudaFree(staging_[g]); staging_[g] = nullptr; }
        for (int s = 0; s < nstreams_[g]; s++)
            if (streams_[g][s]) { cudaStreamDestroy(streams_[g][s]); streams_[g][s] = nullptr; }
        for (int e = 0; e < INFCCL_EVENT_POOL_SIZE; e++)
            if (evpool_[g][e]) { cudaEventDestroy(evpool_[g][e]); evpool_[g][e] = nullptr; }
    }
    cudaSetDevice(saved);
}
const char* PeerTransport::getName() const { return "peer"; }
int PeerTransport::install(const std::string& local_name,
    std::shared_ptr<TransferMetadata> meta, void** args) {
    local_server_name_ = local_name;
    metadata_ = meta;
    int* devlist = args ? (int*)args[0] : nullptr;
    int ndev = args ? *(int*)args[1] : 0;
    if (ndev < 1 || ndev > INFCCL_MAX_DEVS) return ERR_INVALID_ARG;
    int rc = ctx_.init(ndev, devlist);
    if (rc != OK) return rc;
    if (cfg_.probe_bandwidth) ctx_.probeBandwidth();
    if (cfg_.probe_latency) ctx_.probeLatency();
    if (cfg_.probe_optimal_slice) ctx_.probeOptimalSlice();
    const char* env_slice = getenv("INFCCL_SLICE_BYTES");
    if (env_slice) { size_t v = atol(env_slice); if (v >= 1024 && v <= 64*1024*1024) cfg_.slice_bytes = v; }
    const char* env_ns = getenv("INFCCL_NUM_STREAMS");
    if (env_ns) { int v = atoi(env_ns); if (v >= 1 && v <= 4) cfg_.num_streams_per_gpu = v; }
    const char* env_to = getenv("INFCCL_TIMEOUT_MS");
    if (env_to) { int v = atoi(env_to); if (v >= 100) cfg_.timeout_ms = v; }
    int saved; cudaGetDevice(&saved);
    for (int g = 0; g < ctx_.ndev(); g++) {
        int d = ctx_.gpu(g).dev_id;
        cudaSetDevice(d);
        int ns = cfg_.num_streams_per_gpu;
        if (ns < 1) ns = 1; if (ns > 4) ns = 4;
        nstreams_[g] = ns;
        {
            cudaStream_t burn;
            cudaStreamCreateWithFlags(&burn, cudaStreamNonBlocking);
            cudaStreamDestroy(burn);
        }
        for (int s = 0; s < ns; s++) {
            if (cudaStreamCreateWithFlags(&streams_[g][s], cudaStreamNonBlocking) != cudaSuccess)
                { cudaSetDevice(saved); return ERR_CUDA; }
        }
        stream_rr_[g] = 1;
        for (int e = 0; e < INFCCL_EVENT_POOL_SIZE; e++) {
            if (cudaEventCreateWithFlags(&evpool_[g][e], cudaEventDisableTiming) != cudaSuccess)
                { cudaSetDevice(saved); return ERR_CUDA; }
        }
        evnext_[g] = 1;
        for (int o = 0; o < ctx_.ndev(); o++) {
            if (o == g) continue;
            cudaDeviceEnablePeerAccess(ctx_.gpu(o).dev_id, 0);
            cudaGetLastError();
        }
    }
    cudaSetDevice(saved);
    int dl[INFCCL_MAX_DEVS];
    for (int i = 0; i < ctx_.ndev(); i++) dl[i] = ctx_.gpu(i).dev_id;
    worker_.start(ctx_.ndev(), dl, evpool_, evnext_, streams_, nstreams_, stream_rr_,
                  cfg_.timeout_ms, cfg_.max_retry);
    installed_ = true;
    return OK;
}
size_t PeerTransport::selectSliceSize(int src, int dst, size_t total) const {
    if (total <= cfg_.no_slice_threshold) return total;
    if (cfg_.adaptive_slice) {
        size_t opt = ctx_.pair(src, dst).optimal_slice;
        if (opt > 0 && opt >= 1024) return opt;
    }
    return cfg_.slice_bytes;
}
int PeerTransport::submitTransfer(BatchID batch_id,
    const std::vector<TransferRequest>& entries) {
    auto& bd = toBatch(batch_id);
    if ((int)entries.size() > (int)bd.tasks.size()) return ERR_INVALID_ARG;
    for (int ti = 0; ti < (int)entries.size(); ti++) {
        auto& req = entries[ti];
        auto& task = bd.tasks[ti];
        task.orig = req;
        size_t slice_bytes = selectSliceSize(req.src_gpu, req.dst_gpu, req.length);
        if (req.length <= slice_bytes) {
            task.slices.resize(1);
            Slice& s = task.slices[0];
            s = {};
            s.src_addr = req.source; s.dst_addr = req.dest;
            s.length = req.length; s.opcode = req.opcode;
            s.src_gpu = req.src_gpu; s.dst_gpu = req.dst_gpu;
            s.status = Slice::S_PENDING; s.task = &task;
            task.init(1);
        } else {
            int nslice = (req.length + slice_bytes - 1) / slice_bytes;
            task.slices.resize(nslice);
            for (int si = 0; si < nslice; si++) {
                size_t off = (size_t)si * slice_bytes;
                size_t len = slice_bytes;
                if (off + len > req.length) len = req.length - off;
                Slice& s = task.slices[si];
                s = {};
                s.src_addr = (char*)req.source + off;
                s.dst_addr = (char*)req.dest + off;
                s.length = len; s.opcode = req.opcode;
                s.src_gpu = req.src_gpu; s.dst_gpu = req.dst_gpu;
                s.status = Slice::S_PENDING; s.task = &task;
            }
            task.init(nslice);
        }
        int saved_dev; cudaGetDevice(&saved_dev);
        for (int si = 0; si < task.total; si++) {
            Slice& sl = task.slices[si];
            int g = sl.dst_gpu;
            cudaSetDevice(ctx_.gpu(g).dev_id);
            int slot = stream_rr_[g] % nstreams_[g];
            stream_rr_[g]++;
            int ev = evnext_[g] % INFCCL_EVENT_POOL_SIZE;
            evnext_[g]++;
            cudaStream_t st = streams_[g][slot];
            cudaError_t ce = cudaMemcpyPeerAsync(
                sl.dst_addr, ctx_.gpu(sl.dst_gpu).dev_id,
                sl.src_addr, ctx_.gpu(sl.src_gpu).dev_id,
                sl.length, st);
            if (ce != cudaSuccess) {
                sl.markFailed();
                continue;
            }
            cudaEventRecord(evpool_[g][ev], st);
            sl.markPosted(now_us(), slot, ev);
        }
        cudaSetDevice(saved_dev);
        worker_.submitBatch(task.slices.data(), task.total);
    }
    return OK;
}
int PeerTransport::getTransferStatus(BatchID batch_id, size_t task_id,
    TransferStatus& status) {
    auto& bd = toBatch(batch_id);
    if (task_id >= bd.tasks.size()) return ERR_INVALID_ARG;
    auto& task = bd.tasks[task_id];
    if (task.anyFailed()) status.s = FAILED;
    else if (task.allDone()) status.s = COMPLETED;
    else status.s = PENDING;
    status.transferred_bytes = task.transferred_bytes;
    return OK;
}
int PeerTransport::registerLocalMemory(void* addr, size_t length,
    const std::string& location, bool remote_accessible) {
    (void)addr; (void)length; (void)location; (void)remote_accessible;
    return OK;
}
int PeerTransport::unregisterLocalMemory(void* addr) {
    (void)addr;
    return OK;
}
int PeerTransport::ensureStaging(int gpu, size_t bytes) {
    if (staging_bytes_[gpu] >= bytes) return OK;
    int saved; cudaGetDevice(&saved);
    cudaSetDevice(ctx_.gpu(gpu).dev_id);
    if (staging_[gpu]) cudaFree(staging_[gpu]);
    if (cudaMalloc(&staging_[gpu], bytes) != cudaSuccess)
        { cudaSetDevice(saved); return ERR_CUDA; }
    staging_bytes_[gpu] = bytes;
    cudaSetDevice(saved);
    return OK;
}
int PeerTransport::stageFrom(int dst_gpu, void* src_ptr, int src_gpu,
    size_t offset, size_t bytes) {
    if (ensureStaging(dst_gpu, bytes) != OK) return ERR_CUDA;
    cudaStream_t s = streams_[dst_gpu][0];
    cudaMemcpyPeerAsync(staging_[dst_gpu], ctx_.gpu(dst_gpu).dev_id,
        (char*)src_ptr + offset, ctx_.gpu(src_gpu).dev_id, bytes, s);
    return OK;
}
cudaStream_t PeerTransport::selectStream(int gpu, int idx) {
    if (idx < 0) {
        int slot = stream_rr_[gpu] % nstreams_[gpu];
        stream_rr_[gpu]++;
        return streams_[gpu][slot];
    }
    return streams_[gpu][idx % nstreams_[gpu]];
}
}
