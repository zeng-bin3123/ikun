#include "peer_transport.h"
#include <cstring>
#include <cstdlib>
#include <algorithm>
namespace infccl {

PeerTransport::PeerTransport() : installed_(false) {
    memset(streams_, 0, sizeof(streams_));
    memset(stream_rr_, 0, sizeof(stream_rr_));
    memset(nstreams_, 0, sizeof(nstreams_));
    memset(staging_, 0, sizeof(staging_));
    memset(staging_bytes_, 0, sizeof(staging_bytes_));
    memset(evpool_, 0, sizeof(evpool_));
    memset(evnext_, 0, sizeof(evnext_));
    memset(bw_cache_, 0, sizeof(bw_cache_));
    memset(lat_cache_, 0, sizeof(lat_cache_));
    memset(registered_, 0, sizeof(registered_));
    nregistered_ = 0;
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
        for (int p = 0; p < STAGING_POOLS; p++)
            if (staging_[g][p]) { cudaFree(staging_[g][p]); staging_[g][p] = nullptr; }
        for (int s = 0; s < nstreams_[g]; s++)
            if (streams_[g][s]) { cudaStreamDestroy(streams_[g][s]); streams_[g][s] = nullptr; }
        for (int e = 0; e < EVENT_POOL; e++)
            if (evpool_[g][e]) { cudaEventDestroy(evpool_[g][e]); evpool_[g][e] = nullptr; }
    }
    cudaSetDevice(saved);
}

const char* PeerTransport::getName() const { return "peer"; }

int PeerTransport::loadConfig() {
    const char* e;
    e = getenv("INFCCL_SLICE_BYTES");
    if (e) { size_t v = atol(e); if (v >= 1024 && v <= 64*1024*1024) cfg_.slice_bytes = v; }
    e = getenv("INFCCL_NUM_STREAMS");
    if (e) { int v = atoi(e); if (v >= 1 && v <= MAX_STREAMS) cfg_.num_streams_per_gpu = v; }
    e = getenv("INFCCL_TIMEOUT_MS");
    if (e) { int v = atoi(e); if (v >= 100) cfg_.timeout_ms = v; }
    e = getenv("INFCCL_MAX_RETRY");
    if (e) { int v = atoi(e); if (v >= 0 && v <= 10) cfg_.max_retry = v; }
    e = getenv("INFCCL_ADAPTIVE_SLICE");
    if (e && atoi(e)) cfg_.adaptive_slice = 1;
    e = getenv("INFCCL_NO_SLICE_THRESHOLD");
    if (e) { size_t v = atol(e); if (v > 0) cfg_.no_slice_threshold = v; }
    e = getenv("INFCCL_STAGING_SIZE");
    if (e) { size_t v = atol(e); if (v >= 4096) cfg_.staging_pool_bytes = v; }
    return OK;
}

int PeerTransport::initStreams() {
    int saved; cudaGetDevice(&saved);
    for (int g = 0; g < ctx_.ndev(); g++) {
        int d = ctx_.gpu(g).dev_id;
        cudaSetDevice(d);
        int ns = cfg_.num_streams_per_gpu;
        if (ns < 1) ns = 1;
        if (ns > MAX_STREAMS) ns = MAX_STREAMS;
        nstreams_[g] = ns;
        for (int s = 0; s < ns; s++) {
            if (cudaStreamCreate(&streams_[g][s]) != cudaSuccess)
                { cudaSetDevice(saved); return ERR_CUDA; }
        }
        stream_rr_[g] = 0;
    }
    cudaSetDevice(saved);
    return OK;
}

int PeerTransport::initEvents() {
    int saved; cudaGetDevice(&saved);
    for (int g = 0; g < ctx_.ndev(); g++) {
        cudaSetDevice(ctx_.gpu(g).dev_id);
        for (int e = 0; e < EVENT_POOL; e++) {
            if (cudaEventCreateWithFlags(&evpool_[g][e], cudaEventDisableTiming) != cudaSuccess)
                { cudaSetDevice(saved); return ERR_CUDA; }
        }
        evnext_[g] = 0;
    }
    cudaSetDevice(saved);
    return OK;
}

int PeerTransport::initPeerAccess() {
    int saved; cudaGetDevice(&saved);
    for (int g = 0; g < ctx_.ndev(); g++) {
        cudaSetDevice(ctx_.gpu(g).dev_id);
        for (int o = 0; o < ctx_.ndev(); o++) {
            if (o == g) continue;
            cudaDeviceEnablePeerAccess(ctx_.gpu(o).dev_id, 0);
            cudaGetLastError();
        }
    }
    cudaSetDevice(saved);
    return OK;
}

int PeerTransport::initStaging() {
    if (cfg_.staging_pool_bytes == 0) return OK;
    int saved; cudaGetDevice(&saved);
    for (int g = 0; g < ctx_.ndev(); g++) {
        cudaSetDevice(ctx_.gpu(g).dev_id);
        for (int p = 0; p < STAGING_POOLS; p++) {
            if (cudaMalloc(&staging_[g][p], cfg_.staging_pool_bytes) != cudaSuccess)
                staging_[g][p] = nullptr;
            else
                staging_bytes_[g][p] = cfg_.staging_pool_bytes;
        }
    }
    cudaSetDevice(saved);
    return OK;
}

int PeerTransport::cacheBandwidth() {
    for (int i = 0; i < ctx_.ndev(); i++) {
        for (int j = 0; j < ctx_.ndev(); j++) {
            bw_cache_[i][j] = ctx_.pair(i, j).measured_bw_gbps;
            lat_cache_[i][j] = ctx_.pair(i, j).latency_us;
        }
    }
    return OK;
}

int PeerTransport::install(const std::string& local_name,
    std::shared_ptr<TransferMetadata> meta, void** args) {
    local_server_name_ = local_name;
    metadata_ = meta;
    int* devlist = args ? (int*)args[0] : nullptr;
    int ndev = args ? *(int*)args[1] : 0;
    if (ndev < 1 || ndev > INFCCL_MAX_DEVS) return ERR_INVALID_ARG;

    int rc = ctx_.init(ndev, devlist);
    if (rc != OK) return rc;

    loadConfig();

    if (cfg_.probe_bandwidth) ctx_.probeBandwidth();
    if (cfg_.probe_latency) ctx_.probeLatency();
    if (cfg_.probe_optimal_slice) ctx_.probeOptimalSlice();

    rc = initPeerAccess();
    if (rc != OK) return rc;

    rc = initStreams();
    if (rc != OK) return rc;

    rc = initEvents();
    if (rc != OK) return rc;

    rc = initStaging();
    if (rc != OK) return rc;

    cacheBandwidth();

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

int PeerTransport::findGpuByPtr(void* ptr) {
    for (int i = 0; i < nregistered_; i++) {
        if (ptr >= registered_[i].addr &&
            (char*)ptr < (char*)registered_[i].addr + registered_[i].length)
            return registered_[i].gpu;
    }
    cudaPointerAttributes attr;
    if (cudaPointerGetAttributes(&attr, ptr) == cudaSuccess)
        return attr.device;
    return -1;
}

int PeerTransport::postSlicesAsync(TransferTask& task) {
    int saved_dev; cudaGetDevice(&saved_dev);
    uint8_t src_used = 0;
    for (int si = 0; si < task.total; si++) {
        Slice& sl = task.slices[si];
        int g = sl.src_gpu;
        cudaSetDevice(ctx_.gpu(g).dev_id);
        int slot = stream_rr_[g] % nstreams_[g];
        stream_rr_[g]++;
        cudaStream_t st = streams_[g][slot];
        cudaError_t ce = cudaMemcpyPeerAsync(
            sl.dst_addr, ctx_.gpu(sl.dst_gpu).dev_id,
            sl.src_addr, ctx_.gpu(sl.src_gpu).dev_id,
            sl.length, st);
        if (ce != cudaSuccess) {
            sl.markFailed();
            stats_.slices_failed.fetch_add(1);
            continue;
        }
        sl.markPosted(now_us(), slot, 0);
        src_used |= (1 << g);
        stats_.slices_submitted.fetch_add(1);
        stats_.bytes_submitted.fetch_add(sl.length);
    }
    return src_used;
}

int PeerTransport::syncSrcDevices(uint8_t src_used) {
    for (int g = 0; g < ctx_.ndev(); g++) {
        if (!((src_used >> g) & 1)) continue;
        cudaSetDevice(ctx_.gpu(g).dev_id);
        for (int s = 0; s < nstreams_[g]; s++)
            cudaStreamSynchronize(streams_[g][s]);
    }
    return OK;
}

void PeerTransport::completeSlices(TransferTask& task) {
    for (int si = 0; si < task.total; si++) {
        Slice& sl = task.slices[si];
        if (sl.status == Slice::S_POSTED) {
            sl.markSuccess();
            stats_.slices_completed.fetch_add(1);
            stats_.bytes_completed.fetch_add(sl.length);
        }
    }
}

int PeerTransport::submitTransfer(BatchID batch_id,
    const std::vector<TransferRequest>& entries) {
    auto& bd = toBatch(batch_id);
    if ((int)entries.size() > (int)bd.tasks.size()) return ERR_INVALID_ARG;

    int saved_dev; cudaGetDevice(&saved_dev);

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

        uint8_t src_used = postSlicesAsync(task);
        syncSrcDevices(src_used);
        completeSlices(task);
    }

    cudaSetDevice(saved_dev);
    return OK;
}

int PeerTransport::submitTransferAsync(BatchID batch_id,
    const std::vector<TransferRequest>& entries, uint8_t* out_src_used) {
    auto& bd = toBatch(batch_id);
    if ((int)entries.size() > (int)bd.tasks.size()) return ERR_INVALID_ARG;

    uint8_t all_src = 0;
    int saved_dev; cudaGetDevice(&saved_dev);

    for (int ti = 0; ti < (int)entries.size(); ti++) {
        auto& req = entries[ti];
        auto& task = bd.tasks[ti];
        task.orig = req;

        size_t slice_bytes = selectSliceSize(req.src_gpu, req.dst_gpu, req.length);
        int nslice = (req.length <= slice_bytes) ? 1 :
                     (int)((req.length + slice_bytes - 1) / slice_bytes);
        task.slices.resize(nslice);
        for (int si = 0; si < nslice; si++) {
            size_t off = (size_t)si * slice_bytes;
            size_t len = std::min(slice_bytes, req.length - off);
            Slice& s = task.slices[si];
            s = {};
            s.src_addr = (char*)req.source + off;
            s.dst_addr = (char*)req.dest + off;
            s.length = len; s.opcode = req.opcode;
            s.src_gpu = req.src_gpu; s.dst_gpu = req.dst_gpu;
            s.status = Slice::S_PENDING; s.task = &task;
        }
        task.init(nslice);
        all_src |= postSlicesAsync(task);
    }

    cudaSetDevice(saved_dev);
    if (out_src_used) *out_src_used = all_src;
    return OK;
}

int PeerTransport::syncAndComplete(BatchID batch_id, uint8_t src_used) {
    auto& bd = toBatch(batch_id);
    int saved_dev; cudaGetDevice(&saved_dev);
    syncSrcDevices(src_used);
    for (auto& task : bd.tasks)
        completeSlices(task);
    cudaSetDevice(saved_dev);
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
    if (nregistered_ >= MAX_REGISTERED) return ERR_MEMORY;
    int gpu = -1;
    for (int g = 0; g < ctx_.ndev(); g++) {
        cudaPointerAttributes attr;
        if (cudaPointerGetAttributes(&attr, addr) == cudaSuccess &&
            attr.device == ctx_.gpu(g).dev_id) {
            gpu = g; break;
        }
    }
    registered_[nregistered_] = {addr, length, gpu, location, remote_accessible};
    nregistered_++;
    return OK;
}

int PeerTransport::unregisterLocalMemory(void* addr) {
    for (int i = 0; i < nregistered_; i++) {
        if (registered_[i].addr == addr) {
            for (int j = i; j < nregistered_ - 1; j++)
                registered_[j] = registered_[j + 1];
            nregistered_--;
            return OK;
        }
    }
    return ERR_NOT_FOUND;
}

int PeerTransport::ensureStaging(int gpu, size_t bytes, int pool) {
    if (pool < 0 || pool >= STAGING_POOLS) return ERR_INVALID_ARG;
    if (staging_bytes_[gpu][pool] >= bytes) return OK;
    int saved; cudaGetDevice(&saved);
    cudaSetDevice(ctx_.gpu(gpu).dev_id);
    if (staging_[gpu][pool]) cudaFree(staging_[gpu][pool]);
    if (cudaMalloc(&staging_[gpu][pool], bytes) != cudaSuccess)
        { cudaSetDevice(saved); return ERR_CUDA; }
    staging_bytes_[gpu][pool] = bytes;
    cudaSetDevice(saved);
    return OK;
}

int PeerTransport::stageFrom(int dst_gpu, void* src_ptr, int src_gpu,
    size_t offset, size_t bytes, int pool) {
    if (ensureStaging(dst_gpu, bytes, pool) != OK) return ERR_CUDA;
    cudaSetDevice(ctx_.gpu(src_gpu).dev_id);
    cudaStream_t s = streams_[src_gpu][0];
    cudaMemcpyPeerAsync(staging_[dst_gpu][pool], ctx_.gpu(dst_gpu).dev_id,
        (char*)src_ptr + offset, ctx_.gpu(src_gpu).dev_id, bytes, s);
    return OK;
}

int PeerTransport::stageTo(int src_gpu, void* dst_ptr, int dst_gpu,
    size_t offset, size_t bytes, int pool) {
    if (!staging_[src_gpu][pool]) return ERR_MEMORY;
    cudaSetDevice(ctx_.gpu(src_gpu).dev_id);
    cudaStream_t s = streams_[src_gpu][0];
    cudaMemcpyPeerAsync((char*)dst_ptr + offset, ctx_.gpu(dst_gpu).dev_id,
        staging_[src_gpu][pool], ctx_.gpu(src_gpu).dev_id, bytes, s);
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

void PeerTransport::printStats(FILE* out) const {
    fprintf(out, "PeerTransport stats:\n");
    fprintf(out, "  slices submitted: %lu\n", stats_.slices_submitted.load());
    fprintf(out, "  slices completed: %lu\n", stats_.slices_completed.load());
    fprintf(out, "  slices failed:    %lu\n", stats_.slices_failed.load());
    fprintf(out, "  bytes submitted:  %lu\n", stats_.bytes_submitted.load());
    fprintf(out, "  bytes completed:  %lu\n", stats_.bytes_completed.load());
}

float PeerTransport::measuredBandwidth(int src, int dst) const {
    if (src < 0 || src >= ctx_.ndev() || dst < 0 || dst >= ctx_.ndev()) return 0;
    return bw_cache_[src][dst];
}

float PeerTransport::measuredLatency(int src, int dst) const {
    if (src < 0 || src >= ctx_.ndev() || dst < 0 || dst >= ctx_.ndev()) return 0;
    return lat_cache_[src][dst];
}

}
