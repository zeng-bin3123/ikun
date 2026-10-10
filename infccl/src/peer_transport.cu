#include "peer_transport.h"
#include <cstring>
#include <cstdlib>

namespace infccl {

PeerTransport::PeerTransport() : installed_(false) {
    memset(streams_, 0, sizeof(streams_));
    memset(stream_rr_, 0, sizeof(stream_rr_));
    memset(staging_, 0, sizeof(staging_));
    memset(staging2_, 0, sizeof(staging2_));
    memset(staging_bytes_, 0, sizeof(staging_bytes_));
    memset(staging_pool_, 0, sizeof(staging_pool_));
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
        if (staging2_[g]) { cudaFree(staging2_[g]); staging2_[g] = nullptr; }
        for (int p = 0; p < INFCCL_MAX_STAGING_POOLS; p++) {
            if (staging_pool_[g][p].ptr) {
                cudaFree(staging_pool_[g][p].ptr);
                staging_pool_[g][p].ptr = nullptr;
            }
        }
        for (int s = 0; s < 4; s++)
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
    if (ndev < 1 || ndev > 8) return ERR_INVALID_ARG;

    int rc = ctx_.init(ndev, devlist);
    if (rc != OK) return rc;

    if (cfg_.probe_bandwidth) ctx_.probeBandwidth();

    const char* env_slice = getenv("INFCCL_SLICE_BYTES");
    if (env_slice) {
        size_t val = atol(env_slice);
        if (val >= 1024 && val <= 16 * 1024 * 1024) cfg_.slice_bytes = val;
    }

    const char* env_worker = getenv("INFCCL_WORKER_THREAD");
    if (env_worker && atoi(env_worker)) cfg_.enable_worker_thread = 1;

    const char* env_nstreams = getenv("INFCCL_NUM_STREAMS");
    if (env_nstreams) {
        int val = atoi(env_nstreams);
        if (val >= 1 && val <= 4) cfg_.num_streams_per_gpu = val;
    }

    const char* env_adaptive = getenv("INFCCL_ADAPTIVE_SLICE");
    if (env_adaptive && atoi(env_adaptive)) cfg_.adaptive_slice = 1;

    const char* env_no_slice = getenv("INFCCL_NO_SLICE_THRESHOLD");
    if (env_no_slice) {
        size_t val = atol(env_no_slice);
        if (val > 0) cfg_.no_slice_threshold = val;
    }

    int saved; cudaGetDevice(&saved);

    for (int g = 0; g < ctx_.ndev(); g++) {
        int d = ctx_.gpu(g).dev_id;
        cudaSetDevice(d);

        int nstreams = cfg_.num_streams_per_gpu;
        if (nstreams < 1) nstreams = 1;
        if (nstreams > 4) nstreams = 4;
        for (int s = 0; s < nstreams; s++) {
            if (cudaStreamCreateWithFlags(&streams_[g][s], cudaStreamNonBlocking) != cudaSuccess)
                { cudaSetDevice(saved); return ERR_CUDA; }
        }
        stream_rr_[g] = 0;

        for (int e = 0; e < INFCCL_EVENT_POOL_SIZE; e++) {
            if (cudaEventCreateWithFlags(&evpool_[g][e], cudaEventDisableTiming) != cudaSuccess)
                { cudaSetDevice(saved); return ERR_CUDA; }
        }
        evnext_[g] = 0;

        for (int o = 0; o < ctx_.ndev(); o++) {
            if (o == g) continue;
            if (ctx_.pair(g, o).can_p2p) {
                cudaDeviceEnablePeerAccess(ctx_.gpu(o).dev_id, 0);
                cudaGetLastError();
            }
        }

        if (cfg_.staging_pool_size > 0) {
            for (int p = 0; p < INFCCL_MAX_STAGING_POOLS; p++) {
                if (cudaMalloc(&staging_pool_[g][p].ptr, cfg_.staging_pool_size) != cudaSuccess)
                    staging_pool_[g][p].ptr = nullptr;
                else {
                    staging_pool_[g][p].bytes = cfg_.staging_pool_size;
                    staging_pool_[g][p].in_use = 0;
                }
            }
        }
    }

    cudaSetDevice(saved);
    installed_ = true;

    if (cfg_.enable_worker_thread) {
        int devids[8];
        for (int g = 0; g < ctx_.ndev(); g++) devids[g] = ctx_.gpu(g).dev_id;
        worker_.start(ctx_.ndev(), devids);
    }

    return OK;
}

int PeerTransport::registerLocalMemory(void* addr, size_t length,
    const std::string& location, bool remote_accessible) {
    (void)remote_accessible;
    if (!metadata_) return ERR_INVALID_ARG;

    int gpu = -1;
    if (location.size() > 3 && location.substr(0, 3) == "gpu")
        gpu = atoi(location.c_str() + 3);
    if (gpu < 0 || gpu >= ctx_.ndev()) {
        int cur; cudaGetDevice(&cur);
        for (int g = 0; g < ctx_.ndev(); g++)
            if (ctx_.gpu(g).dev_id == cur) { gpu = g; break; }
    }

    BufferDesc buf;
    buf.addr = addr;
    buf.length = length;
    buf.gpu = gpu;
    buf.location = location;

    SegmentID local_id = metadata_->getSegmentID(local_server_name_);
    if (!local_id) {
        auto desc = std::make_shared<SegmentDesc>();
        desc->name = local_server_name_;
        desc->protocol = "peer";
        local_id = metadata_->addLocalSegment(local_server_name_, desc);
    }

    return metadata_->addLocalMemoryBuffer(local_id, buf);
}

int PeerTransport::unregisterLocalMemory(void* addr) {
    SegmentID local_id = metadata_->getSegmentID(local_server_name_);
    if (!local_id) return ERR_NOT_FOUND;
    return metadata_->removeLocalMemoryBuffer(local_id, addr);
}

cudaEvent_t PeerTransport::allocEvent(int gpu) {
    int idx = evnext_[gpu] % INFCCL_EVENT_POOL_SIZE;
    evnext_[gpu]++;
    return evpool_[gpu][idx];
}

int PeerTransport::findGpuByPtr(void* ptr) {
    SegmentID local_id = metadata_->getSegmentID(local_server_name_);
    if (local_id) {
        auto seg = metadata_->getSegmentDescByID(local_id);
        if (seg) {
            for (auto& b : seg->buffers) {
                if ((char*)ptr >= (char*)b.addr &&
                    (char*)ptr < (char*)b.addr + b.length)
                    return b.gpu;
            }
        }
    }
    int cur; cudaGetDevice(&cur);
    for (int g = 0; g < ctx_.ndev(); g++)
        if (ctx_.gpu(g).dev_id == cur) return g;
    return -1;
}

int PeerTransport::ensureStaging(int gpu, size_t bytes) {
    if (staging_[gpu] && staging_bytes_[gpu] >= bytes) return OK;
    int saved; cudaGetDevice(&saved);
    cudaSetDevice(ctx_.gpu(gpu).dev_id);
    if (staging_[gpu]) cudaFree(staging_[gpu]);
    if (staging2_[gpu]) cudaFree(staging2_[gpu]);
    if (cudaMalloc(&staging_[gpu], bytes) != cudaSuccess) {
        staging_[gpu] = nullptr; staging2_[gpu] = nullptr; staging_bytes_[gpu] = 0;
        cudaSetDevice(saved); return ERR_MEMORY;
    }
    if (cudaMalloc(&staging2_[gpu], bytes) != cudaSuccess) {
        cudaFree(staging_[gpu]); staging_[gpu] = nullptr;
        staging2_[gpu] = nullptr; staging_bytes_[gpu] = 0;
        cudaSetDevice(saved); return ERR_MEMORY;
    }
    staging_bytes_[gpu] = bytes;
    cudaSetDevice(saved);
    return OK;
}

void* PeerTransport::acquireStaging(int gpu, size_t bytes) {
    for (int p = 0; p < INFCCL_MAX_STAGING_POOLS; p++) {
        StagingEntry& e = staging_pool_[gpu][p];
        if (e.ptr && e.bytes >= bytes && !e.in_use) {
            e.in_use = 1;
            return e.ptr;
        }
    }
    return nullptr;
}

void PeerTransport::releaseStaging(int gpu, void* ptr) {
    for (int p = 0; p < INFCCL_MAX_STAGING_POOLS; p++) {
        if (staging_pool_[gpu][p].ptr == ptr) {
            staging_pool_[gpu][p].in_use = 0;
            return;
        }
    }
}

int PeerTransport::stageFrom(int dst, void* src_ptr, int src,
    size_t offset, size_t bytes) {
    int rc = ensureStaging(dst, bytes);
    if (rc != OK) return rc;
    int saved; cudaGetDevice(&saved);
    cudaSetDevice(ctx_.gpu(dst).dev_id);
    cudaMemcpyPeerAsync(staging_[dst], ctx_.gpu(dst).dev_id,
        (char*)src_ptr + offset, ctx_.gpu(src).dev_id, bytes, streams_[dst][0]);
    cudaSetDevice(saved);
    return OK;
}

int PeerTransport::postSlice(Slice* s, int src_gpu, int dst_gpu) {
    auto seg = metadata_->getSegmentDescByID(s->target_id);
    if (!seg) return ERR_NOT_FOUND;

    void* target_addr = nullptr;
    uint64_t accum = 0;
    for (auto& b : seg->buffers) {
        if (s->target_offset >= accum && s->target_offset < accum + b.length) {
            target_addr = (char*)b.addr + (s->target_offset - accum);
            dst_gpu = b.gpu;
            break;
        }
        accum += b.length;
    }
    if (!target_addr) return ERR_INVALID_ARG;

    int exec_gpu = src_gpu;
    int saved; cudaGetDevice(&saved);
    cudaSetDevice(ctx_.gpu(exec_gpu).dev_id);

    if (s->opcode == TransferRequest::READ) {
        cudaMemcpyPeerAsync(s->source_addr, ctx_.gpu(src_gpu).dev_id,
            target_addr, ctx_.gpu(dst_gpu).dev_id,
            s->length, selectStream(exec_gpu, 0));
    } else {
        cudaMemcpyPeerAsync(target_addr, ctx_.gpu(dst_gpu).dev_id,
            s->source_addr, ctx_.gpu(src_gpu).dev_id,
            s->length, selectStream(exec_gpu, 0));
    }

    cudaEvent_t ev = allocEvent(exec_gpu);
    cudaEventRecord(ev, selectStream(exec_gpu, 0));
    s->peer.event = ev;
    s->peer.stream = selectStream(exec_gpu, 0);
    s->peer.src_dev = ctx_.gpu(src_gpu).dev_id;
    s->peer.dst_dev = ctx_.gpu(dst_gpu).dev_id;
    s->status = Slice::S_POSTED;

    cudaSetDevice(saved);
    return OK;
}

int PeerTransport::submitTransfer(BatchID batch_id,
    const std::vector<TransferRequest>& entries) {

    BatchDesc* batch = (BatchDesc*)(batch_id);
    if (!batch) return ERR_INVALID_ARG;

    int submitted = 0;
    for (size_t i = 0; i < entries.size(); i++) {
        const auto& req = entries[i];

        int src_gpu = findGpuByPtr(req.source);
        if (src_gpu < 0) return ERR_TRANSPORT;

        auto target_seg = metadata_->getSegmentDescByID(req.target_id);
        if (!target_seg || target_seg->buffers.empty()) return ERR_NOT_FOUND;
        int dst_gpu = target_seg->buffers[0].gpu;

        batch->task_list.emplace_back();
        TransferTask& task = batch->task_list.back();
        task.total_bytes = req.length;

        size_t slice_bytes = selectSliceSize(src_gpu, dst_gpu, req.length);
        size_t off = 0;
        while (off < req.length) {
            size_t chunk = req.length - off;
            if (chunk > slice_bytes) chunk = slice_bytes;

            Slice* s = new Slice();
            memset(s, 0, sizeof(Slice));
            s->source_addr = (char*)req.source + off;
            s->length = chunk;
            s->opcode = req.opcode;
            s->target_id = req.target_id;
            s->target_offset = req.target_offset + off;
            s->status = Slice::S_PENDING;
            s->task = &task;

            int rc = postSlice(s, src_gpu, dst_gpu);
            if (rc != OK) { delete s; return rc; }
            task.slices.push_back(s);

            if (cfg_.enable_worker_thread) worker_.enqueue(s);

            off += chunk;
        }
        submitted++;
    }
    return submitted;
}

int PeerTransport::getTransferStatus(BatchID batch_id, size_t task_id,
    TransferStatus& status) {

    BatchDesc* batch = (BatchDesc*)(batch_id);
    if (!batch || task_id >= batch->task_list.size()) return ERR_INVALID_ARG;

    TransferTask& task = batch->task_list[task_id];

    if (task.is_finished) {
        status.s = (task.failed_slice_count > 0) ? FAILED : COMPLETED;
        status.transferred_bytes = task.transferred_bytes;
        return 1;
    }

    if (!cfg_.enable_worker_thread) {
        for (auto* s : task.slices) {
            if (s->status == Slice::S_POSTED) {
                cudaError_t e = cudaEventQuery(s->peer.event);
                if (e == cudaSuccess) s->markSuccess();
                else if (e != cudaErrorNotReady) s->markFailed();
            }
        }
    }

    uint64_t done = task.success_slice_count + task.failed_slice_count;
    if (done == task.slices.size()) {
        task.is_finished = true;
        status.s = (task.failed_slice_count > 0) ? FAILED : COMPLETED;
        status.transferred_bytes = task.transferred_bytes;
        return 1;
    }

    status.s = PENDING;
    status.transferred_bytes = task.transferred_bytes;
    return 0;
}

int PeerTransport::stageFromOnStream(int dst, void* src_ptr, int src,
    size_t offset, size_t bytes, cudaStream_t s, int buf_idx) {
    int rc = ensureStaging(dst, bytes);
    if (rc != OK) return rc;
    void* target = (buf_idx == 0) ? staging_[dst] : staging2_[dst];
    int saved; cudaGetDevice(&saved);
    cudaSetDevice(ctx_.gpu(dst).dev_id);
    cudaMemcpyPeerAsync(target, ctx_.gpu(dst).dev_id,
        (char*)src_ptr + offset, ctx_.gpu(src).dev_id, bytes, s);
    cudaSetDevice(saved);
    return OK;
}

size_t PeerTransport::selectSliceSize(int src, int dst, size_t total) const {
    if (total <= cfg_.no_slice_threshold) return total;
    if (cfg_.adaptive_slice && src >= 0 && dst >= 0 &&
        src < ctx_.ndev() && dst < ctx_.ndev()) {
        size_t opt = ctx_.optimalSlice(src, dst);
        if (opt > 0) return opt;
    }
    return cfg_.slice_bytes;
}

cudaStream_t PeerTransport::selectStream(int gpu, int hint) {
    int nstreams = cfg_.num_streams_per_gpu;
    if (nstreams <= 1) return streams_[gpu][0];
    if (hint >= 0) return streams_[gpu][hint % nstreams];
    int idx = stream_rr_[gpu] % nstreams;
    stream_rr_[gpu]++;
    return streams_[gpu][idx];
}

}
