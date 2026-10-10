#ifndef INFCCL_PEER_TRANSPORT_H_
#define INFCCL_PEER_TRANSPORT_H_
#include "transport.h"
#include "peer_context.h"
#include "peer_worker.h"

#define INFCCL_EVENT_POOL_SIZE 256

namespace infccl {

struct PeerTransportConfig {
    size_t slice_bytes = 4 * 1024 * 1024;
    size_t no_slice_threshold = 1 * 1024 * 1024;
    int num_streams_per_gpu = 2;
    int timeout_ms = 5000;
    int max_retry = 3;
    int adaptive_slice = 0;
    int probe_bandwidth = 0;
    int probe_latency = 0;
    int probe_optimal_slice = 0;
    size_t staging_pool_bytes = 0;
};

struct TransportStats {
    std::atomic<uint64_t> slices_submitted{0};
    std::atomic<uint64_t> slices_completed{0};
    std::atomic<uint64_t> slices_failed{0};
    std::atomic<uint64_t> bytes_submitted{0};
    std::atomic<uint64_t> bytes_completed{0};
};

struct MemoryRegionEntry {
    void* addr;
    size_t length;
    int gpu;
    std::string location;
    bool remote_accessible;
};

class PeerTransport : public Transport {
public:
    static constexpr int MAX_STREAMS = 4;
    static constexpr int EVENT_POOL = INFCCL_EVENT_POOL_SIZE;
    static constexpr int STAGING_POOLS = 4;
    static constexpr int MAX_REGISTERED = 256;

    PeerTransport();
    PeerTransport(const PeerTransportConfig& cfg);
    ~PeerTransport();

    int submitTransfer(BatchID batch_id,
        const std::vector<TransferRequest>& entries) override;
    int submitTransferAsync(BatchID batch_id,
        const std::vector<TransferRequest>& entries, uint8_t* out_src_used);
    int syncAndComplete(BatchID batch_id, uint8_t src_used);
    int getTransferStatus(BatchID batch_id, size_t task_id,
        TransferStatus& status) override;
    const char* getName() const override;

    PeerContext& context() { return ctx_; }
    PeerWorker& worker() { return worker_; }
    const PeerTransportConfig& config() const { return cfg_; }
    const TransportStats& stats() const { return stats_; }
    int ndev() const { return ctx_.ndev(); }
    int devId(int idx) const { return ctx_.gpu(idx).dev_id; }

    cudaStream_t stream(int gpu) { return streams_[gpu][0]; }
    cudaStream_t selectStream(int gpu, int idx = -1);
    void* stagingBuffer(int gpu, int pool = 0) { return staging_[gpu][pool]; }

    int ensureStaging(int gpu, size_t bytes, int pool = 0);
    int stageFrom(int dst_gpu, void* src_ptr, int src_gpu,
        size_t offset, size_t bytes, int pool = 0);
    int stageTo(int src_gpu, void* dst_ptr, int dst_gpu,
        size_t offset, size_t bytes, int pool = 0);
    int findGpuByPtr(void* ptr);

    float measuredBandwidth(int src, int dst) const;
    float measuredLatency(int src, int dst) const;
    void printStats(FILE* out = stdout) const;

protected:
    int install(const std::string& local_name,
        std::shared_ptr<TransferMetadata> meta, void** args) override;
    int registerLocalMemory(void* addr, size_t length,
        const std::string& location, bool remote_accessible) override;
    int unregisterLocalMemory(void* addr) override;

private:
    int loadConfig();
    int initStreams();
    int initEvents();
    int initPeerAccess();
    int initStaging();
    int cacheBandwidth();
    size_t selectSliceSize(int src, int dst, size_t total) const;
    int postSlicesAsync(TransferTask& task);
    int syncSrcDevices(uint8_t src_used);
    void completeSlices(TransferTask& task);

    PeerTransportConfig cfg_;
    PeerContext ctx_;
    PeerWorker worker_;
    TransportStats stats_;

    cudaStream_t streams_[INFCCL_MAX_DEVS][MAX_STREAMS];
    int nstreams_[INFCCL_MAX_DEVS];
    int stream_rr_[INFCCL_MAX_DEVS];

    cudaEvent_t evpool_[INFCCL_MAX_DEVS][EVENT_POOL];
    int evnext_[INFCCL_MAX_DEVS];

    void* staging_[INFCCL_MAX_DEVS][STAGING_POOLS];
    size_t staging_bytes_[INFCCL_MAX_DEVS][STAGING_POOLS];

    float bw_cache_[INFCCL_MAX_DEVS][INFCCL_MAX_DEVS];
    float lat_cache_[INFCCL_MAX_DEVS][INFCCL_MAX_DEVS];

    MemoryRegionEntry registered_[MAX_REGISTERED];
    int nregistered_;

    bool installed_;
};

}
#endif
