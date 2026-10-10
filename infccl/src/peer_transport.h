#ifndef INFCCL_PEER_TRANSPORT_H_
#define INFCCL_PEER_TRANSPORT_H_
#include "transport.h"
#include "peer_context.h"
#include "peer_worker.h"
#define INFCCL_SLICE_BYTES_DEFAULT (64 * 1024)
#define INFCCL_EVENT_POOL_SIZE 256
namespace infccl {
struct PeerTransportConfig {
    size_t slice_bytes = INFCCL_SLICE_BYTES_DEFAULT;
    size_t no_slice_threshold = 4096;
    int num_streams_per_gpu = 2;
    int timeout_ms = 5000;
    int max_retry = 3;
    int adaptive_slice = 0;
    int probe_bandwidth = 0;
    int probe_latency = 0;
    int probe_optimal_slice = 0;
};
class PeerTransport : public Transport {
public:
    PeerTransport();
    PeerTransport(const PeerTransportConfig& cfg);
    ~PeerTransport();
    int submitTransfer(BatchID batch_id,
        const std::vector<TransferRequest>& entries) override;
    int getTransferStatus(BatchID batch_id, size_t task_id,
        TransferStatus& status) override;
    const char* getName() const override;
    PeerContext& context() { return ctx_; }
    PeerWorker& worker() { return worker_; }
    const PeerTransportConfig& config() const { return cfg_; }
    cudaStream_t stream(int gpu) { return streams_[gpu][0]; }
    cudaStream_t selectStream(int gpu, int idx = -1);
    void* stagingBuffer(int gpu) { return staging_[gpu]; }
    int ensureStaging(int gpu, size_t bytes);
    int stageFrom(int dst_gpu, void* src_ptr, int src_gpu, size_t offset, size_t bytes);
protected:
    int install(const std::string& local_name,
        std::shared_ptr<TransferMetadata> meta, void** args) override;
    int registerLocalMemory(void* addr, size_t length,
        const std::string& location, bool remote_accessible) override;
    int unregisterLocalMemory(void* addr) override;
private:
    size_t selectSliceSize(int src, int dst, size_t total) const;
    PeerTransportConfig cfg_;
    PeerContext ctx_;
    PeerWorker worker_;
    cudaStream_t streams_[INFCCL_MAX_DEVS][4];
    int nstreams_[INFCCL_MAX_DEVS];
    int stream_rr_[INFCCL_MAX_DEVS];
    void* staging_[INFCCL_MAX_DEVS];
    size_t staging_bytes_[INFCCL_MAX_DEVS];
    cudaEvent_t evpool_[INFCCL_MAX_DEVS][INFCCL_EVENT_POOL_SIZE];
    int evnext_[INFCCL_MAX_DEVS];
    bool installed_;
};
}
#endif
