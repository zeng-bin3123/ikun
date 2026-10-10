#ifndef INFCCL_PEER_WORKER_H_
#define INFCCL_PEER_WORKER_H_
#include "transport.h"
#include "common.h"
#include <thread>
#include <mutex>
#include <deque>
#include <atomic>
namespace infccl {
struct WorkerStats {
    std::atomic<uint64_t> slices_posted{0};
    std::atomic<uint64_t> slices_completed{0};
    std::atomic<uint64_t> slices_failed{0};
    std::atomic<uint64_t> slices_retried{0};
    std::atomic<uint64_t> bytes_transferred{0};
    std::atomic<uint64_t> poll_rounds{0};
};
class PeerWorker {
public:
    PeerWorker() : running_(false), ndev_(0), timeout_us_(5000000), max_retry_(3) {}
    ~PeerWorker() { stop(); }
    int start(int ndev, const int* devs,
              cudaEvent_t (*evpool)[256], int* evnext,
              cudaStream_t (*streams)[4], int* nstreams, int* stream_rr,
              int timeout_ms = 5000, int max_retry = 3) {
        if (running_) return OK;
        ndev_ = ndev;
        for (int i = 0; i < ndev; i++) devs_[i] = devs[i];
        evpool_ = evpool;
        evnext_ = evnext;
        streams_ = streams;
        nstreams_ = nstreams;
        stream_rr_ = stream_rr;
        timeout_us_ = (int64_t)timeout_ms * 1000;
        max_retry_ = max_retry;
        running_ = true;
        thread_ = std::thread(&PeerWorker::loop, this);
        return OK;
    }
    void stop() {
        if (!running_) return;
        running_ = false;
        if (thread_.joinable()) thread_.join();
    }
    void submit(Transport::Slice* s) {
        std::lock_guard<std::mutex> lk(mu_);
        pending_.push_back(s);
    }
    void submitBatch(Transport::Slice* arr, int count) {
        std::lock_guard<std::mutex> lk(mu_);
        for (int i = 0; i < count; i++) pending_.push_back(&arr[i]);
    }
    const WorkerStats& stats() const { return stats_; }
private:
    void postOne(Transport::Slice* s) {
        int g = s->src_gpu;
        int slot = stream_rr_[g] % nstreams_[g];
        stream_rr_[g]++;
        int ev = evnext_[g] % 256;
        evnext_[g]++;
        cudaStream_t st = streams_[g][slot];
        cudaError_t e = cudaMemcpyPeerAsync(
            s->dst_addr, devs_[s->dst_gpu],
            s->src_addr, devs_[s->src_gpu],
            s->length, st);
        if (e != cudaSuccess) {
            s->markFailed();
            stats_.slices_failed.fetch_add(1);
            return;
        }
        cudaEventRecord(evpool_[g][ev], st);
        s->markPosted(now_us(), slot, ev);
        stats_.slices_posted.fetch_add(1);
    }
    bool pollOne(Transport::Slice* s) {
        if (s->status != Transport::Slice::S_POSTED) return s->terminal();
        int g = s->src_gpu;
        cudaError_t e = cudaEventQuery(evpool_[g][s->event_slot]);
        if (e == cudaSuccess) {
            s->markSuccess();
            stats_.slices_completed.fetch_add(1);
            stats_.bytes_transferred.fetch_add(s->length);
            return true;
        }
        if (e == cudaErrorNotReady) {
            if (now_us() - s->post_tick > timeout_us_) {
                if (s->retry_count < max_retry_) {
                    s->markTimeout();
                    stats_.slices_retried.fetch_add(1);
                    std::lock_guard<std::mutex> lk(mu_);
                    pending_.push_back(s);
                } else {
                    s->markFailed();
                    stats_.slices_failed.fetch_add(1);
                }
                return true;
            }
            return false;
        }
        s->markFailed();
        stats_.slices_failed.fetch_add(1);
        return true;
    }
    void loop() {
        while (running_) {
            std::deque<Transport::Slice*> batch;
            {
                std::lock_guard<std::mutex> lk(mu_);
                batch.swap(pending_);
            }
            for (auto* s : batch) {
                if (s->status == Transport::Slice::S_PENDING) postOne(s);
                if (!s->terminal()) inflight_.push_back(s);
            }
            std::deque<Transport::Slice*> remain;
            for (auto* s : inflight_) {
                if (!pollOne(s)) remain.push_back(s);
            }
            inflight_.swap(remain);
            stats_.poll_rounds.fetch_add(1);
            if (batch.empty() && inflight_.empty()) {
                for (int i = 0; i < 50; i++) INFCCL_PAUSE();
            }
        }
        for (auto* s : inflight_) {
            if (s->status == Transport::Slice::S_POSTED) {
                int g = s->src_gpu;
                cudaEventSynchronize(evpool_[g][s->event_slot]);
                s->markSuccess();
                stats_.slices_completed.fetch_add(1);
                stats_.bytes_transferred.fetch_add(s->length);
            }
        }
        inflight_.clear();
    }
    std::atomic<bool> running_;
    std::thread thread_;
    std::mutex mu_;
    std::deque<Transport::Slice*> pending_;
    std::deque<Transport::Slice*> inflight_;
    WorkerStats stats_;
    int ndev_;
    int devs_[INFCCL_MAX_DEVS];
    int64_t timeout_us_;
    int max_retry_;
    cudaEvent_t (*evpool_)[256];
    int* evnext_;
    cudaStream_t (*streams_)[4];
    int* nstreams_;
    int* stream_rr_;
};
}
#endif
