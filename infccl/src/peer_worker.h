#ifndef INFCCL_PEER_WORKER_H_
#define INFCCL_PEER_WORKER_H_

#include "transport.h"
#include "common.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <atomic>

namespace infccl {

struct WorkerStats {
    std::atomic<uint64_t> slices_submitted{0};
    std::atomic<uint64_t> slices_completed{0};
    std::atomic<uint64_t> slices_failed{0};
    std::atomic<uint64_t> bytes_transferred{0};
    std::atomic<uint64_t> poll_cycles{0};
};

class PeerWorker {
public:
    PeerWorker() : running_(false), ndev_(0) {
        memset(devs_, 0, sizeof(devs_));
    }

    ~PeerWorker() { stop(); }

    int start(int ndev, const int* devs) {
        if (running_) return OK;
        ndev_ = ndev;
        for (int i = 0; i < ndev; i++) devs_[i] = devs[i];
        running_ = true;
        thread_ = std::thread(&PeerWorker::pollLoop, this);
        return OK;
    }

    void stop() {
        if (!running_) return;
        running_ = false;
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

    void enqueue(Transport::Slice* slice) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            pending_.push_back(slice);
        }
        cv_.notify_one();
        stats_.slices_submitted.fetch_add(1, std::memory_order_relaxed);
    }

    void enqueueBatch(Transport::Slice** slices, int count) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (int i = 0; i < count; i++) pending_.push_back(slices[i]);
        }
        cv_.notify_one();
        stats_.slices_submitted.fetch_add(count, std::memory_order_relaxed);
    }

    int pollOnce() {
        int completed = 0;
        std::deque<Transport::Slice*> local;
        {
            std::lock_guard<std::mutex> lk(mu_);
            local.swap(inflight_);
        }

        std::deque<Transport::Slice*> still_inflight;
        for (auto* s : local) {
            if (s->status != Transport::Slice::S_POSTED) {
                still_inflight.push_back(s);
                continue;
            }
            cudaError_t e = cudaEventQuery(s->peer.event);
            if (e == cudaSuccess) {
                s->markSuccess();
                stats_.slices_completed.fetch_add(1, std::memory_order_relaxed);
                stats_.bytes_transferred.fetch_add(s->length, std::memory_order_relaxed);
                completed++;
            } else if (e == cudaErrorNotReady) {
                still_inflight.push_back(s);
            } else {
                s->markFailed();
                stats_.slices_failed.fetch_add(1, std::memory_order_relaxed);
                completed++;
            }
        }

        {
            std::lock_guard<std::mutex> lk(mu_);
            for (auto* s : still_inflight) inflight_.push_back(s);
        }
        return completed;
    }

    void promoteToInflight() {
        std::lock_guard<std::mutex> lk(mu_);
        while (!pending_.empty()) {
            inflight_.push_back(pending_.front());
            pending_.pop_front();
        }
    }

    bool hasWork() const {
        return !pending_.empty() || !inflight_.empty();
    }

    const WorkerStats& stats() const { return stats_; }

    void resetStats() {
        stats_.slices_submitted.store(0);
        stats_.slices_completed.store(0);
        stats_.slices_failed.store(0);
        stats_.bytes_transferred.store(0);
        stats_.poll_cycles.store(0);
    }

private:
    void pollLoop() {
        while (running_) {
            promoteToInflight();

            bool did_work = false;
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (!inflight_.empty()) did_work = true;
            }

            if (did_work) {
                pollOnce();
                stats_.poll_cycles.fetch_add(1, std::memory_order_relaxed);
            } else {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait_for(lk, std::chrono::microseconds(100),
                    [this]{ return !pending_.empty() || !running_; });
            }
        }

        promoteToInflight();
        while (!inflight_.empty()) {
            pollOnce();
            promoteToInflight();
        }
    }

    std::atomic<bool> running_;
    std::thread thread_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Transport::Slice*> pending_;
    std::deque<Transport::Slice*> inflight_;
    int ndev_;
    int devs_[INFCCL_MAX_DEVS];
    WorkerStats stats_;
};

}
#endif
