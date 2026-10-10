#ifndef INFCCL_PEER_WORKER_H_
#define INFCCL_PEER_WORKER_H_
#include "transport.h"
#include "common.h"
#include <thread>
#include <mutex>
#include <deque>
#include <atomic>
#include <functional>
namespace infccl {

struct WorkerStats {
    std::atomic<uint64_t> slices_submitted{0};
    std::atomic<uint64_t> slices_completed{0};
    std::atomic<uint64_t> slices_failed{0};
    std::atomic<uint64_t> slices_retried{0};
    std::atomic<uint64_t> bytes_transferred{0};
    std::atomic<uint64_t> poll_rounds{0};
    std::atomic<uint64_t> health_checks{0};
    std::atomic<uint64_t> bw_recalibrations{0};
};

struct PeerHealthStatus {
    int gpu;
    int alive;
    float last_bw_gbps;
    float last_latency_us;
    int64_t last_check_us;
    int consecutive_failures;
};

class PeerWorker {
public:
    PeerWorker() : running_(false), ndev_(0), check_interval_us_(5000000) {
        memset(devs_, 0, sizeof(devs_));
        memset(health_, 0, sizeof(health_));
    }
    ~PeerWorker() { stop(); }

    int start(int ndev, const int* devs,
              cudaEvent_t (*evpool)[256], int* evnext,
              cudaStream_t (*streams)[4], int* nstreams, int* stream_rr,
              int timeout_ms = 5000, int max_retry = 3) {
        if (running_) return OK;
        ndev_ = ndev;
        for (int i = 0; i < ndev; i++) {
            devs_[i] = devs[i];
            health_[i].gpu = devs[i];
            health_[i].alive = 1;
            health_[i].last_check_us = now_us();
        }
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

    void setCheckInterval(int64_t us) { check_interval_us_ = us; }

    void enqueueCallback(std::function<void()> fn) {
        std::lock_guard<std::mutex> lk(mu_);
        callbacks_.push_back(fn);
    }

    void submitForPolling(Transport::Slice* arr, int count) {
        std::lock_guard<std::mutex> lk(mu_);
        for (int i = 0; i < count; i++)
            poll_queue_.push_back(&arr[i]);
    }

    const WorkerStats& stats() const { return stats_; }
    const PeerHealthStatus& health(int gpu) const { return health_[gpu]; }
    int isHealthy(int gpu) const { return health_[gpu].alive; }

    void recalibrateBandwidth(int src, int dst) {
        if (src < 0 || src >= ndev_ || dst < 0 || dst >= ndev_) return;
        if (src == dst) return;
        int saved; cudaGetDevice(&saved);

        size_t bytes = 4 * 1024 * 1024;
        float *s_buf, *d_buf;
        cudaSetDevice(devs_[src]);
        if (cudaMalloc(&s_buf, bytes) != cudaSuccess) { cudaSetDevice(saved); return; }
        cudaSetDevice(devs_[dst]);
        if (cudaMalloc(&d_buf, bytes) != cudaSuccess) {
            cudaSetDevice(devs_[src]); cudaFree(s_buf); cudaSetDevice(saved); return;
        }

        cudaSetDevice(devs_[src]);
        cudaStream_t s; cudaStreamCreate(&s);
        cudaMemcpyPeerAsync(d_buf, devs_[dst], s_buf, devs_[src], bytes, s);
        cudaStreamSynchronize(s);

        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        cudaEventRecord(t0, s);
        for (int i = 0; i < 20; i++)
            cudaMemcpyPeerAsync(d_buf, devs_[dst], s_buf, devs_[src], bytes, s);
        cudaEventRecord(t1, s); cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);
        float bw = bytes * 20.0f * 1e-6f / ms;

        cudaEventRecord(t0, s);
        for (int i = 0; i < 100; i++)
            cudaMemcpyPeerAsync(d_buf, devs_[dst], s_buf, devs_[src], 4, s);
        cudaEventRecord(t1, s); cudaEventSynchronize(t1);
        cudaEventElapsedTime(&ms, t0, t1);
        float lat = ms * 1000.0f / 100;

        bw_results_[src][dst] = bw;
        lat_results_[src][dst] = lat;
        stats_.bw_recalibrations.fetch_add(1);

        cudaEventDestroy(t0); cudaEventDestroy(t1);
        cudaStreamDestroy(s);
        cudaSetDevice(devs_[src]); cudaFree(s_buf);
        cudaSetDevice(devs_[dst]); cudaFree(d_buf);
        cudaSetDevice(saved);
    }

    float lastMeasuredBw(int src, int dst) const {
        if (src < 0 || src >= INFCCL_MAX_DEVS || dst < 0 || dst >= INFCCL_MAX_DEVS) return 0;
        return bw_results_[src][dst];
    }

    float lastMeasuredLat(int src, int dst) const {
        if (src < 0 || src >= INFCCL_MAX_DEVS || dst < 0 || dst >= INFCCL_MAX_DEVS) return 0;
        return lat_results_[src][dst];
    }

private:
    void checkHealth(int gpu) {
        int saved; cudaGetDevice(&saved);
        cudaSetDevice(devs_[gpu]);
        int* d_check;
        cudaError_t e = cudaMalloc(&d_check, 4);
        if (e != cudaSuccess) {
            health_[gpu].alive = 0;
            health_[gpu].consecutive_failures++;
            cudaSetDevice(saved);
            return;
        }
        int val = 0;
        cudaMemset(d_check, 0x42, 4);
        cudaMemcpy(&val, d_check, 4, cudaMemcpyDeviceToHost);
        cudaFree(d_check);
        if (val == 0x42424242) {
            health_[gpu].alive = 1;
            health_[gpu].consecutive_failures = 0;
        } else {
            health_[gpu].alive = 0;
            health_[gpu].consecutive_failures++;
        }
        health_[gpu].last_check_us = now_us();
        stats_.health_checks.fetch_add(1);
        cudaSetDevice(saved);
    }

    void loop() {
        while (running_) {
            int64_t now = now_us();

            for (int g = 0; g < ndev_; g++) {
                if (now - health_[g].last_check_us > check_interval_us_)
                    checkHealth(g);
            }

            {
                std::deque<std::function<void()>> cbs;
                {
                    std::lock_guard<std::mutex> lk(mu_);
                    cbs.swap(callbacks_);
                }
                for (auto& fn : cbs) fn();
            }

            {
                std::deque<Transport::Slice*> new_polls;
                {
                    std::lock_guard<std::mutex> lk(mu_);
                    for (auto* s : poll_queue_) inflight_.push_back(s);
                    poll_queue_.clear();
                }
                std::deque<Transport::Slice*> remain;
                for (auto* s : inflight_) {
                    if (s->status != Transport::Slice::S_POSTED) continue;
                    int g = s->src_gpu;
                    cudaSetDevice(devs_[g]);
                    cudaError_t e = cudaEventQuery(evpool_[g][s->event_slot]);
                    if (e == cudaSuccess) {
                        s->markSuccess();
                        stats_.slices_completed.fetch_add(1);
                        stats_.bytes_transferred.fetch_add(s->length);
                    } else if (e == cudaErrorNotReady) {
                        remain.push_back(s);
                    } else {
                        s->markFailed();
                        stats_.slices_failed.fetch_add(1);
                    }
                }
                inflight_.swap(remain);
            }

            stats_.poll_rounds.fetch_add(1);
            if (inflight_.empty()) {
                for (int i = 0; i < 100; i++) INFCCL_PAUSE();
            }
        }
    }

    std::atomic<bool> running_;
    std::thread thread_;
    std::mutex mu_;
    WorkerStats stats_;
    int ndev_;
    int devs_[INFCCL_MAX_DEVS];
    int64_t timeout_us_;
    int max_retry_;
    int64_t check_interval_us_;

    PeerHealthStatus health_[INFCCL_MAX_DEVS];
    float bw_results_[INFCCL_MAX_DEVS][INFCCL_MAX_DEVS];
    float lat_results_[INFCCL_MAX_DEVS][INFCCL_MAX_DEVS];
    std::deque<std::function<void()>> callbacks_;
    std::deque<Transport::Slice*> poll_queue_;
    std::deque<Transport::Slice*> inflight_;

    cudaEvent_t (*evpool_)[256];
    int* evnext_;
    cudaStream_t (*streams_)[4];
    int* nstreams_;
    int* stream_rr_;
};

}
#endif
