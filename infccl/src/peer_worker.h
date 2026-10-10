#ifndef INFCCL_PEER_WORKER_H_
#define INFCCL_PEER_WORKER_H_
#include "transport.h"
#include "common.h"
#include <thread>
#include <mutex>
#include <deque>
#include <atomic>
#include <functional>
#include <vector>
namespace infccl {

struct WorkerStats {
    std::atomic<uint64_t> slices_submitted{0};
    std::atomic<uint64_t> slices_completed{0};
    std::atomic<uint64_t> slices_failed{0};
    std::atomic<uint64_t> slices_retried{0};
    std::atomic<uint64_t> slices_timed_out{0};
    std::atomic<uint64_t> bytes_transferred{0};
    std::atomic<uint64_t> poll_rounds{0};
    std::atomic<uint64_t> poll_hits{0};
    std::atomic<uint64_t> poll_misses{0};
    std::atomic<uint64_t> health_checks{0};
    std::atomic<uint64_t> bw_recalibrations{0};
    std::atomic<uint64_t> callbacks_executed{0};
    std::atomic<uint64_t> drain_count{0};
    std::atomic<int64_t> max_completion_latency_us{0};
    std::atomic<int64_t> total_completion_latency_us{0};
};

struct PeerHealthStatus {
    int gpu;
    int alive;
    float last_bw_gbps;
    float last_latency_us;
    int64_t last_check_us;
    int consecutive_failures;
};

using CompletionCallback = std::function<void(Transport::BatchID, int, bool)>;

struct WorkerConfig {
    int64_t timeout_us = 5000000;
    int max_retry = 3;
    int64_t health_check_interval_us = 5000000;
    int batch_drain_size = 64;
    int spin_count = 100;

    static WorkerConfig fast() {
        WorkerConfig c;
        c.timeout_us = 1000000;
        c.batch_drain_size = 128;
        c.spin_count = 10;
        return c;
    }

    static WorkerConfig conservative() {
        WorkerConfig c;
        c.timeout_us = 10000000;
        c.max_retry = 5;
        c.batch_drain_size = 32;
        c.spin_count = 200;
        return c;
    }

    static WorkerConfig fromEnv() {
        WorkerConfig c;
        const char* e;
        e = getenv("INFCCL_WORKER_TIMEOUT_US");
        if (e) c.timeout_us = atol(e);
        e = getenv("INFCCL_WORKER_MAX_RETRY");
        if (e) c.max_retry = atoi(e);
        e = getenv("INFCCL_WORKER_DRAIN_SIZE");
        if (e) c.batch_drain_size = atoi(e);
        e = getenv("INFCCL_WORKER_SPIN");
        if (e) c.spin_count = atoi(e);
        return c;
    }
};

struct InflightSlice {
    Transport::Slice* slice;
    Transport::BatchID batch_id;
    int task_idx;
    CompletionCallback cb;
    int64_t deadline_us;
};

class PeerWorker {
public:
    PeerWorker() : running_(false), ndev_(0), timeout_us_(5000000),
                   max_retry_(3), check_interval_us_(5000000),
                   batch_drain_size_(64) {
        memset(devs_, 0, sizeof(devs_));
        memset(health_, 0, sizeof(health_));
        memset(bw_results_, 0, sizeof(bw_results_));
        memset(lat_results_, 0, sizeof(lat_results_));
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
    void setBatchDrainSize(int n) { batch_drain_size_ = n > 0 ? n : 1; }
    void setTimeout(int64_t us) { timeout_us_ = us; }
    void setMaxRetry(int n) { max_retry_ = n; }

    void submitForPolling(Transport::Slice* arr, int count) {
        std::lock_guard<std::mutex> lk(mu_);
        for (int i = 0; i < count; i++) {
            InflightSlice is;
            is.slice = &arr[i];
            is.batch_id = 0;
            is.task_idx = 0;
            is.deadline_us = arr[i].post_tick + timeout_us_;
            poll_queue_.push_back(is);
        }
    }

    void submitForPollingWithCallback(Transport::Slice* arr, int count,
        Transport::BatchID bid, int task_idx, CompletionCallback cb) {
        std::lock_guard<std::mutex> lk(mu_);
        for (int i = 0; i < count; i++) {
            InflightSlice is;
            is.slice = &arr[i];
            is.batch_id = bid;
            is.task_idx = task_idx;
            is.cb = cb;
            is.deadline_us = arr[i].post_tick + timeout_us_;
            poll_queue_.push_back(is);
        }
    }

    void enqueueCallback(std::function<void()> fn) {
        std::lock_guard<std::mutex> lk(mu_);
        callbacks_.push_back(fn);
    }

    int inflightCount() const { return inflight_count_.load(); }
    bool hasInflight() const { return inflight_count_.load() > 0; }

    void drain() {
        while (hasInflight()) {
            INFCCL_PAUSE();
        }
        stats_.drain_count.fetch_add(1);
    }

    bool drainWithTimeout(int64_t timeout_us) {
        int64_t deadline = now_us() + timeout_us;
        while (hasInflight()) {
            if (now_us() > deadline) return false;
            INFCCL_PAUSE();
        }
        stats_.drain_count.fetch_add(1);
        return true;
    }

    const WorkerStats& stats() const { return stats_; }
    const PeerHealthStatus& health(int gpu) const { return health_[gpu]; }
    int isHealthy(int gpu) const { return health_[gpu].alive; }

    void recalibrateBandwidth(int src, int dst) {
        if (src < 0 || src >= ndev_ || dst < 0 || dst >= ndev_ || src == dst) return;
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
        health_[src].last_bw_gbps = bw;
        health_[src].last_latency_us = lat;
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

    void printStats(FILE* out = stdout) const {
        fprintf(out, "Worker stats:\n");
        fprintf(out, "  submitted:  %lu\n", stats_.slices_submitted.load());
        fprintf(out, "  completed:  %lu\n", stats_.slices_completed.load());
        fprintf(out, "  failed:     %lu\n", stats_.slices_failed.load());
        fprintf(out, "  retried:    %lu\n", stats_.slices_retried.load());
        fprintf(out, "  timed_out:  %lu\n", stats_.slices_timed_out.load());
        fprintf(out, "  bytes:      %lu\n", stats_.bytes_transferred.load());
        fprintf(out, "  polls:      %lu (hits=%lu misses=%lu)\n",
            stats_.poll_rounds.load(), stats_.poll_hits.load(), stats_.poll_misses.load());
        fprintf(out, "  callbacks:  %lu\n", stats_.callbacks_executed.load());
        fprintf(out, "  drains:     %lu\n", stats_.drain_count.load());
        if (stats_.slices_completed.load() > 0) {
            float avg_lat = (float)stats_.total_completion_latency_us.load() /
                            (float)stats_.slices_completed.load();
            fprintf(out, "  avg completion latency: %.1f us\n", avg_lat);
            fprintf(out, "  max completion latency: %ld us\n",
                stats_.max_completion_latency_us.load());
        }
        fprintf(out, "  inflight:   %d\n", inflight_count_.load());
        fprintf(out, "  health checks: %lu\n", stats_.health_checks.load());
        fprintf(out, "  bw recals:  %lu\n", stats_.bw_recalibrations.load());
    }

private:
    void pollEvents() {
        std::deque<InflightSlice> incoming;
        {
            std::lock_guard<std::mutex> lk(mu_);
            incoming.swap(poll_queue_);
        }
        for (auto& is : incoming) {
            inflight_.push_back(is);
            inflight_count_.fetch_add(1);
        }

        if (inflight_.empty()) return;

        int64_t now = now_us();
        int processed = 0;
        std::deque<InflightSlice> remain;

        for (auto& is : inflight_) {
            if (processed >= batch_drain_size_) {
                remain.push_back(is);
                continue;
            }

            Transport::Slice* s = is.slice;
            if (s->status != Transport::Slice::S_POSTED) {
                if (s->terminal()) {
                    inflight_count_.fetch_sub(1);
                    continue;
                }
                remain.push_back(is);
                continue;
            }

            int g = s->src_gpu;
            cudaSetDevice(devs_[g]);
            cudaError_t e = cudaEventQuery(evpool_[g][s->event_slot]);
            processed++;

            if (e == cudaSuccess) {
                s->markSuccess();
                stats_.slices_completed.fetch_add(1);
                stats_.bytes_transferred.fetch_add(s->length);
                stats_.poll_hits.fetch_add(1);
                int64_t lat = now - s->post_tick;
                stats_.total_completion_latency_us.fetch_add(lat);
                int64_t prev_max = stats_.max_completion_latency_us.load();
                while (lat > prev_max && !stats_.max_completion_latency_us
                    .compare_exchange_weak(prev_max, lat)) {}
                inflight_count_.fetch_sub(1);
                if (is.cb) {
                    is.cb(is.batch_id, is.task_idx, true);
                    stats_.callbacks_executed.fetch_add(1);
                }
            } else if (e == cudaErrorNotReady) {
                stats_.poll_misses.fetch_add(1);
                if (now > is.deadline_us) {
                    if (s->retry_count < max_retry_) {
                        s->status = Transport::Slice::S_PENDING;
                        s->retry_count++;
                        stats_.slices_retried.fetch_add(1);
                        remain.push_back(is);
                    } else {
                        s->markFailed();
                        stats_.slices_timed_out.fetch_add(1);
                        stats_.slices_failed.fetch_add(1);
                        inflight_count_.fetch_sub(1);
                        if (is.cb) {
                            is.cb(is.batch_id, is.task_idx, false);
                            stats_.callbacks_executed.fetch_add(1);
                        }
                    }
                } else {
                    remain.push_back(is);
                }
            } else {
                s->markFailed();
                stats_.slices_failed.fetch_add(1);
                inflight_count_.fetch_sub(1);
                if (is.cb) {
                    is.cb(is.batch_id, is.task_idx, false);
                    stats_.callbacks_executed.fetch_add(1);
                }
            }
        }
        inflight_.swap(remain);
    }

    void runCallbacks() {
        std::deque<std::function<void()>> cbs;
        {
            std::lock_guard<std::mutex> lk(mu_);
            cbs.swap(callbacks_);
        }
        for (auto& fn : cbs) {
            fn();
            stats_.callbacks_executed.fetch_add(1);
        }
    }

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

    void drainOnShutdown() {
        int rounds = 0;
        while (!inflight_.empty() && rounds < 1000) {
            for (auto& is : inflight_) {
                Transport::Slice* s = is.slice;
                if (s->status == Transport::Slice::S_POSTED) {
                    int g = s->src_gpu;
                    cudaSetDevice(devs_[g]);
                    cudaEventSynchronize(evpool_[g][s->event_slot]);
                    s->markSuccess();
                    stats_.slices_completed.fetch_add(1);
                    stats_.bytes_transferred.fetch_add(s->length);
                    inflight_count_.fetch_sub(1);
                    if (is.cb) is.cb(is.batch_id, is.task_idx, true);
                }
            }
            inflight_.clear();

            std::lock_guard<std::mutex> lk(mu_);
            for (auto& is : poll_queue_) {
                Transport::Slice* s = is.slice;
                if (s->status == Transport::Slice::S_POSTED) {
                    int g = s->src_gpu;
                    cudaSetDevice(devs_[g]);
                    cudaEventSynchronize(evpool_[g][s->event_slot]);
                    s->markSuccess();
                    stats_.slices_completed.fetch_add(1);
                    stats_.bytes_transferred.fetch_add(s->length);
                    if (is.cb) is.cb(is.batch_id, is.task_idx, true);
                }
            }
            poll_queue_.clear();
            rounds++;
        }
    }

    void loop() {
        while (running_) {
            pollEvents();
            runCallbacks();

            int64_t now = now_us();
            for (int g = 0; g < ndev_; g++) {
                if (now - health_[g].last_check_us > check_interval_us_)
                    checkHealth(g);
            }

            stats_.poll_rounds.fetch_add(1);
            if (inflight_.empty()) {
                for (int i = 0; i < 100; i++) INFCCL_PAUSE();
            }
        }
        drainOnShutdown();
    }

    std::atomic<bool> running_;
    std::thread thread_;
    std::mutex mu_;
    WorkerStats stats_;
    std::atomic<int> inflight_count_{0};
    int ndev_;
    int devs_[INFCCL_MAX_DEVS];
    int64_t timeout_us_;
    int max_retry_;
    int64_t check_interval_us_;
    int batch_drain_size_;

    PeerHealthStatus health_[INFCCL_MAX_DEVS];
    float bw_results_[INFCCL_MAX_DEVS][INFCCL_MAX_DEVS];
    float lat_results_[INFCCL_MAX_DEVS][INFCCL_MAX_DEVS];

    std::deque<InflightSlice> poll_queue_;
    std::deque<InflightSlice> inflight_;
    std::deque<std::function<void()>> callbacks_;

    cudaEvent_t (*evpool_)[256];
    int* evnext_;
    cudaStream_t (*streams_)[4];
    int* nstreams_;
    int* stream_rr_;
};

}
#endif
