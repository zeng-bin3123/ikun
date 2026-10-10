#ifndef INFCCL_TRANSPORT_H_
#define INFCCL_TRANSPORT_H_
#include "common.h"
#include "transfer_metadata.h"
#include <cuda_runtime.h>
namespace infccl {
class TransferEngine;
class Transport {
    friend class TransferEngine;
public:
    using BatchID = uint64_t;
    static const BatchID INVALID_BATCH_ID = UINT64_MAX;
    struct TransferRequest {
        enum OpCode { READ = 0, WRITE = 1 };
        OpCode opcode;
        void* source;
        void* dest;
        size_t length;
        int src_gpu;
        int dst_gpu;
        SegmentID target_id;
        uint64_t target_offset;
    };
    enum TransferStatusEnum {
        WAITING, PENDING, INVALID, CANCELLED, COMPLETED, TIMEOUT, FAILED
    };
    struct TransferStatus {
        TransferStatusEnum s;
        size_t transferred_bytes;
    };
    struct TransferTask;
    struct Slice {
        enum SliceStatus { S_PENDING=0, S_POSTED=1, S_SUCCESS=2, S_TIMEOUT=3, S_FAILED=4 };
        void* src_addr;
        void* dst_addr;
        size_t length;
        TransferRequest::OpCode opcode;
        int src_gpu;
        int dst_gpu;
        SliceStatus status;
        TransferTask* task;
        int retry_count;
        int stream_slot;
        int event_slot;
        int64_t post_tick;
        void markSuccess() {
            status = S_SUCCESS;
            __sync_fetch_and_add(&task->success_count, 1);
            __sync_fetch_and_add(&task->transferred_bytes, length);
        }
        void markFailed() {
            status = S_FAILED;
            __sync_fetch_and_add(&task->fail_count, 1);
        }
        void markTimeout() {
            status = S_TIMEOUT;
            retry_count++;
        }
        void markPosted(int64_t tick, int si, int ei) {
            status = S_POSTED;
            post_tick = tick;
            stream_slot = si;
            event_slot = ei;
        }
        bool terminal() const { return status == S_SUCCESS || status == S_FAILED; }
    };
    struct TransferTask {
        std::vector<Slice> slices;
        volatile int64_t success_count;
        volatile int64_t fail_count;
        volatile int64_t transferred_bytes;
        int total;
        TransferRequest orig;
        void init(int n) {
            total = n;
            success_count = 0;
            fail_count = 0;
            transferred_bytes = 0;
        }
        bool allDone() const { return (success_count + fail_count) >= total; }
        bool anyFailed() const { return fail_count > 0; }
    };
    struct BatchDesc {
        BatchID id;
        std::vector<TransferTask> tasks;
        int64_t start_us;
        bool freed;
        TransferStatusEnum status() const {
            bool any_pending = false;
            for (auto& t : tasks) {
                if (t.anyFailed()) return FAILED;
                if (!t.allDone()) any_pending = true;
            }
            return any_pending ? PENDING : COMPLETED;
        }
        size_t transferred() const {
            size_t n = 0;
            for (auto& t : tasks) n += t.transferred_bytes;
            return n;
        }
    };
    static inline BatchDesc& toBatch(BatchID id) {
        return *reinterpret_cast<BatchDesc*>(id);
    }
public:
    virtual ~Transport() {}
    BatchID allocateBatchID(size_t batch_size) {
        auto* bd = new BatchDesc();
        bd->id = BatchID(bd);
        bd->tasks.resize(batch_size);
        bd->start_us = now_us();
        bd->freed = false;
        RWSpinlock::WriteGuard guard(batch_lock_);
        batch_set_[bd->id] = std::shared_ptr<BatchDesc>(bd);
        return bd->id;
    }
    int freeBatchID(BatchID batch_id) {
        RWSpinlock::WriteGuard guard(batch_lock_);
        batch_set_.erase(batch_id);
        return OK;
    }
    BatchDesc* getBatch(BatchID id) {
        return reinterpret_cast<BatchDesc*>(id);
    }
    virtual int submitTransfer(BatchID batch_id,
        const std::vector<TransferRequest>& entries) = 0;
    virtual int getTransferStatus(BatchID batch_id, size_t task_id,
        TransferStatus& status) = 0;
    int waitBatch(BatchID batch_id, int timeout_ms = 10000) {
        auto* bd = getBatch(batch_id);
        int64_t deadline = now_us() + (int64_t)timeout_ms * 1000;
        while (bd->status() == PENDING) {
            if (now_us() > deadline) return ERR_TIMEOUT;
            INFCCL_PAUSE();
        }
        return bd->status() == COMPLETED ? OK : ERR_FAIL;
    }
    std::shared_ptr<TransferMetadata>& meta() { return metadata_; }
protected:
    virtual int install(const std::string& local_name,
        std::shared_ptr<TransferMetadata> meta, void** args) {
        local_server_name_ = local_name;
        metadata_ = meta;
        return OK;
    }
    virtual int registerLocalMemory(void* addr, size_t length,
        const std::string& location, bool remote_accessible) = 0;
    virtual int unregisterLocalMemory(void* addr) = 0;
    virtual const char* getName() const = 0;
    std::string local_server_name_;
    std::shared_ptr<TransferMetadata> metadata_;
    RWSpinlock batch_lock_;
    std::unordered_map<BatchID, std::shared_ptr<BatchDesc>> batch_set_;
};
}
#endif
