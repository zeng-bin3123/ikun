#ifndef INFCCL_TRANSPORT_H_
#define INFCCL_TRANSPORT_H_

#include "common.h"
#include "transfer_metadata.h"

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
        SegmentID target_id;
        uint64_t target_offset;
        size_t length;
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
        enum SliceStatus { S_PENDING, S_POSTED, S_SUCCESS, S_TIMEOUT, S_FAILED };

        void* source_addr;
        size_t length;
        TransferRequest::OpCode opcode;
        SegmentID target_id;
        uint64_t target_offset;
        SliceStatus status;
        TransferTask* task;

        union {
            struct {
                void* dest_addr;
                int src_dev;
                int dst_dev;
                cudaStream_t stream;
                cudaEvent_t event;
            } peer;
            struct {
                void* dest_addr;
            } local;
            struct {
                int fd;
                uint64_t dest_addr;
                size_t offset;
            } tcp;
        };

        void markSuccess() {
            status = S_SUCCESS;
            __sync_fetch_and_add(&task->transferred_bytes, length);
            __sync_fetch_and_add(&task->success_slice_count, 1);
        }

        void markFailed() {
            status = S_FAILED;
            __sync_fetch_and_add(&task->failed_slice_count, 1);
        }
    };

    struct TransferTask {
        ~TransferTask() { for (auto s : slices) delete s; slices.clear(); }
        std::vector<Slice*> slices;
        volatile uint64_t success_slice_count = 0;
        volatile uint64_t failed_slice_count = 0;
        volatile uint64_t transferred_bytes = 0;
        volatile bool is_finished = false;
        uint64_t total_bytes = 0;
    };

    struct BatchDesc {
        BatchID id;
        size_t batch_size;
        std::vector<TransferTask> task_list;
        void* context;
    };

    struct BufferEntry {
        void* addr;
        size_t length;
    };

public:
    virtual ~Transport() {}

    BatchID allocateBatchID(size_t batch_size) {
        auto* bd = new BatchDesc();
        if (!bd) return INVALID_BATCH_ID;
        bd->id = BatchID(bd);
        bd->batch_size = batch_size;
        bd->task_list.reserve(batch_size);
        bd->context = nullptr;
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
        return (BatchDesc*)(id);
    }

    virtual int submitTransfer(BatchID batch_id,
        const std::vector<TransferRequest>& entries) = 0;

    virtual int getTransferStatus(BatchID batch_id, size_t task_id,
        TransferStatus& status) = 0;

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
