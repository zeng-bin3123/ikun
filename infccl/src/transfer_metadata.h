#ifndef INFCCL_TRANSFER_METADATA_H_
#define INFCCL_TRANSFER_METADATA_H_

#include "common.h"
#include <cuda_runtime.h>

namespace infccl {

using SegmentID = uint64_t;
const static SegmentID LOCAL_SEGMENT_ID = 0;

struct BufferDesc {
    void* addr;
    uint64_t length;
    int gpu;
    std::string location;
};

struct SegmentDesc {
    std::string name;
    std::string protocol;
    std::vector<BufferDesc> buffers;
};

class TransferMetadata {
public:
    TransferMetadata() : next_id_(1) {}

    SegmentID addLocalSegment(const std::string& name, std::shared_ptr<SegmentDesc> desc) {
        RWSpinlock::WriteGuard guard(lock_);
        SegmentID id = next_id_++;
        id_to_desc_[id] = desc;
        name_to_id_[name] = id;
        return id;
    }

    int removeSegment(const std::string& name) {
        RWSpinlock::WriteGuard guard(lock_);
        auto it = name_to_id_.find(name);
        if (it == name_to_id_.end()) return ERR_NOT_FOUND;
        id_to_desc_.erase(it->second);
        name_to_id_.erase(it);
        return OK;
    }

    std::shared_ptr<SegmentDesc> getSegmentDescByID(SegmentID id) {
        RWSpinlock::ReadGuard guard(lock_);
        auto it = id_to_desc_.find(id);
        return it != id_to_desc_.end() ? it->second : nullptr;
    }

    std::shared_ptr<SegmentDesc> getSegmentDescByName(const std::string& name) {
        RWSpinlock::ReadGuard guard(lock_);
        auto it = name_to_id_.find(name);
        if (it == name_to_id_.end()) return nullptr;
        auto dit = id_to_desc_.find(it->second);
        return dit != id_to_desc_.end() ? dit->second : nullptr;
    }

    SegmentID getSegmentID(const std::string& name) {
        RWSpinlock::ReadGuard guard(lock_);
        auto it = name_to_id_.find(name);
        return it != name_to_id_.end() ? it->second : 0;
    }

    int addLocalMemoryBuffer(SegmentID seg_id, const BufferDesc& buf) {
        RWSpinlock::WriteGuard guard(lock_);
        auto it = id_to_desc_.find(seg_id);
        if (it == id_to_desc_.end()) return ERR_NOT_FOUND;
        for (auto& existing : it->second->buffers)
            if (memOverlap(existing.addr, existing.length, buf.addr, buf.length))
                return ERR_OVERLAP;
        it->second->buffers.push_back(buf);
        return OK;
    }

    int removeLocalMemoryBuffer(SegmentID seg_id, void* addr) {
        RWSpinlock::WriteGuard guard(lock_);
        auto it = id_to_desc_.find(seg_id);
        if (it == id_to_desc_.end()) return ERR_NOT_FOUND;
        auto& bufs = it->second->buffers;
        for (auto bit = bufs.begin(); bit != bufs.end(); ++bit) {
            if (bit->addr == addr) { bufs.erase(bit); return OK; }
        }
        return ERR_NOT_FOUND;
    }

    const BufferDesc* findBufferByAddr(SegmentID seg_id, void* addr) {
        RWSpinlock::ReadGuard guard(lock_);
        auto it = id_to_desc_.find(seg_id);
        if (it == id_to_desc_.end()) return nullptr;
        for (auto& b : it->second->buffers)
            if (addr >= b.addr && (char*)addr < (char*)b.addr + b.length) return &b;
        return nullptr;
    }

    int syncSegmentCache() { return OK; }

private:
    RWSpinlock lock_;
    std::unordered_map<SegmentID, std::shared_ptr<SegmentDesc>> id_to_desc_;
    std::unordered_map<std::string, SegmentID> name_to_id_;
    std::atomic<SegmentID> next_id_;
};

}
#endif
