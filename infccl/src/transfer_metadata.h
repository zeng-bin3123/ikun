#ifndef INFCCL_TRANSFER_METADATA_H_
#define INFCCL_TRANSFER_METADATA_H_

#include "common.h"
#include <cuda_runtime.h>
#include <algorithm>

namespace infccl {

using SegmentID = uint64_t;
const static SegmentID LOCAL_SEGMENT_ID = 0;

struct BufferDesc {
    void* addr;
    uint64_t length;
    int gpu;
    std::string location;
    int access_flags;
    uint64_t registered_at_us;
};

struct DeviceDesc {
    int dev_id;
    char name[64];
    char pci_bus_id[16];
    int numa_node;
    size_t total_mem;
    size_t free_mem;
};

struct SegmentDesc {
    std::string name;
    std::string protocol;
    std::vector<BufferDesc> buffers;
    std::vector<DeviceDesc> devices;
    int64_t created_at_us;
    int64_t updated_at_us;

    const BufferDesc* findBuffer(void* addr) const {
        for (auto& b : buffers)
            if (addr >= b.addr && (char*)addr < (char*)b.addr + b.length)
                return &b;
        return nullptr;
    }

    const BufferDesc* findBufferByGpu(int gpu) const {
        for (auto& b : buffers)
            if (b.gpu == gpu) return &b;
        return nullptr;
    }

    int findDeviceIndex(int dev_id) const {
        for (int i = 0; i < (int)devices.size(); i++)
            if (devices[i].dev_id == dev_id) return i;
        return -1;
    }

    size_t totalRegisteredBytes() const {
        size_t n = 0;
        for (auto& b : buffers) n += b.length;
        return n;
    }

    int bufferCount() const { return (int)buffers.size(); }
    int deviceCount() const { return (int)devices.size(); }
};

class BufferRangeIndex {
public:
    struct Entry {
        uint64_t start;
        uint64_t end;
        int buf_idx;
    };

    void rebuild(const std::vector<BufferDesc>& buffers) {
        entries_.clear();
        entries_.reserve(buffers.size());
        for (int i = 0; i < (int)buffers.size(); i++) {
            uint64_t s = (uint64_t)buffers[i].addr;
            uint64_t e = s + buffers[i].length;
            entries_.push_back({s, e, i});
        }
        std::sort(entries_.begin(), entries_.end(),
            [](const Entry& a, const Entry& b) { return a.start < b.start; });
    }

    int find(void* addr) const {
        uint64_t a = (uint64_t)addr;
        int lo = 0, hi = (int)entries_.size() - 1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if (a < entries_[mid].start) hi = mid - 1;
            else if (a >= entries_[mid].end) lo = mid + 1;
            else return entries_[mid].buf_idx;
        }
        return -1;
    }

    int size() const { return (int)entries_.size(); }

private:
    std::vector<Entry> entries_;
};

class TransferMetadata {
public:
    TransferMetadata() : next_id_(1) {}

    SegmentID addLocalSegment(const std::string& name, std::shared_ptr<SegmentDesc> desc) {
        RWSpinlock::WriteGuard guard(lock_);
        SegmentID id = next_id_++;
        desc->created_at_us = now_us();
        desc->updated_at_us = desc->created_at_us;
        id_to_desc_[id] = desc;
        name_to_id_[name] = id;
        return id;
    }

    int removeSegment(const std::string& name) {
        RWSpinlock::WriteGuard guard(lock_);
        auto it = name_to_id_.find(name);
        if (it == name_to_id_.end()) return ERR_NOT_FOUND;
        SegmentID id = it->second;
        id_to_desc_.erase(id);
        name_to_id_.erase(it);
        range_index_.erase(id);
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
        auto& desc = it->second;
        for (auto& existing : desc->buffers)
            if (memOverlap(existing.addr, existing.length, buf.addr, buf.length))
                return ERR_OVERLAP;
        BufferDesc b = buf;
        b.registered_at_us = now_us();
        desc->buffers.push_back(b);
        desc->updated_at_us = now_us();
        rebuildIndex(seg_id, *desc);
        return OK;
    }

    int addLocalMemoryBufferBatch(SegmentID seg_id, const std::vector<BufferDesc>& bufs) {
        RWSpinlock::WriteGuard guard(lock_);
        auto it = id_to_desc_.find(seg_id);
        if (it == id_to_desc_.end()) return ERR_NOT_FOUND;
        auto& desc = it->second;
        int64_t t = now_us();
        for (auto& buf : bufs) {
            for (auto& existing : desc->buffers)
                if (memOverlap(existing.addr, existing.length, buf.addr, buf.length))
                    return ERR_OVERLAP;
            BufferDesc b = buf;
            b.registered_at_us = t;
            desc->buffers.push_back(b);
        }
        desc->updated_at_us = t;
        rebuildIndex(seg_id, *desc);
        return OK;
    }

    int removeLocalMemoryBuffer(SegmentID seg_id, void* addr) {
        RWSpinlock::WriteGuard guard(lock_);
        auto it = id_to_desc_.find(seg_id);
        if (it == id_to_desc_.end()) return ERR_NOT_FOUND;
        auto& bufs = it->second->buffers;
        for (auto bit = bufs.begin(); bit != bufs.end(); ++bit) {
            if (bit->addr == addr) {
                bufs.erase(bit);
                it->second->updated_at_us = now_us();
                rebuildIndex(seg_id, *it->second);
                return OK;
            }
        }
        return ERR_NOT_FOUND;
    }

    const BufferDesc* findBufferByAddr(SegmentID seg_id, void* addr) {
        RWSpinlock::ReadGuard guard(lock_);
        auto it = id_to_desc_.find(seg_id);
        if (it == id_to_desc_.end()) return nullptr;
        auto ri = range_index_.find(seg_id);
        if (ri != range_index_.end()) {
            int idx = ri->second.find(addr);
            if (idx >= 0 && idx < (int)it->second->buffers.size())
                return &it->second->buffers[idx];
        }
        return it->second->findBuffer(addr);
    }

    int findGpuForAddr(SegmentID seg_id, void* addr) {
        auto* b = findBufferByAddr(seg_id, addr);
        return b ? b->gpu : -1;
    }

    int segmentCount() const {
        RWSpinlock::ReadGuard guard(lock_);
        return (int)id_to_desc_.size();
    }

    int totalBufferCount() const {
        RWSpinlock::ReadGuard guard(lock_);
        int n = 0;
        for (auto& p : id_to_desc_) n += p.second->bufferCount();
        return n;
    }

    size_t totalRegisteredBytes() const {
        RWSpinlock::ReadGuard guard(lock_);
        size_t n = 0;
        for (auto& p : id_to_desc_) n += p.second->totalRegisteredBytes();
        return n;
    }

    void dump(FILE* out = stdout) const {
        RWSpinlock::ReadGuard guard(lock_);
        fprintf(out, "TransferMetadata: %d segments, %d buffers\n",
            (int)id_to_desc_.size(), totalBufferCount());
        for (auto& p : id_to_desc_) {
            auto& desc = p.second;
            fprintf(out, "  segment %lu '%s' proto=%s devs=%d bufs=%d bytes=%zu\n",
                p.first, desc->name.c_str(), desc->protocol.c_str(),
                desc->deviceCount(), desc->bufferCount(),
                desc->totalRegisteredBytes());
            for (auto& b : desc->buffers) {
                fprintf(out, "    buf %p len=%lu gpu=%d loc=%s\n",
                    b.addr, b.length, b.gpu, b.location.c_str());
            }
        }
    }

private:
    void rebuildIndex(SegmentID seg_id, const SegmentDesc& desc) {
        range_index_[seg_id].rebuild(desc.buffers);
    }

    mutable RWSpinlock lock_;
    std::unordered_map<SegmentID, std::shared_ptr<SegmentDesc>> id_to_desc_;
    std::unordered_map<std::string, SegmentID> name_to_id_;
    std::unordered_map<SegmentID, BufferRangeIndex> range_index_;
    std::atomic<SegmentID> next_id_;
};

}
#endif
