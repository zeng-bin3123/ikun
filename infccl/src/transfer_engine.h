#ifndef INFCCL_TRANSFER_ENGINE_H_
#define INFCCL_TRANSFER_ENGINE_H_

#include "transport.h"
#include "peer_transport.h"

namespace infccl {

class TransferEngine {
public:
    TransferEngine(std::shared_ptr<TransferMetadata> meta) : metadata_(meta) {}

    ~TransferEngine() { freeEngine(); }

    int init(const std::string& server_name, int ndev, const int* devlist) {
        local_server_name_ = server_name;
        ndev_ = ndev;
        for (int i = 0; i < ndev; i++) devlist_[i] = devlist ? devlist[i] : i;
        return OK;
    }

    int freeEngine() {
        for (auto* xport : installed_transports_) delete xport;
        installed_transports_.clear();
        local_memory_regions_.clear();
        return OK;
    }

    Transport* installOrGetTransport(const char* proto, void** args) {
        for (auto* x : installed_transports_)
            if (strcmp(x->getName(), proto) == 0) return x;

        Transport* xport = nullptr;
        if (strcmp(proto, "peer") == 0) xport = new PeerTransport();
        if (!xport) return nullptr;

        void* install_args[2] = { devlist_, &ndev_ };
        if (args) { install_args[0] = args[0]; install_args[1] = args[1]; }

        if (xport->install(local_server_name_, metadata_, install_args) != OK) {
            delete xport; return nullptr;
        }

        for (auto& mem : local_memory_regions_) {
            xport->registerLocalMemory(mem.addr, mem.length, mem.location, mem.remote_accessible);
        }

        installed_transports_.push_back(xport);
        return xport;
    }

    int uninstallTransport(const char* proto) {
        for (auto it = installed_transports_.begin(); it != installed_transports_.end(); ++it) {
            if (strcmp((*it)->getName(), proto) == 0) {
                delete *it; installed_transports_.erase(it); return OK;
            }
        }
        return ERR_NOT_FOUND;
    }

    SegmentID openSegment(const std::string& name) {
        return metadata_->getSegmentID(name);
    }

    int registerLocalMemory(void* addr, size_t length,
        const std::string& location, bool remote_accessible = true) {
        for (auto& r : local_memory_regions_)
            if (memOverlap(r.addr, r.length, addr, length)) return ERR_OVERLAP;
        for (auto* xport : installed_transports_) {
            int rc = xport->registerLocalMemory(addr, length, location, remote_accessible);
            if (rc != OK) return rc;
        }
        local_memory_regions_.push_back({addr, length, location, remote_accessible});
        return OK;
    }

    int unregisterLocalMemory(void* addr) {
        for (auto it = local_memory_regions_.begin(); it != local_memory_regions_.end(); ++it) {
            if (it->addr == addr) {
                for (auto* xport : installed_transports_)
                    xport->unregisterLocalMemory(addr);
                local_memory_regions_.erase(it);
                return OK;
            }
        }
        return ERR_NOT_FOUND;
    }

    PeerTransport* getPeerTransport() {
        for (auto* x : installed_transports_)
            if (strcmp(x->getName(), "peer") == 0) return (PeerTransport*)x;
        return nullptr;
    }

    std::shared_ptr<TransferMetadata>& meta() { return metadata_; }

private:
    struct MemoryRegion {
        void* addr;
        size_t length;
        std::string location;
        bool remote_accessible;
    };

    std::vector<Transport*> installed_transports_;
    std::string local_server_name_;
    std::shared_ptr<TransferMetadata> metadata_;
    std::vector<MemoryRegion> local_memory_regions_;
    int ndev_ = 0;
    int devlist_[INFCCL_MAX_DEVS] = {};
};

}
#endif
