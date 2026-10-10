#ifndef INFCCL_TRANSFER_ENGINE_C_H_
#define INFCCL_TRANSFER_ENGINE_C_H_
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef void* infccl_engine_t;
typedef void* infccl_transport_t;
typedef uint64_t infccl_segment_id_t;
typedef uint64_t infccl_batch_id_t;

#define INFCCL_INVALID_BATCH UINT64_MAX
#define INFCCL_OPCODE_READ  0
#define INFCCL_OPCODE_WRITE 1
#define INFCCL_STATUS_WAITING   0
#define INFCCL_STATUS_PENDING   1
#define INFCCL_STATUS_COMPLETED 4
#define INFCCL_STATUS_TIMEOUT   5
#define INFCCL_STATUS_FAILED    6

struct infccl_transfer_request {
    int opcode;
    void* source;
    infccl_segment_id_t target_id;
    uint64_t target_offset;
    uint64_t length;
};

struct infccl_transfer_status {
    int status;
    uint64_t transferred_bytes;
};

struct infccl_bi_v100_caps {
    int warp_size;
    int sm_count;
    int fp64_works;
    float hbm_bw_gbps;
    float peer_bw_gbps;
    float peer_latency_us;
    int peer_memcpy_needs_src_device;
    int peer_memcpy_needs_blocking_stream;
};

infccl_engine_t infccl_create_engine(int ndev, const int* devlist);
int infccl_init_engine(infccl_engine_t engine, const char* server_name);
infccl_transport_t infccl_install_transport(infccl_engine_t engine, const char* proto);
void infccl_destroy_engine(infccl_engine_t engine);
infccl_segment_id_t infccl_open_segment(infccl_engine_t engine, const char* name);
int infccl_register_memory(infccl_engine_t engine, void* addr, size_t length, const char* location);
int infccl_unregister_memory(infccl_engine_t engine, void* addr);
infccl_batch_id_t infccl_alloc_batch(infccl_transport_t xport, size_t batch_size);
int infccl_submit_transfer(infccl_transport_t xport, infccl_batch_id_t batch_id,
    struct infccl_transfer_request* entries, size_t count);
int infccl_submit_peer_transfer(infccl_transport_t xport, infccl_batch_id_t batch_id,
    void* src, int src_gpu, void* dst, int dst_gpu, size_t length);
int infccl_wait_batch(infccl_transport_t xport, infccl_batch_id_t batch_id, int timeout_ms);
int infccl_get_transfer_status(infccl_transport_t xport, infccl_batch_id_t batch_id,
    size_t task_id, struct infccl_transfer_status* status);
int infccl_free_batch(infccl_transport_t xport, infccl_batch_id_t batch_id);
int infccl_probe_caps(int gpu, struct infccl_bi_v100_caps* out);
void infccl_print_caps(int gpu);

#ifdef __cplusplus
}
#endif
#endif
