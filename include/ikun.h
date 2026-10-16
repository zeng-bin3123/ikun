#ifndef IKUN_H
#define IKUN_H

/*
 * ikun.h — GPU-aware Distributed Collective Runner
 *
 * LAYER 2 mapping:
 *   prjname  → ikun
 *   Obj      → collective runner components
 *   error.h  → ikun_error_t
 *   common.h → segment_desc_t, pool_t, catalog_t
 *   transport → ikun_transport_t
 */

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --- Error codes --- */
typedef enum {
    IKUN_OK            =  0,
    IKUN_ERR_INVAL     = -1,
    IKUN_ERR_NOTFOUND  = -2,
    IKUN_ERR_EXISTS    = -3,
    IKUN_ERR_IO        = -4,
    IKUN_ERR_TIMEOUT   = -5,
    IKUN_ERR_PEERDOWN  = -6,
    IKUN_ERR_NOGPU     = -7,
    IKUN_ERR_NOMEM     = -8,
    IKUN_ERR_INTERNAL  = -99
} ikun_error_t;

const char *ikun_strerror(ikun_error_t err);

/* --- Segment Descriptor (catalog entry) --- */
typedef struct {
    char     node_id[64];
    char     buffer_id[128];
    uint64_t offset;
    uint64_t length;
    int      gpu_index;    /* -1 for host memory */
    char     transport[16]; /* "tcp" */
} ikun_segment_t;

/* --- Catalog (segment registry) --- */
#define IKUN_CATALOG_MAX 4096

typedef struct {
    ikun_segment_t entries[IKUN_CATALOG_MAX];
    int count;
} ikun_catalog_t;

ikun_error_t ikun_catalog_init(ikun_catalog_t *cat);
ikun_error_t ikun_catalog_register(ikun_catalog_t *cat, const ikun_segment_t *seg);
ikun_error_t ikun_catalog_lookup(const ikun_catalog_t *cat,
                                  const char *node_id, const char *buffer_id,
                                  ikun_segment_t *out);
ikun_error_t ikun_catalog_unregister(ikun_catalog_t *cat,
                                      const char *node_id, const char *buffer_id);
int ikun_catalog_count(const ikun_catalog_t *cat);

/* --- Memory Pool (per-GPU block allocator) --- */
#define IKUN_POOL_MAX_BLOCKS 1024

typedef struct {
    int      device_id;
    uint64_t block_size;
    int      num_blocks;
    int      in_use[IKUN_POOL_MAX_BLOCKS];  /* 0=free, 1=allocated */
    int      free_count;
} ikun_pool_t;

ikun_error_t ikun_pool_init(ikun_pool_t *pool, int device_id,
                             uint64_t block_size, int num_blocks);
ikun_error_t ikun_pool_alloc(ikun_pool_t *pool, int *block_index);
ikun_error_t ikun_pool_free(ikun_pool_t *pool, int block_index);
void         ikun_pool_stats(const ikun_pool_t *pool,
                              int *total, int *used, int *free_cnt);

/* --- GPU P2P Probe --- */
#define IKUN_MAX_GPUS 16

typedef struct {
    int num_devices;
    int can_access[IKUN_MAX_GPUS][IKUN_MAX_GPUS]; /* 1=P2P ok */
    int device_ids[IKUN_MAX_GPUS];
} ikun_p2p_cap_t;

ikun_error_t ikun_p2p_probe(ikun_p2p_cap_t *cap);
void         ikun_p2p_print(const ikun_p2p_cap_t *cap);
int          ikun_p2p_pair_count(const ikun_p2p_cap_t *cap);

/* --- KV Store (local host-memory store) --- */
#define IKUN_KV_MAX 4096
#define IKUN_KV_KEY_MAX 128
#define IKUN_KV_VAL_MAX (1024*1024)  /* 1MB max value */

typedef struct {
    char     key[IKUN_KV_KEY_MAX];
    uint8_t *value;
    size_t   length;
    int      occupied;
} ikun_kv_entry_t;

typedef struct {
    ikun_kv_entry_t entries[IKUN_KV_MAX];
    int count;
} ikun_kv_store_t;

ikun_error_t ikun_kv_init(ikun_kv_store_t *store);
ikun_error_t ikun_kv_put(ikun_kv_store_t *store, const char *key,
                          const void *value, size_t length);
ikun_error_t ikun_kv_get(const ikun_kv_store_t *store, const char *key,
                          void *buf, size_t buf_size, size_t *out_len);
ikun_error_t ikun_kv_delete(ikun_kv_store_t *store, const char *key);
void         ikun_kv_destroy(ikun_kv_store_t *store);

/* --- Runner (top-level engine) --- */
typedef enum {
    IKUN_STATE_INIT    = 0,
    IKUN_STATE_RUNNING = 1,
    IKUN_STATE_STOPPED = 2
} ikun_state_t;

typedef struct {
    char     node_id[64];
    char     listen_addr[64];
    char     transport_type[16];
    uint64_t block_size;
    int      num_blocks;
    int      gpu_devices[IKUN_MAX_GPUS];
    int      num_gpus;
} ikun_config_t;

typedef struct {
    ikun_config_t  config;
    ikun_state_t   state;
    ikun_catalog_t catalog;
    ikun_pool_t    pools[IKUN_MAX_GPUS];
    ikun_p2p_cap_t p2p_cap;
    ikun_kv_store_t kv;
    int            has_p2p;
} ikun_runner_t;

ikun_error_t ikun_config_default(ikun_config_t *cfg);
ikun_error_t ikun_runner_init(ikun_runner_t *runner, const ikun_config_t *cfg);
ikun_error_t ikun_runner_start(ikun_runner_t *runner);
ikun_error_t ikun_runner_put(ikun_runner_t *runner, const char *key,
                              const void *value, size_t length);
ikun_error_t ikun_runner_get(ikun_runner_t *runner, const char *key,
                              void *buf, size_t buf_size, size_t *out_len);
ikun_error_t ikun_runner_delete(ikun_runner_t *runner, const char *key);
ikun_error_t ikun_runner_stop(ikun_runner_t *runner);
void         ikun_runner_print_stats(const ikun_runner_t *runner);

#ifdef __cplusplus
}
#endif

#endif /* IKUN_H */

