/*
 * ikun.c — GPU-aware Distributed Collective Runner
 *
 * LAYER 1: Implements the catalog (bijection), pool (free monoid),
 *          P2P probe (symmetric relation), and KV store (functor).
 * LAYER 2: Each function preserves structural depth from the reference form.
 */

#include "ikun.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

/* --- Error strings --- */
const char *ikun_strerror(ikun_error_t err) {
    switch (err) {
        case IKUN_OK:           return "ok";
        case IKUN_ERR_INVAL:    return "invalid argument";
        case IKUN_ERR_NOTFOUND: return "not found";
        case IKUN_ERR_EXISTS:   return "already exists";
        case IKUN_ERR_IO:       return "I/O error";
        case IKUN_ERR_TIMEOUT:  return "timeout";
        case IKUN_ERR_PEERDOWN: return "peer down";
        case IKUN_ERR_NOGPU:    return "no GPU available";
        case IKUN_ERR_NOMEM:    return "out of memory";
        case IKUN_ERR_INTERNAL: return "internal error";
        default:                return "unknown error";
    }
}

/* ================================================================
 * Catalog — segment registry
 * ================================================================ */

ikun_error_t ikun_catalog_init(ikun_catalog_t *cat) {
    if (!cat) return IKUN_ERR_INVAL;
    memset(cat, 0, sizeof(*cat));
    return IKUN_OK;
}

static int catalog_find(const ikun_catalog_t *cat,
                         const char *node_id, const char *buffer_id) {
    for (int i = 0; i < cat->count; i++) {
        if (strcmp(cat->entries[i].node_id, node_id) == 0 &&
            strcmp(cat->entries[i].buffer_id, buffer_id) == 0) {
            return i;
        }
    }
    return -1;
}

ikun_error_t ikun_catalog_register(ikun_catalog_t *cat, const ikun_segment_t *seg) {
    if (!cat || !seg) return IKUN_ERR_INVAL;
    if (cat->count >= IKUN_CATALOG_MAX) return IKUN_ERR_NOMEM;

    if (catalog_find(cat, seg->node_id, seg->buffer_id) >= 0)
        return IKUN_ERR_EXISTS;

    cat->entries[cat->count] = *seg;
    cat->count++;
    return IKUN_OK;
}

ikun_error_t ikun_catalog_lookup(const ikun_catalog_t *cat,
                                  const char *node_id, const char *buffer_id,
                                  ikun_segment_t *out) {
    if (!cat) return IKUN_ERR_INVAL;
    int idx = catalog_find(cat, node_id, buffer_id);
    if (idx < 0) return IKUN_ERR_NOTFOUND;
    if (out) *out = cat->entries[idx];
    return IKUN_OK;
}

ikun_error_t ikun_catalog_unregister(ikun_catalog_t *cat,
                                      const char *node_id, const char *buffer_id) {
    if (!cat) return IKUN_ERR_INVAL;
    int idx = catalog_find(cat, node_id, buffer_id);
    if (idx < 0) return IKUN_ERR_NOTFOUND;

    /* Swap with last entry */
    if (idx < cat->count - 1)
        cat->entries[idx] = cat->entries[cat->count - 1];
    cat->count--;
    return IKUN_OK;
}

int ikun_catalog_count(const ikun_catalog_t *cat) {
    return cat ? cat->count : 0;
}

/* ================================================================
 * Memory Pool — per-device block allocator
 * ================================================================ */

ikun_error_t ikun_pool_init(ikun_pool_t *pool, int device_id,
                             uint64_t block_size, int num_blocks) {
    if (!pool || num_blocks <= 0 || num_blocks > IKUN_POOL_MAX_BLOCKS)
        return IKUN_ERR_INVAL;

    pool->device_id = device_id;
    pool->block_size = block_size;
    pool->num_blocks = num_blocks;
    pool->free_count = num_blocks;
    memset(pool->in_use, 0, sizeof(pool->in_use));
    return IKUN_OK;
}

ikun_error_t ikun_pool_alloc(ikun_pool_t *pool, int *block_index) {
    if (!pool || !block_index) return IKUN_ERR_INVAL;
    if (pool->free_count <= 0) return IKUN_ERR_NOMEM;

    for (int i = 0; i < pool->num_blocks; i++) {
        if (!pool->in_use[i]) {
            pool->in_use[i] = 1;
            pool->free_count--;
            *block_index = i;
            return IKUN_OK;
        }
    }
    return IKUN_ERR_NOMEM; /* shouldn't reach here */
}

ikun_error_t ikun_pool_free(ikun_pool_t *pool, int block_index) {
    if (!pool) return IKUN_ERR_INVAL;
    if (block_index < 0 || block_index >= pool->num_blocks) return IKUN_ERR_INVAL;
    if (!pool->in_use[block_index]) return IKUN_ERR_INVAL; /* double free */

    pool->in_use[block_index] = 0;
    pool->free_count++;
    return IKUN_OK;
}

void ikun_pool_stats(const ikun_pool_t *pool,
                      int *total, int *used, int *free_cnt) {
    if (!pool) return;
    if (total) *total = pool->num_blocks;
    if (free_cnt) *free_cnt = pool->free_count;
    if (used) *used = pool->num_blocks - pool->free_count;
}

/* ================================================================
 * GPU P2P Probe
 * ================================================================ */

static int discover_gpus(int *device_ids, int max_gpus) {
    int count = 0;
    char path[64];
    struct stat st;

    /* Iluvatar BI-V100 devices */
    for (int i = 0; i < max_gpus && count < max_gpus; i++) {
        snprintf(path, sizeof(path), "/dev/iluvatar%d", i);
        if (stat(path, &st) == 0) {
            device_ids[count++] = i;
        }
    }

    /* Fallback: NVIDIA devices */
    if (count == 0) {
        for (int i = 0; i < max_gpus && count < max_gpus; i++) {
            snprintf(path, sizeof(path), "/dev/nvidia%d", i);
            if (stat(path, &st) == 0) {
                device_ids[count++] = i;
            }
        }
    }

    /* Fallback: CUDA_VISIBLE_DEVICES */
    if (count == 0) {
        const char *env = getenv("CUDA_VISIBLE_DEVICES");
        if (env && *env) {
            char buf[256];
            strncpy(buf, env, sizeof(buf) - 1);
            buf[sizeof(buf) - 1] = '\0';
            char *tok = strtok(buf, ",");
            while (tok && count < max_gpus) {
                int id = atoi(tok);
                device_ids[count++] = id;
                tok = strtok(NULL, ",");
            }
        }
    }

    return count;
}

/* Read NUMA node for a GPU PCI device via sysfs */
static int get_gpu_numa(int gpu_ordinal) {
    DIR *dir = opendir("/sys/bus/pci/devices");
    if (!dir) return -1;

    struct dirent *ent;
    int gpu_count = 0;
    int result = -1;

    while ((ent = readdir(dir)) != NULL) {
        if (ent->d_name[0] == '.') continue;

        char class_path[512];
        snprintf(class_path, sizeof(class_path),
                 "/sys/bus/pci/devices/%s/class", ent->d_name);

        FILE *f = fopen(class_path, "r");
        if (!f) continue;

        char cls[32] = {0};
        if (fgets(cls, sizeof(cls), f)) {
            /* Trim newline */
            cls[strcspn(cls, "\n")] = '\0';
        }
        fclose(f);

        /* VGA=0x030000, 3D=0x030200, Other=0x038000 */
        if (strcmp(cls, "0x030000") == 0 ||
            strcmp(cls, "0x030200") == 0 ||
            strcmp(cls, "0x038000") == 0) {

            if (gpu_count == gpu_ordinal) {
                char numa_path[512];
                snprintf(numa_path, sizeof(numa_path),
                         "/sys/bus/pci/devices/%s/numa_node", ent->d_name);

                FILE *nf = fopen(numa_path, "r");
                if (nf) {
                    int numa = -1;
                    if (fscanf(nf, "%d", &numa) == 1)
                        result = numa;
                    fclose(nf);
                }
                break;
            }
            gpu_count++;
        }
    }
    closedir(dir);
    return result;
}

ikun_error_t ikun_p2p_probe(ikun_p2p_cap_t *cap) {
    if (!cap) return IKUN_ERR_INVAL;
    memset(cap, 0, sizeof(*cap));

    cap->num_devices = discover_gpus(cap->device_ids, IKUN_MAX_GPUS);
    if (cap->num_devices == 0)
        return IKUN_ERR_NOGPU;

    for (int i = 0; i < cap->num_devices; i++) {
        for (int j = 0; j < cap->num_devices; j++) {
            if (i == j) {
                cap->can_access[i][j] = 1;
                continue;
            }
            /* Same NUMA or unknown → assume P2P capable (PCIe) */
            int numa_i = get_gpu_numa(i);
            int numa_j = get_gpu_numa(j);
            if (numa_i >= 0 && numa_j >= 0)
                cap->can_access[i][j] = (numa_i == numa_j) ? 1 : 0;
            else
                cap->can_access[i][j] = 1; /* conservative: assume P2P works */
        }
    }
    return IKUN_OK;
}

void ikun_p2p_print(const ikun_p2p_cap_t *cap) {
    if (!cap) return;
    printf("P2P Capability Matrix (%d GPUs):\n", cap->num_devices);
    printf("     ");
    for (int j = 0; j < cap->num_devices; j++)
        printf("GPU%-2d ", cap->device_ids[j]);
    printf("\n");

    for (int i = 0; i < cap->num_devices; i++) {
        printf("GPU%d ", cap->device_ids[i]);
        for (int j = 0; j < cap->num_devices; j++) {
            printf("  %c   ", cap->can_access[i][j] ? 'Y' : '-');
        }
        printf("\n");
    }
}

int ikun_p2p_pair_count(const ikun_p2p_cap_t *cap) {
    if (!cap) return 0;
    int count = 0;
    for (int i = 0; i < cap->num_devices; i++)
        for (int j = i + 1; j < cap->num_devices; j++)
            if (cap->can_access[i][j] && cap->can_access[j][i])
                count++;
    return count;
}

/* ================================================================
 * KV Store — local host-memory store
 * ================================================================ */

ikun_error_t ikun_kv_init(ikun_kv_store_t *store) {
    if (!store) return IKUN_ERR_INVAL;
    memset(store, 0, sizeof(*store));
    return IKUN_OK;
}

static int kv_find(const ikun_kv_store_t *store, const char *key) {
    for (int i = 0; i < IKUN_KV_MAX; i++) {
        if (store->entries[i].occupied &&
            strcmp(store->entries[i].key, key) == 0)
            return i;
    }
    return -1;
}

ikun_error_t ikun_kv_put(ikun_kv_store_t *store, const char *key,
                          const void *value, size_t length) {
    if (!store || !key || !value) return IKUN_ERR_INVAL;
    if (length > IKUN_KV_VAL_MAX) return IKUN_ERR_INVAL;
    if (strlen(key) >= IKUN_KV_KEY_MAX) return IKUN_ERR_INVAL;

    /* Check if key exists — update in place */
    int idx = kv_find(store, key);
    if (idx >= 0) {
        free(store->entries[idx].value);
        store->entries[idx].value = (uint8_t *)malloc(length);
        if (!store->entries[idx].value) return IKUN_ERR_NOMEM;
        memcpy(store->entries[idx].value, value, length);
        store->entries[idx].length = length;
        return IKUN_OK;
    }

    /* Find free slot */
    for (int i = 0; i < IKUN_KV_MAX; i++) {
        if (!store->entries[i].occupied) {
            strncpy(store->entries[i].key, key, IKUN_KV_KEY_MAX - 1);
            store->entries[i].key[IKUN_KV_KEY_MAX - 1] = '\0';
            store->entries[i].value = (uint8_t *)malloc(length);
            if (!store->entries[i].value) return IKUN_ERR_NOMEM;
            memcpy(store->entries[i].value, value, length);
            store->entries[i].length = length;
            store->entries[i].occupied = 1;
            store->count++;
            return IKUN_OK;
        }
    }
    return IKUN_ERR_NOMEM;
}

ikun_error_t ikun_kv_get(const ikun_kv_store_t *store, const char *key,
                          void *buf, size_t buf_size, size_t *out_len) {
    if (!store || !key) return IKUN_ERR_INVAL;
    int idx = kv_find(store, key);
    if (idx < 0) return IKUN_ERR_NOTFOUND;

    if (out_len) *out_len = store->entries[idx].length;
    if (buf && buf_size >= store->entries[idx].length)
        memcpy(buf, store->entries[idx].value, store->entries[idx].length);
    else if (buf)
        return IKUN_ERR_NOMEM; /* buffer too small */
    return IKUN_OK;
}

ikun_error_t ikun_kv_delete(ikun_kv_store_t *store, const char *key) {
    if (!store || !key) return IKUN_ERR_INVAL;
    int idx = kv_find(store, key);
    if (idx < 0) return IKUN_ERR_NOTFOUND;

    free(store->entries[idx].value);
    store->entries[idx].value = NULL;
    store->entries[idx].occupied = 0;
    store->count--;
    return IKUN_OK;
}

void ikun_kv_destroy(ikun_kv_store_t *store) {
    if (!store) return;
    for (int i = 0; i < IKUN_KV_MAX; i++) {
        if (store->entries[i].occupied) {
            free(store->entries[i].value);
            store->entries[i].value = NULL;
            store->entries[i].occupied = 0;
        }
    }
    store->count = 0;
}

/* ================================================================
 * Config
 * ================================================================ */

ikun_error_t ikun_config_default(ikun_config_t *cfg) {
    if (!cfg) return IKUN_ERR_INVAL;
    memset(cfg, 0, sizeof(*cfg));

    const char *nid = getenv("IKUN_NODE_ID");
    strncpy(cfg->node_id, nid ? nid : "node-0", sizeof(cfg->node_id) - 1);

    const char *addr = getenv("IKUN_LISTEN_ADDR");
    strncpy(cfg->listen_addr, addr ? addr : "0.0.0.0:19100",
            sizeof(cfg->listen_addr) - 1);

    strncpy(cfg->transport_type, "tcp", sizeof(cfg->transport_type) - 1);
    cfg->block_size = 4 * 1024 * 1024; /* 4MB */
    cfg->num_blocks = 256;

    /* Parse GPU devices */
    const char *gpuenv = getenv("CUDA_VISIBLE_DEVICES");
    if (gpuenv && *gpuenv) {
        char buf[256];
        strncpy(buf, gpuenv, sizeof(buf) - 1);
        buf[sizeof(buf) - 1] = '\0';
        char *tok = strtok(buf, ",");
        while (tok && cfg->num_gpus < IKUN_MAX_GPUS) {
            cfg->gpu_devices[cfg->num_gpus++] = atoi(tok);
            tok = strtok(NULL, ",");
        }
    } else {
        for (int i = 0; i < 8 && i < IKUN_MAX_GPUS; i++) {
            cfg->gpu_devices[i] = i;
            cfg->num_gpus = i + 1;
        }
    }
    return IKUN_OK;
}

/* ================================================================
 * Runner (top-level engine)
 * ================================================================ */

ikun_error_t ikun_runner_init(ikun_runner_t *runner, const ikun_config_t *cfg) {
    if (!runner || !cfg) return IKUN_ERR_INVAL;
    memset(runner, 0, sizeof(*runner));

    runner->config = *cfg;
    runner->state = IKUN_STATE_INIT;

    ikun_catalog_init(&runner->catalog);
    ikun_kv_init(&runner->kv);

    /* Init pools */
    for (int i = 0; i < cfg->num_gpus; i++) {
        ikun_pool_init(&runner->pools[i], cfg->gpu_devices[i],
                       cfg->block_size, cfg->num_blocks);
    }

    /* Probe P2P */
    if (ikun_p2p_probe(&runner->p2p_cap) == IKUN_OK)
        runner->has_p2p = 1;

    return IKUN_OK;
}

ikun_error_t ikun_runner_start(ikun_runner_t *runner) {
    if (!runner) return IKUN_ERR_INVAL;
    if (runner->state != IKUN_STATE_INIT) return IKUN_ERR_INVAL;
    runner->state = IKUN_STATE_RUNNING;
    return IKUN_OK;
}

ikun_error_t ikun_runner_put(ikun_runner_t *runner, const char *key,
                              const void *value, size_t length) {
    if (!runner || runner->state != IKUN_STATE_RUNNING) return IKUN_ERR_INVAL;

    ikun_error_t err = ikun_kv_put(&runner->kv, key, value, length);
    if (err != IKUN_OK) return err;

    /* Register in catalog */
    ikun_catalog_unregister(&runner->catalog, runner->config.node_id, key);

    ikun_segment_t seg;
    memset(&seg, 0, sizeof(seg));
    snprintf(seg.node_id, sizeof(seg.node_id), "%s", runner->config.node_id);
    snprintf(seg.buffer_id, sizeof(seg.buffer_id), "%s", key);
    seg.offset = 0;
    seg.length = (uint64_t)length;
    seg.gpu_index = -1;
    snprintf(seg.transport, sizeof(seg.transport), "%s", runner->config.transport_type);

    return ikun_catalog_register(&runner->catalog, &seg);
}

ikun_error_t ikun_runner_get(ikun_runner_t *runner, const char *key,
                              void *buf, size_t buf_size, size_t *out_len) {
    if (!runner || runner->state != IKUN_STATE_RUNNING) return IKUN_ERR_INVAL;
    return ikun_kv_get(&runner->kv, key, buf, buf_size, out_len);
}

ikun_error_t ikun_runner_delete(ikun_runner_t *runner, const char *key) {
    if (!runner) return IKUN_ERR_INVAL;
    ikun_kv_delete(&runner->kv, key);
    ikun_catalog_unregister(&runner->catalog, runner->config.node_id, key);
    return IKUN_OK;
}

ikun_error_t ikun_runner_stop(ikun_runner_t *runner) {
    if (!runner) return IKUN_ERR_INVAL;
    if (runner->state != IKUN_STATE_RUNNING) return IKUN_OK;
    runner->state = IKUN_STATE_STOPPED;
    ikun_kv_destroy(&runner->kv);
    return IKUN_OK;
}

void ikun_runner_print_stats(const ikun_runner_t *runner) {
    if (!runner) return;
    printf("Runner stats:\n");
    printf("  State:       %d (running=1)\n", runner->state);
    printf("  Keys:        %d\n", runner->kv.count);
    printf("  GPUs:        %d\n", runner->config.num_gpus);
    printf("  P2P pairs:   %d\n",
           runner->has_p2p ? ikun_p2p_pair_count(&runner->p2p_cap) : 0);
    printf("  Catalog:     %d segments\n",
           ikun_catalog_count(&runner->catalog));

    for (int i = 0; i < runner->config.num_gpus; i++) {
        int total, used, fr;
        ikun_pool_stats(&runner->pools[i], &total, &used, &fr);
        printf("  GPU%d pool:   total=%d used=%d free=%d\n",
               runner->config.gpu_devices[i], total, used, fr);
    }
}

