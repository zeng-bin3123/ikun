/*
 * main.c — ikun smoke test and GPU P2P probe
 */

#include "ikun.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int test_catalog(void) {
    printf("--- Catalog CRUD ---\n");
    ikun_catalog_t cat;
    ikun_catalog_init(&cat);

    ikun_segment_t seg = {0};
    strncpy(seg.node_id, "node-0", sizeof(seg.node_id) - 1);
    strncpy(seg.buffer_id, "buf-1", sizeof(seg.buffer_id) - 1);
    seg.offset = 0;
    seg.length = 4096;
    seg.gpu_index = 0;
    strncpy(seg.transport, "tcp", sizeof(seg.transport) - 1);

    if (ikun_catalog_register(&cat, &seg) != IKUN_OK) {
        fprintf(stderr, "  FAIL: register\n"); return 1;
    }
    if (ikun_catalog_count(&cat) != 1) {
        fprintf(stderr, "  FAIL: count != 1\n"); return 1;
    }
    /* Duplicate should fail */
    if (ikun_catalog_register(&cat, &seg) != IKUN_ERR_EXISTS) {
        fprintf(stderr, "  FAIL: duplicate register\n"); return 1;
    }

    ikun_segment_t out;
    if (ikun_catalog_lookup(&cat, "node-0", "buf-1", &out) != IKUN_OK) {
        fprintf(stderr, "  FAIL: lookup\n"); return 1;
    }
    if (out.length != 4096) {
        fprintf(stderr, "  FAIL: wrong length\n"); return 1;
    }

    if (ikun_catalog_unregister(&cat, "node-0", "buf-1") != IKUN_OK) {
        fprintf(stderr, "  FAIL: unregister\n"); return 1;
    }
    if (ikun_catalog_count(&cat) != 0) {
        fprintf(stderr, "  FAIL: count != 0 after unregister\n"); return 1;
    }
    printf("  PASS\n");
    return 0;
}

static int test_pool(void) {
    printf("--- Pool Alloc/Free ---\n");
    ikun_pool_t pool;
    ikun_pool_init(&pool, 0, 4096, 4);

    int total, used, fr;
    ikun_pool_stats(&pool, &total, &used, &fr);
    if (total != 4 || used != 0 || fr != 4) {
        fprintf(stderr, "  FAIL: initial stats\n"); return 1;
    }

    int blocks[4];
    for (int i = 0; i < 4; i++) {
        if (ikun_pool_alloc(&pool, &blocks[i]) != IKUN_OK) {
            fprintf(stderr, "  FAIL: alloc %d\n", i); return 1;
        }
    }

    /* Exhausted */
    int dummy;
    if (ikun_pool_alloc(&pool, &dummy) != IKUN_ERR_NOMEM) {
        fprintf(stderr, "  FAIL: should be exhausted\n"); return 1;
    }

    if (ikun_pool_free(&pool, blocks[0]) != IKUN_OK) {
        fprintf(stderr, "  FAIL: free\n"); return 1;
    }

    /* Double free */
    if (ikun_pool_free(&pool, blocks[0]) != IKUN_ERR_INVAL) {
        fprintf(stderr, "  FAIL: double free should fail\n"); return 1;
    }

    /* Re-alloc */
    if (ikun_pool_alloc(&pool, &dummy) != IKUN_OK) {
        fprintf(stderr, "  FAIL: re-alloc\n"); return 1;
    }

    printf("  PASS\n");
    return 0;
}

static int test_kv(void) {
    printf("--- KV Store ---\n");
    ikun_kv_store_t store;
    ikun_kv_init(&store);

    const char *val = "hello ikun";
    if (ikun_kv_put(&store, "k1", val, strlen(val)) != IKUN_OK) {
        fprintf(stderr, "  FAIL: put\n"); return 1;
    }

    char buf[256];
    size_t len;
    if (ikun_kv_get(&store, "k1", buf, sizeof(buf), &len) != IKUN_OK) {
        fprintf(stderr, "  FAIL: get\n"); return 1;
    }
    if (len != strlen(val) || memcmp(buf, val, len) != 0) {
        fprintf(stderr, "  FAIL: value mismatch\n"); return 1;
    }

    if (ikun_kv_delete(&store, "k1") != IKUN_OK) {
        fprintf(stderr, "  FAIL: delete\n"); return 1;
    }
    if (ikun_kv_get(&store, "k1", buf, sizeof(buf), &len) != IKUN_ERR_NOTFOUND) {
        fprintf(stderr, "  FAIL: get after delete should fail\n"); return 1;
    }

    ikun_kv_destroy(&store);
    printf("  PASS\n");
    return 0;
}

int main(void) {
    printf("=== ikun: GPU-aware Distributed Collective Runner (C) ===\n\n");

    /* Config */
    ikun_config_t cfg;
    ikun_config_default(&cfg);
    printf("Node:      %s\n", cfg.node_id);
    printf("Listen:    %s\n", cfg.listen_addr);
    printf("Transport: %s\n", cfg.transport_type);
    printf("BlockSize: %lu bytes\n", (unsigned long)cfg.block_size);
    printf("NumBlocks: %d per device\n", cfg.num_blocks);
    printf("GPUs:      [");
    for (int i = 0; i < cfg.num_gpus; i++)
        printf("%s%d", i ? "," : "", cfg.gpu_devices[i]);
    printf("]\n\n");

    /* Unit tests */
    int failures = 0;
    failures += test_catalog();
    failures += test_pool();
    failures += test_kv();
    printf("\n");

    /* GPU P2P Probe */
    printf("--- GPU P2P Probe ---\n");
    ikun_p2p_cap_t p2p;
    ikun_error_t err = ikun_p2p_probe(&p2p);
    if (err != IKUN_OK) {
        printf("GPU probe failed: %s\n", ikun_strerror(err));
        printf("Continuing with CPU-only mode...\n");
    } else {
        ikun_p2p_print(&p2p);
        int pairs = ikun_p2p_pair_count(&p2p);
        printf("P2P-capable pairs: %d\n", pairs);
        for (int i = 0; i < p2p.num_devices; i++)
            for (int j = i + 1; j < p2p.num_devices; j++)
                if (p2p.can_access[i][j] && p2p.can_access[j][i])
                    printf("  GPU%d <-> GPU%d\n",
                           p2p.device_ids[i], p2p.device_ids[j]);
    }
    printf("\n");

    /* Runner smoke test */
    printf("--- Collective Runner ---\n");
    printf("Running KV smoke test...\n");

    ikun_runner_t runner;
    if (ikun_runner_init(&runner, &cfg) != IKUN_OK) {
        fprintf(stderr, "FAIL: runner init\n"); return 1;
    }
    if (ikun_runner_start(&runner) != IKUN_OK) {
        fprintf(stderr, "FAIL: runner start\n"); return 1;
    }

    /* Put/Get single key */
    const char *test_val = "hello from ikun collective runner";
    if (ikun_runner_put(&runner, "test-layer-0", test_val, strlen(test_val)) != IKUN_OK) {
        fprintf(stderr, "FAIL: put\n"); return 1;
    }
    char rbuf[256];
    size_t rlen;
    if (ikun_runner_get(&runner, "test-layer-0", rbuf, sizeof(rbuf), &rlen) != IKUN_OK) {
        fprintf(stderr, "FAIL: get\n"); return 1;
    }
    if (rlen != strlen(test_val) || memcmp(rbuf, test_val, rlen) != 0) {
        fprintf(stderr, "FAIL: value mismatch\n"); return 1;
    }
    printf("  Put(\"test-layer-0\") -> Get -> OK\n");

    /* Bulk put */
    for (int i = 0; i < 100; i++) {
        char key[32];
        unsigned char vbuf[1024];
        snprintf(key, sizeof(key), "block-%04d", i);
        for (int j = 0; j < 1024; j++) vbuf[j] = (unsigned char)(i + j);
        if (ikun_runner_put(&runner, key, vbuf, 1024) != IKUN_OK) {
            fprintf(stderr, "FAIL: put block-%04d\n", i); return 1;
        }
    }
    printf("  Put 100 x 1KB blocks -> OK\n");

    /* Bulk verify */
    for (int i = 0; i < 100; i++) {
        char key[32];
        unsigned char vbuf[1024];
        size_t vlen;
        snprintf(key, sizeof(key), "block-%04d", i);
        if (ikun_runner_get(&runner, key, vbuf, sizeof(vbuf), &vlen) != IKUN_OK) {
            fprintf(stderr, "FAIL: get block-%04d\n", i); return 1;
        }
        if (vlen != 1024) {
            fprintf(stderr, "FAIL: block-%04d wrong size %zu\n", i, vlen); return 1;
        }
    }
    printf("  Get 100 x 1KB blocks -> OK\n");

    /* Delete */
    ikun_runner_delete(&runner, "test-layer-0");
    if (ikun_runner_get(&runner, "test-layer-0", rbuf, sizeof(rbuf), &rlen) != IKUN_ERR_NOTFOUND) {
        fprintf(stderr, "FAIL: get after delete should fail\n"); return 1;
    }
    printf("  Delete(\"test-layer-0\") -> Get fails -> OK\n");

    /* Stats */
    printf("\n");
    ikun_runner_print_stats(&runner);

    ikun_runner_stop(&runner);

    printf("\n");
    if (failures > 0) {
        printf("=== %d test(s) FAILED ===\n", failures);
        return 1;
    }
    printf("=== All tests passed ===\n");
    return 0;
}

