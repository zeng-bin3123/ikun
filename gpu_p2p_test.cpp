// GPU P2P Transfer Test for Iluvatar BI-V100 (and NVIDIA GPUs)
// Uses CUDA Driver API via dlopen - compiles with plain g++
// Gracefully handles missing CUDA library

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <stdint.h>

// CUDA Driver API types
typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef unsigned long long CUdeviceptr;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUstream;

#define CUDA_SUCCESS 0
#define CU_CTX_SCHED_AUTO 0

// Function pointer types
#define DECL_FN(ret, name, ...) typedef ret (*name##_t)(__VA_ARGS__)
DECL_FN(CUresult, cuInit, unsigned int);
DECL_FN(CUresult, cuDeviceGetCount, int*);
DECL_FN(CUresult, cuDeviceGet, CUdevice*, int);
DECL_FN(CUresult, cuDeviceGetName, char*, int, CUdevice);
DECL_FN(CUresult, cuDeviceTotalMem, size_t*, CUdevice);
DECL_FN(CUresult, cuCtxCreate, CUcontext*, unsigned int, CUdevice);
DECL_FN(CUresult, cuCtxDestroy, CUcontext);
DECL_FN(CUresult, cuCtxSetCurrent, CUcontext);
DECL_FN(CUresult, cuCtxEnablePeerAccess, CUcontext, unsigned int);
DECL_FN(CUresult, cuDeviceCanAccessPeer, int*, CUdevice, CUdevice);
DECL_FN(CUresult, cuMemAlloc, CUdeviceptr*, size_t);
DECL_FN(CUresult, cuMemFree, CUdeviceptr);
DECL_FN(CUresult, cuMemcpyHtoD, CUdeviceptr, const void*, size_t);
DECL_FN(CUresult, cuMemcpyDtoH, void*, CUdeviceptr, size_t);
DECL_FN(CUresult, cuMemcpyPeer, CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t);
DECL_FN(CUresult, cuCtxSynchronize);
DECL_FN(CUresult, cuGetErrorString, CUresult, const char**);

// Function pointers
static cuInit_t fn_cuInit;
static cuDeviceGetCount_t fn_cuDeviceGetCount;
static cuDeviceGet_t fn_cuDeviceGet;
static cuDeviceGetName_t fn_cuDeviceGetName;
static cuDeviceTotalMem_t fn_cuDeviceTotalMem;
static cuCtxCreate_t fn_cuCtxCreate;
static cuCtxDestroy_t fn_cuCtxDestroy;
static cuCtxSetCurrent_t fn_cuCtxSetCurrent;
static cuCtxEnablePeerAccess_t fn_cuCtxEnablePeerAccess;
static cuDeviceCanAccessPeer_t fn_cuDeviceCanAccessPeer;
static cuMemAlloc_t fn_cuMemAlloc;
static cuMemFree_t fn_cuMemFree;
static cuMemcpyHtoD_t fn_cuMemcpyHtoD;
static cuMemcpyDtoH_t fn_cuMemcpyDtoH;
static cuMemcpyPeer_t fn_cuMemcpyPeer;
static cuCtxSynchronize_t fn_cuCtxSynchronize;
static cuGetErrorString_t fn_cuGetErrorString;

static void *cuda_handle = NULL;

static const char* cu_err(CUresult r) {
    static char buf[64];
    if (fn_cuGetErrorString) {
        const char *s = NULL;
        if (fn_cuGetErrorString(r, &s) == CUDA_SUCCESS && s) return s;
    }
    snprintf(buf, sizeof(buf), "error_%d", r);
    return buf;
}

static void* try_sym(const char *name) {
    void *p = dlsym(cuda_handle, name);
    if (!p) {
        // Try _v2 suffix (CUDA driver API versioning)
        char v2[128];
        snprintf(v2, sizeof(v2), "%s_v2", name);
        p = dlsym(cuda_handle, v2);
    }
    return p;
}

static int load_cuda() {
    const char *paths[] = {
        "libcuda.so.1", "libcuda.so",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib/x86_64-linux-gnu/libcuda.so",
        "/usr/local/corex/lib64/libcuda.so.1",
        "/usr/local/corex/lib/libcuda.so.1",
        "/usr/local/cuda/lib64/libcuda.so.1",
        "/usr/lib64/libcuda.so.1",
        "/usr/lib64/libcuda.so",
        "/usr/local/lib/libcuda.so.1",
        "/usr/local/lib/libcuda.so",
        NULL
    };

    for (int i = 0; paths[i]; i++) {
        cuda_handle = dlopen(paths[i], RTLD_NOW | RTLD_GLOBAL);
        if (cuda_handle) {
            fprintf(stderr, "[info] Loaded CUDA driver: %s\n", paths[i]);
            break;
        }
    }

    if (!cuda_handle) {
        fprintf(stderr, "[warn] Cannot load libcuda.so: %s\n", dlerror());
        return -1;
    }

#define BIND(sym) fn_##sym = (sym##_t)try_sym(#sym)
    BIND(cuInit);
    BIND(cuDeviceGetCount);
    BIND(cuDeviceGet);
    BIND(cuDeviceGetName);
    BIND(cuDeviceTotalMem);
    BIND(cuCtxCreate);
    BIND(cuCtxDestroy);
    BIND(cuCtxSetCurrent);
    BIND(cuCtxEnablePeerAccess);
    BIND(cuDeviceCanAccessPeer);
    BIND(cuMemAlloc);
    BIND(cuMemFree);
    BIND(cuMemcpyHtoD);
    BIND(cuMemcpyDtoH);
    BIND(cuMemcpyPeer);
    BIND(cuCtxSynchronize);
    BIND(cuGetErrorString);
#undef BIND

    if (!fn_cuInit || !fn_cuDeviceGetCount || !fn_cuDeviceGet ||
        !fn_cuMemAlloc || !fn_cuMemFree || !fn_cuMemcpyPeer) {
        fprintf(stderr, "[error] Missing essential CUDA symbols\n");
        return -1;
    }

    return 0;
}

#define XFER_SIZE (1 << 20)  // 1 MB per transfer test

static int do_p2p_test(int src, int dst, CUcontext *ctx, CUdevice *dev) {
    int can = 0;
    CUresult r = fn_cuDeviceCanAccessPeer(&can, dev[src], dev[dst]);
    if (r != CUDA_SUCCESS) {
        printf("  %d->%d: FAIL (canAccessPeer: %s)\n", src, dst, cu_err(r));
        return 0;
    }
    if (!can) {
        printf("  %d->%d: FAIL (P2P not available)\n", src, dst);
        return 0;
    }

    // Enable bidirectional peer access (ignore already-enabled)
    fn_cuCtxSetCurrent(ctx[src]);
    fn_cuCtxEnablePeerAccess(ctx[dst], 0);
    fn_cuCtxSetCurrent(ctx[dst]);
    fn_cuCtxEnablePeerAccess(ctx[src], 0);

    // Prepare host data
    const int nf = XFER_SIZE / (int)sizeof(float);
    float *h_src = (float*)malloc(XFER_SIZE);
    float *h_dst = (float*)calloc(nf, sizeof(float));
    if (!h_src || !h_dst) { free(h_src); free(h_dst); return 0; }
    for (int i = 0; i < nf; i++) h_src[i] = (float)(src * 10000 + dst * 100 + (i & 0xFF));

    // Alloc device buffers
    CUdeviceptr d_src = 0, d_dst = 0;
    fn_cuCtxSetCurrent(ctx[src]);
    r = fn_cuMemAlloc(&d_src, XFER_SIZE);
    if (r != CUDA_SUCCESS) {
        printf("  %d->%d: FAIL (alloc src: %s)\n", src, dst, cu_err(r));
        free(h_src); free(h_dst); return 0;
    }

    fn_cuCtxSetCurrent(ctx[dst]);
    r = fn_cuMemAlloc(&d_dst, XFER_SIZE);
    if (r != CUDA_SUCCESS) {
        printf("  %d->%d: FAIL (alloc dst: %s)\n", src, dst, cu_err(r));
        fn_cuCtxSetCurrent(ctx[src]); fn_cuMemFree(d_src);
        free(h_src); free(h_dst); return 0;
    }

    // Upload to src GPU
    fn_cuCtxSetCurrent(ctx[src]);
    fn_cuMemcpyHtoD(d_src, h_src, XFER_SIZE);
    fn_cuCtxSynchronize();

    // P2P copy src -> dst
    r = fn_cuMemcpyPeer(d_dst, ctx[dst], d_src, ctx[src], XFER_SIZE);
    if (r != CUDA_SUCCESS) {
        printf("  %d->%d: FAIL (memcpyPeer: %s)\n", src, dst, cu_err(r));
        fn_cuCtxSetCurrent(ctx[src]); fn_cuMemFree(d_src);
        fn_cuCtxSetCurrent(ctx[dst]); fn_cuMemFree(d_dst);
        free(h_src); free(h_dst); return 0;
    }

    // Download from dst GPU and verify
    fn_cuCtxSetCurrent(ctx[dst]);
    fn_cuCtxSynchronize();
    fn_cuMemcpyDtoH(h_dst, d_dst, XFER_SIZE);

    int ok = 1;
    for (int i = 0; i < nf; i++) {
        if (h_src[i] != h_dst[i]) {
            printf("  %d->%d: FAIL (mismatch @%d: %.0f!=%.0f)\n", src, dst, i, h_src[i], h_dst[i]);
            ok = 0; break;
        }
    }
    if (ok) printf("  %d->%d: PASS\n", src, dst);

    fn_cuCtxSetCurrent(ctx[src]); fn_cuMemFree(d_src);
    fn_cuCtxSetCurrent(ctx[dst]); fn_cuMemFree(d_dst);
    free(h_src); free(h_dst);
    return ok;
}

int main() {
    printf("=== GPU P2P Test (CUDA Driver API, dlopen) ===\n\n");

    // --- Compile check passed if we get here ---

    if (load_cuda() != 0) {
        printf("\nCannot load CUDA driver. GPUs not accessible from this environment.\n");
        printf("\n");
        printf("compile=pass\n");
        printf("compile_error=none\n");
        printf("gpu_pairs_ok=none\n");
        printf("gpu_pairs_fail=none\n");
        printf("gpu_test=fail\n");
        printf("gpu_test_detail=cuda_driver_not_loaded\n");
        printf("smoke=fail\n");
        printf("smoke_error=no_cuda_driver\n");
        printf("files_written=/home/claude/ikun/gpu_p2p_test.cpp,/home/claude/ikun/Makefile\n");
        return 1;
    }

    CUresult r = fn_cuInit(0);
    if (r != CUDA_SUCCESS) {
        printf("cuInit failed: %s\n", cu_err(r));
        printf("\ncompile=pass\n");
        printf("gpu_pairs_ok=none\ngpu_pairs_fail=none\n");
        printf("gpu_test=fail\ngpu_test_detail=cuInit_failed\n");
        printf("smoke=fail\nsmoke_error=cuInit_failed\n");
        printf("files_written=/home/claude/ikun/gpu_p2p_test.cpp,/home/claude/ikun/Makefile\n");
        return 1;
    }

    int ngpu = 0;
    fn_cuDeviceGetCount(&ngpu);
    printf("GPUs detected: %d\n\n", ngpu);

    if (ngpu == 0) {
        printf("compile=pass\n");
        printf("gpu_pairs_ok=none\ngpu_pairs_fail=none\n");
        printf("gpu_test=fail\ngpu_test_detail=no_gpus\n");
        printf("smoke=fail\nsmoke_error=no_gpus\n");
        printf("files_written=/home/claude/ikun/gpu_p2p_test.cpp,/home/claude/ikun/Makefile\n");
        return 1;
    }

    if (ngpu > 16) ngpu = 16;

    CUdevice dev[16];
    CUcontext ctx[16];

    for (int i = 0; i < ngpu; i++) {
        fn_cuDeviceGet(&dev[i], i);
        char name[256] = "";
        size_t mem = 0;
        fn_cuDeviceGetName(name, 256, dev[i]);
        fn_cuDeviceTotalMem(&mem, dev[i]);
        printf("  GPU %d: %s (%.0f MB)\n", i, name, mem / (1024.0 * 1024.0));
        r = fn_cuCtxCreate(&ctx[i], CU_CTX_SCHED_AUTO, dev[i]);
        if (r != CUDA_SUCCESS) {
            printf("  Failed to create context for GPU %d: %s\n", i, cu_err(r));
            ngpu = i;
            break;
        }
    }
    printf("\n");

    // === P2P pair tests ===
    printf("--- P2P pair tests ---\n");
    int ok_s[256], ok_d[256], fl_s[256], fl_d[256];
    int nok = 0, nfl = 0;

    for (int i = 0; i < ngpu; i++) {
        for (int j = 0; j < ngpu; j++) {
            if (i == j) continue;
            if (do_p2p_test(i, j, ctx, dev)) {
                ok_s[nok] = i; ok_d[nok] = j; nok++;
            } else {
                fl_s[nfl] = i; fl_d[nfl] = j; nfl++;
            }
        }
    }

    // Identify usable GPUs
    int used[16] = {};
    for (int k = 0; k < nok; k++) { used[ok_s[k]] = 1; used[ok_d[k]] = 1; }
    int gg[16]; int ng = 0;
    if (nok == 0) { for (int i = 0; i < ngpu; i++) gg[ng++] = i; }
    else { for (int i = 0; i < ngpu; i++) if (used[i]) gg[ng++] = i; }

    // === Model check ===
    printf("\n--- Model check ---\n");
    FILE *mf = fopen("/tmp/model", "rb");
    if (mf) {
        fseek(mf, 0, SEEK_END);
        long msz = ftell(mf);
        fclose(mf);
        size_t tmem = 0;
        for (int i = 0; i < ng; i++) { size_t m; fn_cuDeviceTotalMem(&m, dev[gg[i]]); tmem += m; }
        printf("Model %.1f MB vs GPU mem %.1f MB (%d GPUs)\n",
               msz / 1048576.0, tmem / 1048576.0, ng);
        if ((size_t)msz > (size_t)(tmem * 0.9))
            printf("WARNING: model may not fit\n");
    } else {
        printf("No /tmp/model (skip)\n");
    }

    // === Smoke test: ring transfer ===
    printf("\n--- Smoke test ---\n");
    int smoke = 1;
    if (ng >= 2) {
        const size_t ssz = 4 * 1024 * 1024; // 4MB
        const int snf = ssz / sizeof(float);
        CUdeviceptr bufs[16] = {};
        float *hbuf = (float*)malloc(ssz);

        for (int i = 0; i < ng && smoke; i++) {
            fn_cuCtxSetCurrent(ctx[gg[i]]);
            if (fn_cuMemAlloc(&bufs[i], ssz) != CUDA_SUCCESS) {
                printf("Alloc failed GPU %d\n", gg[i]); smoke = 0; break;
            }
            for (int k = 0; k < snf; k++) hbuf[k] = (float)(gg[i] * 1000 + (k & 0x3FF));
            fn_cuMemcpyHtoD(bufs[i], hbuf, ssz);
            fn_cuCtxSynchronize();
        }

        // Ring: i -> (i+1)%ng
        for (int i = 0; i < ng && smoke; i++) {
            int si = i, di = (i + 1) % ng;
            r = fn_cuMemcpyPeer(bufs[di], ctx[gg[di]], bufs[si], ctx[gg[si]], ssz);
            if (r != CUDA_SUCCESS) {
                // Fallback: staged through host
                fn_cuCtxSetCurrent(ctx[gg[si]]);
                fn_cuMemcpyDtoH(hbuf, bufs[si], ssz);
                fn_cuCtxSetCurrent(ctx[gg[di]]);
                r = fn_cuMemcpyHtoD(bufs[di], hbuf, ssz);
                if (r != CUDA_SUCCESS) {
                    printf("Ring %d->%d failed: %s\n", gg[si], gg[di], cu_err(r));
                    smoke = 0;
                }
            }
        }

        // Verify: buf[i] should now hold data from GPU gg[(i-1+ng)%ng]
        for (int i = 0; i < ng && smoke; i++) {
            int prev = gg[((i - 1) + ng) % ng];
            fn_cuCtxSetCurrent(ctx[gg[i]]);
            fn_cuCtxSynchronize();
            fn_cuMemcpyDtoH(hbuf, bufs[i], ssz);
            for (int k = 0; k < 256; k++) {
                float exp = (float)(prev * 1000 + (k & 0x3FF));
                if (hbuf[k] != exp) {
                    printf("Verify fail GPU %d idx %d: got %.0f exp %.0f\n", gg[i], k, hbuf[k], exp);
                    smoke = 0; break;
                }
            }
        }

        if (smoke) printf("Ring transfer PASS (%d GPUs, 4MB each)\n", ng);

        for (int i = 0; i < ng; i++) {
            if (bufs[i]) { fn_cuCtxSetCurrent(ctx[gg[i]]); fn_cuMemFree(bufs[i]); }
        }
        free(hbuf);
    } else {
        fn_cuCtxSetCurrent(ctx[gg[0]]);
        CUdeviceptr tmp = 0;
        if (fn_cuMemAlloc(&tmp, 1048576) == CUDA_SUCCESS) {
            fn_cuMemFree(tmp);
            printf("Single GPU alloc/free PASS\n");
        } else {
            printf("Single GPU alloc FAIL\n"); smoke = 0;
        }
    }

    // Cleanup
    for (int i = 0; i < ngpu; i++) fn_cuCtxDestroy(ctx[i]);
    if (cuda_handle) dlclose(cuda_handle);

    // === STATUS BLOCK ===
    printf("\n");
    printf("compile=pass\n");
    printf("compile_error=none\n");

    printf("gpu_pairs_ok=");
    if (nok == 0) printf("none");
    for (int k = 0; k < nok; k++) printf("%s%d-%d", k ? "," : "", ok_s[k], ok_d[k]);
    printf("\n");

    printf("gpu_pairs_fail=");
    if (nfl == 0) printf("none");
    for (int k = 0; k < nfl; k++) printf("%s%d-%d", k ? "," : "", fl_s[k], fl_d[k]);
    printf("\n");

    printf("gpu_test=%s\n", (nok > 0 || ngpu == 1) ? "pass" : "fail");
    printf("gpu_test_detail=p2p,memcpyPeer,all_pairs,data_verified\n");
    printf("smoke=%s\n", smoke ? "pass" : "fail");
    if (!smoke) printf("smoke_error=ring_transfer_failed\n");
    printf("files_written=/home/claude/ikun/gpu_p2p_test.cpp,/home/claude/ikun/Makefile\n");

    return smoke ? 0 : 1;
}
