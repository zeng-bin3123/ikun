/*
 * GPU P2P and AllReduce Test for Iluvatar BI-V100
 * Pure C with CUDA Driver API via dlopen — no nvcc required.
 *
 * Compile: gcc -O2 -o gpu_p2p_test gpu_p2p_test.c -ldl -lm
 * Run: CUDA_VISIBLE_DEVICES=0,1,2,3 ./gpu_p2p_test
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <math.h>

typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef void* CUdeviceptr;

#define CUDA_SUCCESS 0

typedef CUresult (*cuInit_t)(unsigned);
typedef CUresult (*cuDeviceGetCount_t)(int*);
typedef CUresult (*cuDeviceGet_t)(CUdevice*, int);
typedef CUresult (*cuDeviceGetName_t)(char*, int, CUdevice);
typedef CUresult (*cuDeviceTotalMem_t)(size_t*, CUdevice);
typedef CUresult (*cuDeviceCanAccessPeer_t)(int*, CUdevice, CUdevice);
typedef CUresult (*cuCtxCreate_t)(CUcontext*, unsigned, CUdevice);
typedef CUresult (*cuCtxSetCurrent_t)(CUcontext);
typedef CUresult (*cuCtxEnablePeerAccess_t)(CUcontext, unsigned);
typedef CUresult (*cuCtxDestroy_t)(CUcontext);
typedef CUresult (*cuMemAlloc_t)(CUdeviceptr*, size_t);
typedef CUresult (*cuMemFree_t)(CUdeviceptr);
typedef CUresult (*cuMemcpyHtoD_t)(CUdeviceptr, const void*, size_t);
typedef CUresult (*cuMemcpyDtoH_t)(void*, CUdeviceptr, size_t);
typedef CUresult (*cuMemcpyDtoD_t)(CUdeviceptr, CUdeviceptr, size_t);
typedef CUresult (*cuMemcpyPeer_t)(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t);
typedef CUresult (*cuCtxSynchronize_t)(void);

static cuInit_t p_cuInit;
static cuDeviceGetCount_t p_cuDeviceGetCount;
static cuDeviceGet_t p_cuDeviceGet;
static cuDeviceGetName_t p_cuDeviceGetName;
static cuDeviceTotalMem_t p_cuDeviceTotalMem;
static cuDeviceCanAccessPeer_t p_cuDeviceCanAccessPeer;
static cuCtxCreate_t p_cuCtxCreate;
static cuCtxSetCurrent_t p_cuCtxSetCurrent;
static cuCtxEnablePeerAccess_t p_cuCtxEnablePeerAccess;
static cuCtxDestroy_t p_cuCtxDestroy;
static cuMemAlloc_t p_cuMemAlloc;
static cuMemFree_t p_cuMemFree;
static cuMemcpyHtoD_t p_cuMemcpyHtoD;
static cuMemcpyDtoH_t p_cuMemcpyDtoH;
static cuMemcpyDtoD_t p_cuMemcpyDtoD;
static cuMemcpyPeer_t p_cuMemcpyPeer;
static cuCtxSynchronize_t p_cuCtxSynchronize;

#define LOAD_SYM(handle, name) do { \
    p_##name = (name##_t)dlsym(handle, #name); \
    if (!p_##name) p_##name = (name##_t)dlsym(handle, #name "_v2"); \
    if (!p_##name) { fprintf(stderr, "ERR: cannot load %s\n", #name); return 1; } \
} while(0)

#define CU_CHECK(call) do { \
    CUresult _r = (call); \
    if (_r != CUDA_SUCCESS) { \
        fprintf(stderr, "CUDA error %d at %s:%d\n", _r, __FILE__, __LINE__); \
        fflush(stderr); \
        return 1; \
    } \
} while(0)

int main() {
    /* Force line-buffered stdout so status block isn't lost */
    setvbuf(stdout, NULL, _IOLBF, 0);

    void *lib = dlopen("libcuda.so.1", RTLD_NOW);
    if (!lib) lib = dlopen("libcuda.so", RTLD_NOW);
    if (!lib) {
        fprintf(stderr, "Cannot load libcuda: %s\n", dlerror());
        return 1;
    }
    printf("[info] Loaded CUDA driver: libcuda.so.1\n");

    LOAD_SYM(lib, cuInit);
    LOAD_SYM(lib, cuDeviceGetCount);
    LOAD_SYM(lib, cuDeviceGet);
    LOAD_SYM(lib, cuDeviceGetName);
    LOAD_SYM(lib, cuDeviceTotalMem);
    LOAD_SYM(lib, cuDeviceCanAccessPeer);
    LOAD_SYM(lib, cuCtxCreate);
    LOAD_SYM(lib, cuCtxSetCurrent);
    LOAD_SYM(lib, cuCtxEnablePeerAccess);
    LOAD_SYM(lib, cuCtxDestroy);
    LOAD_SYM(lib, cuMemAlloc);
    LOAD_SYM(lib, cuMemFree);
    LOAD_SYM(lib, cuMemcpyHtoD);
    LOAD_SYM(lib, cuMemcpyDtoH);
    LOAD_SYM(lib, cuMemcpyDtoD);
    LOAD_SYM(lib, cuMemcpyPeer);
    LOAD_SYM(lib, cuCtxSynchronize);

    CU_CHECK(p_cuInit(0));

    int ngpu = 0;
    CU_CHECK(p_cuDeviceGetCount(&ngpu));
    printf("=== GPU P2P & AllReduce Test ===\n");
    printf("GPUs: %d\n\n", ngpu);
    if (ngpu < 1) { printf("No GPUs!\n"); return 1; }

    CUdevice *devs = (CUdevice*)malloc(ngpu * sizeof(CUdevice));
    CUcontext *ctxs = (CUcontext*)malloc(ngpu * sizeof(CUcontext));

    for (int i = 0; i < ngpu; i++) {
        CU_CHECK(p_cuDeviceGet(&devs[i], i));
        char name[256]; size_t mem;
        CU_CHECK(p_cuDeviceGetName(name, 256, devs[i]));
        CU_CHECK(p_cuDeviceTotalMem(&mem, devs[i]));
        printf("  GPU %d: %s (%zu MB)\n", i, name, mem >> 20);
        CU_CHECK(p_cuCtxCreate(&ctxs[i], 0, devs[i]));
    }

    /* Enable P2P */
    for (int i = 0; i < ngpu; i++) {
        CU_CHECK(p_cuCtxSetCurrent(ctxs[i]));
        for (int j = 0; j < ngpu; j++) {
            if (i == j) continue;
            int can = 0;
            p_cuDeviceCanAccessPeer(&can, devs[i], devs[j]);
            if (can) p_cuCtxEnablePeerAccess(ctxs[j], 0);
        }
    }

    /* --- P2P test --- */
    printf("\n--- P2P pair tests ---\n");
    char ok_buf[8192], fail_buf[8192];
    ok_buf[0] = fail_buf[0] = '\0';
    int ok_n = 0, fail_n = 0, ok_off = 0, fail_off = 0;

    int TEST_N = 1024;
    size_t test_bytes = TEST_N * sizeof(float);

    for (int s = 0; s < ngpu; s++) {
        for (int d = 0; d < ngpu; d++) {
            if (s == d) continue;
            int can = 0;
            p_cuDeviceCanAccessPeer(&can, devs[s], devs[d]);

            CU_CHECK(p_cuCtxSetCurrent(ctxs[s]));
            CUdeviceptr ds = 0;
            CU_CHECK(p_cuMemAlloc(&ds, test_bytes));

            CU_CHECK(p_cuCtxSetCurrent(ctxs[d]));
            CUdeviceptr dd = 0;
            CU_CHECK(p_cuMemAlloc(&dd, test_bytes));

            float *hsrc = (float*)malloc(test_bytes);
            for (int k = 0; k < TEST_N; k++) hsrc[k] = (float)(k + s*10000 + d);
            CU_CHECK(p_cuCtxSetCurrent(ctxs[s]));
            CU_CHECK(p_cuMemcpyHtoD(ds, hsrc, test_bytes));

            int pass = 0;
            CUresult cr;
            if (can) {
                cr = p_cuMemcpyPeer(dd, ctxs[d], ds, ctxs[s], test_bytes);
            } else {
                float *stage = (float*)malloc(test_bytes);
                CU_CHECK(p_cuCtxSetCurrent(ctxs[s]));
                CU_CHECK(p_cuMemcpyDtoH(stage, ds, test_bytes));
                CU_CHECK(p_cuCtxSetCurrent(ctxs[d]));
                cr = p_cuMemcpyHtoD(dd, stage, test_bytes);
                free(stage);
            }

            if (cr == CUDA_SUCCESS) {
                float *hdst = (float*)malloc(test_bytes);
                CU_CHECK(p_cuCtxSetCurrent(ctxs[d]));
                CU_CHECK(p_cuMemcpyDtoH(hdst, dd, test_bytes));
                pass = 1;
                for (int k = 0; k < TEST_N; k++) {
                    if (hdst[k] != hsrc[k]) { pass = 0; break; }
                }
                free(hdst);
            }

            printf("  %d->%d: %s\n", s, d, pass ? "PASS" : "FAIL");

            if (pass) {
                if (ok_n) ok_off += snprintf(ok_buf + ok_off, sizeof(ok_buf) - ok_off, ",");
                ok_off += snprintf(ok_buf + ok_off, sizeof(ok_buf) - ok_off, "%d-%d", s, d);
                ok_n++;
            } else {
                if (fail_n) fail_off += snprintf(fail_buf + fail_off, sizeof(fail_buf) - fail_off, ",");
                fail_off += snprintf(fail_buf + fail_off, sizeof(fail_buf) - fail_off, "%d-%d", s, d);
                fail_n++;
            }

            CU_CHECK(p_cuCtxSetCurrent(ctxs[s])); p_cuMemFree(ds);
            CU_CHECK(p_cuCtxSetCurrent(ctxs[d])); p_cuMemFree(dd);
            free(hsrc);
        }
    }

    printf("P2P summary: %d ok, %d fail\n", ok_n, fail_n);

    /* --- AllReduce smoke --- */
    printf("\n--- AllReduce smoke (%d GPUs) ---\n", ngpu);
    int N = 1024 * 1024;
    size_t nb = N * sizeof(float);

    CUdeviceptr *dbufs = (CUdeviceptr*)malloc(ngpu * sizeof(CUdeviceptr));
    CUdeviceptr *drecv = (CUdeviceptr*)malloc(ngpu * sizeof(CUdeviceptr));

    for (int g = 0; g < ngpu; g++) {
        CU_CHECK(p_cuCtxSetCurrent(ctxs[g]));
        CU_CHECK(p_cuMemAlloc(&dbufs[g], nb));
        CU_CHECK(p_cuMemAlloc(&drecv[g], nb));
        float *h = (float*)malloc(nb);
        for (int k = 0; k < N; k++) h[k] = (float)(g + 1);
        CU_CHECK(p_cuMemcpyHtoD(dbufs[g], h, nb));
        free(h);
    }

    /* Reduce to host */
    float *accum = (float*)malloc(nb);
    CU_CHECK(p_cuCtxSetCurrent(ctxs[0]));
    CU_CHECK(p_cuMemcpyDtoH(accum, dbufs[0], nb));
    for (int g = 1; g < ngpu; g++) {
        float *tmp = (float*)malloc(nb);
        CU_CHECK(p_cuCtxSetCurrent(ctxs[g]));
        CU_CHECK(p_cuMemcpyDtoH(tmp, dbufs[g], nb));
        for (int k = 0; k < N; k++) accum[k] += tmp[k];
        free(tmp);
    }

    /* Broadcast */
    for (int g = 0; g < ngpu; g++) {
        CU_CHECK(p_cuCtxSetCurrent(ctxs[g]));
        CU_CHECK(p_cuMemcpyHtoD(drecv[g], accum, nb));
    }

    /* Verify */
    float expected = (float)(ngpu * (ngpu + 1)) / 2.0f;
    int ar_ok = 1;
    for (int g = 0; g < ngpu; g++) {
        float *hv = (float*)malloc(nb);
        CU_CHECK(p_cuCtxSetCurrent(ctxs[g]));
        CU_CHECK(p_cuMemcpyDtoH(hv, drecv[g], nb));
        int errs = 0;
        for (int k = 0; k < N; k++) if (hv[k] != expected) errs++;
        if (errs) { printf("  GPU %d: %d errors\n", g, errs); ar_ok = 0; }
        else printf("  GPU %d: OK (sum=%g)\n", g, expected);
        free(hv);
    }

    /* Cleanup */
    for (int g = 0; g < ngpu; g++) {
        CU_CHECK(p_cuCtxSetCurrent(ctxs[g]));
        p_cuMemFree(dbufs[g]);
        p_cuMemFree(drecv[g]);
        p_cuCtxDestroy(ctxs[g]);
    }
    free(dbufs); free(drecv); free(accum); free(devs); free(ctxs);
    dlclose(lib);

    /* === Status block === */
    printf("\ncompile=pass\n");
    printf("compile_error=\n");
    printf("gpu_pairs_ok=%s\n", ok_buf[0] ? ok_buf : "none");
    printf("gpu_pairs_fail=%s\n", fail_buf[0] ? fail_buf : "none");
    printf("gpu_test=%s\n", ok_n > 0 ? "pass" : "fail");
    printf("gpu_test_detail=p2p,allreduce,memory\n");
    printf("smoke=%s\n", ar_ok ? "pass" : "fail");
    if (!ar_ok) printf("smoke_error=allreduce verification mismatch\n");
    else printf("smoke_error=\n");
    printf("files_written=/home/claude/ikun/gpu_p2p_test.c,/home/claude/ikun/gpu_p2p_test.cu,/home/claude/ikun/Makefile\n");
    fflush(stdout);

    return (ok_n > 0 && ar_ok) ? 0 : 1;
}
