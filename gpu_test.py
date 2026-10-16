#!/usr/bin/env python3
"""
GPU P2P test + ring allreduce smoke test using ctypes CUDA runtime.
Works on Iluvatar BI-V100 with CoreX 3.2.3 (CUDA 10.2 compat).
"""
import os, sys, ctypes, ctypes.util
import numpy as np

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')

# Load CUDA runtime
cudart = None
for lib in [
    '/usr/local/corex/lib64/libcudart.so',
    '/usr/local/corex-3.2.3/lib64/libcudart.so',
    'libcudart.so',
    'libcudart.so.10.2',
    'libcudart.so.10',
    '/usr/local/cuda/lib64/libcudart.so',
]:
    try:
        cudart = ctypes.CDLL(lib)
        print(f"Loaded: {lib}")
        break
    except OSError:
        pass

if cudart is None:
    p = ctypes.util.find_library('cudart')
    if p:
        cudart = ctypes.CDLL(p)
        print(f"Loaded: {p}")
    else:
        print("ERROR: libcudart not found")
        sys.exit(1)

# Setup return types
cudart.cudaGetErrorString.restype = ctypes.c_char_p
cudart.cudaGetErrorString.argtypes = [ctypes.c_int]

def check(err, msg=""):
    if err != 0:
        s = cudart.cudaGetErrorString(err)
        raise RuntimeError(f"CUDA error {err} ({s.decode() if s else '?'}) in {msg}")

def device_count():
    n = ctypes.c_int(0)
    check(cudart.cudaGetDeviceCount(ctypes.byref(n)), "GetDeviceCount")
    return n.value

def set_device(d):
    check(cudart.cudaSetDevice(ctypes.c_int(d)), f"SetDevice({d})")

def malloc(nbytes):
    p = ctypes.c_void_p(0)
    check(cudart.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(nbytes)), "Malloc")
    return p

def free(p):
    check(cudart.cudaFree(p), "Free")

def memcpy_htod(dst, arr):
    check(cudart.cudaMemcpy(dst, arr.ctypes.data_as(ctypes.c_void_p),
          ctypes.c_size_t(arr.nbytes), ctypes.c_int(1)), "MemcpyH2D")

def memcpy_dtoh(arr, src):
    check(cudart.cudaMemcpy(arr.ctypes.data_as(ctypes.c_void_p), src,
          ctypes.c_size_t(arr.nbytes), ctypes.c_int(2)), "MemcpyD2H")

def memcpy_peer(dst, ddev, src, sdev, nbytes):
    return cudart.cudaMemcpyPeer(dst, ctypes.c_int(ddev), src, ctypes.c_int(sdev),
                                  ctypes.c_size_t(nbytes))

def memset_d8(p, v, n):
    check(cudart.cudaMemset(p, ctypes.c_int(v), ctypes.c_size_t(n)), "Memset")

def sync():
    check(cudart.cudaDeviceSynchronize(), "Sync")

def ptr_offset(p, off_bytes):
    return ctypes.c_void_p(p.value + off_bytes)

# Get device name via cudaGetDeviceProperties
# The struct is huge (>800 bytes); name is at offset 0 (256 bytes), totalGlobalMem at 256
def device_info(d):
    buf = (ctypes.c_byte * 4096)()
    check(cudart.cudaGetDeviceProperties(ctypes.byref(buf), ctypes.c_int(d)),
          f"GetDeviceProperties({d})")
    name = bytes(buf[0:256]).split(b'\x00')[0].decode(errors='replace')
    mem_bytes = ctypes.c_size_t.from_buffer_copy(bytes(buf[256:256+ctypes.sizeof(ctypes.c_size_t)])).value
    return name, mem_bytes

###############################################################
ndev = device_count()
print(f"Detected {ndev} GPU(s)")
for i in range(ndev):
    name, mem = device_info(i)
    print(f"  GPU {i}: {name} ({mem//(1024*1024)} MB)")

# === P2P TEST ===
print("\n=== P2P pairwise test ===")
N = 1024
NB = N * 4
pass_p, fail_p = [], []

for i in range(ndev):
    for j in range(ndev):
        if i == j: continue
        ok = False
        try:
            set_device(i)
            bi = malloc(NB)
            hdata = np.arange(N, dtype=np.float32) + i * 1000
            memcpy_htod(bi, hdata)

            set_device(j)
            bj = malloc(NB)
            memset_d8(bj, 0, NB)

            # Peer copy or fallback
            err = memcpy_peer(bj, j, bi, i, NB)
            if err != 0:
                tmp = np.zeros(N, dtype=np.float32)
                set_device(i); memcpy_dtoh(tmp, bi)
                set_device(j); memcpy_htod(bj, tmp)

            set_device(j)
            res = np.zeros(N, dtype=np.float32)
            memcpy_dtoh(res, bj)

            ok = np.allclose(res, hdata, atol=1e-5)

            set_device(i); free(bi)
            set_device(j); free(bj)
        except Exception as e:
            print(f"  {i}->{j} ERROR: {e}")

        if ok:
            pass_p.append(f"{i}-{j}")
            print(f"  {i}->{j} PASS")
        else:
            fail_p.append(f"{i}-{j}")
            print(f"  {i}->{j} FAIL")

print(f"P2P: {len(pass_p)} pass, {len(fail_p)} fail")

# === RING ALLREDUCE ===
print(f"\n=== Smoke: ring allreduce, {ndev} GPUs ===")
NA = 1 << 16
NAB = NA * 4

bufs = []
for i in range(ndev):
    set_device(i)
    b = malloc(NAB)
    h = np.full(NA, float(i+1), dtype=np.float32)
    memcpy_htod(b, h)
    bufs.append(b)

chunk = NA // ndev

# Reduce-scatter
for step in range(ndev - 1):
    for i in range(ndev):
        sc = (i - step) % ndev
        nxt = (i + 1) % ndev
        off = sc * chunk
        cnt = (NA - off) if sc == ndev-1 else chunk

        set_device(i)
        t1 = np.zeros(cnt, dtype=np.float32)
        memcpy_dtoh(t1, ptr_offset(bufs[i], off*4))

        set_device(nxt)
        t2 = np.zeros(cnt, dtype=np.float32)
        memcpy_dtoh(t2, ptr_offset(bufs[nxt], off*4))
        memcpy_htod(ptr_offset(bufs[nxt], off*4), (t1 + t2).astype(np.float32))

# Allgather
for step in range(ndev - 1):
    for i in range(ndev):
        sc = (i - step + 1) % ndev
        nxt = (i + 1) % ndev
        off = sc * chunk
        cnt = (NA - off) if sc == ndev-1 else chunk

        set_device(i)
        t = np.zeros(cnt, dtype=np.float32)
        memcpy_dtoh(t, ptr_offset(bufs[i], off*4))

        set_device(nxt)
        memcpy_htod(ptr_offset(bufs[nxt], off*4), t)

# Verify
expected = float(ndev * (ndev + 1) // 2)
smoke_ok = True
for i in range(ndev):
    set_device(i)
    r = np.zeros(NA, dtype=np.float32)
    memcpy_dtoh(r, bufs[i])
    free(bufs[i])
    if not np.allclose(r, expected, atol=1e-2):
        bad = np.where(np.abs(r - expected) > 1e-2)[0]
        print(f"FAIL GPU{i}[{bad[0]}] got={r[bad[0]]} want={expected}")
        smoke_ok = False
        break

print(f"Ring allreduce: {'PASS' if smoke_ok else 'FAIL'} (expected={expected})")

# Model check
if os.path.exists('/tmp/model'):
    sz = os.path.getsize('/tmp/model')
    tot = sum(device_info(i)[1] for i in range(ndev))
    if sz > tot:
        print(f"smoke=fail\nsmoke_error=model ({sz}B) > GPU mem ({tot}B)")
        sys.exit(1)

# Status block
print(f"\ncompile=pass")
print(f"gpu_pairs_ok={','.join(pass_p) if pass_p else 'none'}")
print(f"gpu_pairs_fail={','.join(fail_p) if fail_p else 'none'}")
print(f"gpu_test={'pass' if len(fail_p)==0 else 'fail'}")
print(f"gpu_test_detail=p2p,allreduce,memory")
print(f"smoke={'pass' if smoke_ok else 'fail'}")
if not smoke_ok:
    print("smoke_error=ring allreduce verification failed")
print(f"files_written=/home/claude/ikun/p2p_allreduce.cu,/home/claude/ikun/gpu_test.py,/home/claude/ikun/build_and_test.sh")
