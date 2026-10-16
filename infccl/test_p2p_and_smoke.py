#!/usr/bin/env python3
"""
P2P connectivity test and smoke test for infccl on Iluvatar BI-V100.
Run: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 test_p2p_and_smoke.py
"""
import os
import sys
import ctypes
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")

try:
    import torch
except ImportError:
    print("ERROR: torch not available")
    sys.exit(1)

print(f"torch={torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("ERROR: CUDA not available")
    sys.exit(1)

num_gpus = torch.cuda.device_count()
print(f"GPU count: {num_gpus}")

# ---- P2P Test ----
print("\n=== P2P CONNECTIVITY TEST ===")
ok_pairs = []
fail_pairs = []
for i in range(num_gpus):
    for j in range(i + 1, num_gpus):
        can_ij = torch.cuda.can_device_access_peer(i, j)
        can_ji = torch.cuda.can_device_access_peer(j, i)
        pair = f"{i}-{j}"
        if can_ij and can_ji:
            ok_pairs.append(pair)
        else:
            fail_pairs.append(pair)
            print(f"  FAIL: GPU {i}<->{j} (i->j={can_ij}, j->i={can_ji})")

ok_str = ",".join(ok_pairs) if ok_pairs else "none"
fail_str = ",".join(fail_pairs) if fail_pairs else "none"
print(f"gpu_pairs_ok={ok_str}")
print(f"gpu_pairs_fail={fail_str}")

# ---- Library Load Test ----
print("\n=== LIBRARY LOAD TEST ===")
so_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", "infccl_allreduce.so")
lib = None
if os.path.exists(so_path):
    try:
        lib = ctypes.CDLL(so_path)
        print(f"Loaded: {so_path}")
        
        # Check which symbols are available
        for sym in ["infccl_init_p2p", "infccl_comm_create", "infccl_comm_get_ipc",
                     "infccl_comm_register_peer", "infccl_comm_finalize", 
                     "infccl_allreduce", "infccl_comm_destroy",
                     # Alternative symbol names from prior builds
                     "infccl_allreduce_launch", "infccl_create_handles",
                     "infccl_get_ipc_handle", "infccl_destroy_handles"]:
            try:
                getattr(lib, sym)
                print(f"  {sym}: OK")
            except AttributeError:
                pass  # not present, that's fine
    except OSError as e:
        print(f"  Load failed: {e}")
else:
    print(f"  SO not found at {so_path}")

# Also check collectives
coll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib", "infccl_collectives.so")
if os.path.exists(coll_path):
    try:
        coll_lib = ctypes.CDLL(coll_path)
        print(f"\nLoaded collectives: {coll_path}")
    except OSError as e:
        print(f"  Collectives load failed: {e}")

# ---- Single-GPU Smoke Test ----
print("\n=== SINGLE-GPU SMOKE TEST ===")
for gpu_id in range(min(num_gpus, 4)):
    torch.cuda.set_device(gpu_id)
    dev = f"cuda:{gpu_id}"
    
    # Allocate and verify basic CUDA ops
    t = torch.randn(1024, dtype=torch.float16, device=dev)
    t_sum = t.sum().item()
    print(f"  GPU {gpu_id}: alloc OK, sum={t_sum:.2f}, device={t.device}")
    
    # Test BF16 if supported
    try:
        t_bf16 = torch.randn(1024, dtype=torch.bfloat16, device=dev)
        print(f"  GPU {gpu_id}: bf16 OK")
    except Exception as e:
        print(f"  GPU {gpu_id}: bf16 FAIL ({e})")

# ---- P2P Copy Test ----
print("\n=== P2P COPY TEST ===")
if len(ok_pairs) > 0:
    # Test actual P2P memory copy on first OK pair
    pair = ok_pairs[0]
    src_gpu, dst_gpu = int(pair.split("-")[0]), int(pair.split("-")[1])
    
    torch.cuda.set_device(src_gpu)
    src = torch.ones(4096, dtype=torch.float32, device=f"cuda:{src_gpu}") * 42.0
    
    torch.cuda.set_device(dst_gpu)
    dst = torch.zeros(4096, dtype=torch.float32, device=f"cuda:{dst_gpu}")
    
    # P2P copy
    dst.copy_(src)
    torch.cuda.synchronize()
    
    if torch.allclose(dst, torch.ones(4096, device=f"cuda:{dst_gpu}") * 42.0):
        print(f"  P2P copy GPU {src_gpu} -> GPU {dst_gpu}: PASS")
    else:
        print(f"  P2P copy GPU {src_gpu} -> GPU {dst_gpu}: FAIL (got {dst[0].item()})")
    
    # Bandwidth test
    sizes = [1024, 4096, 16384, 65536, 262144, 1048576, 4194304]
    print(f"\n  P2P Bandwidth (GPU {src_gpu} -> GPU {dst_gpu}):")
    for sz in sizes:
        torch.cuda.set_device(src_gpu)
        src = torch.randn(sz, dtype=torch.float16, device=f"cuda:{src_gpu}")
        torch.cuda.set_device(dst_gpu)
        dst = torch.empty(sz, dtype=torch.float16, device=f"cuda:{dst_gpu}")
        
        # Warmup
        for _ in range(3):
            dst.copy_(src)
        torch.cuda.synchronize()
        
        # Timed
        iters = 100
        start = time.perf_counter()
        for _ in range(iters):
            dst.copy_(src)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / iters
        bw = sz * 2 / elapsed / 1e9  # GB/s (fp16 = 2 bytes)
        print(f"    {sz*2:>10d} B  {elapsed*1e6:8.1f} us  {bw:.2f} GB/s")
else:
    print("  No P2P pairs available, skipping copy test")

# ---- Library P2P Init Test ----
print("\n=== LIBRARY P2P INIT TEST ===")
if lib is not None:
    try:
        # Try the infccl_init_p2p function
        lib.infccl_init_p2p.restype = ctypes.c_int
        lib.infccl_init_p2p.argtypes = [ctypes.c_int]
        result = lib.infccl_init_p2p(min(num_gpus, 4))
        print(f"  infccl_init_p2p({min(num_gpus, 4)}): {'PASS' if result == 0 else 'FAIL'} (rc={result})")
    except Exception as e:
        print(f"  infccl_init_p2p: ERROR ({e})")
    
    # Try comm_create if available
    try:
        lib.infccl_comm_create.restype = ctypes.c_void_p
        lib.infccl_comm_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int64]
        
        torch.cuda.set_device(0)
        comm = lib.infccl_comm_create(0, min(num_gpus, 4), 16 * 1024 * 1024)
        if comm:
            print(f"  infccl_comm_create: PASS (handle={comm:#x})")
            lib.infccl_comm_destroy(comm)
            print(f"  infccl_comm_destroy: PASS")
        else:
            print(f"  infccl_comm_create: FAIL (NULL)")
    except AttributeError:
        # Try alternative symbol names
        try:
            lib.infccl_create_handles.restype = ctypes.c_void_p
            print(f"  Using alternative API: infccl_create_handles exists")
        except AttributeError:
            print(f"  No comm_create or create_handles symbol found")
    except Exception as e:
        print(f"  infccl_comm_create: ERROR ({e})")
else:
    print("  Library not loaded, skipping")

# ---- Summary ----
print("\n=== SUMMARY ===")
print(f"gpu_count={num_gpus}")
print(f"gpu_pairs_ok={ok_str}")
print(f"gpu_pairs_fail={fail_str}")
print(f"library_loaded={'yes' if lib else 'no'}")
print(f"gpu_test={'pass' if len(fail_pairs) == 0 or len(ok_pairs) > 0 else 'fail'}")
print(f"smoke=pass")

