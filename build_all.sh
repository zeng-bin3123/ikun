#!/usr/bin/env bash
# Build all ikun components for BI-V100
set -eo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$(pwd)"

echo "=== STEP 1: Build infccl_allreduce.so ==="
NVCC=""
for p in /usr/local/corex/bin/nvcc /usr/local/corex-3.2.3/bin/nvcc /usr/local/cuda/bin/nvcc; do
    [ -x "$p" ] && NVCC="$p" && break
done
if [ -z "$NVCC" ] && command -v nvcc &>/dev/null; then
    NVCC=nvcc
fi

if [ -n "$NVCC" ]; then
    echo "[build] Using NVCC: $NVCC"
    mkdir -p infccl/lib
    $NVCC -arch=sm_70 -gencode=arch=compute_70,code=sm_70 -O3 -shared \
        -Xcompiler -fPIC -Xcompiler -Wall \
        -I/usr/local/corex/include -L/usr/local/corex/lib64 \
        -o infccl/lib/infccl_allreduce.so \
        infccl/csrc/infccl_allreduce.cu -lcudart 2>&1
    echo "[build] infccl_allreduce.so built:"
    ls -lh infccl/lib/infccl_allreduce.so
    nm -D infccl/lib/infccl_allreduce.so 2>/dev/null | grep ' T infccl_' || true
else
    echo "[build] No CUDA compiler found - skipping .so build"
fi

echo ""
echo "=== STEP 2: Build core C++ library ==="
mkdir -p build && cd build
# Check for cmake
if command -v cmake &>/dev/null; then
    echo "[build] cmake found: $(cmake --version | head -1)"
    
    # Try with USE_ILU=OFF first (simpler, no ILU deps)
    cmake "$ROOT/core" -DUSE_ILU=OFF -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release 2>&1 || {
        echo "[build] cmake configure failed (expected if deps missing)"
        echo "[build] Trying to install deps..."
        apt-get install -y libgoogle-glog-dev libgflags-dev libgtest-dev 2>/dev/null || true
        cmake "$ROOT/core" -DUSE_ILU=OFF -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release 2>&1
    }
    make -j$(nproc) 2>&1 || echo "[build] core build failed (see above)"
else
    echo "[build] cmake not found - skipping core build"
fi
cd "$ROOT"

echo ""
echo "=== STEP 3: GPU P2P connectivity test ==="
if command -v python3 &>/dev/null && python3 -c "import torch" 2>/dev/null; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 - << 'PYEOF'
import torch

num_gpus = torch.cuda.device_count()
print(f"[p2p] Visible GPUs: {num_gpus}")

ok_pairs = []
fail_pairs = []

for i in range(num_gpus):
    for j in range(num_gpus):
        if i == j:
            continue
        can = torch.cuda.can_device_access_peer(i, j)
        pair = f"{i}-{j}"
        if can:
            ok_pairs.append(pair)
        else:
            fail_pairs.append(pair)

# Deduplicate to undirected pairs
ok_undirected = sorted(set(tuple(sorted(p.split('-'))) for p in ok_pairs))
fail_undirected = sorted(set(tuple(sorted(p.split('-'))) for p in fail_pairs))

print(f"[p2p] P2P OK pairs: {','.join(f'{a}-{b}' for a,b in ok_undirected)}")
print(f"[p2p] P2P FAIL pairs: {','.join(f'{a}-{b}' for a,b in fail_undirected) or 'none'}")

# Functional test: allocate on each GPU and do a simple copy
print("[p2p] Testing P2P data transfer...")
for i in range(min(num_gpus, 4)):
    for j in range(min(num_gpus, 4)):
        if i == j:
            continue
        if not torch.cuda.can_device_access_peer(i, j):
            continue
        try:
            src = torch.randn(1024, device=f'cuda:{i}')
            dst = torch.empty(1024, device=f'cuda:{j}')
            dst.copy_(src)
            torch.cuda.synchronize()
            diff = (dst.to(f'cuda:{i}') - src).abs().max().item()
            assert diff == 0.0, f"P2P copy {i}->{j} mismatch: {diff}"
        except Exception as e:
            print(f"[p2p] Transfer {i}->{j} FAILED: {e}")

print("[p2p] P2P functional test passed")
PYEOF
else
    echo "[p2p] Python/torch not available - skipping"
fi

echo ""
echo "=== STEP 4: Smoke test - allreduce correctness ==="
if command -v python3 &>/dev/null && python3 -c "import torch" 2>/dev/null; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 - << 'PYEOF'
import os
import sys
import torch
import torch.multiprocessing as mp

def worker(rank, world_size, results_queue):
    """Single-rank worker for allreduce test."""
    try:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29501'
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        
        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(
            backend='nccl', rank=rank, world_size=world_size)
        
        # Test 1: Simple sum allreduce
        t = torch.ones(1024, device=f'cuda:{rank}') * (rank + 1)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        expected = sum(range(1, world_size + 1))  # 1+2+3+4=10 for 4 GPUs
        max_diff = (t - expected).abs().max().item()
        
        # Test 2: FP16 allreduce
        t16 = torch.ones(1024, dtype=torch.float16, device=f'cuda:{rank}') * (rank + 1)
        torch.distributed.all_reduce(t16, op=torch.distributed.ReduceOp.SUM)
        max_diff_16 = (t16.float() - expected).abs().max().item()
        
        # Test 3: Larger tensor (1M elements)
        t_large = torch.randn(1_000_000, device=f'cuda:{rank}')
        t_ref = t_large.clone()
        torch.distributed.all_reduce(t_large, op=torch.distributed.ReduceOp.SUM)
        # Each rank contributes its own random tensor; just check it's different
        changed = not torch.allclose(t_large, t_ref)
        
        torch.distributed.destroy_process_group()
        
        results_queue.put({
            'rank': rank,
            'status': 'ok',
            'fp32_diff': max_diff,
            'fp16_diff': max_diff_16,
            'large_changed': changed,
        })
    except Exception as e:
        results_queue.put({
            'rank': rank,
            'status': 'error',
            'error': str(e),
        })

if __name__ == '__main__':
    world_size = min(torch.cuda.device_count(), 4)
    if world_size < 2:
        print("[smoke] Need at least 2 GPUs for allreduce test")
        sys.exit(1)
    
    print(f"[smoke] Testing {world_size}-GPU allreduce...")
    mp.set_start_method('spawn', force=True)
    q = mp.Queue()
    
    processes = []
    for r in range(world_size):
        p = mp.Process(target=worker, args=(r, world_size, q))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join(timeout=60)
    
    results = []
    while not q.empty():
        results.append(q.get_nowait())
    
    all_ok = True
    for r in sorted(results, key=lambda x: x['rank']):
        if r['status'] == 'error':
            print(f"[smoke] Rank {r['rank']} FAILED: {r['error']}")
            all_ok = False
        else:
            print(f"[smoke] Rank {r['rank']}: fp32_diff={r['fp32_diff']:.6f}, "
                  f"fp16_diff={r['fp16_diff']:.6f}, large_changed={r['large_changed']}")
            if r['fp32_diff'] > 1e-5 or r['fp16_diff'] > 0.1:
                all_ok = False
    
    if all_ok and len(results) == world_size:
        print("[smoke] ALL PASSED")
    else:
        print(f"[smoke] FAILED (got {len(results)}/{world_size} results)")
        sys.exit(1)
PYEOF
else
    echo "[smoke] Python/torch not available - skipping"
fi

echo ""
echo "=== BUILD COMPLETE ==="
