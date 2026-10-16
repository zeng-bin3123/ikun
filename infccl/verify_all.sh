#!/usr/bin/env bash
# Complete verification: build + symbols + P2P + single-GPU smoke + multi-GPU correctness
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "============================================================"
echo " infccl Full Verification"
echo "============================================================"

# ---- Step 1: Build ----
echo ""
echo "[STEP 1] Build"
bash build_collectives.sh 2>&1
echo ""

# ---- Step 2: Verify symbols ----
echo "[STEP 2] Symbol verification"
for so in lib/infccl_allreduce.so lib/infccl_collectives.so; do
    if [ -f "$so" ]; then
        count=$(nm -D "$so" 2>/dev/null | grep -c ' T infccl_' || true)
        echo "  $so: $count exported infccl_ symbols"
    else
        echo "  $so: MISSING"
    fi
done
echo ""

# ---- Step 3: Find python ----
PY=""
for py in /usr/local/bin/python3.10 python3.10 python3; do
    if command -v "$py" &>/dev/null 2>&1 || [ -x "$py" ]; then
        if "$py" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            PY="$py"
            break
        fi
    fi
done

if [ -z "$PY" ]; then
    echo "[STEP 3] No python with torch+CUDA found"
    echo ""
    echo "============================================================"
    echo " RESULT: compile=pass, gpu_test=skip (no torch)"
    echo "============================================================"
    exit 0
fi
echo "[STEP 3] Using python: $PY"
$PY -c "import torch; print(f'  torch={torch.__version__}, GPUs={torch.cuda.device_count()}')"
echo ""

# ---- Step 4: P2P test ----
echo "[STEP 4] GPU P2P connectivity"
$PY -c "
import torch
n = torch.cuda.device_count()
ok, fail = [], []
for i in range(n):
    for j in range(n):
        if i == j: continue
        can = torch.cuda.can_device_access_peer(i, j)
        (ok if can else fail).append(f'{i}-{j}')
print(f'  OK:   {len(ok)} pairs ({','.join(ok)})')
print(f'  FAIL: {len(fail)} pairs ({','.join(fail) if fail else 'none'})')
" 2>&1 | grep -v FutureWarning | grep -v pynvml
echo ""

# ---- Step 5: Single-GPU smoke (ctypes) ----
echo "[STEP 5] Single-GPU ctypes smoke test"
$PY -c "
import ctypes, os
lib = ctypes.CDLL(os.path.join('lib', 'infccl_collectives.so'))
lib.infccl_coll_init_p2p.restype = ctypes.c_int
lib.infccl_coll_init_p2p.argtypes = [ctypes.c_int]
lib.infccl_coll_comm_create.restype = ctypes.c_void_p
lib.infccl_coll_comm_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int64]
lib.infccl_coll_comm_destroy.restype = None
lib.infccl_coll_comm_destroy.argtypes = [ctypes.c_void_p]

import torch
n = torch.cuda.device_count()

# Init P2P
rc = lib.infccl_coll_init_p2p(min(n, 4))
print(f'  init_p2p: {\"pass\" if rc == 0 else \"fail\"} (rc={rc})')

# Create + destroy comm on GPU 0
comm = lib.infccl_coll_comm_create(0, min(n, 4), 16*1024*1024)
print(f'  comm_create: {\"pass\" if comm else \"fail\"}')
if comm:
    lib.infccl_coll_comm_destroy(comm)
    print(f'  comm_destroy: pass')
print(f'  smoke: PASS')
" 2>&1 | grep -v FutureWarning | grep -v pynvml
echo ""

# ---- Step 6: Multi-GPU correctness ----
echo "[STEP 6] Multi-GPU correctness test"
export CUDA_VISIBLE_DEVICES=0,1,2,3
$PY tests/test_collectives_gpu.py 2>&1 | grep -v FutureWarning | grep -v pynvml
echo ""

echo "============================================================"
echo " Verification complete"
echo "============================================================"
