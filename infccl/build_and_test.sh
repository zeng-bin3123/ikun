#!/usr/bin/env bash
# Build and test infccl collectives on BI-V100
# Outputs structured status block at the end
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

OUTDIR="$(pwd)/lib"
mkdir -p "$OUTDIR"

COREX_ROOT="${COREX_ROOT:-/usr/local/corex}"
[ ! -d "$COREX_ROOT" ] && COREX_ROOT="/usr/local/corex-3.2.3"

CXX="${COREX_ROOT}/bin/clang++"
NVCC="${COREX_ROOT}/bin/nvcc"

COMPILE_STATUS="fail"
COMPILE_ERROR=""
GPU_TEST="skip"
SMOKE="fail"
SMOKE_ERROR=""

echo "================================================================"
echo " infccl collectives — build & test"
echo "================================================================"
echo "COREX_ROOT: $COREX_ROOT"
echo "clang++:    $CXX ($([ -x "$CXX" ] && echo found || echo missing))"
echo "nvcc:       $NVCC ($([ -x "$NVCC" ] && echo found || echo missing))"
echo ""

# ---- Build ----
build_ok=true
for src in csrc/infccl_allreduce.cu csrc/infccl_collectives.cu; do
    name=$(basename "$src" .cu)
    out="$OUTDIR/${name}.so"
    built=false

    # Try clang++ first
    if [ -x "$CXX" ]; then
        echo "[BUILD] $name with clang++ ..."
        if "$CXX" -std=c++17 -O3 -shared -fPIC \
            --cuda-path="${COREX_ROOT}" \
            --cuda-gpu-arch=ivcore10 \
            --no-cuda-version-check \
            -I"${COREX_ROOT}/include" \
            -L"${COREX_ROOT}/lib64" \
            -Wl,-rpath,"${COREX_ROOT}/lib64" \
            -lcudart \
            -o "$out" "$src" 2>&1; then
            echo "[BUILD] $name — OK (clang++)"
            built=true
        else
            echo "[BUILD] $name — clang++ failed, trying nvcc"
        fi
    fi

    # Fallback to nvcc
    if [ "$built" = false ] && [ -x "$NVCC" ]; then
        echo "[BUILD] $name with nvcc ..."
        if "$NVCC" -arch=sm_70 -O3 -shared -Xcompiler -fPIC \
            -I"${COREX_ROOT}/include" \
            -L"${COREX_ROOT}/lib64" \
            -o "$out" "$src" -lcudart 2>&1; then
            echo "[BUILD] $name — OK (nvcc)"
            built=true
        else
            echo "[BUILD] $name — nvcc failed"
        fi
    fi

    if [ "$built" = false ]; then
        build_ok=false
        COMPILE_ERROR="failed to compile $name"
    fi
done

if $build_ok; then
    COMPILE_STATUS="pass"
    echo ""
    echo "[BUILD] Exported symbols:"
    nm -D "$OUTDIR/infccl_allreduce.so" 2>/dev/null | grep ' T infccl_' || true
    nm -D "$OUTDIR/infccl_collectives.so" 2>/dev/null | grep ' T infccl_' || true
    echo ""
    ls -lh "$OUTDIR"/infccl_*.so
else
    echo ""
    echo "[BUILD] FAILED: $COMPILE_ERROR"
fi

# ---- P2P test ----
echo ""
echo "================================================================"
echo " GPU P2P Connectivity Test"
echo "================================================================"

PYTHON="${COREX_ROOT}/lib64/python3/dist-packages/../../../bin/python3"
[ ! -x "$PYTHON" ] && PYTHON="python3"

GPU_PAIRS_OK=""
GPU_PAIRS_FAIL=""

$PYTHON -c "
import torch
n = torch.cuda.device_count()
print(f'CUDA devices: {n}')
ok, fail = [], []
for i in range(n):
    for j in range(n):
        if i == j: continue
        can = torch.cuda.can_device_access_peer(i, j)
        (ok if can else fail).append(f'{i}-{j}')
print(f'P2P_OK={','.join(ok)}')
print(f'P2P_FAIL={','.join(fail)}')
" 2>&1 | while read line; do
    echo "$line"
    if [[ "$line" == P2P_OK=* ]]; then
        echo "${line#P2P_OK=}" > /tmp/infccl_p2p_ok.txt
    fi
    if [[ "$line" == P2P_FAIL=* ]]; then
        echo "${line#P2P_FAIL=}" > /tmp/infccl_p2p_fail.txt
    fi
done

GPU_PAIRS_OK=$(cat /tmp/infccl_p2p_ok.txt 2>/dev/null || echo "")
GPU_PAIRS_FAIL=$(cat /tmp/infccl_p2p_fail.txt 2>/dev/null || echo "")

# ---- Multi-GPU smoke test ----
if [ "$COMPILE_STATUS" = "pass" ]; then
    echo ""
    echo "================================================================"
    echo " Multi-GPU Smoke Test"
    echo "================================================================"

    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

    if $PYTHON tests/test_collectives_gpu.py 2>&1; then
        SMOKE="pass"
        GPU_TEST="pass"
    else
        SMOKE="fail"
        SMOKE_ERROR="multi-gpu test failed"
        GPU_TEST="fail"
    fi
else
    SMOKE_ERROR="compile failed"
    GPU_TEST="skip"
fi

# ---- Output status block ----
echo ""
echo "================================================================"
echo " STATUS BLOCK"
echo "================================================================"
cat << EOF
compile=$COMPILE_STATUS
compile_error=$COMPILE_ERROR
needs_source=false
needs_source_files=
needs_source_reason=
template_fill=adjObj→InfcclColl, prjname→infccl, typenameType→InfcclCollComm, AllGatherKernel→infccl_allgather_kernel, BroadcastKernel→infccl_broadcast_kernel, ReduceKernel→infccl_reduce_kernel, ReduceScatterKernel→infccl_reduce_scatter_kernel, GetBlock→rank%world_size, BcastRole→(rank==root?SEND:RECV), ReduceRole→(rank==root?ROOT:LEAF)
gpu_pairs_ok=$GPU_PAIRS_OK
gpu_pairs_fail=$GPU_PAIRS_FAIL
gpu_test=$GPU_TEST
smoke=$SMOKE
smoke_error=$SMOKE_ERROR
files_written=infccl/csrc/infccl_collectives.cu,infccl/csrc/infccl_allreduce.cu,infccl/python/infccl_collectives.py,infccl/tests/test_collectives_gpu.py,infccl/build_collectives.sh,infccl/build_and_test.sh
EOF

