#!/usr/bin/env bash
# Build infccl shared library for Iluvatar BI-V100
# CoreX SDK uses clang++ with --cuda-gpu-arch=ivcore10
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

OUTDIR="$(pwd)/lib"
mkdir -p "$OUTDIR"

# Detect CoreX root
for candidate in "${COREX_ROOT}" /usr/local/corex /usr/local/corex-3.2.3; do
    [ -z "$candidate" ] && continue
    if [ -x "${candidate}/bin/clang++" ]; then
        COREX_ROOT="$candidate"
        break
    fi
done

CXX="${COREX_ROOT:-/usr/local/corex}/bin/clang++"

if [ ! -x "$CXX" ]; then
    # Fallback to nvcc
    for nvcc_candidate in /usr/local/corex/bin/nvcc /usr/local/corex-3.2.3/bin/nvcc /usr/local/cuda/bin/nvcc nvcc; do
        if command -v "$nvcc_candidate" &>/dev/null 2>&1 || [ -x "$nvcc_candidate" ]; then
            NVCC="$nvcc_candidate"
            break
        fi
    done
    if [ -n "$NVCC" ]; then
        echo "[infccl] Using nvcc: $NVCC"
        $NVCC -arch=sm_70 -O3 -shared -Xcompiler -fPIC \
            -o "$OUTDIR/infccl_allreduce.so" csrc/infccl_allreduce.cu -lcudart
    else
        echo "[infccl] ERROR: No CUDA compiler found" >&2
        exit 1
    fi
else
    echo "[infccl] Using CoreX clang++: $CXX"
    "$CXX" -std=c++17 -O3 -shared -fPIC \
        --cuda-path="${COREX_ROOT}" \
        --cuda-gpu-arch=ivcore10 \
        --no-cuda-version-check \
        -I"${COREX_ROOT}/include" \
        -L"${COREX_ROOT}/lib64" \
        -Wl,-rpath,"${COREX_ROOT}/lib64" \
        -lcudart \
        -o "$OUTDIR/infccl_allreduce.so" csrc/infccl_allreduce.cu
fi

echo "[infccl] Exported symbols:"
nm -D "$OUTDIR/infccl_allreduce.so" 2>/dev/null | grep ' T infccl_' || true

ls -lh "$OUTDIR/infccl_allreduce.so"
echo "[infccl] Build complete."
