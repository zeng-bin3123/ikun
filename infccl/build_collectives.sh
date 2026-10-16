#!/usr/bin/env bash
# Build infccl collectives for Iluvatar BI-V100
# Supports both CoreX clang++ and CoreX nvcc compilers
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

OUTDIR="$(pwd)/lib"
mkdir -p "$OUTDIR"

COREX_ROOT="${COREX_ROOT:-/usr/local/corex}"
# Fallback to corex-3.2.3 if /usr/local/corex doesn't exist
[ ! -d "$COREX_ROOT" ] && COREX_ROOT="/usr/local/corex-3.2.3"

# Prefer clang++ (matches existing repo build pattern), fallback to nvcc
CXX="${COREX_ROOT}/bin/clang++"
NVCC="${COREX_ROOT}/bin/nvcc"

build_with_clang() {
    local src="$1" out="$2"
    echo "[infccl] Building $out (clang++) ..."
    "$CXX" -std=c++17 -O3 -shared -fPIC \
        --cuda-path="${COREX_ROOT}" \
        --cuda-gpu-arch=ivcore10 \
        --no-cuda-version-check \
        -I"${COREX_ROOT}/include" \
        -L"${COREX_ROOT}/lib64" \
        -Wl,-rpath,"${COREX_ROOT}/lib64" \
        -lcudart \
        -o "$out" "$src"
    echo "[infccl] Built: $out"
}

build_with_nvcc() {
    local src="$1" out="$2"
    echo "[infccl] Building $out (nvcc) ..."
    "$NVCC" -arch=sm_70 -O3 -shared -Xcompiler -fPIC \
        -I"${COREX_ROOT}/include" \
        -L"${COREX_ROOT}/lib64" \
        -o "$out" "$src" -lcudart
    echo "[infccl] Built: $out"
}

build_one() {
    local src="$1" out="$2"
    if [ -x "$CXX" ]; then
        if build_with_clang "$src" "$out" 2>&1; then
            return 0
        else
            echo "[infccl] clang++ failed, falling back to nvcc"
        fi
    fi
    if [ -x "$NVCC" ]; then
        build_with_nvcc "$src" "$out"
    else
        echo "[infccl] ERROR: No compiler found at $CXX or $NVCC" >&2
        return 1
    fi
}

echo "[infccl] CoreX root: $COREX_ROOT"
echo "[infccl] clang++: $CXX ($([ -x "$CXX" ] && echo found || echo missing))"
echo "[infccl] nvcc:    $NVCC ($([ -x "$NVCC" ] && echo found || echo missing))"

build_one csrc/infccl_allreduce.cu "$OUTDIR/infccl_allreduce.so"
build_one csrc/infccl_collectives.cu "$OUTDIR/infccl_collectives.so"

echo ""
echo "[infccl] Exported symbols (allreduce):"
nm -D "$OUTDIR/infccl_allreduce.so" 2>/dev/null | grep ' T infccl_' || true

echo "[infccl] Exported symbols (collectives):"
nm -D "$OUTDIR/infccl_collectives.so" 2>/dev/null | grep ' T infccl_' || true

echo ""
ls -lh "$OUTDIR"/infccl_*.so
echo "[infccl] Build complete."
