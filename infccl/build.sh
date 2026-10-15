#!/usr/bin/env bash
# Build infccl shared library for Iluvatar BI-V100
#
# Usage:
#   ./build.sh             # auto-detect compiler
#   NVCC=/path/to/nvcc ./build.sh  # explicit compiler
#
# The BI-V100 uses CoreX SDK 3.2.3 with CUDA 10.2 compatible toolchain.
# SM 7.0 (V100 class) architecture.

set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

OUTDIR="$(pwd)/lib"
mkdir -p "$OUTDIR"

# Detect compiler (prefer CoreX)
if [ -n "$NVCC" ]; then
    echo "[infccl] Using NVCC=$NVCC"
elif [ -x /usr/local/corex/bin/nvcc ]; then
    NVCC=/usr/local/corex/bin/nvcc
    echo "[infccl] Using CoreX nvcc: $NVCC"
elif [ -x /usr/local/corex-3.2.3/bin/nvcc ]; then
    NVCC=/usr/local/corex-3.2.3/bin/nvcc
    echo "[infccl] Using CoreX 3.2.3 nvcc: $NVCC"
elif command -v nvcc &>/dev/null; then
    NVCC=nvcc
    echo "[infccl] Using system nvcc: $(which nvcc)"
else
    echo "[infccl] ERROR: No CUDA compiler found" >&2
    echo "[infccl] Set NVCC=/path/to/nvcc or install CoreX SDK" >&2
    exit 1
fi

# Architecture: BI-V100 = SM 7.0
ARCH_FLAGS="-arch=sm_70 -gencode=arch=compute_70,code=sm_70"

# CoreX-specific: add SDK include/lib paths if available
COREX_ROOT="${COREX_ROOT:-/usr/local/corex}"
EXTRA_FLAGS=""
if [ -d "$COREX_ROOT/include" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS -I${COREX_ROOT}/include"
fi
if [ -d "$COREX_ROOT/lib64" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS -L${COREX_ROOT}/lib64"
fi

echo "[infccl] Building infccl_allreduce.so ..."
echo "[infccl]   Arch: sm_70 (BI-V100)"
echo "[infccl]   Extra: $EXTRA_FLAGS"

$NVCC $ARCH_FLAGS -O3 -shared -Xcompiler -fPIC \
    -Xcompiler -Wall \
    $EXTRA_FLAGS \
    -o "$OUTDIR/infccl_allreduce.so" \
    csrc/infccl_allreduce.cu \
    -lcudart

echo "[infccl] Built: $OUTDIR/infccl_allreduce.so"
ls -lh "$OUTDIR/infccl_allreduce.so"

# Verify exported symbols
echo "[infccl] Exported symbols:"
nm -D "$OUTDIR/infccl_allreduce.so" 2>/dev/null | grep ' T infccl_' || \
    echo "[infccl] (nm not available, skipping symbol check)"

echo "[infccl] Build complete."
