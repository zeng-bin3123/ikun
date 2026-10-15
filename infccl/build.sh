#!/usr/bin/env bash
# Build infccl shared library for Iluvatar BI-V100
#
# Usage:
#   ./build.sh             # auto-detect compiler
#   NVCC=/path/to/nvcc ./build.sh  # explicit compiler
#
# The BI-V100 uses CoreX SDK with CUDA 10.2 compatible toolchain.
# The clang-based compiler is at /usr/local/corex/bin/clang++
# but we can also use the nvcc wrapper if available.

set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

OUTDIR="$(pwd)/lib"
mkdir -p "$OUTDIR"

# Detect compiler
if [ -n "$NVCC" ]; then
    echo "[infccl] Using NVCC=$NVCC"
elif command -v /usr/local/corex/bin/nvcc &>/dev/null; then
    NVCC=/usr/local/corex/bin/nvcc
    echo "[infccl] Using CoreX nvcc: $NVCC"
elif command -v nvcc &>/dev/null; then
    NVCC=nvcc
    echo "[infccl] Using system nvcc: $(which nvcc)"
else
    echo "[infccl] ERROR: No CUDA compiler found" >&2
    exit 1
fi

# Detect architecture
# BI-V100 is SM 7.0 compatible (V100 class)
ARCH_FLAGS="-arch=sm_70"

echo "[infccl] Building infccl_allreduce.so ..."
$NVCC $ARCH_FLAGS -O3 -shared -Xcompiler -fPIC \
    -o "$OUTDIR/infccl_allreduce.so" \
    csrc/infccl_allreduce.cu \
    -lcudart -lcuda

echo "[infccl] Built: $OUTDIR/infccl_allreduce.so"
ls -lh "$OUTDIR/infccl_allreduce.so"

echo "[infccl] Build complete."
