#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p build && cd build
# CoreX auto-detection finds Iluvatar when /usr/local/corex exists.
# Explicit -DWITH_ILUVATAR=ON covers cases where detection fails.
cmake .. \
    -DWITH_NCCL=ON \
    -DWITH_ILUVATAR=ON \
    -DNCCL_LIB=/usr/local/corex/lib64/libnccl.so \
    -DNCCL_INC=/usr/local/corex/include \
    -DCMAKE_CUDA_COMPILER=/usr/local/corex/bin/nvcc \
    -DCMAKE_CXX_FLAGS="-pthread" \
    -DCMAKE_EXE_LINKER_FLAGS="-lpthread" \
    -DCMAKE_SHARED_LINKER_FLAGS="-lpthread" \
    2>&1 | tail -20
make -j$(nproc) 2>&1 | tail -10
echo "=== built ==="
ls -lh src/libinfiniccl.so
