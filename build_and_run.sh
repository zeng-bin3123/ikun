#!/bin/bash
set -eo pipefail
cd /home/claude/ikun

echo "=== Build Phase ==="

# Find nvcc
NVCC=""
for p in /usr/local/corex/bin/nvcc /usr/local/corex-3.2.3/bin/nvcc; do
    if [ -x "$p" ]; then NVCC="$p"; break; fi
done
if [ -z "$NVCC" ]; then
    NVCC=$(which nvcc 2>/dev/null || true)
fi

if [ -n "$NVCC" ]; then
    echo "nvcc found: $NVCC"
    "$NVCC" -O2 gpu_p2p_test.cu \
        -o gpu_p2p_cuda_bin \
        -L/usr/local/iluvatar/lib64 \
        -L/usr/local/corex/lib64 \
        -L/usr/local/corex-3.2.3/lib64 \
        -lcudart 2>&1
    echo "nvcc compile exit: $?"
else
    echo "nvcc not found, skipping .cu build"
fi

# Always build dlopen version
g++ -O2 -Wall -Wno-unused-variable -o gpu_p2p_dlopen_bin gpu_p2p_test.cpp -ldl 2>&1
echo "g++ compile exit: $?"

echo ""
echo "=== Run Phase ==="
export CUDA_VISIBLE_DEVICES=0,1,2,3
export LD_LIBRARY_PATH=/usr/local/iluvatar/lib64:/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib64:${LD_LIBRARY_PATH:-}

echo "--- dlopen version ---"
./gpu_p2p_dlopen_bin 2>&1
echo "dlopen exit: $?"

if [ -x gpu_p2p_cuda_bin ]; then
    echo ""
    echo "--- cuda runtime version ---"
    ./gpu_p2p_cuda_bin 2>&1
    echo "cuda exit: $?"
fi
