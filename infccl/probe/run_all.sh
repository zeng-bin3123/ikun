#!/bin/bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
CXX=/usr/local/corex/bin/clang++
FLAGS="-x cuda --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex-3.2.3 -L/usr/local/corex/lib64 -lcudart -I/usr/local/corex/include"

echo "========== Toolchain =========="
bash "$DIR/probe_toolchain.sh"
echo ""

echo "========== Hardware =========="
$CXX $FLAGS "$DIR/probe_hw.cu" -o /tmp/probe_hw && CUDA_VISIBLE_DEVICES=0,1,2,3 /tmp/probe_hw
echo ""

echo "========== Arch Comparison =========="
for arch in ivcore10 ivcore11 ivcore20; do
    echo "--- $arch ---"
    $CXX -x cuda --cuda-gpu-arch=$arch --cuda-path=/usr/local/corex-3.2.3 -L/usr/local/corex/lib64 -lcudart -I/usr/local/corex/include "$DIR/probe_arch.cu" -o /tmp/probe_$arch -O2 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "  compiled: $(stat -c%s /tmp/probe_$arch) bytes"
        CUDA_VISIBLE_DEVICES=0 timeout 10 /tmp/probe_$arch
    else
        echo "  compile failed"
    fi
done
