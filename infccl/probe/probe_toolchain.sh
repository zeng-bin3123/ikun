#!/bin/bash
echo "========== BI-V100 Toolchain Probe =========="
echo ""
echo "=== nvcc identity ==="
head -c 20 /usr/local/corex/bin/nvcc | xxd | head -1
stat -c "nvcc size: %s bytes" /usr/local/corex/bin/nvcc
stat -c "clang-16 size: %s bytes" /usr/local/corex/bin/clang-16
echo ""
echo "=== llc CPU targets ==="
/usr/local/corex/bin/llc -march=bi -mattr=help 2>&1 | grep -E '^\s+(generic|ivcore|sm_bi)' | head -10
echo ""
echo "=== llc features ==="
/usr/local/corex/bin/llc -march=bi -mattr=help 2>&1 | grep -E '^\s+\S' | grep -v 'generic\|ivcore\|sm_bi' | head -50
echo ""
echo "=== CUDA version ==="
grep -E 'CUDART_VERSION|CUDA_VERSION' /usr/local/corex/include/cuda_runtime_api.h /usr/local/corex/include/cuda.h 2>/dev/null | grep -v defgroup | grep -v END
echo ""
echo "=== cooperative_groups ==="
ls /usr/local/corex/include/cooperative_groups* 2>/dev/null || echo "not found"
echo ""
echo "=== FP16 intrinsics count ==="
grep -c 'hfma\|h2exp\|__hmul\|__hadd\|__hfma2\|__half2half2' /usr/local/corex/include/cuda_fp16.h 2>/dev/null
echo ""
echo "=== libdevice ==="
find /usr/local/corex* -name 'libdevice*.bc' 2>/dev/null
echo ""
echo "=== NCCL version ==="
grep -E 'NCCL_MAJOR|NCCL_MINOR|NCCL_PATCH' /usr/local/corex/include/nccl.h 2>/dev/null | head -3
echo ""
echo "=== PTXToLLVM ==="
/usr/local/corex/bin/PTXToLLVM --help 2>&1 | head -3
echo ""
echo "========== done =========="
