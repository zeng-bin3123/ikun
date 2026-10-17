#!/bin/bash
# build_and_test.sh — Build and test infccl v0.1 on BI-V100
# Run this script on a machine with 4x BI-V100 GPUs and CoreX compiler
set -e

CXX="${CXX:-/usr/local/corex/bin/clang++}"
CUDA_PATH="${CUDA_PATH:-/usr/local/corex-3.2.3}"
ARCH="${CUDA_ARCH:-ivcore10}"

echo "=== infccl v0.1 build ==="
echo "CXX=$CXX"
echo "CUDA_PATH=$CUDA_PATH"
echo "ARCH=$ARCH"

if [ ! -x "$CXX" ]; then
    echo "ERROR: $CXX not found"
    echo "Set CXX=/path/to/clang++"
    exit 1
fi

CXXFLAGS="-x cuda --cuda-gpu-arch=$ARCH --cuda-path=$CUDA_PATH -I$CUDA_PATH/include -Isrc -O2 -std=c++14"
LDFLAGS="-L$CUDA_PATH/lib64 -lcudart -lpthread"
SRCS="src/core.cu src/all_reduce.cu src/reduce.cu src/broadcast.cu src/all_gather.cu src/reduce_scatter.cu"

TESTS="all_reduce_test reduce_scatter_test all_gather_test broadcast_test reduce_test"

echo ""
echo "=== Compiling ==="
for t in $TESTS; do
    echo "  Building $t..."
    $CXX $CXXFLAGS $SRCS test/${t}.cu -o $t $LDFLAGS
    echo "  OK"
done

echo ""
echo "=== Running tests ==="
PASS=0
FAIL=0
for t in $TESTS; do
    echo ""
    echo "--- $t ---"
    if ./$t; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "=== Results: $PASS/$((PASS+FAIL)) passed, $FAIL failed ==="
exit $FAIL
