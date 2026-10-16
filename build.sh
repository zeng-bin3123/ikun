#!/bin/bash
set -e

CMAKE=/usr/local/bin/cmake
CC=gcc
CXX=g++
NVCC=/usr/local/corex/bin/nvcc

echo "=== Build Environment ==="
$CMAKE --version 2>&1 | head -1
$CC --version 2>&1 | head -1
$NVCC --version 2>&1 | tail -1
echo ""

# Build the C collective runner
echo "=== Building ikun C runner ==="
cd /home/claude/ikun
make clean 2>/dev/null || true
make build 2>&1
echo "C_BUILD=$?"
echo ""

# Build the core inference engine
echo "=== Building core inference engine ==="
cd /home/claude/ikun
rm -rf build_core
mkdir -p build_core
cd build_core
$CMAKE ../core \
    -DCMAKE_C_COMPILER=$CC \
    -DCMAKE_CXX_COMPILER=$CXX \
    -DUSE_ILU=ON \
    -DBUILD_TESTING=ON \
    2>&1
echo "CMAKE_CONFIG=$?"

make -j$(nproc) 2>&1 || echo "MAKE_FAILED"
echo "CMAKE_BUILD=$?"

echo ""
echo "=== Running ikun smoke test ==="
cd /home/claude/ikun
CUDA_VISIBLE_DEVICES=0,1,2,3 ./bin/ikun 2>&1
echo "SMOKE=$?"
