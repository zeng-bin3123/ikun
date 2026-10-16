#!/bin/bash
set -e
cd /home/claude/ikun

echo "=== Step 1: Build C runner via Makefile ==="
make clean 2>/dev/null || true
make build
echo "C_BUILD_OK"

echo ""
echo "=== Step 2: CMake configure ==="
rm -rf build_full
mkdir -p build_full
cd build_full
/usr/local/bin/cmake .. \
    -DUSE_ILU=ON \
    -DBUILD_TESTING=ON \
    -DCMAKE_BUILD_TYPE=Release \
    2>&1
echo "CMAKE_CONFIG_OK"

echo ""
echo "=== Step 3: CMake build ==="
make -j$(nproc) 2>&1
echo "CMAKE_BUILD_EXIT=$?"

echo ""
echo "=== Step 4: Built artifacts ==="
find . -type f -executable | sort
find . -name '*.a' -o -name '*.so' | sort

echo ""
echo "=== Step 5: Smoke test ==="
cd /home/claude/ikun
CUDA_VISIBLE_DEVICES=0,1,2,3 ./bin/ikun 2>&1
echo "SMOKE_EXIT=$?"
