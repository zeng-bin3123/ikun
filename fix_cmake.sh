#!/bin/bash
CMAKE=/usr/local/bin/cmake

# CoreX SDK acts as CUDA
export CUDA_HOME=/usr/local/corex
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/corex
export PATH=/usr/local/corex/bin:$PATH

echo "=== Environment ==="
echo "CUDA_HOME=$CUDA_HOME"
echo "NVCC=$(which nvcc)"
nvcc --version 2>&1 | tail -1

echo ""
echo "=== Core CMakeLists.txt (full) ==="
cat /home/claude/ikun/core/CMakeLists.txt

echo ""
echo "=== CMake configure ==="
cd /home/claude/ikun
rm -rf build_fixed
mkdir build_fixed && cd build_fixed
$CMAKE ../core \
    -DUSE_ILU=ON \
    -DBUILD_TESTING=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/corex \
    -DCMAKE_CUDA_COMPILER=/usr/local/corex/bin/nvcc \
    2>&1
echo ""
echo "CMAKE_EXIT=$?"

echo ""
echo "=== Makefile? ==="
ls -la Makefile 2>&1

echo ""
if [ -f Makefile ]; then
    echo "=== Make ==="
    make -j$(nproc) 2>&1 | tail -30
    echo "MAKE_EXIT=$?"
    echo ""
    echo "=== Artifacts ==="
    find . -type f \( -name '*.a' -o -name '*.so' -o -executable \) ! -path '*/CMakeFiles/*' | sort
fi
