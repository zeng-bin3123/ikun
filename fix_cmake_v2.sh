#!/bin/bash
CMAKE=/usr/local/bin/cmake
export CUDA_HOME=/usr/local/corex
export PATH=/usr/local/corex/bin:$PATH

cd /home/claude/ikun
rm -rf build_v2
mkdir build_v2 && cd build_v2

# Try with explicit include paths for CoreX
$CMAKE ../core \
    -DUSE_ILU=ON \
    -DBUILD_TESTING=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/corex \
    -DCMAKE_CUDA_COMPILER=/usr/local/corex/bin/nvcc \
    -DCMAKE_CXX_FLAGS="-I/usr/local/corex/include" \
    2>&1

echo "CMAKE_EXIT=$?"
echo ""
ls Makefile 2>&1

if [ -f Makefile ]; then
    make -j$(nproc) 2>&1 | tail -20
    echo "MAKE_EXIT=$?"
fi
