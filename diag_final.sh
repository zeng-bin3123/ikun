#!/bin/bash
CMAKE=/usr/local/bin/cmake
export CUDA_HOME=/usr/local/corex
export PATH=/usr/local/corex/bin:$PATH

# Save CMakeLists.txt to a readable location
cp /home/claude/ikun/core/CMakeLists.txt /home/claude/ikun/core_cmake_copy.txt 2>/dev/null

# Run cmake USE_ILU=OFF, save output
cd /home/claude/ikun && rm -rf /tmp/b_off && mkdir /tmp/b_off && cd /tmp/b_off
$CMAKE /home/claude/ikun/core -DUSE_ILU=OFF -DCMAKE_BUILD_TYPE=Release > /home/claude/ikun/cmake_off.log 2>&1
echo "OFF_EXIT=$?" >> /home/claude/ikun/cmake_off.log

# Run cmake USE_ILU=ON with CUDA, save output
cd /home/claude/ikun && rm -rf /tmp/b_on && mkdir /tmp/b_on && cd /tmp/b_on
$CMAKE /home/claude/ikun/core \
    -DUSE_ILU=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/corex \
    -DCMAKE_CUDA_COMPILER=/usr/local/corex/bin/nvcc \
    -DCMAKE_CXX_FLAGS="-I/usr/local/corex/include" \
    > /home/claude/ikun/cmake_on.log 2>&1
echo "ON_EXIT=$?" >> /home/claude/ikun/cmake_on.log

# Report sizes
wc -l /home/claude/ikun/core_cmake_copy.txt /home/claude/ikun/cmake_off.log /home/claude/ikun/cmake_on.log
echo "DONE"
