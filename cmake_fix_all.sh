#!/bin/bash
CMAKE=/usr/local/bin/cmake
export CUDA_HOME=/usr/local/corex
export PATH=/usr/local/corex/bin:$PATH

echo "=== FULL core/CMakeLists.txt ==="
cat /home/claude/ikun/core/CMakeLists.txt
echo ""

echo "=== Attempt 1: USE_ILU=OFF (no CUDA needed) ==="
cd /home/claude/ikun && rm -rf /tmp/b1 && mkdir /tmp/b1 && cd /tmp/b1
$CMAKE /home/claude/ikun/core -DUSE_ILU=OFF -DCMAKE_BUILD_TYPE=Release 2>&1
echo "EXIT1=$?"
ls Makefile 2>&1
echo ""

echo "=== Attempt 2: USE_ILU=ON + CUDA paths ==="
cd /home/claude/ikun && rm -rf /tmp/b2 && mkdir /tmp/b2 && cd /tmp/b2
$CMAKE /home/claude/ikun/core \
    -DUSE_ILU=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/corex \
    -DCMAKE_CUDA_COMPILER=/usr/local/corex/bin/nvcc \
    -DCMAKE_CXX_FLAGS="-I/usr/local/corex/include" \
    2>&1
echo "EXIT2=$?"
ls Makefile 2>&1
echo ""

echo "=== Attempt 3: USE_ILU=ON + CUDA as CXX only (no nvcc) ==="
cd /home/claude/ikun && rm -rf /tmp/b3 && mkdir /tmp/b3 && cd /tmp/b3
$CMAKE /home/claude/ikun/core \
    -DUSE_ILU=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-I/usr/local/corex/include" \
    2>&1
echo "EXIT3=$?"
ls Makefile 2>&1

echo ""
echo "=== If any succeeded, build it ==="
for d in /tmp/b1 /tmp/b2 /tmp/b3; do
    if [ -f "$d/Makefile" ]; then
        echo "Building $d..."
        cd "$d" && make -j$(nproc) 2>&1 | tail -10
        echo "MAKE_EXIT=$?"
        find . -type f \( -name '*.a' -o -name '*.so' -o -executable \) ! -path '*/CMakeFiles/*' | sort
        break
    fi
done
