#!/bin/bash
CMAKE=/usr/local/bin/cmake

# Just the core cmake + make, nothing else
cd /home/claude/ikun
rm -rf build_core
mkdir build_core && cd build_core

echo "=== CMAKE CONFIGURE ==="
$CMAKE ../core -DUSE_ILU=ON -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release 2>&1
echo ""
echo "CMAKE_EXIT=$?"

echo ""
echo "=== MAKEFILE CHECK ==="
ls -la Makefile 2>&1

echo ""
echo "=== MAKE ==="
make -j$(nproc) 2>&1
echo ""
echo "MAKE_EXIT=$?"

echo ""
echo "=== ARTIFACTS ==="
find . -type f \( -name '*.a' -o -name '*.so' -o -executable \) ! -path '*/CMakeFiles/*' | sort 2>&1
