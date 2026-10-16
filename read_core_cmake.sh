#!/bin/bash
echo "=== core/CMakeLists.txt (FULL) ==="
cat /home/claude/ikun/core/CMakeLists.txt

echo ""
echo "=== core source files ==="
find /home/claude/ikun/core -type f \( -name '*.cpp' -o -name '*.h' -o -name '*.c' \) | sort

echo ""
echo "=== cmake error after configure ==="
cd /home/claude/ikun
rm -rf build_check
mkdir build_check && cd build_check
/usr/local/bin/cmake ../core -DUSE_ILU=ON 2>&1 | tail -30
echo "EXIT=$?"

echo ""
echo "=== Makefile? ==="
ls Makefile 2>&1
