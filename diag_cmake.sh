#!/bin/bash
CMAKE=/usr/local/bin/cmake

echo "=== Core CMakeLists.txt ==="
cat /home/claude/ikun/core/CMakeLists.txt

echo ""
echo "=== CMake configure (full output) ==="
cd /home/claude/ikun
rm -rf build_diag
mkdir build_diag && cd build_diag
$CMAKE ../core -DUSE_ILU=ON -DBUILD_TESTING=ON 2>&1

echo ""
echo "CMAKE_EXIT=$?"

echo ""
echo "=== Check for Makefile ==="
ls -la Makefile 2>&1

echo ""
echo "=== CMake errors ==="
cat CMakeFiles/CMakeError.log 2>/dev/null | tail -30 || echo "no error log"

echo ""
echo "=== build_full cmake ==="
cd /home/claude/ikun
rm -rf build_full2 && mkdir build_full2 && cd build_full2
$CMAKE .. -DUSE_ILU=ON -DBUILD_TESTING=ON 2>&1
echo "FULL_CMAKE_EXIT=$?"
ls -la Makefile 2>&1

echo ""
echo "=== Make ==="
make -j4 2>&1
echo "MAKE_EXIT=$?"

echo ""
echo "=== Built artifacts ==="
find . -type f \( -executable -o -name '*.a' -o -name '*.so' \) ! -path '*/CMakeFiles/*' 2>/dev/null | sort
