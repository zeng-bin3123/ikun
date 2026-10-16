#!/bin/bash
CMAKE=/usr/local/bin/cmake
cd /home/claude/ikun
rm -rf build_err
mkdir build_err && cd build_err

# Capture FULL cmake output to a file
$CMAKE ../core -DUSE_ILU=ON -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release > /tmp/cmake_out.txt 2>&1
EXIT=$?

echo "CMAKE_EXIT=$EXIT"
echo ""
echo "=== FULL CMAKE OUTPUT ==="
cat /tmp/cmake_out.txt
echo ""
echo "=== MAKEFILE EXISTS? ==="
ls -la Makefile 2>&1
echo ""
echo "=== CMAKE ERROR LOG ==="
cat CMakeFiles/CMakeError.log 2>/dev/null || echo "no error log"
