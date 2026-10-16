#!/bin/bash
cd /home/claude/ikun
rm -rf /tmp/cmake_build
mkdir /tmp/cmake_build && cd /tmp/cmake_build

# Run cmake, capture ALL output
/usr/local/bin/cmake /home/claude/ikun/core \
    -DUSE_ILU=ON \
    -DBUILD_TESTING=ON \
    > /home/claude/ikun/cmake_output.log 2>&1
echo "CMAKE_EXIT=$?" >> /home/claude/ikun/cmake_output.log

# Show the full output
echo "=== CMAKE OUTPUT ==="
cat /home/claude/ikun/cmake_output.log

echo ""
echo "=== MAKEFILE? ==="
ls -la Makefile 2>&1

echo ""
echo "=== CMAKE ERROR LOG ==="
cat CMakeFiles/CMakeError.log 2>&1 | head -50
