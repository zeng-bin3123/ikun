#!/bin/bash
echo "=== core/ source tree ==="
find /home/claude/ikun/core -type f | sort

echo ""
echo "=== core/CMakeLists.txt FULL CONTENT ==="
cat /home/claude/ikun/core/CMakeLists.txt

echo ""
echo "=== core/runtime/ files ==="
ls -la /home/claude/ikun/core/runtime/ 2>&1

echo ""
echo "=== core/distributed_runtime/ files ==="  
ls -la /home/claude/ikun/core/distributed_runtime/ 2>&1

echo ""
echo "=== CoreX SDK location ==="
ls /usr/local/corex/include/*.h 2>/dev/null | head -10
echo "---"
find /usr/local/corex -name 'cuda_runtime.h' 2>/dev/null
echo "---"
echo "CUDA_HOME=$CUDA_HOME"
echo "PATH=$PATH"
