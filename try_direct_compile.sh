#!/bin/bash
export PATH=/usr/local/corex/bin:$PATH

echo "=== Core source files ==="
find /home/claude/ikun/core -name '*.cpp' | sort

echo ""
echo "=== Try compiling each .cpp to .o ==="
for f in $(find /home/claude/ikun/core -name '*.cpp' | sort); do
    echo "--- Compiling: $f ---"
    g++ -c -std=c++17 -O2 \
        -I/home/claude/ikun/core \
        -I/home/claude/ikun/core/common \
        -I/home/claude/ikun/core/config \
        -I/home/claude/ikun/core/framework/config \
        -I/home/claude/ikun/core/framework/kv_cache \
        -I/home/claude/ikun/core/distributed_runtime \
        -I/home/claude/ikun/core/runtime \
        -I/usr/local/corex/include \
        -I/usr/include \
        "$f" -o /tmp/$(basename "$f" .cpp).o 2>&1 | head -5
    echo "EXIT=$?"
    echo ""
done

echo "=== Summary ==="
ls -la /tmp/*.o 2>/dev/null
