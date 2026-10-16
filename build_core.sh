#!/bin/bash
set -e
CMAKE=/usr/local/bin/cmake

echo "=== Environment ==="
$CMAKE --version 2>&1 | head -1
gcc --version 2>&1 | head -1
uname -r
cat /etc/os-release 2>/dev/null | grep PRETTY_NAME

echo ""
echo "=== Step 1: Check dependencies ==="
NEED_INSTALL=""

if ! pkg-config --exists libglog 2>/dev/null; then
    echo "glog: NOT FOUND"
    NEED_INSTALL="$NEED_INSTALL libgoogle-glog-dev"
else
    echo "glog: $(pkg-config --modversion libglog)"
fi

if ! pkg-config --exists gflags 2>/dev/null; then
    if ! ldconfig -p | grep -q libgflags; then
        echo "gflags: NOT FOUND"
        NEED_INSTALL="$NEED_INSTALL libgflags-dev"
    else
        echo "gflags: found via ldconfig"
    fi
else
    echo "gflags: found"
fi

if [ -n "$NEED_INSTALL" ]; then
    echo "Installing: $NEED_INSTALL"
    apt-get update -qq 2>/dev/null && apt-get install -y -qq $NEED_INSTALL 2>&1 | tail -5
    if [ $? -ne 0 ]; then
        echo "WARNING: apt-get failed. Trying pip cmake approach..."
    fi
fi

echo ""
echo "=== Step 2: Build C runner ==="
cd /home/claude/ikun
make build 2>&1
echo "C_BUILD=$?"

echo ""
echo "=== Step 3: CMake configure (core) ==="
cd /home/claude/ikun
rm -rf build_core
mkdir build_core && cd build_core
$CMAKE ../core \
    -DUSE_ILU=ON \
    -DBUILD_TESTING=ON \
    -DCMAKE_BUILD_TYPE=Release \
    2>&1
CMAKE_RC=$?
echo "CMAKE_CONFIG=$CMAKE_RC"

if [ $CMAKE_RC -eq 0 ]; then
    echo ""
    echo "=== Step 4: Make ==="
    make -j$(nproc) 2>&1
    echo "MAKE=$?"
    
    echo ""
    echo "=== Built artifacts ==="
    find . -type f \( -name '*.a' -o -name '*.so' -o -executable \) ! -path '*/CMakeFiles/*' 2>/dev/null | sort
fi

echo ""
echo "=== Step 5: Smoke test ==="
cd /home/claude/ikun
CUDA_VISIBLE_DEVICES=0,1,2,3 ./bin/ikun 2>&1 | grep -E '(PASS|FAIL|tests passed|P2P|GPUs:|GPU[0-9] pool)'
echo "SMOKE=$?"
