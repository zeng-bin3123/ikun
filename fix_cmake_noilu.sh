#!/bin/bash
CMAKE=/usr/local/bin/cmake

cd /home/claude/ikun
rm -rf build_noilu
mkdir build_noilu && cd build_noilu

# Try WITHOUT USE_ILU — should be simpler, no CUDA needed
$CMAKE ../core \
    -DUSE_ILU=OFF \
    -DBUILD_TESTING=ON \
    -DCMAKE_BUILD_TYPE=Release \
    2>&1

echo ""
echo "CMAKE_EXIT=$?"
echo ""
ls Makefile 2>&1

if [ -f Makefile ]; then
    echo "=== Building ==="
    make -j$(nproc) 2>&1 | tail -20
    echo "MAKE_EXIT=$?"
    echo "=== Artifacts ==="
    find . -type f \( -name '*.a' -o -name '*.so' -o -executable \) ! -path '*/CMakeFiles/*' | sort
fi
