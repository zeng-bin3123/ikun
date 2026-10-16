#!/usr/bin/env bash
# Test cmake build in both modes
set -eo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$(pwd)"

echo "=== CMAKE BUILD TEST ==="
echo ""

# --- USE_ILU=OFF ---
echo "--- Build 1: USE_ILU=OFF ---"
BUILD_OFF="$ROOT/build_cmake_off"
rm -rf "$BUILD_OFF"
mkdir -p "$BUILD_OFF"
cd "$BUILD_OFF"
cmake "$ROOT/core" -DUSE_ILU=OFF -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release 2>&1
make -j$(nproc) 2>&1
echo "[cmake OFF] Build successful"
ls -la *.a 2>/dev/null || ls -la lib*.a 2>/dev/null || echo "(static lib location may vary)"
echo ""

# --- USE_ILU=ON ---
echo "--- Build 2: USE_ILU=ON ---"
BUILD_ON="$ROOT/build_cmake_on"
rm -rf "$BUILD_ON"
mkdir -p "$BUILD_ON"
cd "$BUILD_ON"
cmake "$ROOT/core" -DUSE_ILU=ON -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release 2>&1
make -j$(nproc) 2>&1
echo "[cmake ON] Build successful"
ls -la *.a 2>/dev/null || ls -la lib*.a 2>/dev/null || echo "(static lib location may vary)"
echo ""

echo "=== BOTH MODES PASS ==="

