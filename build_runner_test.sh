#!/usr/bin/env bash
# Build and test the ikun collective runner (host-side only, no CUDA needed).
#
# Strategy: compile with stub glog/gflags headers if the system ones are missing.
# When system libs ARE present, link against them.
set -eo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$(pwd)"

BUILD_DIR="$ROOT/build_runner"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Check if we have the real libs
HAS_GLOG=false
HAS_GFLAGS=false
pkg-config --exists libglog 2>/dev/null && HAS_GLOG=true
pkg-config --exists gflags 2>/dev/null && HAS_GFLAGS=true
# Fallback check for gflags (not always in pkg-config)
if ! $HAS_GFLAGS; then
    dpkg -l libgflags-dev &>/dev/null 2>&1 && HAS_GFLAGS=true
fi

echo "[build] glog=$HAS_GLOG gflags=$HAS_GFLAGS"

# Create stub headers if needed
STUB_DIR="$BUILD_DIR/stubs"
mkdir -p "$STUB_DIR/glog" "$STUB_DIR/gflags"

INCLUDE_FLAGS="-I$ROOT/core -I$ROOT/core/common"
LINK_LIBS="-ldl -lpthread"

if $HAS_GLOG; then
    GLOG_CFLAGS="$(pkg-config --cflags libglog 2>/dev/null)" || true
    GLOG_LIBS="$(pkg-config --libs libglog 2>/dev/null)" || GLOG_LIBS="-lglog"
    INCLUDE_FLAGS="$INCLUDE_FLAGS $GLOG_CFLAGS"
    LINK_LIBS="$LINK_LIBS $GLOG_LIBS"
else
    cat > "$STUB_DIR/glog/logging.h" << 'STUBEOF'
#pragma once
#include <iostream>
#define LOG(level) std::cerr << "[" #level "] "
#define CHECK(cond) if (!(cond)) { std::cerr << "CHECK FAILED: " #cond << std::endl; abort(); }
#define CHECK_EQ(a,b) CHECK((a)==(b))
#define CHECK_GT(a,b) CHECK((a)>(b))
#define CHECK_GE(a,b) CHECK((a)>=(b))
#define CHECK_LT(a,b) CHECK((a)<(b))
STUBEOF
    INCLUDE_FLAGS="$INCLUDE_FLAGS -I$STUB_DIR"
fi

if $HAS_GFLAGS; then
    GFLAGS_CFLAGS="$(pkg-config --cflags gflags 2>/dev/null)" || true
    GFLAGS_LIBS="$(pkg-config --libs gflags 2>/dev/null)" || GFLAGS_LIBS="-lgflags"
    INCLUDE_FLAGS="$INCLUDE_FLAGS $GFLAGS_CFLAGS"
    LINK_LIBS="$LINK_LIBS $GFLAGS_LIBS"
else
    cat > "$STUB_DIR/gflags/gflags.h" << 'STUBEOF'
#pragma once
#include <cstdint>
#include <string>
#define DECLARE_int32(name) extern int32_t FLAGS_##name
#define DECLARE_int64(name) extern int64_t FLAGS_##name
#define DECLARE_string(name) extern std::string FLAGS_##name
#define DECLARE_bool(name) extern bool FLAGS_##name
#define DEFINE_int32(name, val, desc) int32_t FLAGS_##name = val
#define DEFINE_int64(name, val, desc) int64_t FLAGS_##name = val
#define DEFINE_string(name, val, desc) std::string FLAGS_##name = val
#define DEFINE_bool(name, val, desc) bool FLAGS_##name = val
STUBEOF
    INCLUDE_FLAGS="$INCLUDE_FLAGS -I$STUB_DIR"
fi

CXXFLAGS="-std=c++17 -Wall -Wextra -O2 -DUSE_ILU"

echo "[build] INCLUDE_FLAGS=$INCLUDE_FLAGS"
echo "[build] LINK_LIBS=$LINK_LIBS"

echo "[build] Compiling ikun_collective_runner.cpp ..."
g++ $CXXFLAGS $INCLUDE_FLAGS \
    -c "$ROOT/core/distributed_runtime/ikun_collective_runner.cpp" \
    -o "$BUILD_DIR/ikun_collective_runner.o" 2>&1

echo "[build] Compiling test_ikun_collective_runner.cpp ..."
g++ $CXXFLAGS $INCLUDE_FLAGS \
    -c "$ROOT/tests/core/test_ikun_collective_runner.cpp" \
    -o "$BUILD_DIR/test_runner.o" 2>&1

echo "[build] Linking test_runner ..."
g++ $BUILD_DIR/ikun_collective_runner.o $BUILD_DIR/test_runner.o \
    -o "$BUILD_DIR/test_runner" $LINK_LIBS 2>&1

echo "[build] Running test_runner ..."
"$BUILD_DIR/test_runner" 2>&1

echo "[build] Done."
