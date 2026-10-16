#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$SCRIPT_DIR/p2p_allreduce.cu"
BIN="$SCRIPT_DIR/p2p_run"

export PATH=/usr/local/corex/bin:/usr/local/corex-3.2.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib64:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NVCC=$(which nvcc 2>/dev/null || echo "")
if [ -z "$NVCC" ]; then echo "ERROR: nvcc not found"; exit 1; fi

echo "Compiler: $NVCC"
$NVCC --version

echo ""
echo "=== Compiling ==="
rm -f "$BIN"
$NVCC -O2 -std=c++14 \
  -I/usr/local/corex/include \
  -L/usr/local/corex/lib64 \
  -Xlinker -rpath=/usr/local/corex/lib64 \
  -o "$BIN" "$SRC" 2>&1
echo "compile=pass"

if [ -x "$BIN" ]; then
  echo "Binary created: $(ls -la $BIN)"
  ldd "$BIN" 2>&1 | grep -i cuda || true
  echo ""
  echo "=== Running CUDA binary ==="
  "$BIN" 2>&1
  echo "RUN_EXIT=$?"
else
  echo "WARNING: Binary not created, falling back to Python test"
  python3 "$SCRIPT_DIR/gpu_test.py" 2>&1
  echo "PY_EXIT=$?"
fi
