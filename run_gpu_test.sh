#!/usr/bin/env bash
# Find the Python that has torch and run the GPU integration test.
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Find Python with torch
PYTHON=""
for p in /usr/local/bin/python3 /usr/local/bin/python3.10 /usr/bin/python3 python3; do
    if [ -x "$p" ] || command -v "$p" &>/dev/null; then
        if "$p" -c "import torch" 2>/dev/null; then
            PYTHON="$p"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "[gpu-test] No Python with torch found"
    exit 1
fi

echo "[gpu-test] Using: $PYTHON ($($PYTHON --version 2>&1))"
echo "[gpu-test] Torch: $($PYTHON -c 'import torch; print(torch.__version__)' 2>&1)"
echo ""

$PYTHON test_gpu_integration.py

