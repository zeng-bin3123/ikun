#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# Find python with torch
PY=""
for candidate in /usr/local/bin/python3.10 /usr/local/bin/python3 python3.10 python3; do
    if [ -x "$candidate" ] 2>/dev/null || command -v "$candidate" &>/dev/null 2>/dev/null; then
        if "$candidate" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            PY="$candidate"
            break
        fi
    fi
done

if [ -z "$PY" ]; then
    echo "ERROR: No python with torch+CUDA found"
    exit 1
fi

echo "Using: $PY ($($PY --version 2>&1))"
echo "torch=$($PY -c 'import torch; print(torch.__version__)')"
echo "GPUs=$($PY -c 'import torch; print(torch.cuda.device_count())')"
echo ""

exec "$PY" tests/test_correctness_minimal.py "$@"
