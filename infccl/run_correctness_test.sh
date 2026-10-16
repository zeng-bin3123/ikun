#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Find python with working torch+CUDA
for py in /usr/local/bin/python3.10 python3.10 python3; do
    if [ -x "$py" ] || command -v "$py" &>/dev/null; then
        if "$py" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            echo "[test] Using: $py"
            "$py" -c "import torch; print(f'torch={torch.__version__}, CUDA GPUs={torch.cuda.device_count()}')"
            echo ""
            exec "$py" tests/test_collectives_gpu.py
        fi
    fi
done
echo "[test] ERROR: No python with torch+CUDA found"
exit 1
