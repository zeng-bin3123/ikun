#!/usr/bin/env bash
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES=0,1,2,3
OUT=/home/claude/ikun/infccl/test_results.txt

PY=""
for c in /usr/local/bin/python3.10 python3.10 python3; do
    if [ -x "$c" ] 2>/dev/null || command -v "$c" &>/dev/null 2>/dev/null; then
        if "$c" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            PY="$c"; break
        fi
    fi
done

[ -z "$PY" ] && echo "NO_PYTHON" > "$OUT" && exit 1

"$PY" tests/test_correctness_minimal.py > "$OUT" 2>&1
echo "TEST_EXIT=$?" >> "$OUT"
