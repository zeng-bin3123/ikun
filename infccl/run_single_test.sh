#!/usr/bin/env bash
cd "$(dirname "${BASH_SOURCE[0]}")"
export CUDA_VISIBLE_DEVICES=0,1,2,3
RESULT_FILE="single_test_results.txt"

PY=""
for c in /usr/local/bin/python3.10 python3.10 python3; do
    if [ -x "$c" ] 2>/dev/null || command -v "$c" &>/dev/null 2>/dev/null; then
        if "$c" -c "import torch" 2>/dev/null; then
            PY="$c"; break
        fi
    fi
done
[ -z "$PY" ] && echo "NO_PYTHON" > "$RESULT_FILE" && cat "$RESULT_FILE" && exit 1

echo "PYTHON=$PY" > "$RESULT_FILE"
"$PY" tests/test_single_process.py >> "$RESULT_FILE" 2>&1
echo "EXIT=$?" >> "$RESULT_FILE"
cat "$RESULT_FILE"
