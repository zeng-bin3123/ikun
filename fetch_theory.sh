#!/bin/bash
# Fetch theory files from GitHub if not present locally
set -e
cd "$(dirname "$0")"

TOKEN="${GITHUB_TOKEN:?Set GITHUB_TOKEN env var}"
REPO="zeng-bin3123/ikun"

for FILE in mooncake_math_theory.tex mooncake_math_theory.pdf; do
    TARGET="infccl/$FILE"
    if [ -f "$TARGET" ] && [ -s "$TARGET" ]; then
        echo "[fetch] $TARGET already exists ($(wc -c < "$TARGET") bytes)"
        continue
    fi
    echo "[fetch] Downloading $FILE from GitHub..."
    curl -sL -H "Authorization: token $TOKEN" \
         -H "Accept: application/vnd.github.v3.raw" \
         "https://api.github.com/repos/$REPO/contents/infccl/$FILE" \
         -o "$TARGET"
    echo "[fetch] Saved $TARGET ($(wc -c < "$TARGET") bytes)"
done

echo "[fetch] Done."
ls -lh infccl/mooncake_math_theory.*
