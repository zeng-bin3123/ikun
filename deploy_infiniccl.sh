#!/bin/bash
# deploy_infiniccl.sh — Copy built libinfiniccl.so to prebuilt directory
# Run after build_bi100.sh succeeds
set -euo pipefail

SRC="third_party/InfiniCCL/build/src/libinfiniccl.so"
DST="ex_engine/prebuilt/libinfiniccl.so"

if [[ -f "$SRC" ]]; then
    cp "$SRC" "$DST"
    echo "[deploy] libinfiniccl.so → $DST ($(stat -c%s "$DST") bytes)"
else
    echo "[deploy] $SRC not found — build InfiniCCL first:"
    echo "  cd third_party/InfiniCCL && bash build_bi100.sh"
    exit 1
fi

# Verify symbols
echo "[deploy] Exported symbols:"
nm -D "$DST" 2>/dev/null | grep -i "infinic\|allreduce\|CommInit" | head -10
