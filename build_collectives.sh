#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
CSRC_DIR="${SCRIPT_DIR}/csrc"

mkdir -p "${BUILD_DIR}"

NVCC=""
for p in /usr/local/corex/bin/nvcc /usr/local/corex-3.2.3/bin/nvcc \
         /usr/local/cuda/bin/nvcc /usr/bin/nvcc; do
    if [ -x "$p" ]; then NVCC="$p"; break; fi
done
[ -z "$NVCC" ] && { echo "nvcc not found"; exit 1; }

CUDA_LIB=""
for lp in /usr/local/corex/lib64 /usr/local/corex-3.2.3/lib64 \
          /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu; do
    if [ -f "${lp}/libcudart.so" ] || [ -f "${lp}/libcudart_static.a" ]; then
        CUDA_LIB="-L${lp}"; break
    fi
done

FLAGS="-O2 -Xcompiler -fPIC -arch=sm_70"

${NVCC} ${FLAGS} -c "${CSRC_DIR}/ikun_collectives.cu" -o "${BUILD_DIR}/ikun_collectives.o"
${NVCC} ${FLAGS} -c "${CSRC_DIR}/ikun_allreduce.cu" -o "${BUILD_DIR}/ikun_allreduce.o"
${NVCC} -shared -o "${BUILD_DIR}/libikun_collectives.so" \
    "${BUILD_DIR}/ikun_collectives.o" "${BUILD_DIR}/ikun_allreduce.o" \
    ${CUDA_LIB} -lcudart

echo "Built: ${BUILD_DIR}/libikun_collectives.so"
