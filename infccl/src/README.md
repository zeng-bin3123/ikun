# infccl v0.1 — Source Code

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              Python Layer (infccl/python/infccl.py)  │
│  InfcclCommunicator → routes by tensor size          │
│    ├─ Custom P2P (csrc/infccl_allreduce.cu)   [v1]  │
│    ├─ Custom staged (src/*.cu)                [v0.1] │
│    └─ System NCCL (libnccl.so.2)              [>8MB] │
├─────────────────────────────────────────────────────┤
│              C/CUDA Layer (infccl/src/)              │
│  infccl.h → public API (NCCL-compatible shape)       │
│  core.cu  → bootstrap, P2P probe, ring topology      │
│  all_reduce.cu, reduce_scatter.cu, ...               │
│    └─ All use staged transport (cudaMemcpyPeer)      │
├─────────────────────────────────────────────────────┤
│              Hardware Layer                          │
│  BI-V100: PCIe Gen3 x16, warpSize=64, CUDA 10.2     │
│  No kernel P2P direct write                          │
│  No __threadfence_system (ivcore10)                  │
│  No cooperative_groups::grid_group                   │
└─────────────────────────────────────────────────────┘
```

## Two CUDA Backends

### csrc/infccl_allreduce.cu (v1 — IPC-based)
- Uses CUDA IPC shared memory
- Three-path routing: fused one-shot / ring / NCCL
- Requires `__threadfence_system` → works on **ivcore11 only**
- Higher performance ceiling (GPU-side barriers)

### src/*.cu (v0.1 — staged transport)
- Uses `cudaMemcpyPeer` (host DMA) for all cross-GPU transfers
- Host-side barriers (pinned memory atomics)
- Works on **both ivcore10 and ivcore11**
- Lower latency for small tensors (simpler barrier)
- NCCL-compatible API shape

## Build

```bash
# v0.1 (staged transport — works everywhere)
cd infccl && make

# v1 (IPC-based — ivcore11 only)
cd infccl && chmod +x build.sh && ./build.sh
```

## Test

```bash
# v0.1 tests
CUDA_VISIBLE_DEVICES=0,1,2,3 make test

# v1 benchmark
python3 bench_allreduce.py
```
