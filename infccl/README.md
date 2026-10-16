# infccl — Iluvatar Inference Collective Communication Library

Optimized collective communication for **Iluvatar BI-V100** (4× PCIe, CUDA 10.2).

## Architecture (v2)

```
┌──────────────────────────────────────────────────────────┐
│                   InfcclCommunicator                      │
│                                                           │
│  ┌────────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Fused One-Shot  │  │  Ring AR     │  │  NCCL Path   │ │
│  │ (≤ 256 KB)     │  │ (256KB~8MB)  │  │ (> 8 MB)     │ │
│  │ all-to-all copy │  │ reduce-scat  │  │ libnccl.so.2 │ │
│  │ → barrier       │  │ + allgather  │  │              │ │
│  │ → local reduce  │  │ P-1 ring     │  │              │ │
│  └──────┬─────────┘  └──────┬───────┘  └──────┬───────┘ │
│         │                    │                  │         │
│  ┌──────▼────────────────────▼──────┐  ┌───────▼───────┐ │
│  │    CUDA IPC P2P shared memory    │  │  System NCCL  │ │
│  │    (pre-resolved, zero-alloc)    │  │               │ │
│  └──────────────────────────────────┘  └───────────────┘ │
│                                                           │
│  ┌───────────────────┐  ┌────────────────────┐           │
│  │ AllReduceCoalescer │  │ Topology-Aware Ring │           │
│  │ (batch N→1 call)  │  │ (P2P-pair ordering) │           │
│  └───────────────────┘  └────────────────────┘           │
└──────────────────────────────────────────────────────────┘
```

## Key Optimizations

### 1. Three-Path Routing (v2)
Tensor size determines the optimal AllReduce strategy:
- **Fused One-Shot (≤ 256KB)**: All-to-all copy → barrier → local reduce
  Single kernel, zero-alloc, template-unrolled for P=2,4,8
- **Ring AllReduce (256KB ~ 8MB)**: Reduce-scatter + allgather in P-1 steps
  Bandwidth-optimal: total wire traffic = 2*(P-1)/P × size
  Topology-aware ring ordering avoids non-P2P GPU pairs
- **NCCL Fallback (> 8MB)**: System libnccl.so.2 for large tensors

### 2. Two-Phase P2P AllReduce (from Mooncake §2)
For tensors < 256KB, bypasses NCCL's ring/tree protocol:
- Phase 1: Each GPU copies data to P2P-visible staging buffer
- Phase 2: Each GPU reduces all staging buffers locally
- **Benefit**: Eliminates NCCL launch overhead (~50μs per call)

### 2. AllReduce Coalescing (from Mooncake §4)
Multiple small allreduces are packed into one:
- Pack N tensors into contiguous buffer
- Single NCCL/P2P allreduce
- Unpack results
- **Benefit**: Reduces N×α overhead to 1×α

### 3. Amortized Barrier Signaling (from Mooncake §2)
Instead of full barrier per allreduce:
- Signal every N operations
- FIFO tracks pending completions
- **Benefit**: Reduces synchronization overhead by factor N

### 4. Topology-Aware Placement (from Mooncake §5)
Staging buffers placed on NUMA-local memory:
- Detects BI-V100 PCIe topology
- Routes through fastest P2P paths
- Falls back gracefully when GPU pairs can't P2P

## Usage

### Direct API
```python
from infccl import InfcclCommunicator

comm = InfcclCommunicator(group=dist_group, device=local_rank)
comm.all_reduce(tensor)  # auto-routes to fastest path
comm.all_reduce_coalesced([t1, t2, t3])  # batched
```

### vLLM Integration
```python
# In patch_ops.sh or env_override.py:
import infccl.patch_vllm_infccl  # auto-patches on import
```

Or set environment:
```bash
export INFCCL_AUTO_PATCH=1
```

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `INFCCL_CUSTOM_AR_THRESHOLD` | `8388608` | Max bytes for custom P2P AR |
| `INFCCL_BATCH_BARRIER` | `1` | Barrier amortization factor |
| `INFCCL_MAX_STAGING` | `16777216` | Staging buffer size per rank |
| `INFCCL_ENABLE_CUSTOM_AR` | `1` | Enable custom allreduce |
| `INFCCL_AUTO_PATCH` | `1` | Auto-patch vLLM on import |
| `INFCCL_FORCE_ENABLE` | `0` | Force enable on non-BI-V100 |

## Build (CUDA kernels)
```bash
cd infccl
chmod +x build.sh
./build.sh
```

## Mathematical Foundation

See `mooncake_math_theory.pdf` for the complete theoretical analysis.

Key theorems applied:
- **Thm 2.1**: CQE reduction factor under amortized signaling
- **Thm 3.1**: String interning complexity reduction
- **Thm 4.1**: Mutex coalescing speedup ≈ n/w
- **Thm 7.1**: M/M/1/K bounded queue delay for admission control
- **Thm 11.1**: Dispatch overhead dominance on BI-V100

## File Structure
```
infccl/
├── __init__.py              # Package exports
├── README.md                # This file
├── build.sh                 # CUDA kernel build script
├── patch_vllm_infccl.py     # vLLM monkey-patch
├── csrc/
│   └── infccl_allreduce.cu  # Custom CUDA allreduce kernels
├── python/
│   ├── __init__.py
│   └── infccl.py            # Main Python module
└── tests/
    └── test_infccl.py       # Unit tests
```
