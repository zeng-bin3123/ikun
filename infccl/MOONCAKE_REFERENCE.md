# Mooncake Reference for infccl Development

## Repository: kvcache-ai/Mooncake (2,430 commits as of 2026-10-16)

## Key Source Files for infccl

### 1. Ring AllReduce GPU Kernel (MOST RELEVANT)
- `mooncake-pg/src/device_comm/device_collective/algorithms/ring/ring_all_reduce.cu` (378 lines)
- `mooncake-pg/include/device_comm/device_collective/algorithms/ring/ring_primitives.cuh`
- `mooncake-pg/include/device_comm/device_collective/algorithms/ring/ring_types.cuh`

**Key patterns to adopt:**
- `runRingTile()`: reduce-scatter (P-1 steps) + allgather (P-1 steps)
- `chooseChannelCount()`: auto-select multi-CTA parallelism based on tensor size
- `kRingPipelineSlots`: double-buffered payload slots for send/recv overlap
- `RingPayloadSlotSchedule`: alternating slot allocation

### 2. Device Transfer (P2P IPC Route)
- `mooncake-pg/src/device_comm/device_transfer/routes/p2p_route/`
- `mooncake-pg/include/device_comm/device_transfer/transfer_lane.cuh`

**Key patterns:**
- Pre-registered IPC handles shared via transfer_lane
- CUDA IPC handle reuse across buffers in one cudaMalloc segment (commit b9474d29)

### 3. Value Primitives (Vectorized Operations)
- `mooncake-pg/include/device_comm/device_primitives/value_primitives.cuh`
- `mooncake-pg/include/device_comm/device_primitives/payload_writer.cuh`

**Key patterns:**
- float4/uint4 wide vector loads for PCIe transfer
- FP32 accumulation for FP16/BF16 reduce

### 4. Communicator (Python API → C++ dispatch)
- `mooncake-pg/src/mooncake_communicator.cpp` (1,951 lines)
- `mooncake-pg/include/mooncake_communicator.h`

**Key patterns:**
- Batched enqueue to reduce lock contention
- Stream-ordered operations
- NCCL fallback path

### 5. Topology Detection
- `mooncake-transfer-engine/src/topology.cpp` (~800 lines)
- PCI distance calculation with NUMA node affinity

## Commit Categories (91 PG/EP commits)

### Collective Communication (PG)
| Date | Hash | Description |
|------|------|-------------|
| 2026-09-30 | ef2b8eb7 | [PG][1/N] Initialize new collective framework (Ring AR GPU kernel) |
| 2026-10-12 | 20fc0bc4 | [PG][2/N] Add IBGDA route to device collectives |
| 2026-07-28 | 1b7e90fc | [PG][2/N] Introduce Control Plane |
| 2026-05-07 | 84df45a9 | Refactor P2PProxy: shared chunk pools + credit-based flow control |
| 2026-03-27 | af733e12 | Add GPU barrier support |
| 2026-02-24 | 99da058d | Implement P2P proxy for low-latency send/recv |
| 2026-02-01 | 4f76f12b | Support gather, scatter, reduce primitives |
| 2026-01-27 | 0b8e1d63 | Support full reduction ops (Product/Min/Max) |

### IPC / Shared Memory
| Date | Hash | Description |
|------|------|-------------|
| 2026-08-20 | b9474d29 | Share CUDA IPC handle across buffers in one segment |
| 2026-08-20 | 8dc90f8c | Fix IPC relocation (don't switch to peer's device ordinal) |
| 2026-05-12 | c3aff0cd | Reuse TCP CUDA staging buffers across chunks |

## Features NOT Applicable to BI-V100
- RDMA/IBGDA transport (no InfiniBand)
- NVLink transport (no NVLink)
- Elastic scaling (fixed 4-GPU topology)
- Complex control plane (single-node, no coordination needed)
- MNNVL support (multi-node NVLink)
- Ascend NPU / ROCm / MUSA / MACA paths

## BI-V100 Constraints to Remember
- CUDA 10.2 max → no __nv_bfloat16, no cooperative_groups::grid_group
- sm_70 compute capability
- PCIe Gen3 x16 only (~16 GB/s per direction)
- P2P may fail between some GPU pairs (0↔3 cross-switch)
- CUDA Graph support is limited
- Iluvatar CoreX compiler (nvcc compatible but version-pinned)
