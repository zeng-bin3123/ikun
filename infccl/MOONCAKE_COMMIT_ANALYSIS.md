# Mooncake Commit Analysis for infccl Development

Scanned all **2,430 commits** from [kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake).
Categorized **464 relevant commits** into 6 buckets for infccl feature development.

## Summary

| Category | Count | Description |
|----------|-------|-------------|
| collective_ops | 8 | AllReduce, ReduceScatter, AllGather, Broadcast kernels |
| p2p_ipc | 97 | P2P communication, IPC handles, staging buffers, peer management |
| topology | 63 | GPU topology detection, PCIe/NVLink routing, NUMA awareness |
| perf_opt | 231 | Batching, coalescing, pipelining, buffer management |
| pg_framework | 66 | Process group framework, PyTorch integration |
| transport | 2 | NCCL, RDMA transport backends |

## Collective Operations (8 commits)

Most directly relevant to infccl kernel development.

| Date | SHA | Message |
|------|-----|---------|
| 2026-10-12 | 20fc0bc4 | [PG][2/N] Add IBGDA route to device collectives (#3914) |
| 2026-10-09 | d603a5af | [PG][Bugfix] Handle Non-Contiguous Tensors in Collectives (#4483) |
| 2026-09-30 | ef2b8eb7 | [PG][1/N] Initialize the new collective framework (#3484) |
| 2026-07-27 | d06806d1 | [PG] Forward collectives through the backend shim for PyTorch 2.13 (#3122) |
| 2026-04-16 | 0a9c9937 | [PG][TENT] Fix CUDA collective wait semantics and NVLink small-transfer completion (#1863) |
| 2026-03-31 | b9a593c5 | [PG][TENT] Fix first-collective hangs on NVLink/MNNVL bootstrap (#1755) |
| 2026-02-01 | 4f76f12b | [PG] Impelemented support for additional collective primitives: **`gather`**, **`scatter`**, and **`reduce`** in the Mooncake backend (#1469) |
| 2025-10-15 | 0bc781df | [Misc] Fix chunked impl of _reduce_scatter_base (#931) |

## P2P / IPC (97 commits — top 30)

| Date | SHA | Message |
|------|-----|---------|
| 2026-10-16 | 9c0528d9 | [Store] Avoid staging for remote GPU reads (#3431) |
| 2026-10-15 | fc4da46e | [TENT][BUGFIX] Respect peer rail policy in TENT allocation (#4435) |
| 2026-10-14 | 3739fbda | [Store] Add Intel XPU host-staging path to the Store client, Python e2e and vLLM-XPU smoke tests (#4 |
| 2026-10-12 | b0a50fad | [Bugfix][TENT] Preserve peer handles after named notifications (#4252) |
| 2026-10-12 | 66bf7412 | [Bugfix][TransferEngine] Keep a stalled peer from wedging the handshake daemon (#4451) |
| 2026-10-10 | 8e231d5b | [EP] Enable MACA legacy EP with shared P2P and IBGDA device paths (#4091) |
| 2026-10-09 | d645d6cd | [PG][Bugfix] Reset P2P Send Timeout on Progress (#4496) |
| 2026-10-08 | b7595de3 | [Bugfix] Reconnect idle TCP lanes after peer close (#4234) |
| 2026-10-07 | a6d66e3b | [Bugfix][P2P-Store] Validate the caller's sizeList against the stored payload layout in GetReplica ( |
| 2026-10-06 | 445c0609 | [Bugfix][TENT] Retry segment update notification after peer restart (#4221) |
| 2026-09-28 | 2196923f | [Doc] Document the WITH_NVIDIA_PEERMEM default and the "Bad address [14]" GPU registration error (#4 |
| 2026-09-20 | 003e0bba | [TENT] Add Intel XPU transport, routing and staging to the Transfer Engine (#4157) |
| 2026-09-16 | 5ebe5fcf | [Bugfix] Honor MC_FORCE_HCA before automatic GPU P2P fallback (#4151) |
| 2026-09-16 | a61a987d | [Store] Centralize MasterService test access in a test-only peer (#4135) |
| 2026-09-15 | fdf45531 | [Bugfix][TENT] Detect dead peers in CheckSegmentStatus under MC_USE_TENT (#4109) |
| 2026-09-15 | f8fca5f3 | [PG] Fix P2P epoch matching and destroy/reinit test ordering (#4106) |
| 2026-09-11 | 4c72064c | [P2P-Store] Back off transfer status polling after a spin budget (#3851) |
| 2026-09-11 | debc2a23 | [TENT] Rotate HP TCP lanes independently per peer (#3987) |
| 2026-09-10 | 5accaaff | [P2PStore] Fix GetReplica metadata recheck retries (#3921) |
| 2026-09-07 | 715d1cfc | [Bugfix][TransferEngine] Prevent peer failures from latching RDMA contexts inactive (#3559) |
| 2026-09-04 | e58ad953 | Bump google.golang.org/grpc in /mooncake-p2p-store/src/p2pstore (#3827) |
| 2026-09-03 | 6957e760 | [P2P-Store] Release batch IDs on all performTransfer exit paths (#3770) |
| 2026-09-03 | 8c1edf46 | [P2P-Store] Unregister buffers on GetReplica failure paths (#3783) |
| 2026-09-02 | 128ba6c8 | [Bugfix][TENT] Stop tebench workers from aborting on peer death (#3768) |
| 2026-09-01 | 1377172a | [Bugfix][TransferEngine] Serialize RDMA endpoint lifecycle by peer owner (#3718) |
| 2026-09-01 | c5f10232 | [TE] Increase kDefaultQueuedTransfersPerPeer and kDefaultPendingAdmissionsPerPeer (#3710) |
| 2026-08-31 | 06e82daa | [Bugfix][TENT] Retry RDMA bootstrap after stale peer endpoints (#3755) |
| 2026-08-31 | b9474d29 | [TENT] Share the CUDA IPC handle across buffers in one cudaMalloc segment (#3679) |
| 2026-08-27 | 8dc90f8c | [TENT] Fix IPC relocation to not switch to the peer's device ordinal (#3678) |
| 2026-08-27 | c3aff0cd | [TransferEngine] Reuse TCP CUDA staging buffers across chunks (#3562) |

## PG Framework (66 commits — top 30)

| Date | SHA | Message |
|------|-----|---------|
| 2026-10-14 | ba6d44d9 | [EP][Bugfix] Validate Legacy Buffer Constructor Arguments (#4537) |
| 2026-10-14 | 1c54e233 | [PG] Use JIT for PG torch adapter (#3673) |
| 2026-10-12 | a0ccd7f2 | [EP] Fix MUSA build: add missing using-declaration for mc_fence_noflush (#4579) |
| 2026-10-10 | abef7875 | [EP] Optimize MUSA legacy EP dispatch and combine (#4125) |
| 2026-10-10 | 6d1960eb | [PG][Bugfix] Keep startup skew out of the queued-send progress window (#4552) |
| 2026-09-09 | a88bd082 | [EP] Support NCCL ElasticBuffer recovery after rank replacement (#3963) |
| 2026-09-07 | d7f99ed0 | [EP] Add NCCL Device API backend for ElasticBuffer (#3449) |
| 2026-09-03 | 886feb8b | [PG] Add support for torch==2.14.0 (#3842) |
| 2026-09-03 | 69d41d5e | [PG] Fix CUDA enqueue-stream serialization (#3798) |
| 2026-08-28 | a11f54e5 | [TransferEngine][EP] Support Kimi K3 & guard against duplicate glog initialization (#3727) |
| 2026-08-28 | ed0a77a8 | [EP] fix: retain tensors for deferred receive hooks (#3732) |
| 2026-08-27 | 6a00c353 | [EP] Migrate Python layer to unified package (#3635) |
| 2026-08-26 | cafc5078 | [Bugfix][PG] Balance CUDA graph capture depth accounting (#3701) |
| 2026-08-13 | bdb04026 | [EP] Fix incorrect target name under condition EP_USE_IDE (#3418) |
| 2026-08-13 | 90c5ce09 | [EP] decouple native EP from torch C++ APIs (#2883) |
| 2026-08-06 | 74e64745 | [PG] Build the device worker with C++17 for CMake 3.22 compatibility (#3308) |
| 2026-08-06 | ee0febde | [EP] Add EP dispatch/combine benchmark with incast routing patterns (#3213) |
| 2026-08-05 | 6be6fd64 | [PG] Decouple PG from PyTorch (#3206) |
| 2026-07-31 | d47f9092 | [PG] Fix single-rank initialization and yalantinglibs dependency (#3164) |
| 2026-07-28 | 1b7e90fc | [PG][2/N] Introduce Control Plane (#2455) |
| 2026-07-21 | 001e4deb | Revert "[PG][1/N] Decouple communication failures from membership changes (#2…" (#3011) |
| 2026-07-20 | ff800ade | [PG][1/N] Decouple communication failures from membership changes (#2338) |
| 2026-07-13 | 4293dce8 | [EP] add DeepEP V2 elastic buffer (#2503) |
| 2026-07-13 | 5507b5ca | [EP] Cap active RoCE QPs for IBGDA kernels (#2544) |
| 2026-06-29 | 4e0d320a | [EP] Fix: Skip inactive ranks in combine reduction loop (#2653) |
| 2026-06-18 | 14375f4f | [EP] support MUSA platform (#2480) |
| 2026-06-11 | 84fc7c37 | [PG] Add experimental MACA PG support (#2365) |
| 2026-06-06 | 27aaae1e | [PG] Build the MUSA PG extension through torchada (#2353) |
| 2026-06-06 | 47b267a5 | [PG] Fix data race: make running_ atomic (#2352) |
| 2026-06-06 | 938d0c70 | [PG] Fix null-deref on MNNVL disconnect and activeRanks leak (#2347) |

## Topology (63 commits — top 20)

| Date | SHA | Message |
|------|-----|---------|
| 2026-10-15 | a2c6c5b7 | [2/3] [Store] Add EGM-backed Store pool over NVLink (#3335) |
| 2026-10-14 | 4a760599 | [TransferEngine] Pair Data Direct registrations and topology with the GPU's PCI root (#4612) |
| 2026-10-13 | a9983dcd | [TE] efa: fall back to any keyed NIC when selectDevice's topology walk misses (#4594) |
| 2026-09-10 | 6c07cb78 | [TENT] Export EGM host memory over multi-node NVLink (opt-in) (#3978) |
| 2026-09-03 | 6d946ef3 | [Bugfix][TENT] Keep RailMonitor state across COW topology refreshes (#3832) |
| 2026-08-25 | 09dfb006 | [Bugfix][TENT] Fix RailMonitor topology use-after-free (#3598) |
| 2026-08-24 | b32bdeba | fix(tent): synchronize with caller's per-thread stream before issuing NVLink/MNNVL batched copy (#35 |
| 2026-08-20 | 266a966b | [transfer-engine] Ship intra-node NVLink transport in x86_64 CUDA wheels (#3547) |
| 2026-08-15 | 0cff074a | [TENT] Support custom NIC priority matrix and native topology dump (#3382) |
| 2026-08-04 | 3c63169a | [TE] perf(efa): optionally register device memory only on topology-local NICs (#3219) |
| 2026-08-04 | 0a857714 | [Store]Support VRAM Segment and NVLink transport within a single node (#3196) |
| 2026-07-14 | 75906c72 | [store] Opt-in topology-aware remote replica scoring in SelectBestReplica (#2516) |
| 2026-07-11 | 6894c2d7 | [TransferEngine] Add show-link diagnostic tool for NIC topology (#2820) |
| 2026-07-06 | 118382f4 | [TE] Fix cross-GPU NVLink/MNNVL event-stream device mismatch (#2722) (#2754) |
| 2026-07-03 | 263090eb | [TE] Capture GPU device at registration time in MnnvlTransport / NVLinkTransport (#2691) |
| 2026-07-02 | a325291c | [CI] Fix build-musa: add missing MUSA mappings for nvlink_transport APIs (#2578) (#2710) |
| 2026-06-29 | fac9d09d | [TENT] Fix stale getTransferStatus in nvlink/mnnvl/ascend transports (#2505) |
| 2026-06-25 | a06c866d | fix(maca): map cudaStreamQuery for the intra-node NVLink build (#2606) |
| 2026-06-24 | d2ebfb08 | [CI] Fix build-musa: add missing MUSA macro mappings for nvlink_transport APIs (#2575) |
| 2026-06-18 | 948e5a8b | [TENT] Fix CUDA event leak in nvlink/mnnvl transports (#2507) |

## Key Mooncake Source Files for infccl Reference

### Ring AllReduce GPU Kernel (MOST RELEVANT)
- `mooncake-pg/src/device_comm/device_collective/algorithms/ring/ring_all_reduce.cu`
- `mooncake-pg/include/device_comm/device_collective/algorithms/ring/ring_primitives.cuh`
- `mooncake-pg/include/device_comm/device_collective/algorithms/ring/ring_types.cuh`

### Device Transfer (P2P IPC Route)
- `mooncake-pg/src/device_comm/device_transfer/routes/p2p_route/`
- `mooncake-pg/include/device_comm/device_transfer/transfer_lane.cuh`

### Value Primitives (Vectorized Operations)
- `mooncake-pg/include/device_comm/device_primitives/value_primitives.cuh`
- `mooncake-pg/include/device_comm/device_primitives/payload_writer.cuh`

### Communicator (Python API → C++ dispatch)
- `mooncake-pg/src/mooncake_communicator.cpp` (1,951 lines)
- `mooncake-pg/include/mooncake_communicator.h`

### Topology Detection
- `mooncake-transfer-engine/src/topology.cpp` (~800 lines)

## BI-V100 Constraints (from Hardware Probe)

| Constraint | Impact on infccl |
|------------|-----------------|
| CUDA 10.2 max | No cooperative_groups::grid_group → single-block ring kernel |
| No __nv_bfloat16 | Manual bf16↔fp32 bit-shift conversion |
| sm_70 / ivcore10-11 | No INT8/FP8 Tensor Cores |
| PCIe Gen3 x16 (~16 GB/s) | Bandwidth-limited, not compute-limited |
| Kernel P2P direct write FAILS | All cross-GPU transfer via cudaMemcpyPeer (staged) |
| warpSize=64 | Need custom warp-shuffle primitives |
| 128KB shared memory | More smem than NVIDIA sm_70 (96KB) |
| __threadfence_system not on ivcore10 | Works on ivcore11; use host barrier fallback on ivcore10 |
| CUDA Graph support limited | Cannot use graph capture for kernel fusion |
| 5.5μs kernel launch overhead | Kernel fusion critical |

---
*Generated 2026-10-17 by scanning kvcache-ai/Mooncake via GitHub API.*
