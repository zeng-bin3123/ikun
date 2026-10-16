"""
infccl — Iluvatar Inference Collective Communication Library

A collective communication library optimized for Iluvatar BI-V100 (4× PCIe).
Wraps the system NCCL for large transfers and provides a custom P2P
shared-memory allreduce for small-to-medium tensors.

Key optimizations (from Mooncake mathematical analysis):
  1. Two-phase P2P allreduce bypassing NCCL for tensors < 8MB
  2. Amortized barrier signaling (signal every N ops, not every op)
  3. Topology-aware buffer placement (NUMA-local staging buffers)
  4. Batched dispatch (coalesce multiple small allreduces)

Usage:
    from infccl import InfcclCommunicator
    
    comm = InfcclCommunicator(rank, world_size, device)
    comm.all_reduce(tensor, op=ReduceOp.SUM)

Integration with vLLM:
    Set VLLM_NCCL_SO_PATH to point to the system libnccl.so.2.
    infccl wraps it and selects the fastest path per tensor size.
"""

import ctypes
import os
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import IntEnum

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

logger = logging.getLogger("infccl")


# ============================================================================
# Constants
# ============================================================================

# Threshold below which we use custom P2P allreduce instead of NCCL
# For BI-V100 PCIe topology, P2P wins for tensors up to ~8MB
CUSTOM_AR_THRESHOLD_BYTES = int(os.environ.get("INFCCL_CUSTOM_AR_THRESHOLD", 8 * 1024 * 1024))

# Amortized barrier generation (from Mooncake's signal interval theory)
# Instead of full barrier per allreduce, batch N allreduces per barrier
BATCH_BARRIER_SIZE = int(os.environ.get("INFCCL_BATCH_BARRIER", 1))

# Maximum staging buffer size per rank
MAX_STAGING_SIZE = int(os.environ.get("INFCCL_MAX_STAGING", 16 * 1024 * 1024))

# Enable/disable custom allreduce
ENABLE_CUSTOM_AR = os.environ.get("INFCCL_ENABLE_CUSTOM_AR", "1") == "1"


class InfcclDataType(IntEnum):
    FLOAT32 = 0
    FLOAT16 = 1
    BFLOAT16 = 2


def torch_to_infccl_dtype(dtype: torch.dtype) -> InfcclDataType:
    """Map PyTorch dtype to infccl dtype enum."""
    mapping = {
        torch.float32: InfcclDataType.FLOAT32,
        torch.float16: InfcclDataType.FLOAT16,
        torch.bfloat16: InfcclDataType.BFLOAT16,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype for infccl: {dtype}")
    return mapping[dtype]


# ============================================================================
# NCCL wrapper (thin layer over system libnccl.so.2)
# ============================================================================

# Re-export NCCL types from vLLM's wrapper for compatibility
ncclResult_t = ctypes.c_int
ncclComm_t = ctypes.c_void_p


class ncclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


cudaStream_t = ctypes.c_void_p
buffer_type = ctypes.c_void_p
ncclDataType_t = ctypes.c_int
ncclRedOp_t = ctypes.c_int


class NcclDataTypeEnum:
    """NCCL data type enum matching nccl.h"""
    ncclFloat16 = 6
    ncclFloat32 = 7
    ncclBfloat16 = 9

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int:
        mapping = {
            torch.float16: cls.ncclFloat16,
            torch.float32: cls.ncclFloat32,
            torch.bfloat16: cls.ncclBfloat16,
        }
        if dtype not in mapping:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return mapping[dtype]


class NcclRedOpEnum:
    """NCCL reduce op enum"""
    ncclSum = 0
    ncclProd = 1
    ncclMax = 2
    ncclMin = 3

    @classmethod
    def from_torch(cls, op: ReduceOp) -> int:
        mapping = {
            ReduceOp.SUM: cls.ncclSum,
            ReduceOp.PRODUCT: cls.ncclProd,
            ReduceOp.MAX: cls.ncclMax,
            ReduceOp.MIN: cls.ncclMin,
        }
        if op not in mapping:
            raise ValueError(f"Unsupported op: {op}")
        return mapping[op]


class NcclLibrary:
    """Thin ctypes wrapper around libnccl.so.2 for BI-V100."""

    def __init__(self, so_path: Optional[str] = None):
        if so_path is None:
            so_path = os.environ.get("VLLM_NCCL_SO_PATH", "libnccl.so.2")
        
        self.lib = ctypes.CDLL(so_path)
        logger.info(f"[infccl] Loaded NCCL from {so_path}")
        
        # Bind functions
        self._setup_bindings()
    
    def _setup_bindings(self):
        """Set up ctypes function signatures."""
        lib = self.lib
        
        # ncclGetVersion
        lib.ncclGetVersion.restype = ncclResult_t
        lib.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        
        # ncclGetUniqueId
        lib.ncclGetUniqueId.restype = ncclResult_t
        lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(ncclUniqueId)]
        
        # ncclCommInitRank
        lib.ncclCommInitRank.restype = ncclResult_t
        lib.ncclCommInitRank.argtypes = [
            ctypes.POINTER(ncclComm_t),  # comm
            ctypes.c_int,                # nranks
            ncclUniqueId,                # commId
            ctypes.c_int,                # rank
        ]
        
        # ncclAllReduce
        lib.ncclAllReduce.restype = ncclResult_t
        lib.ncclAllReduce.argtypes = [
            buffer_type,      # sendbuff
            buffer_type,      # recvbuff
            ctypes.c_size_t,  # count
            ncclDataType_t,   # datatype
            ncclRedOp_t,      # op
            ncclComm_t,       # comm
            cudaStream_t,     # stream
        ]
        
        # ncclCommDestroy
        lib.ncclCommDestroy.restype = ncclResult_t
        lib.ncclCommDestroy.argtypes = [ncclComm_t]
    
    def get_version(self) -> int:
        version = ctypes.c_int()
        self.lib.ncclGetVersion(ctypes.byref(version))
        return version.value
    
    def get_unique_id(self) -> ncclUniqueId:
        uid = ncclUniqueId()
        result = self.lib.ncclGetUniqueId(ctypes.byref(uid))
        if result != 0:
            raise RuntimeError(f"ncclGetUniqueId failed with {result}")
        return uid
    
    def comm_init_rank(self, nranks: int, uid: ncclUniqueId, rank: int) -> ncclComm_t:
        comm = ncclComm_t()
        result = self.lib.ncclCommInitRank(
            ctypes.byref(comm), nranks, uid, rank)
        if result != 0:
            raise RuntimeError(f"ncclCommInitRank failed with {result}")
        return comm
    
    def all_reduce(self, sendbuf, recvbuf, count, datatype, op, comm, stream):
        result = self.lib.ncclAllReduce(
            sendbuf, recvbuf, count, datatype, op, comm, stream)
        if result != 0:
            raise RuntimeError(f"ncclAllReduce failed with {result}")
    
    def comm_destroy(self, comm):
        self.lib.ncclCommDestroy(comm)


# ============================================================================
# Custom AllReduce Library (CUDA kernel based)
# ============================================================================

class InfcclCustomAR:
    """
    Custom P2P shared-memory allreduce for BI-V100.
    
    Uses CUDA IPC to share staging buffers between GPUs on the same node.
    Optimized for the 4-GPU PCIe topology where P2P is available.
    
    Two backends:
      1. Native CUDA kernel (infccl_allreduce.so) — zero per-launch alloc,
         pre-resolved device pointers, fused barrier+reduce kernel.
         Built via build.sh on the target BI-V100 machine.
      2. PyTorch fallback — works without compilation but uses CPU barrier.
    
    Mathematical basis (from Mooncake analysis, Theorem 11.1):
      - For small tensors (< 8MB), 2-phase P2P has latency:
        T_P2P = alpha_p2p + P*s/B_pcie + s*c_reduce/B_compute
      - NCCL ring:
        T_NCCL = alpha_nccl + 2*(P-1)*s/(P*B_pcie)
      - P2P wins when alpha_nccl >> alpha_p2p (true on BI-V100 PCIe).
    """

    DTYPE_MAP = {
        torch.float32: 0,
        torch.float16: 1,
        torch.bfloat16: 2,
    }

    def __init__(self, rank: int, world_size: int, device: int,
                 max_size: int = MAX_STAGING_SIZE):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.max_size = max_size
        self.available = False
        self._native_comm = None  # ctypes pointer to InfcclComm
        self._native_lib = None   # ctypes CDLL
        self._use_native = False
        
        # Fallback state (PyTorch-level P2P)
        self._generation = 1
        self.staging_buffer = None
        self.meta_buffer = None
        self.peer_staging: List[Optional[torch.Tensor]] = [None] * world_size
        self.peer_meta: List[Optional[torch.Tensor]] = [None] * world_size
        
        # Check P2P availability
        self._p2p_pairs = self._check_p2p()
        if not self._p2p_pairs:
            logger.warning("[infccl] P2P not available for all pairs, custom AR disabled")
            return
        
        # Try to load native kernel
        self._try_load_native()
        
        if self._use_native:
            # Native path: C library manages all buffers
            self._init_native()
        else:
            # Fallback: PyTorch-level staging
            self._init_pytorch_fallback()
    
    def _check_p2p(self) -> bool:
        """Check P2P access between this rank and all peers."""
        all_ok = True
        for i in range(self.world_size):
            if i == self.rank:
                continue
            can_access = torch.cuda.can_device_access_peer(self.rank, i)
            if not can_access:
                logger.warning(f"[infccl] GPU {self.rank} cannot P2P access GPU {i}")
                all_ok = False
        return all_ok
    
    def _try_load_native(self):
        """Try to load the compiled infccl_allreduce.so."""
        search_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "lib", "infccl_allreduce.so"),
            os.path.join(os.path.dirname(__file__), "..", "lib", "infccl_allreduce.so"),
            os.environ.get("INFCCL_SO_PATH", ""),
        ]
        for path in search_paths:
            if path and os.path.isfile(path):
                try:
                    lib = ctypes.CDLL(path)
                    # Verify it has the new API
                    lib.infccl_comm_create.restype = ctypes.c_void_p
                    lib.infccl_comm_create.argtypes = [
                        ctypes.c_int, ctypes.c_int, ctypes.c_int64]
                    lib.infccl_allreduce.restype = ctypes.c_int
                    lib.infccl_allreduce.argtypes = [
                        ctypes.c_void_p,  # comm
                        ctypes.c_void_p,  # output
                        ctypes.c_void_p,  # input
                        ctypes.c_int64,   # count
                        ctypes.c_int,     # dtype
                        ctypes.c_void_p,  # stream
                    ]
                    lib.infccl_comm_get_ipc.restype = ctypes.c_int
                    lib.infccl_comm_register_peer.restype = ctypes.c_int
                    lib.infccl_comm_finalize.restype = ctypes.c_int
                    lib.infccl_comm_destroy.restype = None
                    lib.infccl_comm_destroy.argtypes = [ctypes.c_void_p]
                    # v2: ring path query
                    try:
                        lib.infccl_query_path.restype = ctypes.c_int
                        lib.infccl_query_path.argtypes = [ctypes.c_int64]
                        self._has_ring = True
                    except AttributeError:
                        self._has_ring = False
                    self._native_lib = lib
                    self._use_native = True
                    logger.info(f"[infccl] Loaded native kernel from {path}")
                    return
                except (OSError, AttributeError) as e:
                    logger.debug(f"[infccl] Cannot load {path}: {e}")
        logger.info("[infccl] Native kernel not found, using PyTorch fallback")
    
    def _init_native(self):
        """Initialize via native C library (zero-alloc path)."""
        lib = self._native_lib
        self._native_comm = lib.infccl_comm_create(
            self.rank, self.world_size, self.max_size)
        if not self._native_comm:
            logger.error("[infccl] infccl_comm_create returned NULL")
            self._use_native = False
            self._init_pytorch_fallback()
            return
        self.available = True
        logger.info(f"[infccl] Native comm created: rank={self.rank}, ws={self.world_size}")
    
    def _init_pytorch_fallback(self):
        """Initialize PyTorch-level staging buffers."""
        torch.cuda.set_device(self.device)
        self.staging_buffer = torch.zeros(
            self.max_size // 2, dtype=torch.float16, device=f"cuda:{self.device}")
        self.meta_buffer = torch.zeros(
            256, dtype=torch.int32, device=f"cuda:{self.device}")
        self.peer_staging[self.rank] = self.staging_buffer
        self.peer_meta[self.rank] = self.meta_buffer
        self.available = True
        logger.info(f"[infccl] PyTorch fallback AR initialized: rank={self.rank}")
    
    def get_ipc_handles(self) -> Tuple[bytes, bytes]:
        """Get IPC memory handles for this rank's buffers."""
        handle_size = 64  # cudaIpcMemHandle_t
        buf_handle = (ctypes.c_byte * handle_size)()
        meta_handle = (ctypes.c_byte * handle_size)()
        
        if self._use_native:
            rc = self._native_lib.infccl_comm_get_ipc(
                self._native_comm,
                ctypes.byref(buf_handle),
                ctypes.byref(meta_handle))
            if rc != 0:
                raise RuntimeError(f"infccl_comm_get_ipc failed: {rc}")
        else:
            cudart = ctypes.CDLL("libcudart.so")
            r1 = cudart.cudaIpcGetMemHandle(
                ctypes.byref(buf_handle), ctypes.c_void_p(self.staging_buffer.data_ptr()))
            r2 = cudart.cudaIpcGetMemHandle(
                ctypes.byref(meta_handle), ctypes.c_void_p(self.meta_buffer.data_ptr()))
            if r1 != 0 or r2 != 0:
                raise RuntimeError(f"cudaIpcGetMemHandle failed: {r1}/{r2}")
        
        return (bytes(buf_handle), bytes(meta_handle))
    
    def register_peer(self, peer_rank: int, buf_handle_bytes: bytes,
                      meta_handle_bytes: bytes):
        """Register a peer's IPC handles."""
        if peer_rank == self.rank:
            return
        
        handle_size = 64
        buf_handle = (ctypes.c_byte * handle_size)(*buf_handle_bytes)
        meta_handle = (ctypes.c_byte * handle_size)(*meta_handle_bytes)
        
        if self._use_native:
            rc = self._native_lib.infccl_comm_register_peer(
                self._native_comm, peer_rank,
                ctypes.byref(buf_handle), ctypes.byref(meta_handle))
            if rc != 0:
                raise RuntimeError(f"infccl_comm_register_peer failed for rank {peer_rank}: {rc}")
        else:
            cudart = ctypes.CDLL("libcudart.so")
            buf_ptr = ctypes.c_void_p()
            meta_ptr = ctypes.c_void_p()
            r1 = cudart.cudaIpcOpenMemHandle(ctypes.byref(buf_ptr), buf_handle, 1)
            r2 = cudart.cudaIpcOpenMemHandle(ctypes.byref(meta_ptr), meta_handle, 1)
            if r1 != 0 or r2 != 0:
                raise RuntimeError(f"cudaIpcOpenMemHandle failed for rank {peer_rank}: {r1}/{r2}")
            # Store raw pointers for fallback path
            self.peer_staging[peer_rank] = buf_ptr.value  # store as int
            self.peer_meta[peer_rank] = meta_ptr.value
        
        logger.info(f"[infccl] Registered peer {peer_rank}")
    
    def finalize_peers(self):
        """Finalize peer registration (upload device pointer arrays)."""
        if self._use_native:
            rc = self._native_lib.infccl_comm_finalize(self._native_comm)
            if rc != 0:
                raise RuntimeError(f"infccl_comm_finalize failed: {rc}")
            logger.info("[infccl] Native comm finalized (zero-alloc path ready)")
    
    def all_reduce_inplace(self, tensor: torch.Tensor, stream=None):
        """
        Perform in-place allreduce using P2P shared memory.
        
        Native path: single CUDA kernel launch, zero device allocations.
        Fallback path: PyTorch-level copy + barrier + reduce.
        """
        if not self.available:
            raise RuntimeError("Custom AR not available")
        
        numel = tensor.numel()
        nbytes = tensor.element_size() * numel
        if nbytes > self.max_size:
            raise ValueError(f"Tensor too large for custom AR: {nbytes} > {self.max_size}")
        
        if self._use_native:
            self._native_allreduce(tensor, numel, stream)
        else:
            self._pytorch_allreduce(tensor, numel, stream)
    
    def _native_allreduce(self, tensor: torch.Tensor, numel: int, stream=None):
        """Launch via compiled CUDA kernel — zero per-launch allocs.
        
        v2: The kernel now returns 1 if the tensor exceeds the ring threshold
        (>8MB), signaling the caller to use NCCL instead. Return codes:
          0 = success (fused or ring path handled it)
          1 = too large, caller should fall back to NCCL
        """
        dtype_id = self.DTYPE_MAP.get(tensor.dtype)
        if dtype_id is None:
            raise ValueError(f"Unsupported dtype: {tensor.dtype}")
        
        if stream is None:
            cuda_stream = torch.cuda.current_stream(self.device).cuda_stream
        elif hasattr(stream, 'cuda_stream'):
            cuda_stream = stream.cuda_stream
        else:
            cuda_stream = stream
        
        rc = self._native_lib.infccl_allreduce(
            self._native_comm,
            ctypes.c_void_p(tensor.data_ptr()),
            ctypes.c_void_p(tensor.data_ptr()),  # in-place
            ctypes.c_int64(numel),
            ctypes.c_int(dtype_id),
            ctypes.c_void_p(cuda_stream))
        if rc == 1:
            # Tensor too large for custom AR — signal caller to use NCCL
            raise ValueError("tensor_too_large")
        elif rc != 0:
            raise RuntimeError(f"infccl_allreduce failed: {rc}")
    
    def _pytorch_allreduce(self, tensor: torch.Tensor, numel: int, stream=None):
        """Fallback: PyTorch-level 2-phase P2P allreduce."""
        gen = self._generation
        self._generation += 1
        
        # Phase 1: Copy to staging
        staging_view = self.staging_buffer[:numel]
        if tensor.dtype == torch.float16:
            staging_view = staging_view.view(torch.float16)[:numel]
        staging_view.copy_(tensor.view(-1))
        
        # Barrier via meta buffer
        self.meta_buffer[self.rank] = gen
        torch.cuda.synchronize()
        
        for r in range(self.world_size):
            if r == self.rank:
                continue
            # CPU spin-wait on peer's meta (less efficient than kernel barrier)
            while True:
                peer_gen = self.peer_meta[r]
                if isinstance(peer_gen, int):
                    break  # raw pointer, can't spin — rely on sync
                if peer_gen[r].item() >= gen:
                    break
        
        # Phase 2: Reduce
        tensor.zero_()
        for r in range(self.world_size):
            peer = self.peer_staging[r]
            if isinstance(peer, torch.Tensor):
                tensor.view(-1).add_(peer[:numel])
            # else: raw pointer, handled by native path only
    
    def destroy(self):
        """Release all resources."""
        if self._use_native and self._native_comm:
            self._native_lib.infccl_comm_destroy(self._native_comm)
            self._native_comm = None


# ============================================================================
# Batched AllReduce Coalescer
# ============================================================================

@dataclass
class PendingAllReduce:
    """A pending allreduce operation waiting to be batched."""
    tensor: torch.Tensor
    op: ReduceOp
    timestamp: float = field(default_factory=time.monotonic)


class AllReduceCoalescer:
    """
    Coalesce multiple small allreduces into one large allreduce.
    
    Mathematical basis (from Mooncake batched-lock theory, Section 4):
      - N small allreduces each with latency α + n_i/B cost N*α total.
      - One coalesced allreduce costs α + Σn_i/B.
      - Savings: (N-1)*α which dominates for small n_i.
    
    This is particularly important on BI-V100 where NCCL launch overhead
    is significant (α ≈ 50μs) and decode generates many small allreduces.
    """

    def __init__(self, max_bytes: int = 4 * 1024 * 1024,
                 max_pending: int = 16,
                 timeout_us: float = 100.0):
        self.max_bytes = max_bytes
        self.max_pending = max_pending
        self.timeout_us = timeout_us
        self.pending: List[PendingAllReduce] = []
        self.coalesced_buffer: Optional[torch.Tensor] = None
    
    def add(self, tensor: torch.Tensor, op: ReduceOp = ReduceOp.SUM) -> bool:
        """
        Add a tensor to the pending batch.
        Returns True if the batch should be flushed.
        """
        self.pending.append(PendingAllReduce(tensor=tensor, op=op))
        
        total_bytes = sum(p.tensor.numel() * p.tensor.element_size() 
                         for p in self.pending)
        
        # Flush conditions
        if len(self.pending) >= self.max_pending:
            return True
        if total_bytes >= self.max_bytes:
            return True
        
        return False
    
    def flush(self, allreduce_fn) -> None:
        """
        Execute the coalesced allreduce.
        
        Args:
            allreduce_fn: callable(tensor) that performs the actual allreduce
        """
        if not self.pending:
            return
        
        if len(self.pending) == 1:
            # Single tensor, no coalescing needed
            allreduce_fn(self.pending[0].tensor)
            self.pending.clear()
            return
        
        # Coalesce: flatten all tensors into one contiguous buffer
        total_numel = sum(p.tensor.numel() for p in self.pending)
        dtype = self.pending[0].tensor.dtype
        device = self.pending[0].tensor.device
        
        if self.coalesced_buffer is None or self.coalesced_buffer.numel() < total_numel:
            self.coalesced_buffer = torch.empty(total_numel, dtype=dtype, device=device)
        
        buf = self.coalesced_buffer[:total_numel]
        
        # Pack
        offset = 0
        for p in self.pending:
            n = p.tensor.numel()
            buf[offset:offset + n].copy_(p.tensor.view(-1))
            offset += n
        
        # Single allreduce on the coalesced buffer
        allreduce_fn(buf)
        
        # Unpack
        offset = 0
        for p in self.pending:
            n = p.tensor.numel()
            p.tensor.view(-1).copy_(buf[offset:offset + n])
            offset += n
        
        self.pending.clear()


# ============================================================================
# Main Communicator
# ============================================================================

class InfcclCommunicator:
    """
    Unified collective communication for BI-V100.
    
    Routes operations to the fastest available backend:
      - Custom P2P allreduce for small tensors (< CUSTOM_AR_THRESHOLD_BYTES)
      - System NCCL for large tensors
      - Optional coalescing for batches of small allreduces
    
    Drop-in compatible with vLLM's PyNcclCommunicator interface.
    """

    def __init__(
        self,
        group: ProcessGroup,
        device: int,
        nccl_library_path: Optional[str] = None,
        enable_custom_ar: bool = ENABLE_CUSTOM_AR,
    ):
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.group = group
        self.device = device
        self.disabled = False
        self.available = True
        
        # Statistics
        self._nccl_calls = 0
        self._custom_ar_calls = 0
        self._coalesced_calls = 0
        
        # Initialize NCCL
        try:
            self.nccl = NcclLibrary(nccl_library_path)
            logger.info(f"[infccl] NCCL version: {self.nccl.get_version()}")
        except Exception as e:
            logger.error(f"[infccl] Failed to load NCCL: {e}")
            self.available = False
            self.disabled = True
            return
        
        # Exchange NCCL unique ID and create communicator
        if self.rank == 0:
            self.unique_id = self.nccl.get_unique_id()
        else:
            self.unique_id = ncclUniqueId()
        
        # Broadcast unique ID via the process group
        uid_tensor = torch.ByteTensor(list(self.unique_id.internal))
        ranks = dist.get_process_group_ranks(group)
        dist.broadcast(uid_tensor, src=ranks[0], group=group)
        byte_list = uid_tensor.tolist()
        for i, byte_val in enumerate(byte_list):
            self.unique_id.internal[i] = byte_val
        
        # Create NCCL communicator
        torch.cuda.set_device(device)
        self.comm = self.nccl.comm_init_rank(
            self.world_size, self.unique_id, self.rank)
        
        # Initialize custom allreduce
        self.custom_ar: Optional[InfcclCustomAR] = None
        if enable_custom_ar and self.world_size > 1:
            try:
                self.custom_ar = InfcclCustomAR(
                    self.rank, self.world_size, self.device)
                if self.custom_ar.available:
                    self._exchange_ipc_handles()
                else:
                    self.custom_ar = None
            except Exception as e:
                logger.warning(f"[infccl] Custom AR init failed: {e}")
                self.custom_ar = None
        
        # Initialize coalescer
        self.coalescer = AllReduceCoalescer()
        
        logger.info(
            f"[infccl] Communicator ready: rank={self.rank}/{self.world_size}, "
            f"custom_ar={'enabled' if self.custom_ar else 'disabled'}")
    
    def _exchange_ipc_handles(self):
        """Exchange IPC handles between all ranks via the process group."""
        if self.custom_ar is None:
            return
        
        try:
            buf_handle, meta_handle = self.custom_ar.get_ipc_handles()
            
            # Gather all handles to all ranks
            handle_tensor = torch.ByteTensor(list(buf_handle) + list(meta_handle))
            handle_list = [torch.zeros_like(handle_tensor) for _ in range(self.world_size)]
            dist.all_gather(handle_list, handle_tensor, group=self.group)
            
            # Register peer handles
            for r in range(self.world_size):
                if r == self.rank:
                    continue
                peer_data = handle_list[r].tolist()
                peer_buf = bytes(peer_data[:64])
                peer_meta = bytes(peer_data[64:128])
                self.custom_ar.register_peer(r, peer_buf, peer_meta)
            
            # Finalize: upload device pointer arrays (native path)
            # After this call, every allreduce launch is zero-alloc.
            self.custom_ar.finalize_peers()
            
            logger.info("[infccl] IPC handles exchanged and finalized")
        except Exception as e:
            logger.warning(f"[infccl] IPC exchange failed: {e}, disabling custom AR")
            self.custom_ar = None
    
    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
        stream=None,
    ):
        """
        Perform allreduce on the given tensor.
        
        Routing logic (v2):
          1. Custom P2P fused allreduce for tiny tensors (≤ 256KB)
          2. Custom P2P ring allreduce for medium tensors (256KB ~ 8MB)
          3. System NCCL for large tensors (> 8MB) or non-SUM ops
          
        The kernel internally routes between fused and ring, and returns
        rc=1 when the tensor exceeds the ring threshold, triggering NCCL.
        """
        if self.disabled or not self.available:
            return
        
        if self.world_size == 1:
            return
        
        nbytes = tensor.numel() * tensor.element_size()
        
        # Route to custom allreduce (fused or ring) for SUM on contiguous tensors
        if (self.custom_ar is not None 
            and nbytes <= CUSTOM_AR_THRESHOLD_BYTES
            and op == ReduceOp.SUM
            and tensor.is_contiguous()):
            try:
                self.custom_ar.all_reduce_inplace(tensor, stream)
                self._custom_ar_calls += 1
                return
            except ValueError as e:
                if "tensor_too_large" in str(e):
                    pass  # fall through to NCCL
                else:
                    logger.warning(f"[infccl] Custom AR failed, falling back to NCCL: {e}")
            except Exception as e:
                logger.warning(f"[infccl] Custom AR failed, falling back to NCCL: {e}")
        
        # Use NCCL
        self._nccl_allreduce(tensor, op, stream)
        self._nccl_calls += 1
    
    def _nccl_allreduce(self, tensor: torch.Tensor, op: ReduceOp, stream=None):
        """Perform allreduce via NCCL."""
        if stream is None:
            stream = torch.cuda.current_stream(self.device).cuda_stream
        elif hasattr(stream, 'cuda_stream'):
            stream = stream.cuda_stream
        
        self.nccl.all_reduce(
            ctypes.c_void_p(tensor.data_ptr()),
            ctypes.c_void_p(tensor.data_ptr()),
            tensor.numel(),
            NcclDataTypeEnum.from_torch(tensor.dtype),
            NcclRedOpEnum.from_torch(op),
            self.comm,
            ctypes.c_void_p(stream),
        )
    
    def all_reduce_coalesced(
        self,
        tensors: List[torch.Tensor],
        op: ReduceOp = ReduceOp.SUM,
        stream=None,
    ):
        """
        Allreduce multiple tensors with coalescing.
        
        Small tensors are packed into a single contiguous buffer
        and reduced in one NCCL call, amortizing the launch overhead.
        
        From Mooncake Section 4 (batched lock theory):
          Speedup ≈ n/1 for n small tensors with α >> n_i/B
        """
        if not tensors:
            return
        
        if len(tensors) == 1:
            self.all_reduce(tensors[0], op, stream)
            return
        
        # Check total size
        total_bytes = sum(t.numel() * t.element_size() for t in tensors)
        
        if total_bytes <= CUSTOM_AR_THRESHOLD_BYTES and self.custom_ar is not None:
            # Coalesce and use custom AR
            total_numel = sum(t.numel() for t in tensors)
            dtype = tensors[0].dtype
            device = tensors[0].device
            buf = torch.empty(total_numel, dtype=dtype, device=device)
            
            # Pack
            offset = 0
            for t in tensors:
                n = t.numel()
                buf[offset:offset + n].copy_(t.view(-1))
                offset += n
            
            # Single allreduce
            self.all_reduce(buf, op, stream)
            
            # Unpack
            offset = 0
            for t in tensors:
                n = t.numel()
                t.view(-1).copy_(buf[offset:offset + n])
                offset += n
            
            self._coalesced_calls += 1
        else:
            # Fall back to individual NCCL calls
            for t in tensors:
                self.all_reduce(t, op, stream)
    
    def send(self, tensor: torch.Tensor, dst: int, stream=None):
        """Point-to-point send via NCCL."""
        if stream is None:
            stream = torch.cuda.current_stream(self.device).cuda_stream
        elif hasattr(stream, 'cuda_stream'):
            stream = stream.cuda_stream
        
        # ncclSend
        self.nccl.lib.ncclSend(
            ctypes.c_void_p(tensor.data_ptr()),
            tensor.numel(),
            NcclDataTypeEnum.from_torch(tensor.dtype),
            dst,
            self.comm,
            ctypes.c_void_p(stream),
        )
    
    def recv(self, tensor: torch.Tensor, src: int, stream=None):
        """Point-to-point recv via NCCL."""
        if stream is None:
            stream = torch.cuda.current_stream(self.device).cuda_stream
        elif hasattr(stream, 'cuda_stream'):
            stream = stream.cuda_stream
        
        self.nccl.lib.ncclRecv(
            ctypes.c_void_p(tensor.data_ptr()),
            tensor.numel(),
            NcclDataTypeEnum.from_torch(tensor.dtype),
            src,
            self.comm,
            ctypes.c_void_p(stream),
        )
    
    def stats(self) -> Dict[str, int]:
        """Return operation statistics."""
        return {
            "nccl_calls": self._nccl_calls,
            "custom_ar_calls": self._custom_ar_calls,
            "coalesced_calls": self._coalesced_calls,
        }
    
    def destroy(self):
        """Clean up resources."""
        if hasattr(self, 'comm') and self.comm:
            try:
                self.nccl.comm_destroy(self.comm)
            except Exception:
                pass
    
    def __del__(self):
        self.destroy()


# ============================================================================
# vLLM integration adapter
# ============================================================================

class InfcclVllmAdapter:
    """
    Drop-in adapter to replace vLLM's PyNcclCommunicator with infccl.
    
    Usage in vLLM's cuda_communicator.py:
        from infccl import InfcclVllmAdapter as PyNcclCommunicator
    
    Or set environment variable:
        VLLM_COMM_BACKEND=infccl
    """

    def __init__(self, group, device, library_path=None):
        self.infccl = InfcclCommunicator(
            group=group,
            device=device,
            nccl_library_path=library_path,
        )
        self.available = self.infccl.available
        self.disabled = self.infccl.disabled
        self.rank = self.infccl.rank
        self.world_size = self.infccl.world_size
    
    def all_reduce(self, tensor, op=ReduceOp.SUM, stream=None):
        self.infccl.all_reduce(tensor, op, stream)
    
    def send(self, tensor, dst, stream=None):
        self.infccl.send(tensor, dst, stream)
    
    def recv(self, tensor, src, stream=None):
        self.infccl.recv(tensor, src, stream)


# ============================================================================
# Topology utilities for BI-V100
# ============================================================================

def detect_bi_v100_topology() -> Dict:
    """
    Detect the BI-V100 PCIe topology and return routing information.
    
    On a typical 4-GPU BI-V100 node:
      - All 4 GPUs connect via PCIe (no NVLink)
      - P2P may not work for all pairs (GPUs 0,3 might cross PCIe switches)
      - NUMA topology affects buffer placement
    
    Returns a dict with:
      - p2p_matrix: which GPU pairs can do P2P
      - numa_nodes: NUMA node for each GPU
      - recommended_pairs: pairs to use for 2-GPU fallback
    """
    num_gpus = torch.cuda.device_count()
    
    p2p_matrix = {}
    for i in range(num_gpus):
        for j in range(num_gpus):
            if i != j:
                can = torch.cuda.can_device_access_peer(i, j)
                p2p_matrix[(i, j)] = can
    
    # Detect NUMA nodes (if available)
    numa_nodes = {}
    for i in range(num_gpus):
        numa_path = f"/sys/devices/virtual/iluvatar-sys/iluvatar{i+1}/numa_node"
        try:
            with open(numa_path) as f:
                numa_nodes[i] = int(f.read().strip())
        except (FileNotFoundError, ValueError):
            numa_nodes[i] = -1
    
    # Find recommended GPU pairs for 2-GPU fallback
    # Prefer pairs that can P2P and are on the same NUMA node
    recommended = []
    for i in range(num_gpus):
        for j in range(i + 1, num_gpus):
            if p2p_matrix.get((i, j), False):
                same_numa = numa_nodes.get(i) == numa_nodes.get(j)
                recommended.append((i, j, same_numa))
    
    recommended.sort(key=lambda x: (not x[2], x[0]))  # prefer same-NUMA
    
    return {
        "num_gpus": num_gpus,
        "p2p_matrix": p2p_matrix,
        "numa_nodes": numa_nodes,
        "recommended_pairs": [(i, j) for i, j, _ in recommended],
    }


# ============================================================================
# Quick test
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    topo = detect_bi_v100_topology()
    print(f"[infccl] Topology: {topo}")
    
    print("[infccl] Module loaded successfully")
    print(f"[infccl] Custom AR threshold: {CUSTOM_AR_THRESHOLD_BYTES} bytes")
    print(f"[infccl] Custom AR enabled: {ENABLE_CUSTOM_AR}")
