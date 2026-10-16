"""
infccl_collectives — AllGather, Broadcast, Reduce, ReduceScatter for BI-V100

Extends infccl with the remaining collective operations using the same
P2P shared-memory architecture. Each operation has:
  - Native CUDA kernel path (zero per-launch alloc)
  - PyTorch fallback path (works without compilation)
  - Automatic routing based on tensor size

Mathematical foundations (from NCCL THEORY):
  AllGather:     concat(input_0, ..., input_{ws-1}) → all ranks
  Broadcast:     input_root → all ranks
  Reduce:        Σ input_r → root only
  ReduceScatter: Σ then scatter chunks → each rank gets its slice
"""

import ctypes
import os
import logging
from typing import Optional, List, Dict

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

logger = logging.getLogger("infccl.collectives")

# Threshold below which we use custom P2P path instead of NCCL
CUSTOM_COLL_THRESHOLD_BYTES = int(
    os.environ.get("INFCCL_CUSTOM_COLL_THRESHOLD", 8 * 1024 * 1024))

# Max staging buffer per rank
MAX_STAGING_SIZE = int(
    os.environ.get("INFCCL_MAX_STAGING", 16 * 1024 * 1024))

ENABLE_CUSTOM_COLL = os.environ.get("INFCCL_ENABLE_CUSTOM_COLL", "1") == "1"


class InfcclCollDataType:
    FLOAT32 = 0
    FLOAT16 = 1
    BFLOAT16 = 2

    _MAP = {
        torch.float32: 0,
        torch.float16: 1,
        torch.bfloat16: 2,
    }

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int:
        if dtype not in cls._MAP:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return cls._MAP[dtype]


class InfcclCollectiveOps:
    """
    Custom P2P collective operations for BI-V100.

    Mirrors the architecture of InfcclCustomAR but adds AllGather,
    Broadcast, Reduce, and ReduceScatter kernels.
    """

    def __init__(self, rank: int, world_size: int, device: int,
                 max_size: int = MAX_STAGING_SIZE):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.max_size = max_size
        self.available = False
        self._native_comm = None
        self._native_lib = None
        self._use_native = False

        # Check P2P
        self._p2p_ok = self._check_p2p()
        if not self._p2p_ok:
            logger.warning("[infccl_coll] P2P not available, custom collectives disabled")
            return

        self._try_load_native()
        if self._use_native:
            self._init_native()
        else:
            logger.info("[infccl_coll] Native kernel not found, only NCCL fallback available")
            self.available = False

    def _check_p2p(self) -> bool:
        all_ok = True
        for i in range(self.world_size):
            if i == self.rank:
                continue
            if not torch.cuda.can_device_access_peer(self.rank, i):
                logger.warning(f"[infccl_coll] GPU {self.rank} cannot P2P GPU {i}")
                all_ok = False
        return all_ok

    def _try_load_native(self):
        search_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "lib", "infccl_collectives.so"),
            os.environ.get("INFCCL_COLL_SO_PATH", ""),
        ]
        for path in search_paths:
            if path and os.path.isfile(path):
                try:
                    lib = ctypes.CDLL(path)
                    # Bind comm lifecycle
                    lib.infccl_coll_comm_create.restype = ctypes.c_void_p
                    lib.infccl_coll_comm_create.argtypes = [
                        ctypes.c_int, ctypes.c_int, ctypes.c_int64]
                    lib.infccl_coll_comm_get_ipc.restype = ctypes.c_int
                    lib.infccl_coll_register_peer.restype = ctypes.c_int
                    lib.infccl_coll_finalize.restype = ctypes.c_int
                    lib.infccl_coll_comm_destroy.restype = None
                    lib.infccl_coll_comm_destroy.argtypes = [ctypes.c_void_p]

                    # Bind collective ops
                    for fn_name in ['infccl_allgather', 'infccl_broadcast',
                                    'infccl_reduce', 'infccl_reduce_scatter']:
                        fn = getattr(lib, fn_name)
                        fn.restype = ctypes.c_int

                    lib.infccl_allgather.argtypes = [
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_int64, ctypes.c_int, ctypes.c_void_p]
                    lib.infccl_broadcast.argtypes = [
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
                    lib.infccl_reduce.argtypes = [
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
                    lib.infccl_reduce_scatter.argtypes = [
                        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                        ctypes.c_int64, ctypes.c_int, ctypes.c_void_p]

                    self._native_lib = lib
                    self._use_native = True
                    logger.info(f"[infccl_coll] Loaded native kernel from {path}")
                    return
                except (OSError, AttributeError) as e:
                    logger.debug(f"[infccl_coll] Cannot load {path}: {e}")

    def _init_native(self):
        lib = self._native_lib
        self._native_comm = lib.infccl_coll_comm_create(
            self.rank, self.world_size, self.max_size)
        if not self._native_comm:
            logger.error("[infccl_coll] comm_create returned NULL")
            self._use_native = False
            return
        self.available = True
        logger.info(f"[infccl_coll] Native comm created: rank={self.rank}, ws={self.world_size}")

    def get_ipc_handles(self):
        handle_size = 64
        buf_h = (ctypes.c_byte * handle_size)()
        bar_h = (ctypes.c_byte * handle_size)()
        if self._use_native:
            rc = self._native_lib.infccl_coll_comm_get_ipc(
                self._native_comm, ctypes.byref(buf_h), ctypes.byref(bar_h))
            if rc != 0:
                raise RuntimeError(f"infccl_coll_comm_get_ipc failed: {rc}")
        return (bytes(buf_h), bytes(bar_h))

    def register_peer(self, peer_rank: int, buf_bytes: bytes, bar_bytes: bytes):
        if peer_rank == self.rank:
            return
        handle_size = 64
        buf_h = (ctypes.c_byte * handle_size)(*buf_bytes)
        bar_h = (ctypes.c_byte * handle_size)(*bar_bytes)
        if self._use_native:
            rc = self._native_lib.infccl_coll_register_peer(
                self._native_comm, peer_rank,
                ctypes.byref(buf_h), ctypes.byref(bar_h))
            if rc != 0:
                raise RuntimeError(f"register_peer failed for rank {peer_rank}: {rc}")

    def finalize_peers(self):
        if self._use_native:
            rc = self._native_lib.infccl_coll_finalize(self._native_comm)
            if rc != 0:
                raise RuntimeError(f"finalize failed: {rc}")
            logger.info("[infccl_coll] Finalized (zero-alloc path ready)")

    def _get_stream(self, stream):
        if stream is None:
            return torch.cuda.current_stream(self.device).cuda_stream
        elif hasattr(stream, 'cuda_stream'):
            return stream.cuda_stream
        return stream

    def allgather(self, output: torch.Tensor, input: torch.Tensor, stream=None):
        """
        AllGather: each rank contributes input, output = concat of all inputs.
        output must have size = input.numel() * world_size.
        """
        if not self.available or not self._use_native:
            raise RuntimeError("Native collectives not available")

        dtype_id = InfcclCollDataType.from_torch(input.dtype)
        cuda_stream = self._get_stream(stream)

        rc = self._native_lib.infccl_allgather(
            self._native_comm,
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_void_p(input.data_ptr()),
            ctypes.c_int64(input.numel()),
            ctypes.c_int(dtype_id),
            ctypes.c_void_p(cuda_stream))
        if rc != 0:
            raise RuntimeError(f"infccl_allgather failed: {rc}")

    def broadcast(self, output: torch.Tensor, input: torch.Tensor,
                  root: int = 0, stream=None):
        """
        Broadcast: root's input → all ranks' output.
        """
        if not self.available or not self._use_native:
            raise RuntimeError("Native collectives not available")

        dtype_id = InfcclCollDataType.from_torch(input.dtype)
        cuda_stream = self._get_stream(stream)

        rc = self._native_lib.infccl_broadcast(
            self._native_comm,
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_void_p(input.data_ptr()),
            ctypes.c_int64(input.numel()),
            ctypes.c_int(dtype_id),
            ctypes.c_int(root),
            ctypes.c_void_p(cuda_stream))
        if rc != 0:
            raise RuntimeError(f"infccl_broadcast failed: {rc}")

    def reduce(self, output: torch.Tensor, input: torch.Tensor,
               root: int = 0, stream=None):
        """
        Reduce: sum of all ranks' input → root's output.
        """
        if not self.available or not self._use_native:
            raise RuntimeError("Native collectives not available")

        dtype_id = InfcclCollDataType.from_torch(input.dtype)
        cuda_stream = self._get_stream(stream)

        rc = self._native_lib.infccl_reduce(
            self._native_comm,
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_void_p(input.data_ptr()),
            ctypes.c_int64(input.numel()),
            ctypes.c_int(dtype_id),
            ctypes.c_int(root),
            ctypes.c_void_p(cuda_stream))
        if rc != 0:
            raise RuntimeError(f"infccl_reduce failed: {rc}")

    def reduce_scatter(self, output: torch.Tensor, input: torch.Tensor,
                       stream=None):
        """
        ReduceScatter: reduce then scatter. Each rank gets a chunk.
        input.numel() must be divisible by world_size.
        output.numel() == input.numel() / world_size.
        """
        if not self.available or not self._use_native:
            raise RuntimeError("Native collectives not available")

        dtype_id = InfcclCollDataType.from_torch(input.dtype)
        cuda_stream = self._get_stream(stream)

        rc = self._native_lib.infccl_reduce_scatter(
            self._native_comm,
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_void_p(input.data_ptr()),
            ctypes.c_int64(input.numel()),
            ctypes.c_int(dtype_id),
            ctypes.c_void_p(cuda_stream))
        if rc != 0:
            raise RuntimeError(f"infccl_reduce_scatter failed: {rc}")

    def destroy(self):
        if self._use_native and self._native_comm:
            self._native_lib.infccl_coll_comm_destroy(self._native_comm)
            self._native_comm = None

    def __del__(self):
        self.destroy()


class InfcclFullCommunicator:
    """
    Full-featured communicator combining allreduce + all collectives.

    Routes each operation to the fastest path:
      - Custom P2P for small tensors (< threshold)
      - NCCL for large tensors
    """

    def __init__(self, group: ProcessGroup, device: int,
                 nccl_library_path: Optional[str] = None):
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.group = group
        self.device = device

        # Stats
        self._stats = {
            'allgather_custom': 0, 'allgather_nccl': 0,
            'broadcast_custom': 0, 'broadcast_nccl': 0,
            'reduce_custom': 0, 'reduce_nccl': 0,
            'reduce_scatter_custom': 0, 'reduce_scatter_nccl': 0,
        }

        # Load NCCL
        from .infccl import NcclLibrary, ncclUniqueId, NcclDataTypeEnum, NcclRedOpEnum
        self._NcclDataTypeEnum = NcclDataTypeEnum
        self._NcclRedOpEnum = NcclRedOpEnum

        try:
            self.nccl = NcclLibrary(nccl_library_path)
        except Exception as e:
            logger.error(f"[infccl_coll] Failed to load NCCL: {e}")
            self.available = False
            return

        # Exchange NCCL unique ID
        if self.rank == 0:
            self.nccl_uid = self.nccl.get_unique_id()
        else:
            self.nccl_uid = ncclUniqueId()

        uid_tensor = torch.ByteTensor(list(self.nccl_uid.internal))
        ranks = dist.get_process_group_ranks(group)
        dist.broadcast(uid_tensor, src=ranks[0], group=group)
        for i, b in enumerate(uid_tensor.tolist()):
            self.nccl_uid.internal[i] = b

        torch.cuda.set_device(device)
        self.nccl_comm = self.nccl.comm_init_rank(
            self.world_size, self.nccl_uid, self.rank)

        # Initialize custom collectives
        self.custom_coll: Optional[InfcclCollectiveOps] = None
        if ENABLE_CUSTOM_COLL and self.world_size > 1:
            try:
                self.custom_coll = InfcclCollectiveOps(
                    self.rank, self.world_size, self.device)
                if self.custom_coll.available:
                    self._exchange_coll_ipc()
                else:
                    self.custom_coll = None
            except Exception as e:
                logger.warning(f"[infccl_coll] Custom coll init failed: {e}")
                self.custom_coll = None

        self.available = True
        logger.info(
            f"[infccl_coll] Full communicator ready: rank={self.rank}/{self.world_size}, "
            f"custom_coll={'enabled' if self.custom_coll else 'disabled'}")

    def _exchange_coll_ipc(self):
        if self.custom_coll is None:
            return
        try:
            buf_h, bar_h = self.custom_coll.get_ipc_handles()
            handle_tensor = torch.ByteTensor(list(buf_h) + list(bar_h))
            handle_list = [torch.zeros_like(handle_tensor)
                           for _ in range(self.world_size)]
            dist.all_gather(handle_list, handle_tensor, group=self.group)
            for r in range(self.world_size):
                if r == self.rank:
                    continue
                data = handle_list[r].tolist()
                self.custom_coll.register_peer(r, bytes(data[:64]), bytes(data[64:128]))
            self.custom_coll.finalize_peers()
            logger.info("[infccl_coll] IPC handles exchanged")
        except Exception as e:
            logger.warning(f"[infccl_coll] IPC exchange failed: {e}")
            self.custom_coll = None

    def _get_nccl_stream(self, stream):
        if stream is None:
            return torch.cuda.current_stream(self.device).cuda_stream
        elif hasattr(stream, 'cuda_stream'):
            return stream.cuda_stream
        return stream

    def _should_use_custom(self, nbytes: int) -> bool:
        return (self.custom_coll is not None
                and nbytes <= CUSTOM_COLL_THRESHOLD_BYTES)

    # ---- AllGather ----
    def all_gather(self, output: torch.Tensor, input: torch.Tensor,
                   stream=None):
        nbytes = input.numel() * input.element_size()
        if self._should_use_custom(nbytes):
            try:
                self.custom_coll.allgather(output, input, stream)
                self._stats['allgather_custom'] += 1
                return
            except Exception as e:
                logger.warning(f"[infccl_coll] Custom allgather failed: {e}")

        # NCCL fallback
        cuda_stream = self._get_nccl_stream(stream)
        self.nccl.lib.ncclAllGather(
            ctypes.c_void_p(input.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            input.numel(),
            self._NcclDataTypeEnum.from_torch(input.dtype),
            self.nccl_comm,
            ctypes.c_void_p(cuda_stream))
        self._stats['allgather_nccl'] += 1

    # ---- Broadcast ----
    def broadcast(self, tensor: torch.Tensor, root: int = 0, stream=None):
        nbytes = tensor.numel() * tensor.element_size()
        if self._should_use_custom(nbytes):
            try:
                self.custom_coll.broadcast(tensor, tensor, root, stream)
                self._stats['broadcast_custom'] += 1
                return
            except Exception as e:
                logger.warning(f"[infccl_coll] Custom broadcast failed: {e}")

        cuda_stream = self._get_nccl_stream(stream)
        self.nccl.lib.ncclBroadcast(
            ctypes.c_void_p(tensor.data_ptr()),
            ctypes.c_void_p(tensor.data_ptr()),
            tensor.numel(),
            self._NcclDataTypeEnum.from_torch(tensor.dtype),
            root,
            self.nccl_comm,
            ctypes.c_void_p(cuda_stream))
        self._stats['broadcast_nccl'] += 1

    # ---- Reduce ----
    def reduce(self, tensor: torch.Tensor, root: int = 0, stream=None):
        nbytes = tensor.numel() * tensor.element_size()
        if self._should_use_custom(nbytes):
            try:
                self.custom_coll.reduce(tensor, tensor, root, stream)
                self._stats['reduce_custom'] += 1
                return
            except Exception as e:
                logger.warning(f"[infccl_coll] Custom reduce failed: {e}")

        cuda_stream = self._get_nccl_stream(stream)
        self.nccl.lib.ncclReduce(
            ctypes.c_void_p(tensor.data_ptr()),
            ctypes.c_void_p(tensor.data_ptr()),
            tensor.numel(),
            self._NcclDataTypeEnum.from_torch(tensor.dtype),
            0,  # ncclSum
            root,
            self.nccl_comm,
            ctypes.c_void_p(cuda_stream))
        self._stats['reduce_nccl'] += 1

    # ---- ReduceScatter ----
    def reduce_scatter(self, output: torch.Tensor, input: torch.Tensor,
                       stream=None):
        nbytes = input.numel() * input.element_size()
        if self._should_use_custom(nbytes):
            try:
                self.custom_coll.reduce_scatter(output, input, stream)
                self._stats['reduce_scatter_custom'] += 1
                return
            except Exception as e:
                logger.warning(f"[infccl_coll] Custom reduce_scatter failed: {e}")

        cuda_stream = self._get_nccl_stream(stream)
        self.nccl.lib.ncclReduceScatter(
            ctypes.c_void_p(input.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            output.numel(),
            self._NcclDataTypeEnum.from_torch(input.dtype),
            0,  # ncclSum
            self.nccl_comm,
            ctypes.c_void_p(cuda_stream))
        self._stats['reduce_scatter_nccl'] += 1

    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def destroy(self):
        if self.custom_coll:
            self.custom_coll.destroy()
        if hasattr(self, 'nccl_comm') and self.nccl_comm:
            try:
                self.nccl.comm_destroy(self.nccl_comm)
            except Exception:
                pass

    def __del__(self):
        self.destroy()

