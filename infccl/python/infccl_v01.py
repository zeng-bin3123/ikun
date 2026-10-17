"""
infccl_v01.py — Python bridge to the v0.1 C library (src/*.cu)

This wraps the NCCL-compatible C API from infccl/src/infccl.h
for use from Python. Unlike the v1 IPC-based backend, this
uses staged transport (cudaMemcpyPeer) and works on ivcore10.

Usage:
    from infccl.python.infccl_v01 import InfcclV01

    v01 = InfcclV01(nranks=4)
    v01.all_reduce(tensor, stream=stream)
"""

import ctypes
import os
import logging
import torch

logger = logging.getLogger("infccl.v01")


# Map infccl.h enums
class InfcclDataType:
    FLOAT16  = 0
    FLOAT32  = 1
    BFLOAT16 = 2
    INT32    = 3

    _TORCH_MAP = {
        torch.float16:  0,
        torch.float32:  1,
        torch.bfloat16: 2,
        torch.int32:    3,
    }

    @classmethod
    def from_torch(cls, dtype):
        if dtype not in cls._TORCH_MAP:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return cls._TORCH_MAP[dtype]


class InfcclRedOp:
    SUM  = 0
    MAX  = 1
    MIN  = 2
    PROD = 3


class InfcclV01:
    """
    Wrapper around the v0.1 shared library built from infccl/src/.

    This is the staged-transport backend that works on ivcore10
    (no __threadfence_system required).

    NOTE: This wrapper is for single-process multi-GPU usage (testing).
    In production multi-process (vLLM), each process calls the C API
    directly after sharing the barrier flags via shared memory.
    """

    def __init__(self, so_path=None):
        self.lib = None
        self.comms = []
        self._load(so_path)

    def _load(self, so_path=None):
        """Load the compiled .so library."""
        if so_path is None:
            candidates = [
                os.path.join(os.path.dirname(__file__), "..", "lib",
                             "libinfccl.so"),
                os.path.join(os.path.dirname(__file__), "..",
                             "libinfccl.so"),
            ]
            for p in candidates:
                if os.path.isfile(p):
                    so_path = p
                    break

        if so_path is None or not os.path.isfile(so_path):
            logger.warning("[infccl v0.1] Shared library not found. "
                           "Build with: cd infccl && make lib")
            return

        try:
            lib = ctypes.CDLL(so_path)
            self._bind(lib)
            self.lib = lib
            logger.info(f"[infccl v0.1] Loaded {so_path}")
        except OSError as e:
            logger.error(f"[infccl v0.1] Failed to load: {e}")

    def _bind(self, lib):
        """Set up ctypes function signatures."""
        # infcclCommInitAll(comms, nranks)
        # Not in the public API but useful for testing
        pass  # single-process init handled in Python

    @property
    def available(self):
        return self.lib is not None

    @staticmethod
    def ensure_contiguous(tensor):
        """Handle non-contiguous tensors (Mooncake d603a5af)."""
        if tensor.is_contiguous():
            return tensor, False
        logger.debug("[infccl v0.1] Making tensor contiguous")
        return tensor.contiguous(), True

    @staticmethod
    def restore_non_contiguous(original, result, was_copied):
        """Copy result back if we had to make a contiguous copy."""
        if was_copied:
            original.copy_(result)


def detect_infccl_v01():
    """Check if v0.1 library is available."""
    v01 = InfcclV01()
    return v01.available
