"""
infccl — Iluvatar Inference Collective Communication Library

Optimized collective communications for Iluvatar BI-V100 GPUs.
"""

from infccl.python.infccl import (
    InfcclCommunicator,
    InfcclVllmAdapter,
    InfcclCustomAR,
    AllReduceCoalescer,
    NcclLibrary,
    detect_bi_v100_topology,
    CUSTOM_AR_THRESHOLD_BYTES,
    ENABLE_CUSTOM_AR,
)

__version__ = "0.1.0"
__all__ = [
    "InfcclCommunicator",
    "InfcclVllmAdapter",
    "InfcclCustomAR",
    "AllReduceCoalescer",
    "NcclLibrary",
    "detect_bi_v100_topology",
]
