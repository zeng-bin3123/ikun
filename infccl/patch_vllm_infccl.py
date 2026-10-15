"""
Monkey-patch vLLM to use infccl instead of the default NCCL path.

This script is imported early in the BI-V100 deployment pipeline
(by patch_ops.sh or env_override.py) to redirect collective operations
through infccl's optimized paths.

Integration points:
  1. PyNcclCommunicator → InfcclVllmAdapter
  2. Custom allreduce → infccl's P2P allreduce for small tensors
  3. AllReduce coalescing for decode-phase small tensor reductions

Usage:
    import infccl.patch_vllm_infccl  # auto-patches on import
    
    # Or explicitly:
    from infccl.patch_vllm_infccl import patch_vllm
    patch_vllm()
"""

import os
import sys
import logging

logger = logging.getLogger("infccl.patch")


def patch_vllm():
    """
    Patch vLLM's NCCL communicator to use infccl.
    
    This patches:
      1. vllm.distributed.device_communicators.pynccl.PyNcclCommunicator
      2. The custom allreduce path in cuda_communicator.py
    """
    
    # Only patch if we're on BI-V100
    if not _is_bi_v100():
        logger.info("[infccl] Not BI-V100 hardware, skipping patch")
        return False
    
    try:
        # Add infccl to path
        infccl_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if infccl_root not in sys.path:
            sys.path.insert(0, infccl_root)
        
        from infccl.python.infccl import InfcclVllmAdapter, InfcclCommunicator
        
        # Patch PyNcclCommunicator
        try:
            import vllm.distributed.device_communicators.pynccl as pynccl_mod
            
            # Save original for fallback
            pynccl_mod._OriginalPyNcclCommunicator = pynccl_mod.PyNcclCommunicator
            
            # Replace with infccl adapter
            pynccl_mod.PyNcclCommunicator = InfcclVllmAdapter
            
            logger.info("[infccl] Patched PyNcclCommunicator → InfcclVllmAdapter")
        except ImportError:
            logger.warning("[infccl] Could not import vllm pynccl module")
        
        # Patch custom allreduce to use infccl's version
        try:
            import vllm.distributed.device_communicators.custom_all_reduce as car_mod
            
            # The custom_ar flag may be False on BI-V100 because the
            # C++ custom allreduce ops aren't available. infccl provides
            # a pure-Python+CUDA alternative.
            if not getattr(car_mod, 'custom_ar', False):
                logger.info("[infccl] vLLM custom_ar=False, infccl provides alternative")
                # We don't patch CustomAllreduce directly because it's
                # already handled by the InfcclCommunicator routing logic.
                # The InfcclCommunicator.all_reduce() will route small
                # tensors to P2P allreduce automatically.
        except ImportError:
            pass
        
        logger.info("[infccl] vLLM patching complete")
        return True
        
    except Exception as e:
        logger.error(f"[infccl] Patching failed: {e}")
        return False


def _is_bi_v100() -> bool:
    """Detect if we're running on Iluvatar BI-V100."""
    # Check for Iluvatar device files
    if os.path.exists("/dev/iluvatar0"):
        return True
    
    # Check for CoreX SDK
    if os.path.exists("/usr/local/corex"):
        return True
    
    # Check environment variable
    if os.environ.get("INFCCL_FORCE_ENABLE", "0") == "1":
        return True
    
    return False


def unpatch_vllm():
    """Restore original vLLM NCCL communicator."""
    try:
        import vllm.distributed.device_communicators.pynccl as pynccl_mod
        if hasattr(pynccl_mod, '_OriginalPyNcclCommunicator'):
            pynccl_mod.PyNcclCommunicator = pynccl_mod._OriginalPyNcclCommunicator
            logger.info("[infccl] Restored original PyNcclCommunicator")
    except ImportError:
        pass


# Auto-patch on import if enabled
if os.environ.get("INFCCL_AUTO_PATCH", "1") == "1":
    patch_vllm()

