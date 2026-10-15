# ex_engine.moe — MoE expert computation for BI-V100
#
# This is a LIGHTWEIGHT wrapper that exposes naive_batched_moe_forward
# without pulling in vllm's Triton-dependent fused_moe internals.
# The original upstream __init__.py imports from
# vllm.model_executor.layers.fused_moe.* which requires Triton and
# crashes on BI-V100. We keep only what qwen3_5.py actually needs.

from .naive_batched_experts import naive_batched_moe_forward

__all__ = ["naive_batched_moe_forward"]

# Lazy imports for anything else — only load if explicitly requested
def __getattr__(name):
    """Lazy-load upstream vllm fused_moe components on demand."""
    # These are only needed if someone explicitly imports them from ex_engine.moe
    # In normal operation, only naive_batched_moe_forward is used.
    _upstream_names = {
        "FusedMoE", "FusedMoERouter", "FusedMoEConfig",
        "FusedMoEQuantConfig", "FusedMoEParallelConfig",
        "FusedMoEMethodBase", "MoEActivation",
        "UnquantizedFusedMoEMethod", "FusedMoeWeightScaleSupported",
        "override_config", "get_config",
    }
    if name in _upstream_names:
        try:
            import importlib
            mod = importlib.import_module("vllm.model_executor.layers.fused_moe")
            return getattr(mod, name)
        except (ImportError, AttributeError):
            raise AttributeError(f"ex_engine.moe has no attribute {name!r} "
                                 f"(vllm fused_moe not available)")
    raise AttributeError(f"module 'ex_engine.moe' has no attribute {name!r}")
