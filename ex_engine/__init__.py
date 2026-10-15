# ex_engine — BI-V100 inference acceleration engine
# Graceful import: don't crash if sub-modules have missing dependencies
try:
    from .python.ex_loader import EXEngine, get_engine
except Exception:
    EXEngine = None
    get_engine = None

__all__ = ["EXEngine", "get_engine"]
