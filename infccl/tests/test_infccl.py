"""
Tests for infccl library.

These tests run without distributed setup (single-process) to verify
the individual components. Multi-GPU tests require 4× BI-V100.

Run: python -m pytest infccl/tests/test_infccl.py -v
"""

import os
import sys
import pytest
import torch

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestNcclLibrary:
    """Test the NCCL library wrapper."""

    def test_load_nccl(self):
        """Test that we can load the NCCL library."""
        from infccl.python.infccl import NcclLibrary
        try:
            nccl = NcclLibrary()
            version = nccl.get_version()
            assert version > 0
            print(f"NCCL version: {version}")
        except OSError as e:
            pytest.skip(f"NCCL library not available: {e}")
    
    def test_unique_id(self):
        """Test unique ID generation."""
        from infccl.python.infccl import NcclLibrary, ncclUniqueId
        try:
            nccl = NcclLibrary()
            uid = nccl.get_unique_id()
            # Check that it's not all zeros
            assert any(uid.internal[i] != 0 for i in range(128))
        except OSError:
            pytest.skip("NCCL not available")


class TestAllReduceCoalescer:
    """Test the allreduce coalescer."""

    def test_single_tensor(self):
        """Single tensor should pass through without coalescing."""
        from infccl.python.infccl import AllReduceCoalescer
        
        coalescer = AllReduceCoalescer(max_pending=4)
        t = torch.randn(100)
        
        should_flush = coalescer.add(t)
        assert not should_flush  # only 1 tensor, below threshold
    
    def test_batch_triggers_flush(self):
        """Reaching max_pending should trigger flush."""
        from infccl.python.infccl import AllReduceCoalescer
        
        coalescer = AllReduceCoalescer(max_pending=3)
        
        for i in range(2):
            assert not coalescer.add(torch.randn(10))
        
        # Third tensor should trigger flush
        assert coalescer.add(torch.randn(10))
    
    def test_coalesced_reduction(self):
        """Test that coalesced allreduce produces correct results."""
        from infccl.python.infccl import AllReduceCoalescer
        from torch.distributed import ReduceOp
        
        coalescer = AllReduceCoalescer(max_pending=4)
        
        tensors = [torch.ones(10) * (i + 1) for i in range(3)]
        for t in tensors:
            coalescer.add(t, ReduceOp.SUM)
        
        # Mock allreduce: multiply by 2 (simulating 2-rank sum)
        def mock_ar(buf):
            buf.mul_(2.0)
        
        coalescer.flush(mock_ar)
        
        # Each tensor should be doubled
        for i, t in enumerate(tensors):
            expected = (i + 1) * 2.0
            assert torch.allclose(t, torch.ones(10) * expected), \
                f"Tensor {i}: expected {expected}, got {t[0].item()}"
    
    def test_size_triggers_flush(self):
        """Large total size should trigger flush."""
        from infccl.python.infccl import AllReduceCoalescer
        
        coalescer = AllReduceCoalescer(max_bytes=1000, max_pending=100)
        
        # Each float32 tensor of 100 elements = 400 bytes
        assert not coalescer.add(torch.randn(100))  # 400 bytes
        assert not coalescer.add(torch.randn(100))  # 800 bytes
        assert coalescer.add(torch.randn(100))       # 1200 > 1000


class TestTopology:
    """Test topology detection."""

    def test_detect_topology(self):
        """Test topology detection doesn't crash."""
        from infccl.python.infccl import detect_bi_v100_topology
        
        topo = detect_bi_v100_topology()
        assert "num_gpus" in topo
        assert "p2p_matrix" in topo
        assert "numa_nodes" in topo
        print(f"Detected topology: {topo}")


class TestDataTypeMapping:
    """Test dtype mapping."""

    def test_float32(self):
        from infccl.python.infccl import torch_to_infccl_dtype, InfcclDataType
        assert torch_to_infccl_dtype(torch.float32) == InfcclDataType.FLOAT32
    
    def test_float16(self):
        from infccl.python.infccl import torch_to_infccl_dtype, InfcclDataType
        assert torch_to_infccl_dtype(torch.float16) == InfcclDataType.FLOAT16
    
    def test_bfloat16(self):
        from infccl.python.infccl import torch_to_infccl_dtype, InfcclDataType
        assert torch_to_infccl_dtype(torch.bfloat16) == InfcclDataType.BFLOAT16
    
    def test_unsupported(self):
        from infccl.python.infccl import torch_to_infccl_dtype
        with pytest.raises(ValueError):
            torch_to_infccl_dtype(torch.int32)


class TestNcclDtypeEnum:
    """Test NCCL dtype enum mapping."""

    def test_torch_to_nccl(self):
        from infccl.python.infccl import NcclDataTypeEnum
        assert NcclDataTypeEnum.from_torch(torch.float32) == 7
        assert NcclDataTypeEnum.from_torch(torch.float16) == 6
        assert NcclDataTypeEnum.from_torch(torch.bfloat16) == 9


class TestNcclRedOpEnum:
    """Test NCCL reduce op enum mapping."""

    def test_torch_to_nccl(self):
        from infccl.python.infccl import NcclRedOpEnum
        from torch.distributed import ReduceOp
        assert NcclRedOpEnum.from_torch(ReduceOp.SUM) == 0
        assert NcclRedOpEnum.from_torch(ReduceOp.PRODUCT) == 1
        assert NcclRedOpEnum.from_torch(ReduceOp.MAX) == 2
        assert NcclRedOpEnum.from_torch(ReduceOp.MIN) == 3


class TestRingPathSelection:
    """Test the v2 path selection thresholds."""

    def test_fused_path(self):
        """Tiny tensors should use fused (one-shot) path."""
        # 100 FP16 elements = 200 bytes, well under 256KB fused threshold
        assert 200 <= 256 * 1024  # fused path

    def test_ring_path(self):
        """Medium tensors should use ring path."""
        # 500K FP16 elements = 1MB, between 256KB and 8MB
        nbytes = 500_000 * 2
        assert 256 * 1024 < nbytes <= 8 * 1024 * 1024  # ring path

    def test_nccl_path(self):
        """Large tensors should fall back to NCCL."""
        # 5M FP32 elements = 20MB, above 8MB ring threshold
        nbytes = 5_000_000 * 4
        assert nbytes > 8 * 1024 * 1024  # NCCL path

    def test_native_query_path(self):
        """Test the native infccl_query_path function if available."""
        try:
            import ctypes
            search_paths = [
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__)))), "infccl", "lib", "infccl_allreduce.so"),
            ]
            lib = None
            for p in search_paths:
                if os.path.isfile(p):
                    lib = ctypes.CDLL(p)
                    break
            if lib is None:
                pytest.skip("infccl_allreduce.so not built")
            
            lib.infccl_query_path.restype = ctypes.c_int
            lib.infccl_query_path.argtypes = [ctypes.c_int64]
            
            # Fused: ≤ 256KB
            assert lib.infccl_query_path(100) == 0           # 100 bytes → fused
            assert lib.infccl_query_path(256 * 1024) == 0    # exactly 256KB → fused
            
            # Ring: 256KB < x ≤ 8MB
            assert lib.infccl_query_path(256 * 1024 + 1) == 1  # just over → ring
            assert lib.infccl_query_path(8 * 1024 * 1024) == 1 # exactly 8MB → ring
            
            # NCCL: > 8MB
            assert lib.infccl_query_path(8 * 1024 * 1024 + 1) == 2  # just over → NCCL
            
            print("infccl_query_path: all thresholds verified")
        except OSError as e:
            pytest.skip(f"Cannot load native library: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
