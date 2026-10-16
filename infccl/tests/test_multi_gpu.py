#!/usr/bin/env python3
"""
Multi-GPU correctness tests for infccl on BI-V100.

Run with torchrun (requires 2+ GPUs):
  torchrun --nproc_per_node=4 infccl/tests/test_multi_gpu.py

Or with 2 GPUs:
  torchrun --nproc_per_node=2 infccl/tests/test_multi_gpu.py

Tests verify:
  1. Fused AllReduce (≤256KB) — correctness across all dtypes
  2. Ring AllReduce (256KB~8MB) — correctness and bandwidth
  3. ReduceScatter — output is 1/P of the reduced input
  4. AllGather — output is concatenation of all ranks' inputs
  5. AllReduce Coalescing — batched small tensors
  6. NCCL fallback — large tensors route correctly
"""

import os
import sys
import time
import math
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ── Test infrastructure ──

class TestResult:
    def __init__(self, name):
        self.name = name
        self.passed = False
        self.error = None
        self.latency_us = 0.0

    def __repr__(self):
        status = "PASS" if self.passed else f"FAIL: {self.error}"
        return f"  [{status}] {self.name} ({self.latency_us:.0f} μs)"


def check_close(a, b, name, atol=1e-2, rtol=1e-2):
    """Check two tensors are close, with relaxed tolerance for FP16/BF16."""
    if not torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol):
        max_diff = (a.float() - b.float()).abs().max().item()
        raise AssertionError(
            f"{name}: max_diff={max_diff:.6f} > atol={atol} "
            f"(shapes: {a.shape} vs {b.shape})")


# ── Tests ──

def test_fused_allreduce(rank, world_size, comm, dtype=torch.float16):
    """Test fused (one-shot) allreduce for small tensors."""
    result = TestResult(f"fused_allreduce_{dtype}")
    try:
        # Each rank fills tensor with (rank + 1)
        size = 1024  # 2KB for FP16 — well within fused threshold
        tensor = torch.full((size,), rank + 1, dtype=dtype, device=f"cuda:{rank}")
        expected_sum = sum(range(1, world_size + 1))  # 1+2+...+P

        t0 = time.perf_counter()
        comm.all_reduce(tensor)
        torch.cuda.synchronize()
        result.latency_us = (time.perf_counter() - t0) * 1e6

        expected = torch.full((size,), expected_sum, dtype=dtype, device=f"cuda:{rank}")
        check_close(tensor, expected, "fused_allreduce")
        result.passed = True
    except Exception as e:
        result.error = str(e)
    return result


def test_ring_allreduce(rank, world_size, comm, dtype=torch.float16):
    """Test ring allreduce for medium tensors (256KB~8MB)."""
    result = TestResult(f"ring_allreduce_{dtype}")
    try:
        # 512K FP16 elements = 1MB — in ring territory
        size = 512 * 1024
        tensor = torch.full((size,), rank + 1, dtype=dtype, device=f"cuda:{rank}")
        expected_sum = sum(range(1, world_size + 1))

        t0 = time.perf_counter()
        comm.all_reduce(tensor)
        torch.cuda.synchronize()
        result.latency_us = (time.perf_counter() - t0) * 1e6

        expected = torch.full((size,), expected_sum, dtype=dtype, device=f"cuda:{rank}")
        check_close(tensor, expected, "ring_allreduce")
        result.passed = True
    except Exception as e:
        result.error = str(e)
    return result


def test_ring_allreduce_random(rank, world_size, comm, dtype=torch.float16):
    """Test ring allreduce with random data to catch numerical issues."""
    result = TestResult(f"ring_allreduce_random_{dtype}")
    try:
        size = 256 * 1024  # 512KB — ring path
        torch.manual_seed(42)  # same seed on all ranks for verification
        
        # Each rank has different random data
        torch.manual_seed(42 + rank)
        my_data = torch.randn(size, dtype=dtype, device=f"cuda:{rank}")
        
        # Compute expected sum (sum of all ranks' random data)
        all_data = []
        for r in range(world_size):
            torch.manual_seed(42 + r)
            all_data.append(torch.randn(size, dtype=dtype, device=f"cuda:{rank}"))
        expected = torch.stack(all_data).float().sum(dim=0).to(dtype)
        
        t0 = time.perf_counter()
        comm.all_reduce(my_data)
        torch.cuda.synchronize()
        result.latency_us = (time.perf_counter() - t0) * 1e6

        # FP16 accumulation across 4 ranks can have noticeable error
        check_close(my_data, expected, "ring_allreduce_random", atol=0.05, rtol=0.02)
        result.passed = True
    except Exception as e:
        result.error = str(e)
    return result


def test_large_nccl_fallback(rank, world_size, comm, dtype=torch.float16):
    """Test that large tensors correctly fall back to NCCL."""
    result = TestResult(f"nccl_fallback_{dtype}")
    try:
        # 8M FP16 elements = 16MB — should route to NCCL
        size = 8 * 1024 * 1024
        tensor = torch.full((size,), rank + 1, dtype=dtype, device=f"cuda:{rank}")
        expected_sum = sum(range(1, world_size + 1))

        t0 = time.perf_counter()
        comm.all_reduce(tensor)
        torch.cuda.synchronize()
        result.latency_us = (time.perf_counter() - t0) * 1e6

        expected = torch.full((size,), expected_sum, dtype=dtype, device=f"cuda:{rank}")
        check_close(tensor, expected, "nccl_fallback")
        result.passed = True
    except Exception as e:
        result.error = str(e)
    return result


def test_coalesced_allreduce(rank, world_size, comm):
    """Test coalesced allreduce (multiple small tensors batched)."""
    result = TestResult("coalesced_allreduce")
    try:
        tensors = [torch.full((256,), rank + 1 + i * 0.1,
                   dtype=torch.float16, device=f"cuda:{rank}")
                   for i in range(8)]

        t0 = time.perf_counter()
        comm.all_reduce_coalesced(tensors)
        torch.cuda.synchronize()
        result.latency_us = (time.perf_counter() - t0) * 1e6

        # Each tensor should be sum of all ranks' values
        for i, t in enumerate(tensors):
            val = sum(r + 1 + i * 0.1 for r in range(world_size))
            expected = torch.full((256,), val, dtype=torch.float16,
                                  device=f"cuda:{rank}")
            check_close(t, expected, f"coalesced_tensor_{i}", atol=0.1)

        result.passed = True
    except Exception as e:
        result.error = str(e)
    return result


def test_bf16_allreduce(rank, world_size, comm):
    """Test BF16 allreduce (critical for BI-V100 CUDA 10.2 compatibility)."""
    result = TestResult("bf16_allreduce")
    try:
        size = 4096  # 8KB — fused path
        tensor = torch.full((size,), rank + 1, dtype=torch.bfloat16,
                            device=f"cuda:{rank}")
        expected_sum = sum(range(1, world_size + 1))

        t0 = time.perf_counter()
        comm.all_reduce(tensor)
        torch.cuda.synchronize()
        result.latency_us = (time.perf_counter() - t0) * 1e6

        expected = torch.full((size,), expected_sum, dtype=torch.bfloat16,
                              device=f"cuda:{rank}")
        check_close(tensor, expected, "bf16_allreduce", atol=0.5)
        result.passed = True
    except Exception as e:
        result.error = str(e)
    return result


def test_bf16_ring_allreduce(rank, world_size, comm):
    """Test BF16 ring allreduce (medium tensors)."""
    result = TestResult("bf16_ring_allreduce")
    try:
        size = 256 * 1024  # 512KB — ring path
        tensor = torch.full((size,), rank + 1, dtype=torch.bfloat16,
                            device=f"cuda:{rank}")
        expected_sum = sum(range(1, world_size + 1))

        t0 = time.perf_counter()
        comm.all_reduce(tensor)
        torch.cuda.synchronize()
        result.latency_us = (time.perf_counter() - t0) * 1e6

        expected = torch.full((size,), expected_sum, dtype=torch.bfloat16,
                              device=f"cuda:{rank}")
        check_close(tensor, expected, "bf16_ring_allreduce", atol=0.5)
        result.passed = True
    except Exception as e:
        result.error = str(e)
    return result


def test_reduce_scatter(rank, world_size, comm):
    """Test ReduceScatter primitive."""
    result = TestResult("reduce_scatter")
    try:
        input_size = 1024 * world_size  # divisible by P
        shard_size = 1024
        
        # Each rank fills input with (rank + 1)
        input_t = torch.full((input_size,), rank + 1, dtype=torch.float16,
                             device=f"cuda:{rank}")
        output_t = torch.zeros(shard_size, dtype=torch.float16,
                               device=f"cuda:{rank}")
        
        expected_sum = sum(range(1, world_size + 1))

        t0 = time.perf_counter()
        comm.reduce_scatter(output_t, input_t)
        torch.cuda.synchronize()
        result.latency_us = (time.perf_counter() - t0) * 1e6

        # Each rank's shard should be the sum
        expected = torch.full((shard_size,), expected_sum, dtype=torch.float16,
                              device=f"cuda:{rank}")
        check_close(output_t, expected, "reduce_scatter")
        result.passed = True
    except Exception as e:
        result.error = str(e)
    return result


def test_all_gather(rank, world_size, comm):
    """Test AllGather primitive."""
    result = TestResult("all_gather")
    try:
        shard_size = 1024
        total_size = shard_size * world_size
        
        # Each rank contributes its shard filled with (rank + 1)
        input_t = torch.full((shard_size,), rank + 1, dtype=torch.float16,
                             device=f"cuda:{rank}")
        output_t = torch.zeros(total_size, dtype=torch.float16,
                               device=f"cuda:{rank}")

        t0 = time.perf_counter()
        comm.all_gather(output_t, input_t)
        torch.cuda.synchronize()
        result.latency_us = (time.perf_counter() - t0) * 1e6

        # Output should be [1,1,...,2,2,...,3,3,...,4,4,...] for 4 ranks
        for r in range(world_size):
            shard = output_t[r * shard_size : (r + 1) * shard_size]
            expected = torch.full((shard_size,), r + 1, dtype=torch.float16,
                                  device=f"cuda:{rank}")
            check_close(shard, expected, f"all_gather_shard_{r}")

        result.passed = True
    except Exception as e:
        result.error = str(e)
    return result


# ── Bandwidth benchmark ──

def benchmark_paths(rank, world_size, comm, warmup=5, repeat=50):
    """Quick bandwidth benchmark across the three paths."""
    if rank != 0:
        # All ranks must participate but only rank 0 prints
        pass

    configs = [
        ("fused (2KB)",     1024,    torch.float16),
        ("fused (32KB)",    16384,   torch.float16),
        ("fused (256KB)",   131072,  torch.float16),
        ("ring (512KB)",    262144,  torch.float16),
        ("ring (2MB)",      1048576, torch.float16),
        ("ring (8MB)",      4194304, torch.float16),
        ("nccl (32MB)",    16777216, torch.float16),
    ]

    if rank == 0:
        print(f"\n  {'Path':<20} {'Size':>10} {'Latency':>10} {'BW':>10}")
        print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10}")

    for name, size, dtype in configs:
        tensor = torch.randn(size, dtype=dtype, device=f"cuda:{rank}")
        
        for _ in range(warmup):
            comm.all_reduce(tensor)
        torch.cuda.synchronize()
        dist.barrier()

        t0 = time.perf_counter()
        for _ in range(repeat):
            comm.all_reduce(tensor)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        avg_us = (elapsed / repeat) * 1e6
        nbytes = size * tensor.element_size()
        algo_bw = (nbytes * 2 * (world_size - 1) / world_size) / (elapsed / repeat) / 1e9

        if rank == 0:
            print(f"  {name:<20} {nbytes/1024:>8.0f}KB {avg_us:>8.1f}μs {algo_bw:>8.2f}GB/s")


# ── Main ──

def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"  infccl Multi-GPU Correctness Tests")
        print(f"  {world_size} GPUs, CUDA device: {torch.cuda.get_device_name(rank)}")
        print(f"{'='*60}")

    # Create infccl communicator
    from infccl.python.infccl import InfcclCommunicator
    group = dist.group.WORLD
    comm = InfcclCommunicator(group=group, device=rank)

    if rank == 0:
        stats_prefix = "custom_ar" if comm.custom_ar else "nccl_only"
        print(f"  Backend: {stats_prefix}")
        print()

    # Run tests
    results = []

    # Fused path tests
    results.append(test_fused_allreduce(rank, world_size, comm, torch.float16))
    results.append(test_fused_allreduce(rank, world_size, comm, torch.float32))
    
    # Ring path tests
    results.append(test_ring_allreduce(rank, world_size, comm, torch.float16))
    results.append(test_ring_allreduce(rank, world_size, comm, torch.float32))
    results.append(test_ring_allreduce_random(rank, world_size, comm))

    # BF16 tests
    results.append(test_bf16_allreduce(rank, world_size, comm))
    results.append(test_bf16_ring_allreduce(rank, world_size, comm))

    # NCCL fallback
    results.append(test_large_nccl_fallback(rank, world_size, comm))

    # Coalescing
    results.append(test_coalesced_allreduce(rank, world_size, comm))

    # ReduceScatter and AllGather
    results.append(test_reduce_scatter(rank, world_size, comm))
    results.append(test_all_gather(rank, world_size, comm))

    # Print results
    dist.barrier()
    if rank == 0:
        print("  Test Results:")
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        for r in results:
            print(r)
        print(f"\n  {passed}/{total} tests passed")

        if passed < total:
            print("\n  FAILED TESTS:")
            for r in results:
                if not r.passed:
                    print(f"    {r.name}: {r.error}")

    # Bandwidth benchmark
    dist.barrier()
    if rank == 0:
        print(f"\n  Bandwidth Benchmark:")
    benchmark_paths(rank, world_size, comm)

    # Print stats
    dist.barrier()
    if rank == 0:
        stats = comm.stats()
        print(f"\n  Stats: {stats}")
        print(f"{'='*60}\n")

    comm.destroy()
    dist.destroy_process_group()

    # Exit with error if any test failed
    if rank == 0 and passed < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
