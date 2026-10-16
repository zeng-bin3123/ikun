#!/usr/bin/env python3
"""
Benchmark: infccl custom allreduce vs NCCL on BI-V100.

Run with torchrun:
  torchrun --nproc_per_node=4 infccl/bench_allreduce.py

Measures latency for various tensor sizes across three paths:
  Path 0 (Fused):  ≤ 256KB — all-to-all copy → barrier → local reduce
  Path 1 (Ring):   256KB~8MB — reduce-scatter + allgather in P-1 steps
  Path 2 (NCCL):   > 8MB — system NCCL fallback

Expected results on 4× BI-V100 PCIe:
  - Fused wins for tiny tensors (α-dominated, ~10-15 μs)
  - Ring wins for medium tensors (bandwidth-optimal, 2*(P-1)/P wire traffic)
  - NCCL wins for large tensors (> 8MB, ring/tree with kernel overlap)
"""

import os
import sys
import time
import torch
import torch.distributed as dist

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def benchmark_nccl_allreduce(rank, world_size, sizes, warmup=10, repeat=100):
    """Benchmark NCCL allreduce latency."""
    results = {}
    
    for size in sizes:
        tensor = torch.randn(size, dtype=torch.float16, device=f"cuda:{rank}")
        
        # Warmup
        for _ in range(warmup):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(repeat):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        avg_us = (elapsed / repeat) * 1e6
        nbytes = size * 2  # fp16
        bandwidth_gbps = (nbytes * 2 * (world_size - 1) / world_size) / (elapsed / repeat) / 1e9
        
        results[size] = {
            "avg_us": avg_us,
            "bandwidth_gbps": bandwidth_gbps,
            "nbytes": nbytes,
        }
        
        if rank == 0:
            print(f"  NCCL     size={size:>10,}  ({nbytes/1024:>8.1f} KB)  "
                  f"avg={avg_us:>8.1f} μs  bw={bandwidth_gbps:>6.2f} GB/s")
    
    return results


def benchmark_torch_allreduce_coalesced(rank, world_size, sizes, warmup=10, repeat=100):
    """Benchmark coalesced allreduce (pack + NCCL + unpack)."""
    results = {}
    
    for num_tensors in [4, 8, 16]:
        per_tensor_size = 1024  # 1K elements each
        tensors = [torch.randn(per_tensor_size, dtype=torch.float16, 
                               device=f"cuda:{rank}") for _ in range(num_tensors)]
        total_size = num_tensors * per_tensor_size
        
        # Warmup
        for _ in range(warmup):
            # Naive: individual allreduces
            for t in tensors:
                dist.all_reduce(t)
        torch.cuda.synchronize()
        
        # Benchmark naive
        start = time.perf_counter()
        for _ in range(repeat):
            for t in tensors:
                dist.all_reduce(t)
        torch.cuda.synchronize()
        naive_elapsed = time.perf_counter() - start
        naive_us = (naive_elapsed / repeat) * 1e6
        
        # Benchmark coalesced
        coalesced_buf = torch.empty(total_size, dtype=torch.float16, 
                                     device=f"cuda:{rank}")
        
        for _ in range(warmup):
            offset = 0
            for t in tensors:
                coalesced_buf[offset:offset + per_tensor_size].copy_(t)
                offset += per_tensor_size
            dist.all_reduce(coalesced_buf)
            offset = 0
            for t in tensors:
                t.copy_(coalesced_buf[offset:offset + per_tensor_size])
                offset += per_tensor_size
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(repeat):
            offset = 0
            for t in tensors:
                coalesced_buf[offset:offset + per_tensor_size].copy_(t)
                offset += per_tensor_size
            dist.all_reduce(coalesced_buf)
            offset = 0
            for t in tensors:
                t.copy_(coalesced_buf[offset:offset + per_tensor_size])
                offset += per_tensor_size
        torch.cuda.synchronize()
        coalesced_elapsed = time.perf_counter() - start
        coalesced_us = (coalesced_elapsed / repeat) * 1e6
        
        speedup = naive_us / coalesced_us
        
        if rank == 0:
            print(f"  Coalesce n={num_tensors:>2} × {per_tensor_size} elems: "
                  f"naive={naive_us:>8.1f} μs  coalesced={coalesced_us:>8.1f} μs  "
                  f"speedup={speedup:.2f}x")
        
        results[num_tensors] = {
            "naive_us": naive_us,
            "coalesced_us": coalesced_us,
            "speedup": speedup,
        }
    
    return results


def main():
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    torch.cuda.set_device(rank)
    
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"infccl AllReduce Benchmark — {world_size} GPUs")
        print(f"{'='*70}")
    
    # Test sizes: from 256 elements to 16M elements (fp16)
    sizes = [
        256,       # 512 B
        1024,      # 2 KB
        4096,      # 8 KB
        16384,     # 32 KB
        65536,     # 128 KB
        262144,    # 512 KB
        1048576,   # 2 MB
        4194304,   # 8 MB
        16777216,  # 32 MB
    ]
    
    if rank == 0:
        print(f"\n--- NCCL AllReduce Latency ---")
    dist.barrier()
    nccl_results = benchmark_nccl_allreduce(rank, world_size, sizes)
    
    if rank == 0:
        print(f"\n--- Coalesced vs Naive AllReduce ---")
    dist.barrier()
    coal_results = benchmark_torch_allreduce_coalesced(rank, world_size, sizes)
    
    if rank == 0:
        print(f"\n--- Summary ---")
        print(f"Custom AR threshold recommendation:")
        
        # Find the crossover point where NCCL latency > ~100μs
        # (below that, P2P can win)
        for size, res in sorted(nccl_results.items()):
            marker = " ← P2P zone" if res["avg_us"] > 80 and res["nbytes"] < 8*1024*1024 else ""
            print(f"  {res['nbytes']/1024:>8.1f} KB: {res['avg_us']:>8.1f} μs{marker}")
        
        print(f"\nCoalescing benefit:")
        for n, res in sorted(coal_results.items()):
            print(f"  {n} tensors: {res['speedup']:.2f}x faster when coalesced")
        
        print(f"\n{'='*70}")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
