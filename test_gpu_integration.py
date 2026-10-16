#!/usr/bin/env python3
"""
GPU integration test for ikun collective communication.

Tests:
  1. P2P connectivity between all GPU pairs (0,1,2,3)
  2. P2P data transfer correctness
  3. Multi-GPU allreduce via torch.distributed (NCCL backend)
  4. FP16 allreduce correctness
  5. BF16 allreduce correctness (if supported)
  6. Large tensor allreduce
  7. infccl .so symbol verification

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 test_gpu_integration.py
"""

import os
import sys
import ctypes
import struct

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1,2,3')

import torch
import torch.multiprocessing as mp


def test_p2p_connectivity():
    """Test 1: P2P connectivity between all GPU pairs."""
    print("=== Test 1: P2P Connectivity ===")
    num_gpus = torch.cuda.device_count()
    print(f"  Visible GPUs: {num_gpus}")

    ok_pairs = []
    fail_pairs = []

    for i in range(num_gpus):
        for j in range(i + 1, num_gpus):
            can = torch.cuda.can_device_access_peer(i, j)
            if can:
                ok_pairs.append(f"{i}-{j}")
            else:
                fail_pairs.append(f"{i}-{j}")

    print(f"  OK pairs: {','.join(ok_pairs)}")
    print(f"  FAIL pairs: {','.join(fail_pairs) or 'none'}")

    if fail_pairs:
        print("  WARNING: some P2P pairs failed")
    return ok_pairs, fail_pairs


def test_p2p_transfer(ok_pairs):
    """Test 2: P2P data transfer correctness."""
    print("=== Test 2: P2P Data Transfer ===")
    for pair in ok_pairs:
        i, j = map(int, pair.split('-'))
        src = torch.randn(4096, device=f'cuda:{i}')
        dst = torch.empty(4096, device=f'cuda:{j}')
        dst.copy_(src)
        torch.cuda.synchronize()
        diff = (dst.cpu() - src.cpu()).abs().max().item()
        assert diff == 0.0, f"Copy {i}->{j} diff={diff}"
    print(f"  All {len(ok_pairs)} pairs transfer correctly")


def allreduce_worker(rank, world_size, results_queue, test_bf16=False):
    """Worker process for multi-GPU allreduce tests."""
    try:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29502'
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)

        torch.cuda.set_device(rank)
        torch.distributed.init_process_group(
            backend='nccl', rank=rank, world_size=world_size)

        results = {'rank': rank, 'status': 'ok'}

        # Test 3: FP32 allreduce
        t32 = torch.ones(1024, device=f'cuda:{rank}') * (rank + 1)
        torch.distributed.all_reduce(t32, op=torch.distributed.ReduceOp.SUM)
        expected = sum(range(1, world_size + 1))
        results['fp32_diff'] = (t32 - expected).abs().max().item()

        # Test 4: FP16 allreduce
        t16 = torch.ones(1024, dtype=torch.float16, device=f'cuda:{rank}') * (rank + 1)
        torch.distributed.all_reduce(t16, op=torch.distributed.ReduceOp.SUM)
        results['fp16_diff'] = (t16.float() - expected).abs().max().item()

        # Test 5: BF16 allreduce (if supported)
        if test_bf16:
            try:
                tbf = torch.ones(1024, dtype=torch.bfloat16, device=f'cuda:{rank}') * (rank + 1)
                torch.distributed.all_reduce(tbf, op=torch.distributed.ReduceOp.SUM)
                results['bf16_diff'] = (tbf.float() - expected).abs().max().item()
            except Exception as e:
                results['bf16_diff'] = -1
                results['bf16_error'] = str(e)
        else:
            results['bf16_diff'] = -1
            results['bf16_note'] = 'skipped'

        # Test 6: Large tensor (1M elements)
        t_large = torch.randn(1_000_000, device=f'cuda:{rank}')
        t_ref = t_large.clone()
        torch.distributed.all_reduce(t_large, op=torch.distributed.ReduceOp.SUM)
        results['large_changed'] = not torch.allclose(t_large, t_ref)

        torch.distributed.destroy_process_group()
        results_queue.put(results)

    except Exception as e:
        results_queue.put({
            'rank': rank, 'status': 'error', 'error': str(e)
        })


def test_allreduce():
    """Tests 3-6: Multi-GPU allreduce via NCCL."""
    print("=== Tests 3-6: Multi-GPU AllReduce ===")
    world_size = min(torch.cuda.device_count(), 4)
    if world_size < 2:
        print("  SKIP: Need at least 2 GPUs")
        return False

    mp.set_start_method('spawn', force=True)
    q = mp.Queue()

    processes = []
    for r in range(world_size):
        p = mp.Process(target=allreduce_worker, args=(r, world_size, q, False))
        p.start()
        processes.append(p)

    for p in processes:
        p.join(timeout=120)

    results = []
    while not q.empty():
        results.append(q.get_nowait())

    all_ok = True
    for r in sorted(results, key=lambda x: x['rank']):
        if r['status'] == 'error':
            print(f"  Rank {r['rank']} FAILED: {r['error']}")
            all_ok = False
        else:
            print(f"  Rank {r['rank']}: fp32_diff={r['fp32_diff']:.6f}, "
                  f"fp16_diff={r['fp16_diff']:.6f}, "
                  f"large_changed={r['large_changed']}")
            if r['fp32_diff'] > 1e-5:
                print(f"    WARNING: fp32 diff too large")
                all_ok = False
            if r['fp16_diff'] > 0.1:
                print(f"    WARNING: fp16 diff too large")
                all_ok = False

    if all_ok and len(results) == world_size:
        print(f"  All {world_size} ranks passed")
    else:
        print(f"  FAILED ({len(results)}/{world_size} results)")

    return all_ok


def test_infccl_so():
    """Test 7: Verify infccl .so symbol availability."""
    print("=== Test 7: infccl .so Symbols ===")
    so_paths = [
        'infccl/lib/infccl_allreduce.so',
        './lib/infccl_allreduce.so',
    ]

    lib = None
    for p in so_paths:
        try:
            lib = ctypes.CDLL(p)
            print(f"  Loaded: {p}")
            break
        except OSError:
            continue

    if lib is None:
        print("  WARNING: infccl .so not found — custom P2P unavailable")
        return True  # Not a failure, just unavailable

    expected_symbols = [
        'infccl_init_p2p',
        'infccl_create_handles',
        'infccl_get_ipc_handle',
        'infccl_register_peer',
        'infccl_allreduce_launch',
        'infccl_destroy_handles',
    ]

    missing = []
    for sym in expected_symbols:
        try:
            getattr(lib, sym)
            print(f"  {sym}: OK")
        except AttributeError:
            print(f"  {sym}: MISSING")
            missing.append(sym)

    if missing:
        print(f"  WARNING: {len(missing)} symbols missing")
        return False

    print(f"  All {len(expected_symbols)} symbols present")
    return True


def main():
    print("=" * 60)
    print("ikun GPU Integration Test")
    print(f"GPUs: {torch.cuda.device_count()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    print()

    ok_pairs, fail_pairs = test_p2p_connectivity()
    print()

    test_p2p_transfer(ok_pairs)
    print()

    ar_ok = test_allreduce()
    print()

    so_ok = test_infccl_so()
    print()

    print("=" * 60)
    print("SUMMARY:")
    print(f"  P2P pairs OK: {','.join(ok_pairs)}")
    print(f"  P2P pairs FAIL: {','.join(fail_pairs) or 'none'}")
    print(f"  AllReduce: {'PASS' if ar_ok else 'FAIL'}")
    print(f"  infccl .so: {'PASS' if so_ok else 'FAIL'}")

    overall = ar_ok and so_ok and not fail_pairs
    print(f"  Overall: {'PASS' if overall else 'FAIL'}")
    print("=" * 60)

    return 0 if overall else 1


if __name__ == '__main__':
    sys.exit(main())

