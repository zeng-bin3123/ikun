#!/usr/bin/env python3
"""
GPU P2P + Smoke Tests for infccl collectives.

Tests:
  1. P2P access between all GPU pairs
  2. Single-GPU kernel launch (compile verification)
  3. Multi-GPU correctness via multiprocessing

Run:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python infccl/tests/test_collectives_gpu.py
"""

import os
import sys
import ctypes
import json
import time
import traceback

import torch
import torch.multiprocessing as mp

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def test_p2p_pairs():
    """Test P2P access between all GPU pairs."""
    num_gpus = torch.cuda.device_count()
    print(f"\n[P2P TEST] Found {num_gpus} GPUs")

    ok_pairs = []
    fail_pairs = []

    for i in range(num_gpus):
        for j in range(num_gpus):
            if i == j:
                continue
            can = torch.cuda.can_device_access_peer(i, j)
            pair_str = f"{i}-{j}"
            if can:
                ok_pairs.append(pair_str)
            else:
                fail_pairs.append(pair_str)

    print(f"[P2P TEST] OK pairs: {','.join(ok_pairs) if ok_pairs else 'none'}")
    print(f"[P2P TEST] FAIL pairs: {','.join(fail_pairs) if fail_pairs else 'none'}")

    return ok_pairs, fail_pairs


def test_compile_and_load():
    """Test that the .so loads and exports the right symbols."""
    lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "lib", "infccl_collectives.so")
    if not os.path.exists(lib_path):
        print(f"[COMPILE TEST] {lib_path} not found, attempting build...")
        build_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "build_collectives.sh")
        ret = os.system(f"bash {build_script}")
        if ret != 0:
            return False, "build failed"
        if not os.path.exists(lib_path):
            return False, "build produced no .so"

    try:
        lib = ctypes.CDLL(lib_path)
        # Check all exported symbols
        required = [
            'infccl_coll_comm_create', 'infccl_coll_comm_destroy',
            'infccl_coll_comm_get_ipc', 'infccl_coll_register_peer',
            'infccl_coll_finalize', 'infccl_coll_init_p2p',
            'infccl_allgather', 'infccl_broadcast',
            'infccl_reduce', 'infccl_reduce_scatter',
        ]
        missing = []
        for sym in required:
            if not hasattr(lib, sym):
                missing.append(sym)
        if missing:
            return False, f"missing symbols: {missing}"
        print(f"[COMPILE TEST] All {len(required)} symbols found in {lib_path}")
        return True, "ok"
    except OSError as e:
        return False, str(e)


def _worker_multi_gpu(rank, world_size, results_dict):
    """Worker process for multi-GPU correctness test."""
    try:
        torch.cuda.set_device(rank)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29501'

        import torch.distributed as dist
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

        lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "lib", "infccl_collectives.so")
        lib = ctypes.CDLL(lib_path)

        # Bind API
        lib.infccl_coll_init_p2p.restype = ctypes.c_int
        lib.infccl_coll_init_p2p.argtypes = [ctypes.c_int]
        lib.infccl_coll_comm_create.restype = ctypes.c_void_p
        lib.infccl_coll_comm_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int64]
        lib.infccl_coll_comm_get_ipc.restype = ctypes.c_int
        lib.infccl_coll_register_peer.restype = ctypes.c_int
        lib.infccl_coll_finalize.restype = ctypes.c_int
        lib.infccl_allgather.restype = ctypes.c_int
        lib.infccl_allgather.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int, ctypes.c_void_p]
        lib.infccl_broadcast.restype = ctypes.c_int
        lib.infccl_broadcast.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        lib.infccl_reduce.restype = ctypes.c_int
        lib.infccl_reduce.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        lib.infccl_reduce_scatter.restype = ctypes.c_int
        lib.infccl_reduce_scatter.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int, ctypes.c_void_p]
        lib.infccl_coll_comm_destroy.restype = None
        lib.infccl_coll_comm_destroy.argtypes = [ctypes.c_void_p]

        # Init P2P
        if rank == 0:
            lib.infccl_coll_init_p2p(world_size)
        dist.barrier()

        # Create comm
        max_size = 16 * 1024 * 1024
        comm = lib.infccl_coll_comm_create(rank, world_size, max_size)
        assert comm, "comm_create returned NULL"

        # Exchange IPC handles
        handle_size = 64
        buf_h = (ctypes.c_byte * handle_size)()
        bar_h = (ctypes.c_byte * handle_size)()
        lib.infccl_coll_comm_get_ipc(comm, ctypes.byref(buf_h), ctypes.byref(bar_h))

        my_handles = torch.ByteTensor(list(bytes(buf_h)) + list(bytes(bar_h)))
        all_handles = [torch.zeros_like(my_handles) for _ in range(world_size)]
        dist.all_gather(all_handles, my_handles)

        for r in range(world_size):
            if r == rank:
                continue
            data = all_handles[r].tolist()
            peer_buf = (ctypes.c_byte * handle_size)(*data[:64])
            peer_bar = (ctypes.c_byte * handle_size)(*data[64:128])
            lib.infccl_coll_register_peer(comm, r,
                                          ctypes.byref(peer_buf), ctypes.byref(peer_bar))

        lib.infccl_coll_finalize(comm)
        dist.barrier()

        stream_ptr = torch.cuda.current_stream(rank).cuda_stream
        results = {}
        N = 1024  # elements per test

        # ---- Test AllGather ----
        send_buf = torch.full((N,), float(rank + 1), dtype=torch.float32, device=f'cuda:{rank}')
        recv_buf = torch.zeros(N * world_size, dtype=torch.float32, device=f'cuda:{rank}')

        lib.infccl_allgather(comm,
                             ctypes.c_void_p(recv_buf.data_ptr()),
                             ctypes.c_void_p(send_buf.data_ptr()),
                             ctypes.c_int64(N), ctypes.c_int(0),
                             ctypes.c_void_p(stream_ptr))
        torch.cuda.synchronize(rank)

        ag_ok = True
        for r in range(world_size):
            chunk = recv_buf[r * N:(r + 1) * N]
            expected = float(r + 1)
            if not torch.allclose(chunk, torch.full_like(chunk, expected)):
                ag_ok = False
                break
        results['allgather'] = 'pass' if ag_ok else 'fail'

        # ---- Test Broadcast ----
        dist.barrier()
        root = 0
        if rank == root:
            bcast_buf = torch.full((N,), 42.0, dtype=torch.float32, device=f'cuda:{rank}')
        else:
            bcast_buf = torch.zeros(N, dtype=torch.float32, device=f'cuda:{rank}')
        bcast_out = torch.zeros(N, dtype=torch.float32, device=f'cuda:{rank}')

        lib.infccl_broadcast(comm,
                             ctypes.c_void_p(bcast_out.data_ptr()),
                             ctypes.c_void_p(bcast_buf.data_ptr()),
                             ctypes.c_int64(N), ctypes.c_int(0), ctypes.c_int(root),
                             ctypes.c_void_p(stream_ptr))
        torch.cuda.synchronize(rank)

        bcast_ok = torch.allclose(bcast_out, torch.full_like(bcast_out, 42.0))
        results['broadcast'] = 'pass' if bcast_ok else 'fail'

        # ---- Test Reduce ----
        dist.barrier()
        reduce_in = torch.full((N,), float(rank + 1), dtype=torch.float32, device=f'cuda:{rank}')
        reduce_out = torch.zeros(N, dtype=torch.float32, device=f'cuda:{rank}')

        lib.infccl_reduce(comm,
                          ctypes.c_void_p(reduce_out.data_ptr()),
                          ctypes.c_void_p(reduce_in.data_ptr()),
                          ctypes.c_int64(N), ctypes.c_int(0), ctypes.c_int(0),
                          ctypes.c_void_p(stream_ptr))
        torch.cuda.synchronize(rank)

        if rank == 0:
            expected_sum = sum(range(1, world_size + 1))
            reduce_ok = torch.allclose(reduce_out,
                                       torch.full_like(reduce_out, float(expected_sum)))
        else:
            reduce_ok = True  # only root has valid output
        results['reduce'] = 'pass' if reduce_ok else 'fail'

        # ---- Test ReduceScatter ----
        dist.barrier()
        total = N * world_size
        rs_in = torch.zeros(total, dtype=torch.float32, device=f'cuda:{rank}')
        # Each rank fills its section with (rank+1)
        for r in range(world_size):
            rs_in[r * N:(r + 1) * N] = float(rank + 1)
        rs_out = torch.zeros(N, dtype=torch.float32, device=f'cuda:{rank}')

        lib.infccl_reduce_scatter(comm,
                                   ctypes.c_void_p(rs_out.data_ptr()),
                                   ctypes.c_void_p(rs_in.data_ptr()),
                                   ctypes.c_int64(total), ctypes.c_int(0),
                                   ctypes.c_void_p(stream_ptr))
        torch.cuda.synchronize(rank)

        # Each rank's chunk should be sum of all ranks' contribution = ws*(ws+1)/2
        # Actually each rank put (rank+1) in ALL chunks, so each chunk sums to Σ(rank+1) = ws*(ws+1)/2
        expected_rs = float(world_size * (world_size + 1) // 2)
        rs_ok = torch.allclose(rs_out, torch.full_like(rs_out, expected_rs))
        results['reduce_scatter'] = 'pass' if rs_ok else 'fail'

        # Cleanup
        lib.infccl_coll_comm_destroy(comm)
        dist.destroy_process_group()

        results_dict[rank] = results

    except Exception as e:
        results_dict[rank] = {'error': f"{e}\n{traceback.format_exc()}"}


def test_multi_gpu_correctness(world_size: int):
    """Run multi-GPU correctness test."""
    print(f"\n[MULTI-GPU TEST] Testing with {world_size} GPUs")

    manager = mp.Manager()
    results_dict = manager.dict()

    try:
        mp.spawn(_worker_multi_gpu, args=(world_size, results_dict),
                 nprocs=world_size, join=True)
    except Exception as e:
        print(f"[MULTI-GPU TEST] spawn failed: {e}")
        return False, str(e)

    all_pass = True
    for rank in range(world_size):
        r = results_dict.get(rank, {'error': 'no result'})
        if 'error' in r:
            print(f"  Rank {rank}: ERROR - {r['error'][:200]}")
            all_pass = False
        else:
            for op, status in r.items():
                tag = "PASS" if status == "pass" else "FAIL"
                print(f"  Rank {rank} {op}: {tag}")
                if status != "pass":
                    all_pass = False

    return all_pass, "ok" if all_pass else "correctness failure"


def main():
    print("=" * 60)
    print("infccl Collectives GPU Test Suite")
    print("=" * 60)

    num_gpus = torch.cuda.device_count()
    print(f"CUDA devices visible: {num_gpus}")

    # 1. P2P test
    ok_pairs, fail_pairs = test_p2p_pairs()

    # 2. Compile test
    compile_ok, compile_err = test_compile_and_load()
    print(f"\n[COMPILE] {'PASS' if compile_ok else 'FAIL'}: {compile_err}")

    if not compile_ok:
        print(json.dumps({
            'compile': 'fail',
            'compile_error': compile_err,
            'gpu_pairs_ok': ','.join(ok_pairs),
            'gpu_pairs_fail': ','.join(fail_pairs),
            'gpu_test': 'skip',
            'smoke': 'fail',
            'smoke_error': 'compile failed',
        }, indent=2))
        return

    # 3. Multi-GPU test (use smallest viable group from P2P pairs)
    # Find connected GPUs
    if num_gpus >= 2 and len(ok_pairs) >= 2:
        # Test with 2 GPUs first
        test_ws = min(num_gpus, 4)  # up to 4 GPUs
        if test_ws > 1:
            gpu_ok, gpu_err = test_multi_gpu_correctness(test_ws)
        else:
            gpu_ok, gpu_err = False, "need at least 2 GPUs"
    else:
        gpu_ok = False
        gpu_err = "insufficient P2P pairs"

    smoke = compile_ok and gpu_ok

    print(f"\n{'=' * 60}")
    print(f"RESULTS:")
    print(f"  compile:     {'PASS' if compile_ok else 'FAIL'}")
    print(f"  gpu_test:    {'PASS' if gpu_ok else 'FAIL'}")
    print(f"  smoke:       {'PASS' if smoke else 'FAIL'}")
    print(f"  gpu_ok:      {','.join(ok_pairs)}")
    print(f"  gpu_fail:    {','.join(fail_pairs)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()

