#!/usr/bin/env python3
"""Minimal multi-GPU correctness test for infccl collectives."""
import os, sys, ctypes, traceback
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

LIB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "lib", "infccl_collectives.so")

def bind_lib():
    """Load and bind all ctypes signatures."""
    lib = ctypes.CDLL(LIB_PATH)
    lib.infccl_coll_init_p2p.restype = ctypes.c_int
    lib.infccl_coll_init_p2p.argtypes = [ctypes.c_int]
    lib.infccl_coll_comm_create.restype = ctypes.c_void_p
    lib.infccl_coll_comm_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int64]
    lib.infccl_coll_comm_get_ipc.restype = ctypes.c_int
    lib.infccl_coll_comm_get_ipc.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.infccl_coll_register_peer.restype = ctypes.c_int
    lib.infccl_coll_register_peer.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                               ctypes.c_void_p, ctypes.c_void_p]
    lib.infccl_coll_finalize.restype = ctypes.c_int
    lib.infccl_coll_finalize.argtypes = [ctypes.c_void_p]
    lib.infccl_coll_comm_destroy.restype = None
    lib.infccl_coll_comm_destroy.argtypes = [ctypes.c_void_p]
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
    return lib

def worker(rank, world_size, results):
    try:
        torch.cuda.set_device(rank)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29503'
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

        lib = bind_lib()

        # Init P2P (all ranks call it — idempotent)
        lib.infccl_coll_init_p2p(world_size)
        dist.barrier()

        # Create comm + exchange IPC handles
        comm = lib.infccl_coll_comm_create(rank, world_size, 16 * 1024 * 1024)
        assert comm, "comm_create failed"

        HS = 64
        buf_h = (ctypes.c_byte * HS)()
        bar_h = (ctypes.c_byte * HS)()
        lib.infccl_coll_comm_get_ipc(comm, ctypes.byref(buf_h), ctypes.byref(bar_h))

        my_handles = torch.ByteTensor(list(bytes(buf_h)) + list(bytes(bar_h)))
        all_handles = [torch.zeros_like(my_handles) for _ in range(world_size)]
        dist.all_gather(all_handles, my_handles)

        for r in range(world_size):
            if r == rank:
                continue
            d = all_handles[r].tolist()
            pb = (ctypes.c_byte * HS)(*d[:HS])
            mb = (ctypes.c_byte * HS)(*d[HS:2 * HS])
            lib.infccl_coll_register_peer(comm, r, ctypes.byref(pb), ctypes.byref(mb))
        lib.infccl_coll_finalize(comm)
        dist.barrier()

        sp = torch.cuda.current_stream(rank).cuda_stream
        N = 1024
        dev = f'cuda:{rank}'
        r = {}

        # ---- AllGather ----
        send = torch.full((N,), float(rank + 1), dtype=torch.float32, device=dev)
        recv = torch.zeros(N * world_size, dtype=torch.float32, device=dev)
        lib.infccl_allgather(comm, recv.data_ptr(), send.data_ptr(), N, 0, sp)
        torch.cuda.synchronize()
        ag_ok = all(
            torch.allclose(recv[i * N:(i + 1) * N],
                           torch.full((N,), float(i + 1), device=dev))
            for i in range(world_size))
        r['allgather'] = 'PASS' if ag_ok else 'FAIL'
        dist.barrier()

        # ---- Broadcast (root=0) ----
        if rank == 0:
            bcast_in = torch.full((N,), 42.0, dtype=torch.float32, device=dev)
        else:
            bcast_in = torch.zeros(N, dtype=torch.float32, device=dev)
        bcast_out = torch.zeros(N, dtype=torch.float32, device=dev)
        lib.infccl_broadcast(comm, bcast_out.data_ptr(), bcast_in.data_ptr(),
                             N, 0, 0, sp)
        torch.cuda.synchronize()
        r['broadcast'] = 'PASS' if torch.allclose(
            bcast_out, torch.full((N,), 42.0, device=dev)) else 'FAIL'
        dist.barrier()

        # ---- Reduce (root=0) ----
        red_in = torch.full((N,), float(rank + 1), dtype=torch.float32, device=dev)
        red_out = torch.zeros(N, dtype=torch.float32, device=dev)
        lib.infccl_reduce(comm, red_out.data_ptr(), red_in.data_ptr(),
                          N, 0, 0, sp)
        torch.cuda.synchronize()
        if rank == 0:
            expected = float(world_size * (world_size + 1) // 2)
            r['reduce'] = 'PASS' if torch.allclose(
                red_out, torch.full((N,), expected, device=dev)) else 'FAIL'
        else:
            r['reduce'] = 'PASS'
        dist.barrier()

        # ---- ReduceScatter ----
        total = N * world_size
        rs_in = torch.zeros(total, dtype=torch.float32, device=dev)
        for i in range(world_size):
            rs_in[i * N:(i + 1) * N] = float(rank + 1)
        rs_out = torch.zeros(N, dtype=torch.float32, device=dev)
        lib.infccl_reduce_scatter(comm, rs_out.data_ptr(), rs_in.data_ptr(),
                                  total, 0, sp)
        torch.cuda.synchronize()
        expected_rs = float(world_size * (world_size + 1) // 2)
        r['reduce_scatter'] = 'PASS' if torch.allclose(
            rs_out, torch.full((N,), expected_rs, device=dev)) else 'FAIL'

        lib.infccl_coll_comm_destroy(comm)
        dist.destroy_process_group()
        results[rank] = r
    except Exception as e:
        results[rank] = {'error': f"{e}\n{traceback.format_exc()}"}


def main():
    ws = min(torch.cuda.device_count(), 4)
    print(f"Testing with {ws} GPUs")

    manager = mp.Manager()
    results = manager.dict()

    try:
        mp.spawn(worker, args=(ws, results), nprocs=ws, join=True)
    except Exception as e:
        print(f"spawn failed: {e}")
        # Print any partial results
        for rank in range(ws):
            r = results.get(rank, None)
            if r:
                print(f"  Rank {rank}: {r}")
        return 1

    all_pass = True
    for rank in range(ws):
        r = results.get(rank, {'error': 'no result'})
        if 'error' in r:
            print(f"  Rank {rank}: ERROR — {r['error'][:300]}")
            all_pass = False
        else:
            for op, status in r.items():
                print(f"  Rank {rank} {op}: {status}")
                if status != 'PASS':
                    all_pass = False

    print(f"\nOVERALL: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    sys.exit(main())
