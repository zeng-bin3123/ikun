#!/usr/bin/env python3
"""Minimal multi-GPU correctness test for infccl collectives."""
import os, sys, ctypes, traceback
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

LIB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "lib", "infccl_collectives.so")

def worker(rank, world_size, results):
    try:
        torch.cuda.set_device(rank)
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29502'
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

        lib = ctypes.CDLL(LIB_PATH)
        # Bind types
        lib.infccl_coll_init_p2p.restype = ctypes.c_int
        lib.infccl_coll_comm_create.restype = ctypes.c_void_p
        lib.infccl_coll_comm_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int64]
        lib.infccl_coll_comm_get_ipc.restype = ctypes.c_int
        lib.infccl_coll_register_peer.restype = ctypes.c_int
        lib.infccl_coll_finalize.restype = ctypes.c_int
        for fn in ['infccl_allgather','infccl_broadcast','infccl_reduce','infccl_reduce_scatter']:
            getattr(lib, fn).restype = ctypes.c_int
        lib.infccl_coll_comm_destroy.restype = None
        lib.infccl_coll_comm_destroy.argtypes = [ctypes.c_void_p]

        if rank == 0:
            lib.infccl_coll_init_p2p(world_size)
        dist.barrier()

        # Create comm + exchange IPC
        comm = lib.infccl_coll_comm_create(rank, world_size, 16*1024*1024)
        assert comm, "comm_create failed"

        HS = 64
        buf_h = (ctypes.c_byte * HS)()
        bar_h = (ctypes.c_byte * HS)()
        lib.infccl_coll_comm_get_ipc(comm, ctypes.byref(buf_h), ctypes.byref(bar_h))

        my_handles = torch.ByteTensor(list(bytes(buf_h)) + list(bytes(bar_h)))
        all_handles = [torch.zeros_like(my_handles) for _ in range(world_size)]
        dist.all_gather(all_handles, my_handles)

        for r in range(world_size):
            if r == rank: continue
            d = all_handles[r].tolist()
            pb = (ctypes.c_byte * HS)(*d[:HS])
            mb = (ctypes.c_byte * HS)(*d[HS:2*HS])
            lib.infccl_coll_register_peer(comm, r, ctypes.byref(pb), ctypes.byref(mb))
        lib.infccl_coll_finalize(comm)
        dist.barrier()

        sp = torch.cuda.current_stream(rank).cuda_stream
        N = 1024
        r = {}

        # AllGather
        send = torch.full((N,), float(rank+1), dtype=torch.float32, device=f'cuda:{rank}')
        recv = torch.zeros(N*world_size, dtype=torch.float32, device=f'cuda:{rank}')
        lib.infccl_allgather(comm, ctypes.c_void_p(recv.data_ptr()),
            ctypes.c_void_p(send.data_ptr()), ctypes.c_int64(N), 0, ctypes.c_void_p(sp))
        torch.cuda.synchronize()
        ag_ok = all(torch.allclose(recv[i*N:(i+1)*N], 
                    torch.full((N,), float(i+1), device=f'cuda:{rank}')) for i in range(world_size))
        r['allgather'] = 'PASS' if ag_ok else 'FAIL'

        dist.barrier()

        # Broadcast (root=0)
        bcast = torch.full((N,), 42.0, dtype=torch.float32, device=f'cuda:{rank}') if rank==0 \
                else torch.zeros(N, dtype=torch.float32, device=f'cuda:{rank}')
        bout = torch.zeros(N, dtype=torch.float32, device=f'cuda:{rank}')
        lib.infccl_broadcast(comm, ctypes.c_void_p(bout.data_ptr()),
            ctypes.c_void_p(bcast.data_ptr()), ctypes.c_int64(N), 0, 0, ctypes.c_void_p(sp))
        torch.cuda.synchronize()
        r['broadcast'] = 'PASS' if torch.allclose(bout, torch.full((N,), 42.0, device=f'cuda:{rank}')) else 'FAIL'

        dist.barrier()

        # Reduce (root=0)
        rin = torch.full((N,), float(rank+1), dtype=torch.float32, device=f'cuda:{rank}')
        rout = torch.zeros(N, dtype=torch.float32, device=f'cuda:{rank}')
        lib.infccl_reduce(comm, ctypes.c_void_p(rout.data_ptr()),
            ctypes.c_void_p(rin.data_ptr()), ctypes.c_int64(N), 0, 0, ctypes.c_void_p(sp))
        torch.cuda.synchronize()
        if rank == 0:
            expected = float(world_size*(world_size+1)//2)
            r['reduce'] = 'PASS' if torch.allclose(rout, torch.full((N,), expected, device='cuda:0')) else 'FAIL'
        else:
            r['reduce'] = 'PASS'  # only root has valid output

        dist.barrier()

        # ReduceScatter
        total = N * world_size
        rsin = torch.zeros(total, dtype=torch.float32, device=f'cuda:{rank}')
        for i in range(world_size):
            rsin[i*N:(i+1)*N] = float(rank+1)
        rsout = torch.zeros(N, dtype=torch.float32, device=f'cuda:{rank}')
        lib.infccl_reduce_scatter(comm, ctypes.c_void_p(rsout.data_ptr()),
            ctypes.c_void_p(rsin.data_ptr()), ctypes.c_int64(total), 0, ctypes.c_void_p(sp))
        torch.cuda.synchronize()
        expected_rs = float(world_size*(world_size+1)//2)
        r['reduce_scatter'] = 'PASS' if torch.allclose(rsout, 
            torch.full((N,), expected_rs, device=f'cuda:{rank}')) else 'FAIL'

        lib.infccl_coll_comm_destroy(comm)
        dist.destroy_process_group()
        results[rank] = r
    except Exception as e:
        results[rank] = {'error': str(e)}

def main():
    ws = min(torch.cuda.device_count(), 4)
    print(f"Testing with {ws} GPUs")
    
    manager = mp.Manager()
    results = manager.dict()
    mp.spawn(worker, args=(ws, results), nprocs=ws, join=True)

    all_pass = True
    for rank in range(ws):
        r = results.get(rank, {'error': 'no result'})
        if 'error' in r:
            print(f"  Rank {rank}: ERROR — {r['error'][:200]}")
            all_pass = False
        else:
            for op, status in r.items():
                print(f"  Rank {rank} {op}: {status}")
                if status != 'PASS': all_pass = False

    print(f"\nOVERALL: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return 0 if all_pass else 1

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    sys.exit(main())
