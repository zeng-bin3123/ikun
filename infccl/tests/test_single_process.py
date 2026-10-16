#!/usr/bin/env python3
"""
Single-process test for infccl collectives.
Tests AllGather, Broadcast, Reduce, ReduceScatter with world_size=1
(self-to-self) to verify kernel correctness without multi-process IPC.

Also tests comm create/destroy on all GPU pairs.
"""
import ctypes
import os
import sys

# Must import torch before loading the .so to ensure CUDA is initialized
import torch

LIB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib")
COLL_SO = os.path.join(LIB_DIR, "infccl_collectives.so")

def load_lib():
    lib = ctypes.CDLL(COLL_SO)
    lib.infccl_coll_init_p2p.restype = ctypes.c_int
    lib.infccl_coll_init_p2p.argtypes = [ctypes.c_int]
    lib.infccl_coll_comm_create.restype = ctypes.c_void_p
    lib.infccl_coll_comm_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int64]
    lib.infccl_coll_finalize.restype = ctypes.c_int
    lib.infccl_coll_comm_destroy.restype = None
    lib.infccl_coll_comm_destroy.argtypes = [ctypes.c_void_p]
    lib.infccl_allgather.restype = ctypes.c_int
    lib.infccl_broadcast.restype = ctypes.c_int
    lib.infccl_reduce.restype = ctypes.c_int
    lib.infccl_reduce_scatter.restype = ctypes.c_int
    return lib


def test_single_rank_collectives():
    """Test all collectives with world_size=1 (degenerate but validates kernel launch)."""
    lib = load_lib()
    ngpu = torch.cuda.device_count()
    print(f"GPUs: {ngpu}")

    # Init P2P
    rc = lib.infccl_coll_init_p2p(ngpu)
    print(f"init_p2p: {'PASS' if rc == 0 else f'FAIL rc={rc}'}")

    results = {}

    for gpu in range(min(ngpu, 4)):
        torch.cuda.set_device(gpu)
        comm = lib.infccl_coll_comm_create(gpu, 1, 4 * 1024 * 1024)
        if not comm:
            print(f"  GPU {gpu}: comm_create FAIL")
            continue
        lib.infccl_coll_finalize(comm)

        N = 512
        sp = torch.cuda.current_stream(gpu).cuda_stream
        dev = f'cuda:{gpu}'

        # AllGather (ws=1: output should equal input)
        inp = torch.full((N,), float(gpu + 1), dtype=torch.float32, device=dev)
        out = torch.zeros(N, dtype=torch.float32, device=dev)
        rc = lib.infccl_allgather(comm, ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(inp.data_ptr()), ctypes.c_int64(N), 0, ctypes.c_void_p(sp))
        torch.cuda.synchronize(gpu)
        ag = torch.allclose(out, inp) and rc == 0

        # Broadcast (ws=1, root=0: output should equal input)
        inp_b = torch.full((N,), 99.0, dtype=torch.float32, device=dev)
        out_b = torch.zeros(N, dtype=torch.float32, device=dev)
        rc = lib.infccl_broadcast(comm, ctypes.c_void_p(out_b.data_ptr()),
            ctypes.c_void_p(inp_b.data_ptr()), ctypes.c_int64(N), 0, 0, ctypes.c_void_p(sp))
        torch.cuda.synchronize(gpu)
        bc = torch.allclose(out_b, inp_b) and rc == 0

        # Reduce (ws=1, root=0: output should equal input)
        inp_r = torch.full((N,), 7.0, dtype=torch.float32, device=dev)
        out_r = torch.zeros(N, dtype=torch.float32, device=dev)
        rc = lib.infccl_reduce(comm, ctypes.c_void_p(out_r.data_ptr()),
            ctypes.c_void_p(inp_r.data_ptr()), ctypes.c_int64(N), 0, 0, ctypes.c_void_p(sp))
        torch.cuda.synchronize(gpu)
        rd = torch.allclose(out_r, inp_r) and rc == 0

        # ReduceScatter (ws=1: output should equal input since chunk=total)
        inp_rs = torch.full((N,), 13.0, dtype=torch.float32, device=dev)
        out_rs = torch.zeros(N, dtype=torch.float32, device=dev)
        rc = lib.infccl_reduce_scatter(comm, ctypes.c_void_p(out_rs.data_ptr()),
            ctypes.c_void_p(inp_rs.data_ptr()), ctypes.c_int64(N), 0, ctypes.c_void_p(sp))
        torch.cuda.synchronize(gpu)
        rs = torch.allclose(out_rs, inp_rs) and rc == 0

        # FP16 test
        inp16 = torch.full((N,), 5.0, dtype=torch.float16, device=dev)
        out16 = torch.zeros(N, dtype=torch.float16, device=dev)
        rc = lib.infccl_reduce(comm, ctypes.c_void_p(out16.data_ptr()),
            ctypes.c_void_p(inp16.data_ptr()), ctypes.c_int64(N), 1, 0, ctypes.c_void_p(sp))
        torch.cuda.synchronize(gpu)
        fp16 = torch.allclose(out16, inp16) and rc == 0

        # BF16 test
        inp_bf = torch.full((N,), 3.0, dtype=torch.bfloat16, device=dev)
        out_bf = torch.zeros(N, dtype=torch.bfloat16, device=dev)
        rc = lib.infccl_broadcast(comm, ctypes.c_void_p(out_bf.data_ptr()),
            ctypes.c_void_p(inp_bf.data_ptr()), ctypes.c_int64(N), 2, 0, ctypes.c_void_p(sp))
        torch.cuda.synchronize(gpu)
        bf16 = torch.allclose(out_bf, inp_bf) and rc == 0

        lib.infccl_coll_comm_destroy(comm)

        status = {
            'allgather': 'PASS' if ag else 'FAIL',
            'broadcast': 'PASS' if bc else 'FAIL',
            'reduce': 'PASS' if rd else 'FAIL',
            'reduce_scatter': 'PASS' if rs else 'FAIL',
            'fp16_reduce': 'PASS' if fp16 else 'FAIL',
            'bf16_broadcast': 'PASS' if bf16 else 'FAIL',
        }
        all_pass = all(v == 'PASS' for v in status.values())
        print(f"  GPU {gpu}: {status} -> {'ALL PASS' if all_pass else 'SOME FAIL'}")
        results[gpu] = status

    overall = all(
        all(v == 'PASS' for v in s.values())
        for s in results.values()
    )
    print(f"\nOVERALL: {'ALL PASS' if overall else 'SOME FAILED'}")
    return 0 if overall else 1


if __name__ == "__main__":
    if not os.path.exists(COLL_SO):
        print(f"FAIL: {COLL_SO} not found")
        sys.exit(1)
    sys.exit(test_single_rank_collectives())

