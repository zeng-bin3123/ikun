#!/usr/bin/env python3
"""
Standalone build verification + P2P test for infccl collectives.
Runs on the local machine with torch + CoreX SDK.
"""
import os
import sys
import ctypes
import json

def main():
    results = {}

    # 1. Check for .so files
    lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")
    ar_so = os.path.join(lib_dir, "infccl_allreduce.so")
    coll_so = os.path.join(lib_dir, "infccl_collectives.so")

    results['ar_so_exists'] = os.path.isfile(ar_so)
    results['coll_so_exists'] = os.path.isfile(coll_so)

    if not results['coll_so_exists']:
        # Try building
        build_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "build_collectives.sh")
        ret = os.system(f"bash {build_script}")
        results['build_exit'] = ret
        results['coll_so_exists'] = os.path.isfile(coll_so)

    # 2. Load and verify symbols
    if results['coll_so_exists']:
        try:
            lib = ctypes.CDLL(coll_so)
            required_syms = [
                'infccl_coll_comm_create', 'infccl_coll_comm_destroy',
                'infccl_coll_init_p2p', 'infccl_coll_finalize',
                'infccl_allgather', 'infccl_broadcast',
                'infccl_reduce', 'infccl_reduce_scatter',
            ]
            missing = [s for s in required_syms if not hasattr(lib, s)]
            results['symbols_ok'] = len(missing) == 0
            results['missing_symbols'] = missing
            results['compile'] = 'pass'
        except OSError as e:
            results['compile'] = 'fail'
            results['compile_error'] = str(e)
    else:
        results['compile'] = 'fail'
        results['compile_error'] = 'no .so file'

    # 3. P2P test
    try:
        import torch
        num_gpus = torch.cuda.device_count()
        results['num_gpus'] = num_gpus
        ok_pairs = []
        fail_pairs = []
        for i in range(num_gpus):
            for j in range(num_gpus):
                if i == j:
                    continue
                can = torch.cuda.can_device_access_peer(i, j)
                pair = f"{i}-{j}"
                (ok_pairs if can else fail_pairs).append(pair)
        results['gpu_pairs_ok'] = ','.join(ok_pairs)
        results['gpu_pairs_fail'] = ','.join(fail_pairs)
        results['gpu_test'] = 'pass' if len(fail_pairs) == 0 else 'partial'
    except ImportError:
        results['gpu_test'] = 'skip'
        results['gpu_pairs_ok'] = ''
        results['gpu_pairs_fail'] = ''

    # 4. Smoke test: load .so on GPU
    if results.get('compile') == 'pass':
        try:
            import torch
            lib = ctypes.CDLL(coll_so)
            lib.infccl_coll_init_p2p.restype = ctypes.c_int
            lib.infccl_coll_init_p2p.argtypes = [ctypes.c_int]
            rc = lib.infccl_coll_init_p2p(min(num_gpus, 4))
            results['p2p_init'] = 'pass' if rc == 0 else f'fail (rc={rc})'

            # Create a comm on GPU 0
            lib.infccl_coll_comm_create.restype = ctypes.c_void_p
            lib.infccl_coll_comm_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int64]
            comm = lib.infccl_coll_comm_create(0, min(num_gpus, 4), 16 * 1024 * 1024)
            results['comm_create'] = 'pass' if comm else 'fail'

            if comm:
                lib.infccl_coll_comm_destroy.restype = None
                lib.infccl_coll_comm_destroy.argtypes = [ctypes.c_void_p]
                lib.infccl_coll_comm_destroy(comm)
                results['comm_destroy'] = 'pass'

            results['smoke'] = 'pass'
        except Exception as e:
            results['smoke'] = 'fail'
            results['smoke_error'] = str(e)
    else:
        results['smoke'] = 'fail'
        results['smoke_error'] = 'compile failed'

    # Output
    print(json.dumps(results, indent=2))

    # Also output status block format
    print("\n--- STATUS BLOCK ---")
    print(f"compile={results.get('compile', 'fail')}")
    print(f"compile_error={results.get('compile_error', '')}")
    print(f"needs_source=false")
    print(f"gpu_pairs_ok={results.get('gpu_pairs_ok', '')}")
    print(f"gpu_pairs_fail={results.get('gpu_pairs_fail', '')}")
    print(f"gpu_test={results.get('gpu_test', 'skip')}")
    print(f"smoke={results.get('smoke', 'fail')}")
    print(f"smoke_error={results.get('smoke_error', '')}")

if __name__ == "__main__":
    main()
