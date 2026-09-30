#!/usr/bin/env python3
"""all_reduce 延迟微基准 —— M0「先做尺子」(issue #1)

    torchrun --nproc_per_node=4 bench/latency_micro.py

- 5 档消息大小 4KB→1MB，每档 200 次计时调用，整轮扫描重跑 3 次
- stdout 恰好 6 行（仅 rank0 打印）：1 行汇总 + 5 行每档结果
- 单次计时：synchronize → t0 → all_reduce(in-place) → synchronize → t1
- 每档 p50 取所有 rank 里最大的那个（最慢的 rank 决定延迟）
- 方差 = 每档 (max(p50) - min(p50)) / median(p50)，取 5 档中最差的一档
  （用极差而不是标准差，更严格）
- 只有 CUDA 设备 + nccl 后端才写 bench/results/gate-NNN.json；
  gloo/CPU 只打印，从不写门禁文件，避免把非真实硬件的数字当实测提交
"""
import argparse
import json
import os
import socket
import statistics
import subprocess
import time
from datetime import datetime, timezone

SIZES = [4 << 10, 16 << 10, 64 << 10, 256 << 10, 1 << 20]  # 4KB 16KB 64KB 256KB 1MB
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------- 纯函数（不依赖 torch，便于单测） ----------

def size_label(nbytes):
    return f"{nbytes >> 20}MB" if nbytes >= 1 << 20 else f"{nbytes >> 10}KB"


def percentile(samples, q):
    s = sorted(samples)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def range_variance_pct(values):
    """(max - min) / median，百分比。"""
    med = statistics.median(values)
    return 0.0 if med == 0 else (max(values) - min(values)) / med * 100.0


def format_report(meta, p50_runs, p99_runs, threshold_pct=5.0):
    """p50_runs/p99_runs: [run][size] 微秒。返回要打印的行（恰好 1 + len(SIZES) 行）。"""
    n_sizes = len(p50_runs[0])
    per_size_var = [range_variance_pct([r[i] for r in p50_runs]) for i in range(n_sizes)]
    worst = max(per_size_var)
    verdict = "✓" if worst <= threshold_pct else "✗"
    head = (f"allreduce {meta['dtype']} · {meta['world']} ranks · {meta['backend']} · "
            f"{meta['iters']} iters × {len(p50_runs)} runs · "
            f"p50 var {worst:.1f}% (≤{threshold_pct:g}% {verdict}) · {meta['sink']}")
    lines = [head]
    for i, nbytes in enumerate(SIZES[:n_sizes]):
        p50s = [r[i] for r in p50_runs]
        runs = "/".join(f"{v:.1f}" for v in p50s)
        lines.append(f"{size_label(nbytes):>6}  p50 {statistics.median(p50s):8.1f}us  "
                     f"p99 {statistics.median([r[i] for r in p99_runs]):8.1f}us  "
                     f"runs {runs}  var {per_size_var[i]:.1f}%")
    return lines, worst


def git_state():
    def run(*args):
        return subprocess.run(["git", "-C", REPO, *args], capture_output=True,
                              text=True).stdout.strip()
    sha = run("rev-parse", "HEAD")
    dirty = [l for l in run("status", "--porcelain", "--untracked-files=no").splitlines()
             if "bench/results/" not in l]
    return sha, bool(dirty)


def build_result(sha, dirty, host, world, lines, worst, p50_runs):
    """门禁文件内容。metrics 键名必须和 issue gate-spec 一致。"""
    col = lambda i: [r[i] for r in p50_runs]
    return {
        "commit": sha,
        "host": host,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cmd": f"torchrun --nproc_per_node={world} bench/latency_micro.py",
        "metrics": {
            "latency_p50_variance_pct": round(worst, 2),
            "output_lines": len(lines),
            **{f"allreduce_{size_label(n).lower()}_us": round(statistics.median(col(i)), 2)
               for i, n in enumerate(SIZES)},
        },
        "runs": [round(v, 2) for v in col(0)],
        "runs_by_size": {size_label(n): [round(v, 2) for v in col(i)]
                         for i, n in enumerate(SIZES)},
        "dirty_tree": dirty,
        "doc": "bench/results/README.md",
    }


# ---------- 基准本体 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--reruns", type=int, default=3)
    ap.add_argument("--issue", type=int, default=1, help="写 bench/results/gate-NNN.json")
    ap.add_argument("--no-write", action="store_true", help="只打印，不写门禁文件")
    args = ap.parse_args()

    import torch
    import torch.distributed as dist

    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend=backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    try:
        if use_cuda:
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
            device, dtype = torch.device("cuda"), torch.float16
            sync = torch.cuda.synchronize
        else:
            device, dtype = torch.device("cpu"), torch.float32  # gloo 不支持 fp16 allreduce
            sync = lambda: None
        elem = torch.tensor([], dtype=dtype).element_size()

        # 正确性自检：ones 求和应等于 world
        chk = torch.ones(SIZES[0] // elem, dtype=dtype, device=device)
        dist.all_reduce(chk)
        sync()
        if not torch.all(chk == world):
            raise RuntimeError(f"all_reduce 结果错误: 期望 {world}, 得到 {chk[0].item()}")

        # 计时用全 0 张量，重复 in-place 求和不会溢出
        bufs = [torch.zeros(n // elem, dtype=dtype, device=device) for n in SIZES]
        p50 = torch.zeros(args.reruns, len(SIZES), dtype=torch.float32)  # fp32: 各家 ccl 都支持
        p99 = torch.zeros_like(p50)
        for r in range(args.reruns):
            for i, buf in enumerate(bufs):
                for _ in range(args.warmup):
                    dist.all_reduce(buf)
                sync()
                dist.barrier()
                samples = []
                for _ in range(args.iters):
                    sync()
                    t0 = time.perf_counter()
                    dist.all_reduce(buf)
                    sync()
                    samples.append((time.perf_counter() - t0) * 1e6)
                p50[r, i] = percentile(samples, 0.50)
                p99[r, i] = percentile(samples, 0.99)

        # 取最慢 rank
        stats = torch.stack([p50, p99]).to(device)
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)
        p50_runs, p99_runs = stats[0].cpu().tolist(), stats[1].cpu().tolist()

        if rank == 0:
            write = use_cuda and not args.no_write
            out = os.path.join(REPO, "bench", "results", f"gate-{args.issue:03d}.json")
            meta = dict(dtype=str(dtype).replace("torch.", ""), world=world,
                        backend=backend, iters=args.iters,
                        sink=f"→ {os.path.relpath(out, REPO)}" if write
                        else "dry-run, 未写门禁文件")
            lines, worst = format_report(meta, p50_runs, p99_runs)
            for line in lines:
                print(line, flush=True)
            if write:
                sha, dirty = git_state()
                host = f"{socket.gethostname()} ({world}x {torch.cuda.get_device_name(0)}, {backend})"
                result = build_result(sha, dirty, host, world, lines, worst, p50_runs)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with open(out, "w") as f:
                    json.dump(result, f, ensure_ascii=False, indent=1)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
