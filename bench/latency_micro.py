#!/usr/bin/env python3
r"""all_reduce 延迟微基准 —— M0「先做尺子」(issue #1)

    torchrun --nproc_per_node=4 bench/latency_micro.py

理论与推导见 docs/ALLREDUCE_LATENCY_MODEL.md。要点：
- 模型 T(S) = α + β·S（Hockney α-β）。decode 包 4KB ≪ S* = α/β ≈ 5.6MiB，
  所以本基准测的核心量是 α；β 由大包带宽基准定更准（本扫描里 β 相对误差 ~11%）
- 5 档 4KB→1MB，每档 200 次，整轮重跑 3 次；stdout 恰好 6 行（仅 rank0）
- 单次计时 synchronize→t0→all_reduce(in-place)→synchronize→t1；另测空 synchronize 的
  开销 ε_sync，拟合 α 前扣除
- 分位数取所有 rank 中最大的（最慢 rank 决定延迟）
- 方差 = 每档 (max−min)/median，取最差档。3 次独立正态重跑的极差 95% 分位是 3.314σ，
  故 P(方差≤5%) ≥ 95% ⇔ σ_rel ≤ 1.51%
- 只有 CUDA+nccl 才写 bench/results/gate-NNN.json；gloo/CPU 只打印
"""
import argparse
import json
import math
import os
import socket
import statistics
import subprocess
import time
from datetime import datetime, timezone

SIZES = [4 << 10, 16 << 10, 64 << 10, 256 << 10, 1 << 20]  # 4KB 16KB 64KB 256KB 1MB
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Z75 = 0.6744897501960817      # Φ⁻¹(0.75)
DOC = "docs/ALLREDUCE_LATENCY_MODEL.md"


# ---------- 纯函数（不依赖 torch，便于单测） ----------

def size_label(nbytes):
    return f"{nbytes >> 20}MB" if nbytes >= 1 << 20 else f"{nbytes >> 10}KB"


def percentile(samples, q):
    s = sorted(samples)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def range_variance_pct(values):
    """(max − min) / median，百分比。"""
    med = statistics.median(values)
    return 0.0 if med == 0 else (max(values) - min(values)) / med * 100.0


def fit_alpha_beta(sizes_bytes, t_us):
    """加权最小二乘 T = α + β·S，权重 w=1/T²（乘性噪声 ⇒ 最小化相对残差）。
    返回 (α [us], β [us/byte])。"""
    w = [1.0 / (t * t) for t in t_us]
    sw = sum(w)
    sx = sum(wi * x for wi, x in zip(w, sizes_bytes))
    sy = sum(wi * y for wi, y in zip(w, t_us))
    sxx = sum(wi * x * x for wi, x in zip(w, sizes_bytes))
    sxy = sum(wi * x * y for wi, x, y in zip(w, sizes_bytes, t_us))
    beta = (sw * sxy - sx * sy) / (sw * sxx - sx * sx)
    return (sy - beta * sx) / sw, beta


def jitter_s(p25, p75):
    """单次延迟按对数正态建模时 log 标准差的稳健估计：s = ln(p75/p25) / (2·z₀.₇₅)。"""
    return 0.0 if p25 <= 0 or p75 <= p25 else math.log(p75 / p25) / (2 * Z75)


def median_cv(s, n):
    """n 个样本中位数的渐近相对标准差：√(π/2)·s/√n。"""
    return math.sqrt(math.pi / 2) * s / math.sqrt(n)


def min_iters(s, sigma_rel):
    """使 median_cv(s, n) ≤ sigma_rel 的最小 n。"""
    return math.ceil((math.sqrt(math.pi / 2) * s / sigma_rel) ** 2) if sigma_rel > 0 else 0


def range_cdf(w, n=3, lo=-8.0, hi=8.0, steps=4000):
    """n 个 iid N(0,1) 的极差 W 的 CDF：P(W≤w) = n∫φ(x)[Φ(x+w)−Φ(x)]^{n−1}dx（梯形积分）。"""
    phi = lambda z: math.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    Phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
    h = (hi - lo) / steps
    tot = 0.0
    for i in range(steps + 1):
        x = lo + i * h
        f = n * phi(x) * (Phi(x + w) - Phi(x)) ** (n - 1)
        tot += f * (0.5 if i in (0, steps) else 1.0)
    return tot * h


def decode_comm_ms(alpha_us, beta_us_per_byte, layers, hidden, dtype_bytes=2, batch=1):
    """decode 每步 TP allreduce 总耗时：N·(α + β·b·h·s_d)，N = 2L + 1。"""
    n_ar = 2 * layers + 1
    return n_ar * (alpha_us + beta_us_per_byte * batch * hidden * dtype_bytes) / 1000.0, n_ar


def analyze(q, eps_us, iters, layers, hidden, dtype_bytes, threshold_pct=5.0):
    """q: {'p25','p50','p75','p99'} → [run][size] 微秒（原始，含 ε_sync）。"""
    k = len(q["p50"][0])
    col = lambda key, i: [r[i] for r in q[key]]
    med = {key: [statistics.median(col(key, i)) for i in range(k)] for key in q}
    var = [range_variance_pct(col("p50", i)) for i in range(k)]
    s = [jitter_s(med["p25"][i], med["p75"][i]) for i in range(k)]
    net = [max(t - eps_us, 1e-9) for t in med["p50"]]
    alpha, beta = fit_alpha_beta(SIZES[:k], net)
    comm_ms, n_ar = decode_comm_ms(alpha, max(beta, 0.0), layers, hidden, dtype_bytes)
    sigma_max = threshold_pct / 3.314 / 100          # 95% 通过所需的 σ_rel 上限
    return dict(
        median=med, per_size_var=var, worst=max(var), s=s, net_p50=net,
        alpha_us=alpha, beta_us_per_byte=beta,
        B_GBps=(1e-3 / beta) if beta > 0 else float("inf"),
        decode_comm_ms=comm_ms, n_ar=n_ar, eps_us=eps_us,
        sigma_samp_pct=[median_cv(x, iters) * 100 for x in s],
        n_min=max(min_iters(x, sigma_max) for x in s),
        pass_prob_no_drift=min(range_cdf(threshold_pct / max(median_cv(x, iters) * 100, 1e-9))
                               for x in s),
        threshold_pct=threshold_pct,
    )


def format_report(meta, an):
    """恰好 1 + len(SIZES) 行。"""
    v = "✓" if an["worst"] <= an["threshold_pct"] else "✗"
    warn = f" ⚠iters≥{an['n_min']}" if an["n_min"] > meta["iters"] else ""
    B = f"{an['B_GBps']:.1f}GB/s" if math.isfinite(an["B_GBps"]) else "n/a"
    head = (f"allreduce {meta['dtype']} ×{meta['world']} {meta['backend']} · "
            f"{meta['iters']}×{meta['runs']} · ε_sync {an['eps_us']:.1f}us · "
            f"α {an['alpha_us']:.1f}us B {B} · "
            f"decode {an['n_ar']}×T(4KB)={an['decode_comm_ms']:.2f}ms · "
            f"p50 var {an['worst']:.1f}% (≤{an['threshold_pct']:g}% {v}){warn} · {meta['sink']}")
    lines = [head]
    for i, n in enumerate(SIZES[:len(an["s"])]):
        runs = "/".join(f"{x:.1f}" for x in meta["p50_runs_col"](i))
        lines.append(f"{size_label(n):>6}  p50 {an['median']['p50'][i]:8.1f}us  "
                     f"p99 {an['median']['p99'][i]:8.1f}us  s {an['s'][i] * 100:4.1f}%  "
                     f"runs {runs}  var {an['per_size_var'][i]:.1f}%")
    return lines


def build_result(sha, dirty, host, world, lines, an, p50_runs):
    """门禁文件。metrics 里 gate-spec 声明的键名必须原样存在。"""
    col = lambda i: [r[i] for r in p50_runs]
    r2 = lambda x: round(x, 2)
    return {
        "commit": sha,
        "host": host,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cmd": f"torchrun --nproc_per_node={world} bench/latency_micro.py",
        "metrics": {
            "latency_p50_variance_pct": r2(an["worst"]),
            "output_lines": len(lines),  # 只计本脚本打印的行
            **{f"allreduce_{size_label(n).lower()}_us": r2(an["median"]["p50"][i])
               for i, n in enumerate(SIZES)},
            "alpha_us": r2(an["alpha_us"]),
            "eps_sync_us": r2(an["eps_us"]),
            "decode_comm_ms": r2(an["decode_comm_ms"]),
        },
        "model": {
            "beta_us_per_MiB": r2(an["beta_us_per_byte"] * (1 << 20)),
            "B_eff_GBps": r2(an["B_GBps"]) if math.isfinite(an["B_GBps"]) else None,
            "n_allreduce_per_step": an["n_ar"],
            "jitter_s_pct": [r2(x * 100) for x in an["s"]],
            "sigma_samp_pct": [r2(x) for x in an["sigma_samp_pct"]],
            "n_min_iters": an["n_min"],
            "pass_prob_no_drift": round(an["pass_prob_no_drift"], 4),
        },
        "runs": [r2(v) for v in col(0)],
        "runs_by_size": {size_label(n): [r2(v) for v in col(i)] for i, n in enumerate(SIZES)},
        "dirty_tree": dirty,
        "doc": DOC,
    }


def git_state():
    def run(*args):
        return subprocess.run(["git", "-C", REPO, *args], capture_output=True,
                              text=True).stdout.strip()
    dirty = [l for l in run("status", "--porcelain", "--untracked-files=no").splitlines()
             if "bench/results/" not in l]
    return run("rev-parse", "HEAD"), bool(dirty)


# ---------- 基准本体 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--reruns", type=int, default=3)
    ap.add_argument("--layers", type=int, default=40, help="decode 模型层数 L")
    ap.add_argument("--hidden", type=int, default=2048, help="hidden size h")
    ap.add_argument("--issue", type=int, default=1, help="写 bench/results/gate-NNN.json")
    ap.add_argument("--no-write", action="store_true", help="只打印，不写门禁文件")
    args = ap.parse_args()

    import torch
    import torch.distributed as dist

    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    if use_cuda:  # 必须在建通信组之前绑卡，否则各 rank 的 barrier/communicator 可能都落在 GPU0
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group(backend=backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    try:
        if use_cuda:
            device, dtype, sync = torch.device("cuda"), torch.float16, torch.cuda.synchronize
        else:  # gloo 不支持 fp16 allreduce
            device, dtype, sync = torch.device("cpu"), torch.float32, (lambda: None)
        elem = torch.tensor([], dtype=dtype).element_size()

        # 正确性自检：ones 求和应等于 world
        chk = torch.ones(SIZES[0] // elem, dtype=dtype, device=device)
        dist.all_reduce(chk)
        sync()
        if not torch.all(chk == world):
            raise RuntimeError(f"all_reduce 结果错误: 期望 {world}, 得到 {chk[0].item()}")

        # ε_sync：空队列时 synchronize + 计时本身的开销
        eps = []
        for _ in range(args.iters):
            sync()
            t0 = time.perf_counter()
            sync()
            eps.append((time.perf_counter() - t0) * 1e6)

        bufs = [torch.zeros(n // elem, dtype=dtype, device=device) for n in SIZES]  # 全 0，不溢出
        qs = (0.25, 0.50, 0.75, 0.99)
        stats = torch.zeros(len(qs), args.reruns, len(SIZES), dtype=torch.float32)
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
                for j, qq in enumerate(qs):
                    stats[j, r, i] = percentile(samples, qq)

        # 取最慢 rank（fp32：各家 ccl 都支持）
        stats = stats.to(device)
        eps_t = torch.tensor([percentile(eps, 0.5)], dtype=torch.float32, device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)
        dist.all_reduce(eps_t, op=dist.ReduceOp.MAX)
        q = {k: stats[j].cpu().tolist() for j, k in enumerate(("p25", "p50", "p75", "p99"))}

        if rank == 0:
            write = use_cuda and not args.no_write
            out = os.path.join(REPO, "bench", "results", f"gate-{args.issue:03d}.json")
            an = analyze(q, eps_t.item(), args.iters, args.layers, args.hidden, elem)
            meta = dict(dtype=str(dtype).replace("torch.", ""), world=world, backend=backend,
                        iters=args.iters, runs=args.reruns,
                        p50_runs_col=lambda i: [r[i] for r in q["p50"]],
                        sink=f"→ {os.path.relpath(out, REPO)}" if write else "dry-run 未写门禁文件")
            lines = format_report(meta, an)
            for line in lines:
                print(line, flush=True)
            if write:
                sha, dirty = git_state()
                host = f"{socket.gethostname()} ({world}x {torch.cuda.get_device_name(0)}, {backend})"
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with open(out, "w") as f:
                    json.dump(build_result(sha, dirty, host, world, lines, an, q["p50"]),
                              f, ensure_ascii=False, indent=1)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
