#!/usr/bin/env python3
r"""all_reduce 延迟微基准 —— M0「先做尺子」(issue #1)

    python3 bench/latency_micro.py            # 自带启动器（推荐，不用 torchrun）
    python3 bench/latency_micro.py --cpu      # CPU/gloo 演练，从不碰 GPU

理论与推导见 docs/ALLREDUCE_LATENCY_MODEL.md（§1–8 模型，§9 安全协议）。

安全协议（BI-V100/corex：中断在途 NCCL 通信会永久损坏驱动，且 GPU 资源不释放）：
- 控制面全部走 gloo（CPU）：rendezvous、每阶段前的共识、统计汇总都不经过 GPU
- 数据面（nccl）只跑被测的 all_reduce；进入每个数据阶段前，所有 rank 先在 CPU 上达成共识
  （都健康、都没收到停止信号），否则所有 rank 一起在 CPU 侧退出，GPU 上无在途操作
- 从不中断在途通信：不用 torchrun（它会在一个 worker 失败时杀掉其余 worker）；
  关闭 torch NCCL 看门狗的 abort/kill；SIGINT/SIGTERM 只置标志，在阶段边界统一退出；
  疑似卡住只报告、写现场文件，不自动 kill
- 数据阶段总暴露约 1 秒；其余时间出任何错都只会发生在 CPU 侧

测量要点：
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
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from datetime import timedelta
from datetime import datetime, timezone

SIZES = [4 << 10, 16 << 10, 64 << 10, 256 << 10, 1 << 20]  # 4KB 16KB 64KB 256KB 1MB
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Z75 = 0.6744897501960817      # Φ⁻¹(0.75)
DOC = "docs/ALLREDUCE_LATENCY_MODEL.md"

# 禁止 torch 在超时/心跳丢失时 abort 通信或杀进程（新旧两套变量名都设）
SAFE_ENV = {
    "TORCH_NCCL_ASYNC_ERROR_HANDLING": "0", "NCCL_ASYNC_ERROR_HANDLING": "0",
    "TORCH_NCCL_ENABLE_MONITORING": "0",
    "TORCH_NCCL_BLOCKING_WAIT": "0", "NCCL_BLOCKING_WAIT": "0",
}
EXIT_OK, EXIT_FAIL, EXIT_LOST, EXIT_STOP, EXIT_HUNG = 0, 2, 3, 4, 5


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
        "cmd": ("torchrun --nproc_per_node={} bench/latency_micro.py" if "TORCHELASTIC_RUN_ID"
                in os.environ else "python3 bench/latency_micro.py --nproc {}").format(world),
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


def verdict(fail, stop):
    """共识向量 → 决定。fail/stop 是各 rank 取 MAX 后的结果。"""
    return "fail" if fail else ("stop" if stop else "go")


def git_state():
    def run(*args):
        return subprocess.run(["git", "-C", REPO, *args], capture_output=True,
                              text=True).stdout.strip()
    dirty = [l for l in run("status", "--porcelain", "--untracked-files=no").splitlines()
             if "bench/results/" not in l]
    return run("rev-parse", "HEAD"), bool(dirty)


# ---------- 安全护栏 ----------

class Guard:
    """每个 rank 一个：信号只置标志；看门狗线程只报告不 kill。"""

    def __init__(self, rank, hang_after):
        self.rank, self.hang_after, self.stop = rank, hang_after, False
        self.stage_name, self._t = "init", time.monotonic()
        self.in_data_phase = False
        self._trace = os.environ.get("LATENCY_MICRO_TRACE")   # 测试用：记录进入过的数据阶段
        try:
            os.remove(f"/tmp/latency_micro.rank{rank}.hang.json")  # 清掉上次的现场文件
        except OSError:
            pass
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, self._on_signal)
        threading.Thread(target=self._watch, daemon=True).start()

    def log(self, msg):
        print(f"[latency_micro r{self.rank}] {msg}", file=sys.stderr, flush=True)

    def _on_signal(self, signum, _frame):
        if not self.stop:
            self.log(f"收到信号 {signum}：将在下一个阶段边界与所有 rank 一起退出。不要 kill -9")
        self.stop = True

    def stage(self, name, data=False):
        self.stage_name, self._t, self.in_data_phase = name, time.monotonic(), data
        if data and self._trace:
            with open(os.path.join(self._trace, f"rank{self.rank}.txt"), "a") as f:
                f.write(name + "\n")

    def _watch(self):
        reported = None
        while True:
            time.sleep(1.0)
            dt = time.monotonic() - self._t
            if dt > self.hang_after and reported != self.stage_name:
                reported = self.stage_name
                self.log(f"疑似卡住：阶段 {self.stage_name} 已 {dt:.0f}s（pid {os.getpid()}）。"
                         f"不会自动 kill —— 中断在途通信是驱动损坏的已知诱因。保留现场，见 {DOC} §9")
                try:
                    with open(f"/tmp/latency_micro.rank{self.rank}.hang.json", "w") as f:
                        json.dump({"rank": self.rank, "pid": os.getpid(), "stage": self.stage_name,
                                   "data_phase": self.in_data_phase, "seconds": round(dt)}, f)
                except OSError:
                    pass


def _fault(rank, where):
    """测试用故障注入：LATENCY_MICRO_FAULT=rank:where:{exit|raise|signal}，在共识之前触发。"""
    spec = os.environ.get("LATENCY_MICRO_FAULT", "")
    if spec.count(":") != 2:
        return
    r, w, kind = spec.split(":")
    if int(r) != rank or w != where:
        return
    if kind == "exit":
        os._exit(9)
    if kind == "raise":
        raise RuntimeError(f"injected fault at {where}")
    if kind == "signal":
        os.kill(os.getpid(), signal.SIGTERM)


# ---------- worker ----------

def worker(args):
    for k, v in SAFE_ENV.items():
        os.environ.setdefault(k, v)
    import torch
    import torch.distributed as dist

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    g = Guard(rank, args.hang_after)
    if "TORCHELASTIC_RUN_ID" in os.environ and rank == 0:
        g.log("警告：torchrun 会在任一 worker 失败时杀掉其余 worker，可能中断在途通信；"
              "推荐直接 python3 bench/latency_micro.py")
    use_cuda = torch.cuda.is_available() and not args.cpu

    def finish(code):
        """非正常退出：只在本 rank GPU 上无在途操作时调用；不 destroy、不 abort。"""
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)

    # S0 rendezvous：控制面 gloo，超时后抛异常而不是挂死
    g.stage("rendezvous")
    dist.init_process_group("gloo", timeout=timedelta(seconds=args.ctrl_timeout))

    def agree(ok, where):
        """CPU 共识。返回 go/fail/stop/lost。"""
        vec = torch.tensor([0 if ok else 1, 1 if g.stop else 0], dtype=torch.int32)
        try:
            dist.all_reduce(vec, op=dist.ReduceOp.MAX)
        except Exception as e:  # 对端已退出或超时：只发生在 CPU 侧
            g.log(f"共识 @{where} 失败（对端缺席）：{type(e).__name__}")
            return "lost"
        return verdict(*vec.tolist())

    def bail(v, where):
        g.log(f"@{where} → {v}，所有 rank 在 CPU 侧退出，未进入下一个数据阶段")
        finish({"fail": EXIT_FAIL, "stop": EXIT_STOP, "lost": EXIT_LOST}[v])

    # S1 本地健康检查（不涉及任何通信）+ 预分配全部缓冲区
    ok = True
    try:
        _fault(rank, "health")
        if use_cuda:
            torch.cuda.set_device(local)
            device, dtype, sync = torch.device("cuda", local), torch.float16, torch.cuda.synchronize
        else:  # gloo 不支持 fp16 allreduce
            device, dtype, sync = torch.device("cpu"), torch.float32, (lambda: None)
        g.stage("health")
        x = torch.ones(1024, dtype=dtype, device=device) * 2
        sync()
        if float(x.sum().item()) != 2048.0:
            raise RuntimeError("本地 GPU 运算结果错误")
        elem = torch.tensor([], dtype=dtype).element_size()
        chk = torch.ones(SIZES[0] // elem, dtype=dtype, device=device)
        bufs = [torch.zeros(n // elem, dtype=dtype, device=device) for n in SIZES]  # 全 0 不溢出
        eps = []  # ε_sync：空队列 synchronize + 计时开销
        for _ in range(args.iters):
            sync()
            t0 = time.perf_counter()
            sync()
            eps.append((time.perf_counter() - t0) * 1e6)
    except Exception as e:
        g.log(f"本地准备失败：{type(e).__name__}: {e}")
        ok = False
    v = agree(ok, "health")
    if v != "go":
        bail(v, "health")

    # 数据面：nccl（GPU）/ gloo（--cpu 演练）。超时设到极大，配合 SAFE_ENV 永不 abort
    data = dist.new_group(backend="nccl" if use_cuda else "gloo", timeout=timedelta(hours=24))

    def data_phase(name, fn):
        """共识 → 数据阶段 → 同步。阶段内异常说明对端可能正卡在与本 rank 的通信里：原地停车。"""
        try:
            _fault(rank, name)
            ok = True
        except Exception as e:
            g.log(f"@{name} 前出错：{type(e).__name__}: {e}")
            ok = False
        v = agree(ok, name)
        if v != "go":
            bail(v, name)
        g.stage(name, data=True)
        try:
            out = fn()
            sync()
        except Exception as e:
            g.log(f"数据阶段 {name} 内异常：{type(e).__name__}: {e}。对端可能正等本 rank，"
                  f"原地停车不退出（见 {DOC} §9）")
            g.stage(f"parked@{name}")
            while True:
                time.sleep(3600)
        g.stage(f"after:{name}")
        return out

    # S2 canary：首个数据集合 = NCCL 通信器初始化 + 正确性自检
    def canary():
        dist.all_reduce(chk, group=data)
        return chk
    data_phase("canary", canary)
    ok = bool(torch.all(chk == world))
    if not ok:
        g.log(f"canary 结果错误：期望 {world}，得到 {chk[0].item()}")
    v = agree(ok, "canary-check")  # 所有 rank 都调用，结果错误也在 CPU 侧统一退出
    if v != "go":
        bail(v, "canary-check")

    # S3 扫描：每个 (run, size) 是一个独立数据阶段
    qs = (0.25, 0.50, 0.75, 0.99)
    stats = torch.zeros(len(qs), args.reruns, len(SIZES), dtype=torch.float32)  # CPU
    for r in range(args.reruns):
        for i, buf in enumerate(bufs):
            def sweep(buf=buf):
                for _ in range(args.warmup):
                    dist.all_reduce(buf, group=data)
                sync()
                samples = []
                for _ in range(args.iters):
                    sync()
                    t0 = time.perf_counter()
                    dist.all_reduce(buf, group=data)
                    sync()
                    samples.append((time.perf_counter() - t0) * 1e6)
                return samples
            samples = data_phase(f"run{r}/{size_label(SIZES[i])}", sweep)
            for j, qq in enumerate(qs):
                stats[j, r, i] = percentile(samples, qq)

    # S4 汇总：全部走 gloo（CPU），取最慢 rank
    v = agree(True, "done")
    if v != "go":
        bail(v, "done")
    g.stage("gather")
    eps_t = torch.tensor([percentile(eps, 0.5)], dtype=torch.float32)
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    dist.all_reduce(eps_t, op=dist.ReduceOp.MAX)
    q = {k: stats[j].tolist() for j, k in enumerate(("p25", "p50", "p75", "p99"))}

    if rank == 0:
        write = use_cuda and not args.no_write
        out = os.path.join(REPO, "bench", "results", f"gate-{args.issue:03d}.json")
        an = analyze(q, eps_t.item(), args.iters, args.layers, args.hidden, elem)
        meta = dict(dtype=str(dtype).replace("torch.", ""), world=world,
                    backend="nccl" if use_cuda else "gloo", iters=args.iters, runs=args.reruns,
                    p50_runs_col=lambda i: [row[i] for row in q["p50"]],
                    sink=f"→ {os.path.relpath(out, REPO)}" if write else "dry-run 未写门禁文件")
        lines = format_report(meta, an)
        for line in lines:
            print(line, flush=True)
        if write:
            sha, dirty = git_state()
            host = f"{socket.gethostname()} ({world}x {torch.cuda.get_device_name(local)}, nccl)"
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w") as f:
                json.dump(build_result(sha, dirty, host, world, lines, an, q["p50"]),
                          f, ensure_ascii=False, indent=1)

    # 正常收尾：所有 rank 都已同步、无在途操作，才销毁通信器
    agree(True, "teardown")
    g.stage("teardown")
    dist.destroy_process_group(data)
    dist.destroy_process_group()
    return EXIT_OK


# ---------- 启动器（不 import torch，不碰 GPU，永不 kill 子进程） ----------

def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _proc_state(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            return next((l.split(":", 1)[1].strip() for l in f if l.startswith("State")), "?")
    except OSError:
        return "gone"


def launch(args):
    port, procs = _free_port(), []
    for r in range(args.nproc):
        env = dict(os.environ, RANK=str(r), LOCAL_RANK=str(r), WORLD_SIZE=str(args.nproc),
                   MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
        procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__), *sys.argv[1:]],
                                      env=env))

    def on_signal(signum, _frame):  # 只转告，由 worker 在阶段边界统一退出
        print(f"[latency_micro launcher] 收到信号 {signum}，已转告 worker 在阶段边界退出；"
              f"不会 kill", file=sys.stderr, flush=True)
        for p in procs:
            if p.poll() is None:
                try:
                    p.send_signal(signal.SIGTERM)
                except OSError:
                    pass
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, on_signal)

    deadline = time.monotonic() + args.deadline
    while time.monotonic() < deadline and any(p.poll() is None for p in procs):
        time.sleep(0.2)
    alive = [(r, p) for r, p in enumerate(procs) if p.poll() is None]
    if alive:
        for r, p in alive:
            print(f"[latency_micro launcher] rank{r} pid {p.pid} 仍在运行（{_proc_state(p.pid)}），"
                  f"现场文件 /tmp/latency_micro.rank{r}.hang.json。未 kill，见 {DOC} §9",
                  file=sys.stderr, flush=True)
        return EXIT_HUNG
    codes = [p.returncode for p in procs]
    return next((c for c in codes if c), EXIT_OK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nproc", type=int, default=4, help="启动器模式下的 rank 数")
    ap.add_argument("--cpu", action="store_true", help="CPU/gloo 演练，不碰 GPU")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--reruns", type=int, default=3)
    ap.add_argument("--layers", type=int, default=40, help="decode 模型层数 L")
    ap.add_argument("--hidden", type=int, default=2048, help="hidden size h")
    ap.add_argument("--issue", type=int, default=1, help="写 bench/results/gate-NNN.json")
    ap.add_argument("--no-write", action="store_true", help="只打印，不写门禁文件")
    ap.add_argument("--ctrl-timeout", type=float, default=120, help="CPU 共识等待对端的秒数")
    ap.add_argument("--hang-after", type=float, default=60, help="单阶段超过此秒数即报告疑似卡住")
    ap.add_argument("--deadline", type=float, default=900, help="启动器等待总秒数（到点只报告不 kill）")
    args = ap.parse_args()
    if "RANK" in os.environ:
        sys.exit(worker(args))
    sys.exit(launch(args))


if __name__ == "__main__":
    main()
