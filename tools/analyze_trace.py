#!/usr/bin/env python3
r"""analyze_trace —— decode trace 的 ≤25 行摘要（issue #4；移植自 EinsteinAuto/ikun PR 10 并重写输出）

    python3 tools/analyze_trace.py trace.json[.gz] [--tp 4] [--top 12] [--format text|json] [--gate 4]

【为什么各分类之和恰好 100%】
取所选 GPU 上全部活动（kernel / memcpy / memset）的区间 [a_i, b_i) 与类别 c(i)，把所有端点排序为
t_0 < t_1 < … < t_m。每个基本区间 [t_j, t_{j+1}) 上活跃集合 A_j 不变，定义

    T_c    = Σ_{j: A_j≠∅} (t_{j+1} − t_j) · |{i ∈ A_j : c(i) = c}| / |A_j|
    T_idle = Σ_{j: A_j=∅} (t_{j+1} − t_j)

则 Σ_c T_c + T_idle = Σ_j (t_{j+1} − t_j) = t_m − t_0（望远镜求和）。分类是窗口的一个**划分**，
多 stream 重叠时按并发数均分，不会重复计时。显示时用最大余数法舍入到 0.1%，显示值之和恰为 100.0。

CPU 侧事件（aten::item、aten::to、cudaLaunchKernel）在另一条时间线上，与 GPU 时间重叠，
单列且不计入 100%。（旧版把两者混加：0923 文档的分类合计只有 92.5%。）

【通信结构（issue #6）】TP=p 的 ring allreduce 每次 s = 2(p−1) 步，每步 2 次 fenceWait + 1 次 fenceOps；
reduce-scatter / all-gather 各有 p−1 个 elementwise-sum / kernelCopy。于是

    N = fenceOps / s，   校验 fenceWait = 2sN，elementwise-sum = kernelCopy = (p−1)N，
    α_trace = (四类总耗时) / N   —— 与 docs/ALLREDUCE_LATENCY_MODEL.md §3 同一口径
"""
import argparse
import gzip
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

MAX_LINES = 25
FIXED_LINES = 6            # 头 + 空闲 + 合计 + 通信 2 行 + CPU 1 行
GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CATEGORY_RULES = [
    ("fenceflagwait", "NCCL:fenceWait"), ("fenceops", "NCCL:fenceOps"),
    ("nccl", "NCCL:other"), ("general_gemm", "GEMM"),
    ("direct_w13", "MoE:w13"), ("direct_w2_reduce", "MoE:w2_reduce"),
    ("topk_gating", "MoE:topk_softmax"),
    ("gdn_packed_decode", "GDN:packed_decode"),
    ("causal_conv", "GDN:causal_conv"), ("gated_rms_norm", "GDN:gated_rms_norm"),
    ("fused_add_rms_norm", "Norm:fused_add_rms"),
    ("fused_qknorm_rope", "Norm:qknorm_rope"),
    ("rms_norm_kernel", "Norm:rms_norm"),
    ("kernelcopy", "Copy:kernelCopy"),
    ("act_and_mul", "Activation:silu_and_mul"),
    ("silu", "Activation:silu"), ("sigmoid", "Activation:sigmoid"),
    ("cached_kv_attention", "PagedAttention"),
    ("reshape_and_cache", "KVCache:reshape"),
    ("reduce_kernel", "Reduce"), ("embedding", "Embedding"),
    ("arange", "Utility:arange"), ("cat", "Utility:cat"),
]


# ---------- 分类 ----------

def categorize_kernel(name):
    nl = name.lower()
    for pat, cat in _CATEGORY_RULES:
        if pat in nl:
            return cat
    if "elementwise" in nl and "sum" in nl:
        return "Elementwise:sum(NCCL)"
    if "elementwise" in nl:
        return "Elementwise:other"
    return "Other"


def categorize(event):
    if event.get("cat") in ("gpu_memcpy", "gpu_memset"):
        return "Memcpy"
    return categorize_kernel(event.get("name", "")).split(":")[0]


# ---------- 数学核心（纯函数） ----------

def partition(intervals):
    """扫描线划分。intervals: [(a, b, label)]。
    返回 (T: {label: us}, idle, span, busy, raw)，满足 Σ T + idle == span。"""
    pts = [(a, 1, lab) for a, b, lab in intervals if b > a] + \
          [(b, -1, lab) for a, b, lab in intervals if b > a]
    if not pts:
        return {}, 0.0, 0.0, 0.0, 0.0
    pts.sort(key=lambda x: x[0])
    T, active, n, idle = defaultdict(float), Counter(), 0, 0.0
    i, t_prev = 0, pts[0][0]
    while i < len(pts):
        t = pts[i][0]
        dt = t - t_prev
        if dt > 0:
            if n == 0:
                idle += dt
            else:
                for lab, k in active.items():
                    if k:
                        T[lab] += dt * k / n
        while i < len(pts) and pts[i][0] == t:
            _, d, lab = pts[i]
            active[lab] += d
            n += d
            i += 1
        t_prev = t
    span = pts[-1][0] - pts[0][0]
    raw = sum(b - a for a, b, _ in intervals if b > a)
    return dict(T), idle, span, span - idle, raw


def round_to_total(values, total=1000):
    """最大余数法：把非负数按比例分成和恰为 total 的整数（total=1000 ⇒ 0.1% 精度）。"""
    s = sum(values.values())
    if s <= 0:
        return {k: 0 for k in values}
    raw = {k: v / s * total for k, v in values.items()}
    out = {k: math.floor(x) for k, x in raw.items()}
    rem = max(0, total - sum(out.values()))
    for k in sorted(raw, key=lambda k: raw[k] - out[k], reverse=True)[:rem]:
        out[k] += 1
    return out


def comm_structure(kernels, p):
    """kernels: [(name, dur)]。返回 N、逐项校验、α_trace。"""
    s = 2 * (p - 1)
    agg = {k: [0, 0.0] for k in ("fenceWait", "fenceOps", "elemSum", "kernelCopy")}
    for name, dur in kernels:
        nl = name.lower()
        key = ("fenceWait" if "fenceflagwait" in nl else "fenceOps" if "fenceops" in nl else
               "kernelCopy" if "kernelcopy" in nl else
               "elemSum" if ("elementwise" in nl and "sum" in nl) else None)
        if key:
            agg[key][0] += 1
            agg[key][1] += dur
    fo = agg["fenceOps"][0]
    if fo == 0:
        return None
    N = fo // s
    expect = {"fenceOps": s * N, "fenceWait": 2 * s * N, "elemSum": (p - 1) * N,
              "kernelCopy": (p - 1) * N}
    checks = {k: (agg[k][0], expect[k], agg[k][0] == expect[k]) for k in expect}
    exact = fo % s == 0 and all(ok for _, _, ok in checks.values())
    keys = list(agg) if exact else ["fenceWait", "fenceOps"]
    total = sum(agg[k][1] for k in keys)
    return dict(p=p, s=s, N=N, checks=checks, exact=exact,
                alpha_us=total / max(N, 1), comm_us=total, alpha_basis="+".join(keys))


# ---------- 读取与汇总 ----------

def load_trace(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def summarize(data, tp=None, top=12, device=None):
    events = data.get("traceEvents", [])
    by_dev = defaultdict(list)
    for e in events:
        if e.get("cat") in GPU_CATS and e.get("dur", 0) > 0:
            by_dev[e.get("args", {}).get("device", e.get("pid"))].append(e)
    if not by_dev:
        raise SystemExit("trace 里没有 GPU 活动（kernel/gpu_memcpy/gpu_memset）")
    if device is None:  # 默认取活动时间最多的那张卡
        device = max(by_dev, key=lambda d: sum(e["dur"] for e in by_dev[d]))
    evs = by_dev[device]
    T, idle, span, busy, raw = partition([(e["ts"], e["ts"] + e["dur"], categorize(e)) for e in evs])
    counts = Counter(categorize(e) for e in evs)

    top = max(1, min(top, MAX_LINES - FIXED_LINES - 1))
    order = sorted(T, key=lambda k: -T[k])
    rows = {k: T[k] for k in order[:top]}
    rest = order[top:]
    if rest:
        rows[f"其他({len(rest)} 类)"] = sum(T[k] for k in rest)
        counts[f"其他({len(rest)} 类)"] = sum(counts[k] for k in rest)
    rows["空闲"] = idle
    counts["空闲"] = 0
    shown = round_to_total(rows)

    p = tp or data.get("distributedInfo", {}).get("world_size") or 4
    comm = comm_structure([(e.get("name", ""), e["dur"]) for e in evs if e.get("cat") == "kernel"], p)

    cpu = {}
    for label, names, cat in (("aten::item", {"aten::item"}, "cpu_op"),
                              ("aten::to", {"aten::to", "aten::_to_copy"}, "cpu_op"),
                              ("cudaLaunchKernel", {"cudaLaunchKernel"}, "cuda_runtime")):
        sel = [e for e in events if e.get("cat") == cat and e.get("name") in names]
        cpu[label] = (len(sel), sum(e.get("dur", 0) for e in sel))

    names = [d.get("name", "?") for d in data.get("deviceProperties", []) if d.get("id") == device]
    exact_sum = (sum(T.values()) + idle) / span * 100 if span else 0.0
    return dict(device=device, gpu=names[0] if names else "?", span_us=span, busy_us=busy,
                overlap=raw / busy if busy else 0.0, rows=rows, counts=dict(counts),
                shown_tenths=shown, comm=comm, cpu=cpu,
                sum_err_pct=max(abs(exact_sum - 100), abs(sum(shown.values()) / 10 - 100)))


def format_summary(sm, trace_name):
    L = [f"trace {trace_name} · device {sm['device']}（{sm['gpu']}）· 窗口 {sm['span_us'] / 1000:.2f}ms · "
         f"GPU 忙 {sm['busy_us'] / max(sm['span_us'], 1e-9):.1%} · 重叠系数 {sm['overlap']:.2f}"]
    for k in sm["rows"]:
        L.append(f"  {sm['shown_tenths'][k] / 10:5.1f}%  {sm['rows'][k] / 1000:8.2f}ms  "
                 f"{sm['counts'].get(k, 0):6d}×  {k}")
    L.append(f"  {sum(sm['shown_tenths'].values()) / 10:5.1f}%  合计（分类是 GPU 窗口的划分；CPU 侧见末行）")
    c = sm["comm"]
    if c:
        chk = " · ".join(f"{k} {got}={exp // max(c['N'], 1) if c['N'] else 0}×{c['N']} {'✓' if ok else '✗'}"
                         for k, (got, exp, ok) in c["checks"].items())
        L.append(f"通信 TP={c['p']} ring {c['s']} 步：N={c['N']} 次 allreduce · {chk}")
        L.append(f"α_trace={c['alpha_us']:.1f}us（{c['alpha_basis']}）· 通信合计 {c['comm_us'] / 1000:.2f}ms"
                 f"（窗口 {c['comm_us'] / max(sm['span_us'], 1e-9):.1%}）· 若窗口=1 个 decode step，"
                 f"通信归零上限 {1e6 / max(sm['span_us'] - c['comm_us'], 1e-9):.1f} tok/s")
    else:
        L += ["通信：未发现 fenceOps（非 ring/NCCL trace）", ""]
    L.append("CPU 侧（另一条时间线，不计入 100%）：" + " · ".join(
        f"{k} {n}× {us / 1000:.2f}ms" for k, (n, us) in sm["cpu"].items()))
    assert len(L) <= MAX_LINES, len(L)
    return L


def git_sha():
    return subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description="decode trace ≤25 行摘要")
    ap.add_argument("trace")
    ap.add_argument("--tp", type=int, default=None, help="TP 大小，默认读 distributedInfo.world_size")
    ap.add_argument("--top", type=int, default=12, help="显示的分类数，其余合并")
    ap.add_argument("--device", default=None, help="选哪张卡，默认活动最多的")
    ap.add_argument("--format", choices=["text", "json"], default="text")
    ap.add_argument("--gate", type=int, default=None, help="写 bench/results/gate-NNN.json")
    args = ap.parse_args(argv)

    data = load_trace(args.trace)
    dev = None if args.device is None else (int(args.device) if args.device.isdigit() else args.device)
    sm = summarize(data, args.tp, args.top, dev)
    lines = format_summary(sm, os.path.basename(args.trace))
    if args.format == "json":
        out = {k: v for k, v in sm.items() if k != "shown_tenths"}
        out["shown_pct"] = {k: v / 10 for k, v in sm["shown_tenths"].items()}
        print(json.dumps(out, ensure_ascii=False, default=str))
    else:
        print("\n".join(lines))
    if args.gate:
        with open(args.trace, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        res = {"commit": git_sha(), "host": socket.gethostname(),
               "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "cmd": "python3 tools/analyze_trace.py " + " ".join(argv or sys.argv[1:]),
               "metrics": {"summary_lines": len(lines),
                           "category_sum_pct_err": round(sm["sum_err_pct"], 6)},
               "trace": os.path.basename(args.trace), "trace_sha256": digest}
        path = os.path.join(REPO, "bench", "results", f"gate-{args.gate:03d}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
