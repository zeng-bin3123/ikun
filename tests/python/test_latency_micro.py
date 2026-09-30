"""Tests for bench/latency_micro.py (issue #1). No torch / GPU needed.

数值与 docs/ALLREDUCE_LATENCY_MODEL.md 锁定：文档里的每个结论都在这里有对应断言。
"""
import importlib.util
import math
import os

import pytest

_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "bench", "latency_micro.py")
_spec = importlib.util.spec_from_file_location("latency_micro", _PATH)
lm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lm)

MiB = 1 << 20
# 实测带宽基准（4×BI-V100，每档 20 轮），打印值 = 2·S·R/t/2^30
BW_SIZES = [1 * MiB, 16 * MiB, 64 * MiB, 256 * MiB]
BW_GBPS = [6.1, 31.7, 37.5, 39.1]
BW_T_US = [2 * s / (g * 2**30) * 1e6 for s, g in zip(BW_SIZES, BW_GBPS)]
# issue #1 正文 gate-spec 原文
GATE_SPEC = [{"m": "latency_p50_variance_pct", "op": "le", "t": 5, "u": "%"},
             {"m": "output_lines", "op": "le", "t": 6, "u": "line"}]


def test_five_sizes_4kb_to_1mb():
    assert [lm.size_label(n) for n in lm.SIZES] == ["4KB", "16KB", "64KB", "256KB", "1MB"]


def test_percentile_and_range():
    assert lm.percentile([1, 2, 3, 4], 0.5) == pytest.approx(2.5)
    assert lm.range_variance_pct([27.1, 27.4, 27.8]) == pytest.approx(0.7 / 27.4 * 100)


def test_fit_recovers_exact_line():
    a, b = 268.5, 47.54 / MiB
    xs = lm.SIZES
    alpha, beta = lm.fit_alpha_beta(xs, [a + b * x for x in xs])
    assert alpha == pytest.approx(a) and beta == pytest.approx(b)


def test_fit_on_measured_bandwidth_matches_doc():
    """§2：α ≈ 268.5us，B_eff ≈ 22.1 GB/s，ring 链路 B = 1.5·B_eff ≈ 33 GB/s。"""
    alpha, beta = lm.fit_alpha_beta(BW_SIZES, BW_T_US)
    assert alpha == pytest.approx(268.5, abs=0.5)
    assert beta * MiB == pytest.approx(47.54, abs=0.05)
    assert 1e-3 / beta == pytest.approx(22.06, abs=0.05)
    assert alpha / (beta * MiB) == pytest.approx(5.65, abs=0.02)   # S* [MiB]


def test_trace_microstructure_predicts_alpha():
    """§3：81 次 allreduce 的 fence+copy+elementwise 总和 / 81 ≈ α_fit（差 < 1%）。"""
    n_ar = 2 * 40 + 1
    assert (972 / n_ar, 486 / n_ar, 243 / n_ar) == (12, 6, 3)
    alpha_trace = (10.9 + 5.1 + 2.9 + 2.8) * 1000 / n_ar
    alpha_fit, _ = lm.fit_alpha_beta(BW_SIZES, BW_T_US)
    assert abs(alpha_trace - alpha_fit) / alpha_fit < 0.01


def test_fence_counts_match_two_independent_configs():
    """§3.1 零参数外推：同一公式同时命中 L=40/p=4 的 972 与 L=4/p=2 的 36。"""
    big = lm.fence_counts(layers=40, p=4)
    assert (big["n_allreduce"], big["steps"]) == (81, 6)
    assert (big["fenceWait"], big["fenceOps"]) == (972, 486)
    assert big["elemSum"] == big["kernelCopy"] == 243
    mini = lm.fence_counts(layers=4, p=2)      # PR10 session 5 实测 36，未用于标定
    assert mini["fenceWait"] == 36
    # 被排除的替代假设
    assert 2 * 6 * 40 != 972 and 2 * 6 * 9 != 36


def test_fence_hypothesis_is_uniquely_identified():
    """§3.1 可辨识性：400 组 (N=aL+b, s=c(p-1)+d) 中只有 (2,1,2,0) 同时命中两点。"""
    obs = [(40, 4, 972), (4, 2, 36)]
    hits = [(a, b, c, d)
            for a in range(4) for b in range(-2, 3) for c in range(4) for d in range(-2, 3)
            if all((c * (p - 1) + d) > 0 and (a * L + b) > 0
                   and 2 * (c * (p - 1) + d) * (a * L + b) == o for L, p, o in obs)]
    assert hits == [(2, 1, 2, 0)]


def test_infiniccl_overhead_confirms_n81():
    """§3.2：12.5→12.2 tok/s ⇒ 每次 allreduce 多 24.3us，与 PRD 的 ~25us/dispatch 吻合。"""
    d_ms = 1000 / 12.2 - 1000 / 12.5
    assert d_ms == pytest.approx(1.97, abs=0.01)
    assert d_ms * 1000 / 81 == pytest.approx(24.3, abs=0.2)     # N=81 → 吻合
    assert d_ms * 1000 / 40 > 45                                 # N=40 → 与 25us/call 矛盾


def test_decode_comm_and_amdahl_bound():
    """§4：T_comm(b=1) ≈ 21.76ms；通信归零的上限 ≈ 17.1 tok/s。"""
    ms, n = lm.decode_comm_ms(268.5, 47.54 / MiB, layers=40, hidden=2048)
    assert n == 81 and ms == pytest.approx(21.76, abs=0.01)
    assert 1000 / (80.2 - ms) == pytest.approx(17.1, abs=0.05)


def test_range_distribution_constants():
    """§6：3 个 iid N(0,1) 的极差 95% 分位 = 3.314 ⇒ σ_rel ≤ 1.51%。"""
    assert lm.range_cdf(3.314) == pytest.approx(0.95, abs=1e-3)
    assert 5 / 3.314 == pytest.approx(1.51, abs=0.005)


def test_jitter_and_median_cv():
    s = 0.17
    p25, p75 = 100 * math.exp(-lm.Z75 * s), 100 * math.exp(lm.Z75 * s)
    assert lm.jitter_s(p25, p75) == pytest.approx(s)
    assert lm.median_cv(s, 200) == pytest.approx(0.01506, abs=1e-4)
    assert lm.min_iters(s, 0.0151) <= 200 < lm.min_iters(0.25, 0.0151)


def _q(p50):
    return {"p25": [[x * 0.95 for x in r] for r in p50], "p50": p50,
            "p75": [[x * 1.05 for x in r] for r in p50], "p99": p50}


def _meta(p50):
    return dict(dtype="float16", world=4, backend="nccl", iters=200, runs=3, sink="dry-run",
                p50_runs_col=lambda i: [r[i] for r in p50])


def test_report_is_six_lines_and_takes_worst_size():
    p50 = [[268.7, 269.2, 271.5, 280.4, 316.0],
           [268.7, 269.2, 271.5, 280.4, 348.0],   # 1MB 抖 10%
           [268.7, 269.2, 271.5, 280.4, 320.0]]
    an = lm.analyze(_q(p50), 4.0, 200, 40, 2048, 2)
    lines = lm.format_report(_meta(p50), an)
    assert len(lines) == 6
    assert an["worst"] == pytest.approx(32 / 320 * 100)
    assert "✗" in lines[0]


def test_gate_json_matches_spec_and_readme_fields():
    p50 = [[268.7, 269.2, 271.5, 280.4, 316.0]] * 3
    an = lm.analyze(_q(p50), 0.0, 200, 40, 2048, 2)
    lines = lm.format_report(_meta(p50), an)
    res = lm.build_result("a" * 40, False, "node (4x BI-V100, nccl)", 4, lines, an, p50)
    for f in ("commit", "host", "timestamp", "metrics"):
        assert res[f]
    for c in GATE_SPEC:
        v = res["metrics"][c["m"]]
        assert isinstance(v, (int, float)) and v <= c["t"]
    assert res["metrics"]["alpha_us"] == pytest.approx(268.5, abs=0.5)
    assert res["doc"] == lm.DOC
