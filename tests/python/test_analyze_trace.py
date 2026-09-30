"""Tests for tools/analyze_trace.py (issue #4, 并为 #6 的计数口径做准备)。纯 CPU。"""
import gzip
import importlib.util
import json
import os

import pytest

_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "tools", "analyze_trace.py")
_spec = importlib.util.spec_from_file_location("analyze_trace", _PATH)
at = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(at)


def K(name, ts, dur, stream=7, cat="kernel"):
    return {"ph": "X", "cat": cat, "name": name, "pid": 0, "tid": stream, "ts": ts, "dur": dur,
            "args": {"device": 0, "stream": stream}}


def decode_like_trace(n_ar=81, p=4, drop_one_fence=False):
    """按 0923 trace 的结构合成：每次 allreduce 走 2(p-1) 步，每步 2 fenceWait + 1 fenceOps
    + 1 个 elementwise-sum（RS）或 kernelCopy（AG），耗时 11/11/11/12us；中间夹 GEMM/MoE 与空隙。"""
    ev, t = [], 0.0
    for a in range(n_ar):
        ev.append(K("general_gemm_f16", t, 130.0)); t += 135.0
        for step in range(2 * (p - 1)):
            for nm in ("fenceFlagWaitE", "fenceFlagWaitE", "fenceOps"):
                ev.append(K(nm, t, 11.0)); t += 11.0
            nm = "legacy::elementWiseKernel<sum>" if step < p - 1 else "kernelCopy"
            ev.append(K(nm, t, 12.0)); t += 12.0
        ev.append(K("direct_w13_moe", t, 40.0)); t += 45.0
    if drop_one_fence:
        ev = [e for i, e in enumerate(ev) if not (e["name"] == "fenceFlagWaitE" and i == 1)]
    ev.append(K("Memcpy DtoD", t, 50.0, cat="gpu_memcpy"))
    ev += [{"ph": "X", "cat": "cpu_op", "name": "aten::to", "ts": 0, "dur": 9e6, "pid": 1, "tid": 1},
           {"ph": "X", "cat": "cpu_op", "name": "aten::item", "ts": 5, "dur": 370, "pid": 1, "tid": 1},
           {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 5, "dur": 4, "pid": 1, "tid": 1}]
    return {"traceEvents": ev, "distributedInfo": {"world_size": p},
            "deviceProperties": [{"id": 0, "name": "Iluvatar BI-V100"}]}


# ---- 划分定理 ----

def test_partition_sums_to_span_with_gaps():
    T, idle, span, busy, raw = at.partition([(0, 10, "A"), (15, 20, "B"), (30, 31, "A")])
    assert (T["A"], T["B"], idle, span) == (11, 5, 15, 31)
    assert sum(T.values()) + idle == pytest.approx(span)


def test_partition_splits_overlap_instead_of_double_counting():
    T, idle, span, busy, raw = at.partition([(0, 10, "GEMM"), (5, 15, "MoE")])
    assert T == {"GEMM": 7.5, "MoE": 7.5} and idle == 0 and span == 15
    assert raw / busy == pytest.approx(20 / 15)          # 重叠系数


def test_partition_identity_random():
    import random
    rnd = random.Random(0)
    iv = [(a, a + rnd.uniform(0.1, 50), rnd.choice("ABCDE"))
          for a in (rnd.uniform(0, 1000) for _ in range(2000))]
    T, idle, span, _, _ = at.partition(iv)
    assert abs(sum(T.values()) + idle - span) < 1e-6 * span


# ---- 舍入 ----

def test_largest_remainder_sums_exactly():
    r = at.round_to_total({"a": 1, "b": 1, "c": 1})
    assert sum(r.values()) == 1000 and sorted(r.values()) == [333, 333, 334]
    r = at.round_to_total({f"k{i}": 1 + i * 1e-3 for i in range(17)})
    assert sum(r.values()) == 1000


# ---- 摘要：行数上界、合计 100%、CPU 不计入 ----

def test_summary_on_decode_like_trace():
    sm = at.summarize(decode_like_trace())
    lines = at.format_summary(sm, "t.json")
    assert len(lines) <= 25
    assert sum(sm["shown_tenths"].values()) == 1000
    assert sm["sum_err_pct"] < 1e-9
    assert sm["overlap"] == pytest.approx(1.0)          # 串行合成 trace
    assert "aten::to 1× 9000.00ms" in lines[-1]         # CPU 侧单列，未进入 100%


def test_many_categories_are_folded_and_lines_bounded():
    names = [r[0] for r in at._CATEGORY_RULES] + ["weird_kernel_%d" % i for i in range(300)]
    ev = [K(n + "_x", i * 10.0, 5.0) for i, n in enumerate(names)]
    for top in (3, 12, 100):
        sm = at.summarize({"traceEvents": ev}, top=top)
        lines = at.format_summary(sm, "t.json")
        assert len(lines) <= 25 and sum(sm["shown_tenths"].values()) == 1000


# ---- 通信结构（#6 口径）----

def test_comm_structure_recovers_n81_and_alpha():
    c = at.summarize(decode_like_trace())["comm"]
    assert c["N"] == 81 and c["exact"]
    assert {k: v[:2] for k, v in c["checks"].items()} == {
        "fenceOps": (486, 486), "fenceWait": (972, 972), "elemSum": (243, 243), "kernelCopy": (243, 243)}
    assert c["alpha_us"] == pytest.approx(6 * (3 * 11 + 12))   # 270us = 6 步 × 45us


def test_comm_structure_flags_mismatch():
    c = at.summarize(decode_like_trace(drop_one_fence=True))["comm"]
    assert not c["exact"] and not c["checks"]["fenceWait"][2]
    assert c["alpha_basis"] == "fenceWait+fenceOps"


def test_other_tp_sizes():
    c = at.summarize(decode_like_trace(n_ar=10, p=2))["comm"]
    assert (c["s"], c["N"], c["exact"]) == (2, 10, True)


# ---- CLI ----

def test_cli_gz_text_and_gate(tmp_path, capsys, monkeypatch):
    f = tmp_path / "t.json.gz"
    with gzip.open(f, "wt") as g:
        json.dump(decode_like_trace(), g)
    monkeypatch.setattr(at, "REPO", str(tmp_path))
    assert at.main([str(f), "--gate", "4"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) <= 25
    res = json.loads((tmp_path / "bench" / "results" / "gate-004.json").read_text())
    assert res["metrics"]["summary_lines"] == len(out)
    assert res["metrics"]["category_sum_pct_err"] <= 1
    assert len(res["trace_sha256"]) == 64
