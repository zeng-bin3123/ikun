"""Tests for bench/latency_micro.py pure helpers (issue #1). No torch / GPU needed."""
import importlib.util
import os

import pytest

_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "bench", "latency_micro.py")
_spec = importlib.util.spec_from_file_location("latency_micro", _PATH)
lm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lm)

META = dict(dtype="float16", world=4, backend="nccl", iters=200, sink="dry-run")


def test_five_sizes_4kb_to_1mb():
    assert [lm.size_label(n) for n in lm.SIZES] == ["4KB", "16KB", "64KB", "256KB", "1MB"]


def test_percentile_interpolates():
    assert lm.percentile([1, 2, 3, 4], 0.5) == pytest.approx(2.5)
    assert lm.percentile([5], 0.99) == 5


def test_range_variance_pct():
    assert lm.range_variance_pct([27.1, 27.4, 27.8]) == pytest.approx(0.7 / 27.4 * 100)
    assert lm.range_variance_pct([0, 0, 0]) == 0.0


def test_report_is_six_lines_and_takes_worst_size():
    p50 = [[27.0, 30.0, 40.0, 80.0, 300.0],
           [27.0, 30.0, 40.0, 80.0, 330.0],   # 1MB 抖 10%
           [27.0, 30.0, 40.0, 80.0, 310.0]]
    lines, worst = lm.format_report(META, p50, p50)
    assert len(lines) == 6
    assert worst == pytest.approx(30 / 310 * 100)
    assert "✗" in lines[0]


def test_report_passes_when_stable():
    p50 = [[27.1] * 5, [27.4] * 5, [27.8] * 5]
    lines, worst = lm.format_report(META, p50, p50)
    assert worst < 5 and "✓" in lines[0]


# issue #1 正文里的 gate-spec 原文
GATE_SPEC = [{"m": "latency_p50_variance_pct", "op": "le", "t": 5, "u": "%"},
             {"m": "output_lines", "op": "le", "t": 6, "u": "line"}]


def test_gate_json_matches_spec_and_readme_fields():
    p50 = [[27.1] * 5, [27.4] * 5, [27.8] * 5]
    lines, worst = lm.format_report(META, p50, p50)
    res = lm.build_result("a" * 40, False, "node (4x BI-V100, nccl)", 4, lines, worst, p50)
    for f in ("commit", "host", "timestamp", "metrics"):
        assert res[f]
    for c in GATE_SPEC:
        v = res["metrics"][c["m"]]
        assert isinstance(v, (int, float)) and v <= c["t"]
    assert res["metrics"]["output_lines"] == 6
    assert res["runs"] == [27.1, 27.4, 27.8]
