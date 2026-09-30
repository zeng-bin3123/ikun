"""故障注入：证明安全协议的不变式（issue #1，需要 torch，CPU/gloo 即可跑）。

不变式：任一 rank 进入数据阶段 P_k ⇒ 所有 rank 都在 CPU 共识 C_k 上投了 go。
等价可测形式：所有 rank 的"已进入数据阶段"序列完全相同，且不含故障发生处的阶段；
并且所有进程都在有限时间内自行退出（启动器从不 kill）。
"""
import os
import subprocess
import sys
import time

import pytest

pytest.importorskip("torch")

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..", "bench", "latency_micro.py")
FULL = ["canary"] + [f"run{r}/{s}" for r in range(2)
                     for s in ("4KB", "16KB", "64KB", "256KB", "1MB")]


def run(tmp_path, fault=None):
    env = dict(os.environ, LATENCY_MICRO_TRACE=str(tmp_path), OMP_NUM_THREADS="1")
    env.pop("LATENCY_MICRO_FAULT", None)
    if fault:
        env["LATENCY_MICRO_FAULT"] = fault
    t0 = time.monotonic()
    p = subprocess.run([sys.executable, SCRIPT, "--cpu", "--nproc", "4", "--iters", "10",
                        "--warmup", "1", "--reruns", "2", "--ctrl-timeout", "30",
                        "--deadline", "150"], env=env, capture_output=True, text=True, timeout=200)
    traces = []
    for r in range(4):
        f = tmp_path / f"rank{r}.txt"
        traces.append(f.read_text().split() if f.exists() else [])
    return p, traces, time.monotonic() - t0


def assert_lockstep(traces, upto):
    expected = FULL[:FULL.index(upto)] if upto else FULL
    assert all(t == expected for t in traces), traces


def test_happy_path(tmp_path):
    p, traces, _ = run(tmp_path)
    assert p.returncode == 0, p.stderr[-800:]
    assert len(p.stdout.splitlines()) == 6
    assert_lockstep(traces, None)


def test_rank_crash_between_phases_all_exit_on_cpu(tmp_path):
    """rank2 在 run0/64KB 前崩溃（os._exit）：其余 rank 不得进入 run0/64KB，且全部自行退出。"""
    p, traces, dt = run(tmp_path, "2:run0/64KB:exit")
    assert p.returncode in (3, 9), p.stderr[-800:]      # 3 = 对端缺席（CPU 侧检测到）
    assert "仍在运行" not in p.stderr                   # 启动器没有等到 deadline
    assert_lockstep(traces, "run0/64KB")


def test_exception_before_canary_nobody_touches_data_plane(tmp_path):
    p, traces, _ = run(tmp_path, "1:canary:raise")
    assert p.returncode == 2, p.stderr[-800:]
    assert traces == [[], [], [], []]                   # 数据面（NCCL 初始化）从未被触发


def test_local_health_failure_blocks_everything(tmp_path):
    p, traces, _ = run(tmp_path, "0:health:raise")
    assert p.returncode == 2 and traces == [[], [], [], []]


def test_sigterm_mid_run_stops_at_phase_boundary(tmp_path):
    """rank3 收到 SIGTERM：不立即退出，所有 rank 在 run1/4KB 前一起停下。"""
    p, traces, _ = run(tmp_path, "3:run1/4KB:signal")
    assert p.returncode == 4, p.stderr[-800:]
    assert "不要 kill -9" in p.stderr
    assert_lockstep(traces, "run1/4KB")
