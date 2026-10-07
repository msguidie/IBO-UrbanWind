"""Windowed step profiling and a per-epoch resource monitor.

Profiling runs in windows only: steps [10, 30] after start,
then `periodic_len` steps every `periodic_every`. Per step: data_wait_ms
(DataLoader stall), h2d_ms, forward_ms, backward_ms, optim_ms, total_step_ms,
plus gpu_mem and (best-effort) gpu_util%/cpu_util%. Emits
profiling_step{a}-{b}.json and prints the bottleneck verdicts.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import torch

try:
    import psutil
    _PROC = psutil.Process()
    _PROC.cpu_percent(None)  # prime
except ImportError:
    psutil = None
    _PROC = None

SEGMENTS = ("forward", "backward", "optim")


def _gpu_util_percent() -> float | None:
    """Utilization of THIS process's GPU only (remapped through
    CUDA_VISIBLE_DEVICES) — a node-wide max would hide rank-0 bubbles behind
    peer GPUs busy-waiting in NCCL."""
    try:
        idx = torch.cuda.current_device() if torch.cuda.is_available() else 0
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd:
            ids = [g.strip() for g in cvd.split(",") if g.strip()]
            phys = ids[idx] if idx < len(ids) else ids[0]
        else:
            phys = str(idx)
        out = subprocess.run(
            ["nvidia-smi", "-i", phys, "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2)
        vals = [float(v) for v in out.stdout.split() if v.strip()]
        return vals[0] if vals else None
    except Exception:
        return None


def _cpu_util_percent() -> float | None:
    """Main process + children (DataLoader workers), best-effort."""
    if _PROC is None:
        return None
    try:
        total = _PROC.cpu_percent(None)
        for ch in _PROC.children(recursive=True):
            try:
                total += ch.cpu_percent(None)
            except Exception:
                pass
        return total
    except Exception:
        return None


class StepProfiler:
    def __init__(self, out_dir: str | Path, first_window: tuple[int, int],
                 periodic_every: int, periodic_len: int, enabled: bool = True):
        self.out_dir = Path(out_dir)
        self.first_window = first_window
        self.periodic_every = max(int(periodic_every), 1)
        self.periodic_len = int(periodic_len)
        self.enabled = enabled
        self.use_cuda = torch.cuda.is_available()
        self.records: list[dict] = []
        self.window_start: int | None = None
        self.baseline_step_ms: float | None = None
        self._events: dict[str, tuple] = {}
        self._t_step0 = 0.0
        self._t_data0 = 0.0
        self._data_wait_ms = 0.0

    def active(self, step: int) -> bool:
        if not self.enabled:
            return False
        a, b = self.first_window
        if a <= step <= b:
            return True
        if step >= self.periodic_every:
            phase = step % self.periodic_every
            return phase < self.periodic_len
        return False

    # -- data wait: call around the DataLoader fetch --------------------
    def data_wait_begin(self):
        self._t_data0 = time.perf_counter()

    def data_wait_end(self):
        self._data_wait_ms = (time.perf_counter() - self._t_data0) * 1000.0

    # -- timed segments ---------------------------------------------------
    def step_begin(self, step: int):
        if self.window_start is None:
            self.window_start = step
        self._events = {}
        self._t_step0 = time.perf_counter()

    def mark_begin(self, name: str):
        if self.use_cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self._events[name] = (e, None)
        else:
            self._events[name] = (time.perf_counter(), None)

    def mark_end(self, name: str):
        begin, _ = self._events[name]
        if self.use_cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self._events[name] = (begin, e)
        else:
            self._events[name] = (begin, time.perf_counter())

    def step_end(self, step: int):
        if self.use_cuda:
            torch.cuda.synchronize()
        rec = {"step": step, "data_wait_ms": self._data_wait_ms}
        for name, (b, e) in self._events.items():
            if e is None:
                continue
            rec[f"{name}_ms"] = b.elapsed_time(e) if self.use_cuda \
                else (e - b) * 1000.0
        rec["total_step_ms"] = (time.perf_counter() - self._t_step0) * 1000.0
        if self.use_cuda:
            rec["gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 1024 ** 2
        cpu = _cpu_util_percent()
        if cpu is not None:
            rec["cpu_util_pct"] = cpu
        self.records.append(rec)

    def window_close(self, step: int):
        """Call when active(step) was True and active(step+1) is False."""
        if not self.records:
            self.window_start = None
            return
        a, b = self.window_start, step
        util = _gpu_util_percent()
        mean = {k: float(sum(r.get(k, 0.0) for r in self.records) / len(self.records))
                for k in self.records[-1]}
        summary = {"window": [a, b], "mean": mean, "gpu_util_pct": util,
                   "steps": self.records}

        verdicts = []
        if mean.get("data_wait_ms", 0) > mean.get("forward_ms", 0):
            verdicts.append("CPU bottleneck: data_wait > forward — add DataLoader workers")
        if util is not None and util < 85:
            verdicts.append(f"GPU util {util:.0f}% < 85% — pipeline bubble, check sync points")
        if self.baseline_step_ms is None:
            self.baseline_step_ms = mean.get("total_step_ms")
        elif mean.get("total_step_ms", 0) > 1.1 * self.baseline_step_ms:
            verdicts.append(f"step time drift >10% vs baseline "
                            f"{self.baseline_step_ms:.1f} ms")
        summary["verdicts"] = verdicts

        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.out_dir / f"profiling_step{a}-{b}.json", "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, indent=1)
        line = " | ".join(f"{k}={v:.1f}" for k, v in mean.items()
                          if k.endswith("_ms"))
        print(f"[profile {a}-{b}] {line}" +
              ("".join(f"\n  !! {v}" for v in verdicts)), flush=True)
        self.records = []
        self.window_start = None


class EpochMonitor:
    """Per-epoch GPU peak memory + CPU RSS (psutil, /proc fallback)."""

    @staticmethod
    def snapshot() -> dict:
        out = {}
        if torch.cuda.is_available():
            out["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 1024 ** 3
            torch.cuda.reset_peak_memory_stats()
        if _PROC is not None:
            out["cpu_rss_gb"] = _PROC.memory_info().rss / 1024 ** 3
        else:
            try:
                import resource
                out["cpu_rss_gb"] = resource.getrusage(
                    resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
            except ImportError:
                pass
        return out
