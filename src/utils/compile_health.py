"""Compile / CUDA-Graph health verification.

"reduce-overhead" in the config is a request, not a fact. Every producer
that declares compiled inference checks, after its warmup call, what
actually happened and fails on a silent downgrade: a latency or FLOPs
number measured on an accidental eager fallback is a wrong number.

Signals (torch._dynamo.utils.counters, torch 2.x):
  - counters["graph_break"]      any entry  -> the graph broke somewhere
  - counters["stats"]            calls_captured / unique_graphs -> compile
                                 actually produced graphs
  - counters["inductor"]         any '*cudagraph*skip*' entry -> inductor
                                 refused CUDA-Graph capture while
                                 reduce-overhead was requested

reset_compile_state() clears both dynamo caches and the counters so a
verification window sees only its own compilation.
"""
from __future__ import annotations

import torch


def reset_compile_state() -> None:
    torch._dynamo.reset()
    try:
        from torch._dynamo.utils import counters
        counters.clear()
    except Exception:
        pass


def _counters():
    from torch._dynamo.utils import counters
    return counters


def compile_health(where: str, expect_cudagraphs: bool) -> dict:
    """Inspect the dynamo/inductor counters accumulated since the last
    reset_compile_state(). Returns the status dict that gets recorded into
    telemetry; raises RuntimeError on any detected downgrade."""
    try:
        counters = _counters()
    except Exception as exc:
        raise RuntimeError(
            f"[{where}] cannot read torch._dynamo counters "
            f"({type(exc).__name__}: {exc}) — compile status is "
            f"UNVERIFIABLE on this torch build; refusing to proceed with an "
            f"unverified compiled producer") from exc

    breaks = dict(counters.get("graph_break", {}))
    n_breaks = int(sum(breaks.values()))
    stats = dict(counters.get("stats", {}))
    captured = int(stats.get("calls_captured", 0))
    unique = int(stats.get("unique_graphs", 0))
    skips = {k: int(v) for k, v in dict(counters.get("inductor", {})).items()
             if "cudagraph" in k and "skip" in k.lower()}
    n_skips = int(sum(skips.values()))

    status = {"where": where, "graph_breaks": n_breaks,
              "unique_graphs": unique, "calls_captured": captured,
              "cudagraph_skips": n_skips,
              "cudagraphs_expected": bool(expect_cudagraphs),
              "verified": True}
    if n_breaks:
        raise RuntimeError(
            f"[{where}] compile health FAILED: {n_breaks} graph break(s): "
            f"{sorted(breaks)[:5]} — the producer declared fullgraph "
            f"compiled inference; a broken graph must be fixed, not "
            f"silently tolerated")
    if unique == 0 and captured == 0:
        raise RuntimeError(
            f"[{where}] compile health FAILED: no compiled graph was "
            f"captured during warmup — the producer ran eager while "
            f"declaring compiled inference")
    if expect_cudagraphs and n_skips:
        raise RuntimeError(
            f"[{where}] compile health FAILED: inductor skipped CUDA-Graph "
            f"capture {n_skips} time(s) ({sorted(skips)[:5]}) while "
            f"reduce-overhead was requested — fix the capture blocker "
            f"before measuring latency")
    return status
