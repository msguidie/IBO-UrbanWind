"""SWA helpers, on disk.

Every epoch rank 0 writes an fp32 model snapshot to
epoch_weights/epoch_{N:04d}.pt. The final weights (swa_model.pt) average the
snapshots of [swa_start, num_epochs], computed after training in one
sequential pass over the files (_TrailingWindowAccumulator: a running fp32
sum, each file read once). average_epoch_range is an independent, naive
implementation of the same average, kept as a reference.
"""
from __future__ import annotations

import os
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn


def unwrap_model(model: nn.Module) -> nn.Module:
    """Peel torch.compile (_orig_mod) and DDP (module) wrappers."""
    for _ in range(4):
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        elif hasattr(model, "module") and isinstance(getattr(model, "module"), nn.Module):
            model = model.module
        else:
            break
    return model


def _atomic_save(payload, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, str(path))


def _cpu_copy(v: torch.Tensor) -> torch.Tensor:
    """Detached CPU copy, fp32 for floats. Always a COPY, never an alias.

    `.to("cpu")` is a no-op when the tensor is ALREADY on the CPU, so on a
    CPU-only run the naive version hands the live buffer to the background
    writer, which then serializes it while training keeps mutating it."""
    out = v.detach().to("cpu", dtype=torch.float32) if v.is_floating_point() \
        else v.detach().cpu()
    return out.clone() if out.data_ptr() == v.data_ptr() else out


def epoch_weights_payload(model: nn.Module, epoch: int) -> dict:
    """What save_epoch_weights writes, WITHOUT writing it.

    Split out so the training loop can take the snapshot synchronously — it
    has to read the parameters before the next epoch's optimizer.step()
    moves them — and hand the torch.save itself to the background writer
    (ckpt_writer.AsyncCheckpointWriter)."""
    inner = unwrap_model(model)
    sd = {k: _cpu_copy(v) for k, v in inner.state_dict().items()}
    return {"format": "epoch_weights", "epoch": int(epoch),
            "model_state_dict": sd}


def save_epoch_weights(path: str | Path, model: nn.Module, epoch: int):
    """Raw (unaveraged) fp32 model-only snapshot for one epoch."""
    _atomic_save(epoch_weights_payload(model, epoch), path)


def available_cpu_count() -> int:
    """Cores usable by this process: the CPU affinity mask where available
    (a batch scheduler may allocate only part of a node, while
    os.cpu_count() reports the whole machine), else os.cpu_count().
    sched_getaffinity does not exist on Windows."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _accumulate(total: dict[str, torch.Tensor], nonfloat: dict[str, torch.Tensor],
                sd: dict[str, torch.Tensor], sign: float) -> None:
    """total[k] += sign * sd[k] for every floating tensor (fp32 running sum).
    Non-floating buffers are unconditionally overwritten when sign > 0 (this
    snapshot becomes the newest one seen) and left untouched when sign < 0:
    eviction only ever removes the OLDEST snapshot in a window, never the
    one "latest" is tracking, so there is nothing to undo there."""
    for k, v in sd.items():
        if v.is_floating_point():
            vf = v.to(dtype=torch.float32)
            if k not in total:
                total[k] = vf.clone() if sign > 0 else -vf
            else:
                total[k] = total[k] + sign * vf
        elif sign > 0:
            nonfloat[k] = v.clone()


class _TrailingWindowAccumulator:
    """Running average over a trailing window of state dicts, fed one at a
    time via push() in strictly increasing (epoch) order.

    width=None means "accumulate everything pushed, never evict" — for a
    family with a single target at the very end of the scan (e.g. the final
    SWA average, which only ever starts accumulating once the scan reaches
    swa_start and is read out once at num_epochs). A finite width covers
    both the growing phase (fewer than `width` snapshots pushed so far —
    eviction never triggers, so the running sum is naturally "everything so
    far") and the steady sliding phase, and width=1 is the degenerate
    "latest snapshot only, no averaging" case — all three are the same
    push()/average() calls, no special-casing needed.
    """

    def __init__(self, width: int | None):
        self.width = width
        self.buf: deque[dict[str, torch.Tensor]] = deque()
        self.total: dict[str, torch.Tensor] = {}
        self.nonfloat: dict[str, torch.Tensor] = {}
        self.count = 0

    def push(self, sd: dict[str, torch.Tensor]) -> None:
        if self.width is not None:
            if len(self.buf) == self.width:
                _accumulate(self.total, self.nonfloat, self.buf.popleft(), sign=-1.0)
                self.count -= 1
            self.buf.append(sd)  # kept only so a later push() can evict it
        _accumulate(self.total, self.nonfloat, sd, sign=1.0)
        self.count += 1

    def average(self) -> dict[str, torch.Tensor]:
        assert self.count > 0, "accumulator never received a snapshot"
        out = {k: v / self.count for k, v in self.total.items()}
        out.update(self.nonfloat)
        return out


def ordered_prefetch(pool, items, load_fn, lookahead: int):
    """Yield load_fn(item) for item in items, STRICTLY in order, with at
    most `lookahead` submitted-but-not-yet-consumed tasks in flight at any
    time.

    Deliberately NOT concurrent.futures.Executor.map(): map() submits every
    item to the pool immediately (CPython's Executor.map calls self.submit
    for every item up front, before returning), regardless of how fast the
    consumer drains results. That is fine when each result is small, but
    every result here is a full fp32 model state dict — if disk reads keep
    completing faster than this generator's consumer processes them, map()
    would let an unbounded number of already-loaded state dicts pile up in
    memory. This generator instead keeps a fixed-size window of in-flight
    futures: one new submission per consumed result.
    """
    it = iter(items)
    pending: deque = deque()
    for _ in range(lookahead):
        try:
            pending.append(pool.submit(load_fn, next(it)))
        except StopIteration:
            break
    for item in it:
        yield pending.popleft().result()
        pending.append(pool.submit(load_fn, item))
    while pending:
        yield pending.popleft().result()


def average_epoch_range(epoch_weights_dir: str | Path, lo: int, hi: int
                        ) -> dict[str, torch.Tensor]:
    """Average epoch_{lo..hi}.pt (inclusive) in fp32 — independently
    re-reading and re-summing every file in [lo, hi] on every call.

    Deliberately naive reference implementation (O(window width) disk reads
    and additions on EVERY call, nothing shared across calls): used by tests
    to check _TrailingWindowAccumulator's output, never by the training
    loop itself — calling this once per target checkpoint is exactly the
    pattern _TrailingWindowAccumulator replaces (see module docstring).

    Non-floating buffers take the LAST file's value (matches
    _TrailingWindowAccumulator's "keep latest" convention)."""
    epoch_weights_dir = Path(epoch_weights_dir)
    paths = [epoch_weights_dir / f"epoch_{n:04d}.pt" for n in range(lo, hi + 1)]
    missing = [p for p in paths if not p.exists()]
    assert not missing, f"missing epoch weight files for SWA window: {missing}"

    total: dict[str, torch.Tensor] = {}
    nonfloat: dict[str, torch.Tensor] = {}
    for p in paths:
        sd = torch.load(str(p), map_location="cpu", weights_only=True)["model_state_dict"]
        for k, v in sd.items():
            if v.is_floating_point():
                vf = v.to(dtype=torch.float32)
                total[k] = total.get(k, torch.zeros_like(vf)) + vf
            else:
                nonfloat[k] = v.clone()
    n = len(paths)
    out = {k: v / n for k, v in total.items()}
    out.update(nonfloat)
    return out
