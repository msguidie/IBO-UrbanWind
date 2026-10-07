"""Checkpoints.

Resume state is two files per epoch:
  - epoch_weights/epoch_{N:04d}.pt (swa.py): fp32 model weights, also the
    addends of the final SWA average;
  - full_epoch{N:04d}.pt (this module): optimizer, scheduler and epoch.
A resume needs both halves of the same epoch. Older epochs are pruned as
new ones are written, except those inside the SWA window.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import torch

from .swa import unwrap_model

_FULL_EPOCH_RE = re.compile(r"^full_epoch(\d+)\.pt$")


def find_latest_trainer_state(ckpt_dir: str | Path,
                              epoch_weights_dir: str | Path | None = None
                              ) -> tuple[int, Path] | None:
    """Newest RESUMABLE point in ckpt_dir as (epoch, trainer_state_path), or
    None if there is nothing to resume from.

    A resume needs BOTH halves in sync — full_epoch{N}.pt (optimizer +
    scheduler, this module) and epoch_weights/epoch_{N}.pt (model weights,
    swa.save_epoch_weights) — because they are written by two separate calls
    and a job killed between them leaves a half-pair. So candidates are
    walked newest-first and the first one whose weights file actually exists
    wins; a lone full_epoch{N}.pt is skipped rather than trusted (it would
    fail later inside load_full_checkpoint with a FileNotFoundError, after
    the packs were already loaded — minutes of wasted node time on an
    unattended job).

    Epoch numbers come from the FILENAME, not from sorting: zero-padding
    makes lexical order agree with numeric order today, but the pairing is
    what matters here and reading the number explicitly makes that
    independent of the padding width."""
    ckpt_dir = Path(ckpt_dir)
    weights_dir = Path(epoch_weights_dir) if epoch_weights_dir is not None \
        else ckpt_dir / "epoch_weights"
    cands: list[tuple[int, Path]] = []
    for p in ckpt_dir.glob("full_epoch*.pt"):
        m = _FULL_EPOCH_RE.match(p.name)
        if m:
            cands.append((int(m.group(1)), p))
    for epoch, path in sorted(cands, reverse=True):
        if (weights_dir / f"epoch_{epoch:04d}.pt").exists():
            return epoch, path
    return None


def _atomic_save(payload, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, str(path))


def _state_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _state_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        seq = [_state_to_cpu(v) for v in obj]
        return type(obj)(seq) if isinstance(obj, tuple) else seq
    return obj


#: Public name for the tmp+rename write, so ckpt_writer can queue it by
#: reference. The atomicity is what lets the background writer be safe: a
#: killed job leaves either the old file or the new one, never a torn .pt.
write_payload = _atomic_save


def trainer_state_payload(optimizer, scheduler, epoch: int,
                          world_size: int | None = None,
                          batch_size: int | None = None,
                          shard_hash: str | None = None,
                          recipe: str | None = None,
                          profile: str | None = None,
                          grad_accum: int | None = None,
                          config_hash: str | None = None,
                          global_opt_step: int | None = None) -> dict:
    """What save_trainer_state writes, WITHOUT writing it — see
    swa.epoch_weights_payload for why the two halves are split. _state_to_cpu
    already deep-copies every tensor, so the result is safe to hand to
    another thread while training continues.

    recipe/profile/grad_accum/config_hash are identity fields: resuming
    refuses a mismatch, so a checkpoint never continues as a different
    model, hardware profile, step geometry or experiment."""
    return {
        "format": "trainer_state",
        "epoch": int(epoch),
        "world_size": int(world_size) if world_size is not None else None,
        "batch_size": int(batch_size) if batch_size is not None else None,
        "shard_hash": shard_hash,
        "recipe": recipe,
        "profile": profile,
        "grad_accum": int(grad_accum) if grad_accum is not None else None,
        "config_hash": config_hash,
        "global_opt_step": int(global_opt_step)
        if global_opt_step is not None else None,
        "optimizer_state_dict": _state_to_cpu(optimizer.state_dict()),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
    }


def prune_epoch_weights(epoch_weights_dir: Path, epoch: int,
                         rolling_keep: int, swa_start: int) -> None:
    """Delete the epoch weight files this run can no longer need.

    Two populations, two reasons to keep a file, and nothing else:

      - epoch >= swa_start: an ADDEND of the final SWA average. Every one
        of them survives to the end of the loop, because the average is
        only formed there.
      - the newest `rolling_keep` epochs: the model half of a RESUME PAIR.
        checkpoint.prune_rolling keeps that many full_epoch{N}.pt (the
        optimizer/scheduler half) and find_latest_trainer_state only trusts
        an N whose weights file is also present, so the two prunes must
        agree exactly — keeping fewer weights than trainer states would
        silently push a resume back an epoch at a time, keeping more would
        just be dead bytes.

    Everything else is deleted as soon as it is passed, so the disk holds
    at most rolling_keep + (num_epochs - swa_start + 1) weight files.
    """
    if rolling_keep <= 0:
        return
    cutoff = epoch - rolling_keep          # keep (cutoff, epoch]
    for old in epoch_weights_dir.glob("epoch_*.pt"):
        try:
            n = int(old.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        if n <= cutoff and n < swa_start:
            try:
                old.unlink()
            except OSError:
                pass


def write_epoch_artifacts(weights_payload: dict, weights_path,
                          state_payload: dict, state_path,
                          ckpt_dir, rolling_keep: int,
                          epoch: int, swa_start: int) -> None:
    """One epoch's entire disk footprint, in the ONE order a resume can trust:
    model weights, then the trainer state that pairs with them, then the
    prune. find_latest_trainer_state walks full_epoch*.pt newest-first and
    skips any whose weights file is missing, so this order makes an
    interrupted write degrade to "resume from N-1" instead of a hard failure.

    Submitted to the background writer as a SINGLE unit so its queue depth
    counts epochs rather than files."""
    write_payload(weights_payload, weights_path)
    write_payload(state_payload, state_path)
    prune_rolling(ckpt_dir, rolling_keep)
    prune_epoch_weights(Path(weights_path).parent, epoch, rolling_keep,
                        swa_start)


def save_trainer_state(path, optimizer, scheduler, epoch: int,
                       world_size: int | None = None,
                       batch_size: int | None = None,
                       shard_hash: str | None = None):
    """Optimizer + scheduler + epoch only — no model weights (those live in
    the matching epoch_weights/epoch_{epoch}.pt file). world_size/batch_size
    are recorded so --resume can refuse a mismatched relaunch: the LR
    schedule's step geometry is steps_per_epoch = f(shard size, batch_size),
    which changes with either, so resuming a world-4 run on 3 GPUs would
    silently continue with a different schedule. shard_hash
    covers ONLY the pool manifest's per-rank estate-name assignment —
    resuming against a re-partitioned file would silently change both the
    data each rank sees and steps_per_epoch; predicted byte numbers are
    deliberately NOT covered (advisory, regenerable)."""
    _atomic_save(trainer_state_payload(optimizer, scheduler, epoch,
                                       world_size=world_size,
                                       batch_size=batch_size,
                                       shard_hash=shard_hash), path)


def verify_swa_payload(path: str | Path, expected_identity: dict) -> dict:
    """Load swa_model.pt and check that it is the complete SWA of the
    expected run (recipe, config hash, epochs, window). Raises on a missing,
    corrupt or mismatched file. Returns the payload."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — evaluation reads swa_model.pt; finish "
            f"training first.")
    try:
        payload = torch.load(str(path), map_location="cpu",
                             weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"{path}: swa_model.pt failed to load "
                           f"({type(exc).__name__}: {exc})") from exc
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise RuntimeError(f"{path}: not an SWA payload (no "
                           f"model_state_dict) — refusing")
    fmt = payload.get("format")
    if fmt == "swa_model":
        checks = ("recipe", "family", "config_hash", "num_epochs",
                  "swa_window", "mode")
        for key in checks:
            exp = expected_identity.get(key)
            got = payload.get(key)
            if exp is not None and got != exp:
                raise RuntimeError(
                    f"{path}: SWA identity mismatch on {key!r}: payload has "
                    f"{got!r}, this run expects {exp!r}")
        lo, hi = payload["swa_window"]
        if payload.get("swa_count") != hi - lo + 1:
            raise RuntimeError(
                f"{path}: SWA window incomplete — payload averaged "
                f"{payload.get('swa_count')} snapshots over "
                f"[{lo}, {hi}] (expected {hi - lo + 1})")
        return payload
    raise RuntimeError(
        f"{path}: payload format {fmt!r} is not an SWA payload written by "
        f"the trainer")


def load_state_strict(model: torch.nn.Module, sd: dict) -> None:
    """Upcast floating tensors to fp32, then load with strict=True."""
    sd = {k: (v.float() if isinstance(v, torch.Tensor)
              and v.is_floating_point() else v) for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)


def load_model_weights(path, model):
    """Strictly load model weights from a checkpoint file (a payload with
    "model_state_dict", or a bare state dict)."""
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and \
        "model_state_dict" in ckpt else ckpt
    load_state_strict(unwrap_model(model), sd)
    return ckpt.get("epoch") if isinstance(ckpt, dict) else None


def load_full_checkpoint(path, model, optimizer, scheduler,
                         epoch_weights_dir: str | Path,
                         world_size: int | None = None,
                         batch_size: int | None = None,
                         shard_hash: str | None = None,
                         recipe: str | None = None,
                         profile: str | None = None,
                         grad_accum: int | None = None,
                         config_hash: str | None = None) -> int:
    """Resume: trainer-state file (optimizer/scheduler/epoch) + the matching
    per-epoch model weights file. Only a trainer_state file carries the
    optimizer and scheduler.
    Also rejects a world_size/batch_size mismatch when both the checkpoint
    and the caller carry the value (steps_per_epoch — and with it the whole
    LR schedule geometry — depends on both; see save_trainer_state)."""
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    if ckpt.get("format") != "trainer_state" or "optimizer_state_dict" not in ckpt:
        raise ValueError(f"{path} is not a trainer-state checkpoint — cannot --resume from it")
    for key, current in (("world_size", world_size), ("batch_size", batch_size),
                         ("grad_accum", grad_accum)):
        saved = ckpt.get(key)
        if saved is not None and current is not None and int(saved) != int(current):
            raise ValueError(
                f"--resume {key} mismatch: checkpoint was written at "
                f"{key}={saved}, this launch has {key}={current}. "
                f"steps_per_epoch (and the LR schedule) depend on it — "
                f"relaunch with the original {key}.")
    for key, current in (("recipe", recipe), ("profile", profile),
                         ("config_hash", config_hash)):
        saved = ckpt.get(key)
        if saved is not None and current is not None and \
                str(saved) != str(current):
            raise ValueError(
                f"--resume {key} mismatch: checkpoint was written with "
                f"{key}={saved!r}, this launch has {key}={current!r}. "
                f"Cross-{key} resumes are refused — start a fresh run.")
    saved_hash = ckpt.get("shard_hash")
    if saved_hash is not None and shard_hash is not None \
            and saved_hash != shard_hash:
        raise ValueError(
            "--resume shard mismatch: the pool manifest was re-partitioned "
            "since this checkpoint was written (assignment hash differs). "
            "Resuming would silently change each rank's data and "
            "steps_per_epoch — restore the original manifest or "
            "start a fresh run.")
    epoch = int(ckpt["epoch"])
    weights_path = Path(epoch_weights_dir) / f"epoch_{epoch:04d}.pt"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"resume needs matching model weights at {weights_path} "
            f"(epoch {epoch}, from {path}) — was epoch_weights/ deleted?")
    w = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    load_state_strict(unwrap_model(model), w["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return epoch


def prune_rolling(ckpt_dir: str | Path, keep: int):
    fulls = sorted(Path(ckpt_dir).glob("full_epoch*.pt"))
    for old in fulls[:-keep] if keep > 0 else fulls:
        try:
            old.unlink()
        except OSError:
            pass
