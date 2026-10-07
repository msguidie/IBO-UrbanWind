"""The trainer: one loop for every recipe in the registry.

Everything model-specific arrives through the Recipe (src/recipes): the
resident dataset over the estate packs, the collate, the forward + loss, the
optimizer and scheduler at the model's own granularity, and the compile
plan. The loop owns what must be identical across models:

  - DDP (compile first, then wrap), bf16 autocast without a GradScaler,
    gradient clipping at 1.0;
  - gradient accumulation (training.grad_accum, resolved from the recipe's
    frozen effective batch and the launch world size; DDP allreduce is
    suppressed on non-boundary microsteps via no_sync, loss scaled by
    1/accum so the accumulated gradient is the effective-batch mean);
  - the LR scheduler stepped at the recipe's declared granularity ("step" =
    per optimizer update, "epoch" = per epoch), with a resume-time position
    assert (scheduler.last_epoch must equal the resumed unit count — a
    drifted schedule fails at resume, not 300 epochs later);
  - the async checkpoint writer (ckpt_writer.py): the GPU->CPU snapshot is
    synchronous, the torch.save + prune runs on the background thread, so
    disk I/O overlaps the next epoch's GPU work; writer errors re-raise on
    the training thread;
  - every-epoch epoch_weights snapshots (SWA addends + resume pairs, pruned
    as written), every-epoch trainer state (recipe/profile/config-hash
    identity fields refused on mismatched resume), the .train.lock,
    --auto-resume, --max-hours epoch-boundary stop + .needs_resume;
  - the disk-based SWA over the last 25% window, VERIFIED by a strict load
    into a fresh model instance BEFORE the window is deleted;
  - run telemetry recorded into cfg.run (standardized peak training memory
    after warmup + the first full optimizer update; exact per-optimizer-
    step train FLOPs from a metered eager forward + the recipe's manual
    non-Linear terms).

Training only trains; `main.py post --run-dir <run>` evaluates a finished
run separately, rebuilding the model from its config_used.yaml.
"""
from __future__ import annotations

import gc
import json
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ..config import Config, config_hash
from ..data.dataloader import CudaPrefetcher, RESIDENT_MAX_WORKERS, \
    build_dataloader
from ..data.estate import SURF_CHANNEL_NAMES, VOL_CHANNEL_NAMES
from ..evaluation.flops import FlopMeter
from ..manifest import dirs_by_estate, load_pool_manifest, \
    train_estates_for_rank
from ..recipes.base import Recipe, cpu_defloat_bf16
from ..utils.seed import (effective_train_seed, seed_everything,
                          train_seed_salt)
from .checkpoint import (find_latest_trainer_state, load_full_checkpoint,
                         load_model_weights, load_state_strict,
                         trainer_state_payload, write_epoch_artifacts)
from .ckpt_writer import AsyncCheckpointWriter
from .ddp import all_reduce_sum, barrier, cleanup_ddp, init_ddp, \
    is_distributed
from .profiler import EpochMonitor, StepProfiler
from .swa import (_atomic_save, _TrailingWindowAccumulator,
                  available_cpu_count, epoch_weights_payload, ordered_prefetch)

NEEDS_RESUME_MARKER = ".needs_resume"
TRAINING_COMPLETE_EVENT = "training_complete"


def read_training_complete_record(ckpt_dir: Path) -> dict | None:
    """Last training_complete event in training_log.jsonl, or None."""
    log_path = Path(ckpt_dir) / "training_log.jsonl"
    if not log_path.exists():
        return None
    record = None
    try:
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("event") == TRAINING_COMPLETE_EVENT:
                    record = entry
    except OSError:
        return None
    return record


def training_complete_strict(cfg: Config, recipe: Recipe, ckpt_dir: Path
                             ) -> bool:
    """Training counts as COMPLETE only when ALL of:
      - swa_model.pt exists, is a legitimate identity-stamped SWA for this
        exact config (verify_swa_payload), and STRICT-LOADS into a fresh
        model instance;
      - training_log.jsonl carries a training_complete record whose
        identity (config_hash, recipe, epochs) matches.
    Returns True only on full validation. A missing SWA with no complete
    record returns False (not started / interrupted). Anything half-present
    or mismatched RAISES — a partially complete run must be looked at, not
    silently restarted or silently skipped."""
    from .checkpoint import verify_swa_payload
    swa_path = ckpt_dir / "swa_model.pt"
    record = read_training_complete_record(ckpt_dir)
    if not swa_path.exists() and record is None:
        return False
    cfg_hash = cfg.run.config_hash or config_hash(cfg)
    identity = {"recipe": recipe.name, "family": recipe.family,
                "config_hash": cfg_hash, "mode": cfg.run.mode,
                "num_epochs": int(cfg.training.num_epochs),
                "swa_window": [int(cfg.training.swa_start),
                               int(cfg.training.num_epochs)]}
    payload = verify_swa_payload(swa_path, identity)   # raises on mismatch
    fresh = recipe.build_model(cfg)
    load_state_strict(fresh, payload["model_state_dict"])
    del fresh
    if record is None:
        raise RuntimeError(
            f"{ckpt_dir}: swa_model.pt verifies but training_log.jsonl has "
            f"no {TRAINING_COMPLETE_EVENT} record — the completion write was "
            f"interrupted. Inspect the run; do not silently continue.")
    for key, exp in (("config_hash", cfg_hash), ("recipe", recipe.name),
                     ("epochs", int(cfg.training.num_epochs))):
        if record.get(key) != exp:
            raise RuntimeError(
                f"{ckpt_dir}: training_complete record mismatch on {key!r} "
                f"({record.get(key)!r} != {exp!r}) — refusing to treat this "
                f"run as complete")
    return True


# ---------------------------------------------------------------------------
# SWA (disk-based; identity-stamped payload; BatchNorm recalibrated on real
# data BEFORE the final save; verified strict-load before the window and
# every other training artifact are deleted)
# ---------------------------------------------------------------------------

def swa_identity(cfg: Config, recipe: Recipe, swa_start: int) -> dict:
    """The identity stamped into swa_model.pt and re-verified by every
    consumer (evaluation, auto-resume completion)."""
    return {
        "recipe": recipe.name, "family": recipe.family,
        "config_hash": cfg.run.config_hash or config_hash(cfg),
        "code_digest": cfg.run.code_digest,
        "mode": cfg.run.mode, "profile": cfg.hardware.profile,
        "world_size": int(cfg.run.world_size),
        "effective_batch": int(cfg.run.effective_batch),
        "num_epochs": int(cfg.training.num_epochs),
        "swa_window": [int(swa_start), int(cfg.training.num_epochs)],
    }


def _compute_swa_checkpoint(cfg: Config, recipe: Recipe, ckpt_dir: Path,
                            epoch_weights_dir: Path, swa_start: int,
                            recalib_batches=None, device=None,
                            end_epoch: int | None = None) -> Path:
    """Average the window, RECALIBRATE BatchNorm running stats on real
    training batches (models that carry BN — Erwin; a plain average of old
    running buffers is never published), stamp the run identity into the
    payload, strict-load-verify into a fresh model, and only then delete the
    window. Returns the swa path. Raises (leaving everything on disk) on any
    failure — there is no fallback."""
    tc = cfg.training
    end = int(end_epoch) if end_epoch is not None else int(tc.num_epochs)
    swa_path = ckpt_dir / "swa_model.pt"
    if not (tc.swa_enabled and swa_start <= end):
        raise RuntimeError(
            "SWA is required: swa_enabled is off or the window is empty — "
            "check training.num_epochs / swa_fraction.")

    epochs = range(swa_start, end + 1)
    missing = [e for e in epochs
               if not (epoch_weights_dir / f"epoch_{e:04d}.pt").exists()]
    assert not missing, (
        f"missing epoch weight files inside the SWA window "
        f"[{swa_start}, {end}]: {missing[:5]}"
        f"{'...' if len(missing) > 5 else ''}")

    def load(epoch: int) -> dict:
        path = epoch_weights_dir / f"epoch_{epoch:04d}.pt"
        return torch.load(str(path), map_location="cpu",
                          weights_only=True)["model_state_dict"]

    n_threads = min(32, max(4, available_cpu_count()))
    fam = _TrailingWindowAccumulator(None)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        for sd in ordered_prefetch(pool, epochs, load, n_threads):
            fam.push(sd)
    averaged = fam.average()

    # ---- BatchNorm recalibration (Erwin) ---------------------------------
    fresh = recipe.build_model(cfg)
    load_state_strict(fresh, averaged)
    has_bn = any(isinstance(m, nn.modules.batchnorm._BatchNorm)
                 for m in fresh.modules())
    bn_recalibrated = False
    if has_bn:
        if recalib_batches is None:
            raise RuntimeError(
                "model carries BatchNorm but no recalibration data was "
                "provided — refusing to publish an SWA whose running stats "
                "are a plain average of stale buffers")
        dev = device if device is not None else torch.device("cpu")
        fresh = fresh.to(dev)
        n_bn = _bn_recalibrate(fresh, recipe, cfg, recalib_batches, dev)
        fresh = fresh.to("cpu")
        bn_recalibrated = True
        print(f"[train] SWA BatchNorm recalibration: {n_bn} BN modules, "
              f"running stats recomputed on real training batches",
              flush=True)
        averaged = {k: (v.detach().to("cpu", dtype=torch.float32)
                        if v.is_floating_point() else v.detach().cpu())
                    for k, v in fresh.state_dict().items()}
    del fresh

    identity = swa_identity(cfg, recipe, swa_start)
    identity["swa_window"] = [int(swa_start), end]
    payload = {"format": "swa_model", "model_state_dict": averaged,
               "swa_count": fam.count, "epoch": end,
               "bn_recalibrated": bn_recalibrated, **identity}
    _atomic_save(payload, swa_path)
    print(f"[train] SWA over epochs [{swa_start}, {end}] "
          f"({fam.count} snapshots) -> {swa_path.name} in "
          f"{time.time() - t0:.1f}s", flush=True)

    # VERIFY before deleting the window: the payload must strict-load into a
    # SECOND fresh model instance of THIS run's config. If this raises, the
    # window stays on disk for diagnosis — nothing is lost.
    verify = recipe.build_model(cfg)
    load_model_weights(swa_path, verify)
    del verify
    print("[train] swa_model.pt verified: strict load into a fresh model "
          "instance OK", flush=True)

    n_removed = 0
    for old in epoch_weights_dir.glob("epoch_*.pt"):
        try:
            old.unlink()
            n_removed += 1
        except OSError:
            pass
    try:
        epoch_weights_dir.rmdir()
    except OSError:
        pass
    print(f"[train] removed {n_removed} epoch snapshots; swa_model.pt is the "
          f"only model weights left", flush=True)
    return swa_path


def _bn_recalibrate(model, recipe: Recipe, cfg: Config, batches, dev) -> int:
    """torch.optim.swa_utils.update_bn semantics on the recipe's own forward:
    reset running stats, momentum=None (cumulative average), one no-grad
    train-mode pass over real batches."""
    bns = [m for m in model.modules()
           if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    momenta = {}
    for m in bns:
        m.reset_running_stats()
        momenta[m] = m.momentum
        m.momentum = None
    was_training = model.training
    model.train()
    use_cuda = dev.type == "cuda"
    with torch.no_grad():
        for batch in batches:
            batch = recipe.batch_to_device(batch, dev)
            if not use_cuda:
                batch = cpu_defloat_bf16(batch)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=use_cuda):
                recipe.forward_loss(model, batch, cfg)
    for m in bns:
        m.momentum = momenta[m]
    if not was_training:
        model.eval()
    return len(bns)


# ---------------------------------------------------------------------------
# run_dir guards (lock + stale takeover)
# ---------------------------------------------------------------------------

def _lock_owner_desc(info: dict) -> str:
    return (f"pid={info.get('pid')} host={info.get('host')} "
            f"job={info.get('job_id') or '-'}")


def _lock_owner_is_alive(info: dict) -> bool | None:
    if info.get("host") != socket.gethostname():
        return None
    pid = info.get("pid")
    if not isinstance(pid, int):
        return None
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None


def _read_run_lock(lock_path: Path) -> dict:
    try:
        with open(lock_path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_run_lock(lock_path: Path) -> None:
    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, json.dumps({
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "job_id": os.environ.get("PBS_JOBID") or os.environ.get("SLURM_JOB_ID"),
            "started": time.time(),
        }).encode())
    finally:
        os.close(fd)


def _acquire_run_lock(ckpt_dir: Path, steal_stale: bool = False) -> bool:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    lock_path = ckpt_dir / ".train.lock"
    try:
        _write_run_lock(lock_path)
        return True
    except FileExistsError:
        pass
    info = _read_run_lock(lock_path)
    alive = _lock_owner_is_alive(info)
    if alive is True:
        print(f"[train] lock held by a LIVE process ({_lock_owner_desc(info)})",
              flush=True)
        return False
    if not steal_stale:
        print(f"[train] lock present ({_lock_owner_desc(info)}); owner is "
              f"{'dead' if alive is False else 'on another host, undecidable'}."
              f" Pass --auto-resume to take over a stale lock automatically.",
              flush=True)
        return False
    print(f"[train] STALE lock taken over: previous owner "
          f"{_lock_owner_desc(info)} is "
          f"{'dead' if alive is False else 'on another host and not this job'}"
          f" — the expected state after a walltime/OOM kill.", flush=True)
    lock_path.unlink(missing_ok=True)
    try:
        _write_run_lock(lock_path)
        return True
    except FileExistsError:
        return False


def _release_run_lock(ckpt_dir: Path) -> None:
    (ckpt_dir / ".train.lock").unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# the trainer
# ---------------------------------------------------------------------------

def train(cfg: Config, recipe: Recipe, resume_path: str | None = None,
          cleanup: bool = True, auto_resume: bool = False,
          max_hours: float | None = None) -> int:
    """Resume resolution, run_dir guards and a guaranteed lock release around
    the training loop. Returns this process's rank."""
    t0_wall = time.time()
    rank, world, local = init_ddp()
    ckpt_dir = Path(cfg.checkpoint.save_dir)

    if auto_resume and resume_path is None:
        found = find_latest_trainer_state(ckpt_dir, ckpt_dir / "epoch_weights")
        if found is not None:
            last_epoch, resume_path = found[0], str(found[1])
            if rank == 0:
                print(f"[train] --auto-resume: continuing from epoch "
                      f"{last_epoch} ({Path(resume_path).name})", flush=True)
        elif training_complete_strict(cfg, recipe, ckpt_dir):
            # completion is VERIFIED, never inferred from a marker file:
            # identity-stamped SWA strict-loaded + the training_complete
            # record matched. Half-present state raises inside the check.
            if rank == 0:
                print(f"[train] --auto-resume: {ckpt_dir} verified complete "
                      f"(SWA identity + strict load + complete record), "
                      f"nothing to do", flush=True)
            if cleanup:
                cleanup_ddp()
            return rank
        elif rank == 0:
            print(f"[train] --auto-resume: no resumable checkpoint in "
                  f"{ckpt_dir}, starting fresh", flush=True)

    if resume_path is None and (ckpt_dir / "swa_model.pt").exists():
        raise RuntimeError(
            f"{ckpt_dir} already holds swa_model.pt — a completed run. Use a "
            f"fresh --run-dir, or --auto-resume for a verified no-op.")
    if rank == 0 and not _acquire_run_lock(ckpt_dir, steal_stale=auto_resume):
        raise RuntimeError(
            f"{ckpt_dir}/.train.lock exists and its owner is alive — "
            f"another process is using this run_dir.")
    try:
        barrier()
        return _train_impl(cfg, recipe, resume_path, cleanup, rank, world,
                           local, ckpt_dir, max_hours=max_hours,
                           t0_wall=t0_wall)
    finally:
        if rank == 0:
            _release_run_lock(ckpt_dir)


def _meter_train_flops(cfg: Config, recipe: Recipe, model, dataset,
                       device, use_cuda: bool) -> float:
    """One metered EAGER forward on a real sample -> exact
    Linear/Conv/norm/activation FLOPs + the recipe's manual non-hookable
    terms; scaled to a full optimizer update (fwd + 2x bwd, every microstep,
    every rank). With activation checkpointing ON, the recompute that
    ACTUALLY happens in backward (one extra forward of the checkpointed
    blocks + their manual terms) is added — the paper number reports work
    executed, not work idealized away."""
    batch = recipe.collate()([dataset[(i, 1)] for i
                              in range(cfg.training.batch_size)])
    batch = recipe.batch_to_device(batch, device)
    if not use_cuda:
        batch = cpu_defloat_bf16(batch)
    was_training = model.training
    model.eval()      # BN running stats must not move during the meter pass
    inner = model
    from .swa import unwrap_model
    inner = unwrap_model(model)
    ckpt_mods = recipe.checkpointed_modules(inner) \
        if cfg.training.activation_checkpointing else []
    block_meter_ctx = FlopMeter(nn.ModuleList(ckpt_mods)) if ckpt_mods \
        else None
    with FlopMeter(model) as meter, torch.no_grad():
        ctx = block_meter_ctx if block_meter_ctx is not None \
            else nullcontext()
        with ctx:
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=use_cuda):
                recipe.forward_loss(model, batch, cfg)
        manual_sum = 0.0
        for name, val in recipe.manual_train_flops(cfg).items():
            meter.add(name, val * cfg.training.batch_size)
            manual_sum += val * cfg.training.batch_size
        fwd = meter.total
        breakdown = meter.breakdown()
    recompute = 0.0
    if block_meter_ctx is not None:
        # the checkpointed blocks run forward a SECOND time in backward;
        # their manual terms (attention/segment math inside the blocks) run
        # again with them — the recipes that support checkpointing keep all
        # their manual terms inside the checkpointed blocks by construction
        recompute = block_meter_ctx.total + manual_sum
        breakdown["ckpt_recompute_fwd"] = recompute
    per_opt_step = (fwd * 3.0 + recompute) * cfg.training.grad_accum \
        * max(cfg.run.world_size, 1)
    breakdown["per_opt_step"] = per_opt_step
    breakdown["scaling"] = {"fwd_bwd_multiplier": 3.0,
                            "ckpt_recompute": recompute,
                            "grad_accum": cfg.training.grad_accum,
                            "world_size": cfg.run.world_size}
    if was_training:
        model.train()
    del batch
    return per_opt_step, breakdown


def _train_impl(cfg: Config, recipe: Recipe, resume_path: str | None,
                cleanup: bool, rank: int, world: int, local: int,
                ckpt_dir: Path, max_hours: float | None = None,
                t0_wall: float | None = None) -> int:
    # The TRAINING seed, not the top-level (preprocessing) one — identical
    # unless training.seed is set; see utils.seed.effective_train_seed.
    train_seed = effective_train_seed(cfg)
    seed_everything(train_seed)
    if rank == 0:
        salt = train_seed_salt(cfg)
        print(f"[train] training seed {train_seed} (top-level seed "
              f"{cfg.seed}); per-step draws "
              + (f"salted '{salt}'" if salt else "UNSALTED (base stream)"),
              flush=True)
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{local}") if use_cuda else torch.device("cpu")
    cfg.training.world_size = world
    tc = cfg.training
    accum = int(tc.grad_accum)

    pool = load_pool_manifest(cfg.data.pool_manifest_path, world=world)
    recipe.check_pool(cfg, pool)
    estate_shard = train_estates_for_rank(pool, rank)
    shard_hash = pool["assignment_sha256"]
    from ..manifest import roster_sha256
    roster_hash = roster_sha256(pool)
    # the manifest identity is part of the config hash — it must be bound
    # BEFORE hashing, and a persisted value must match the file on disk
    for field_name, live in (("manifest_hash", shard_hash),
                             ("roster_hash", roster_hash)):
        recorded = getattr(cfg.run, field_name)
        if recorded and recorded != live:
            raise RuntimeError(
                f"pool manifest changed since this run was configured: "
                f"{field_name} recorded {recorded} but the file now hashes "
                f"to {live} — restore the original manifest or start a "
                f"fresh run dir")
        setattr(cfg.run, field_name, live)
    cfg_hash = cfg.run.config_hash or config_hash(cfg)
    if cfg.run.config_hash and config_hash(cfg) != cfg.run.config_hash:
        raise RuntimeError(
            f"config identity drifted between configuration and training: "
            f"recorded {cfg.run.config_hash}, recomputed {config_hash(cfg)}")
    cfg.run.config_hash = cfg_hash
    if rank == 0:
        print(f"[train] recipe={recipe.name} (family {recipe.family}) "
              f"profile={cfg.hardware.profile} pool={pool['pool']}\n"
              f"[train] grid={cfg.preprocess.kdtree_weighting} "
              f"({cfg.data.cache_estate_dir})\n"
              f"[train] {len(estate_shard)} estates/rank, "
              f"{pool['case_count_per_rank']} cases/rank (strict)",
              flush=True)

    num_workers = min(tc.num_workers, RESIDENT_MAX_WORKERS) \
        if tc.num_workers > 0 else 0
    train_ds = recipe.build_train_dataset(cfg, estate_shard,
                                          dirs_by_estate(pool))
    n_flat = len(train_ds)
    assert n_flat % tc.batch_size == 0, \
        "flat sample count must be a multiple of batch_size"
    micro_per_epoch = n_flat // tc.batch_size
    assert micro_per_epoch % accum == 0, (
        f"microsteps/epoch {micro_per_epoch} not divisible by grad_accum "
        f"{accum} — the effective-batch geometry does not tile this shard")
    opt_steps_per_epoch = micro_per_epoch // accum

    model = recipe.build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        print(f"[train] {recipe.tex_row_name}: {n_params:,} trainable "
              f"params ({n_params / 1e6:.2f}M)", flush=True)

    # exact train FLOPs from a metered eager pass (before compile — hooks
    # cannot see inside a compiled graph)
    flops_per_opt_step, flops_breakdown = _meter_train_flops(
        cfg, recipe, model, train_ds, device, use_cuda)
    cfg.run.train_flops_per_opt_step = float(flops_per_opt_step)
    if rank == 0:
        print(f"[train] train FLOPs/optimizer step = "
              f"{flops_per_opt_step / 1e12:.3f} TF "
              f"(fwd x3 x accum{accum} x world{world}; "
              f"linear metered exactly, manual terms: "
              f"{sorted(flops_breakdown['manual'])})", flush=True)
        _atomic_save({"format": "train_flops", **flops_breakdown},
                     ckpt_dir / ".train_flops.pt")
        # the breakdown also goes into training_log.jsonl (part of the final
        # trio) — .train_flops.pt is a transient debugging aid deleted at
        # completion
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with open(ckpt_dir / "training_log.jsonl", "a",
                  encoding="utf-8") as f:
            f.write(json.dumps({"event": "train_flops_breakdown",
                                **flops_breakdown}) + "\n")

    from ..utils.env import quiet_inductor_perf_hints
    quiet_inductor_perf_hints()   # per-subgraph cudagraph HINTS off; the
    #                               real status is verified, not assumed
    compile_mode, fullgraph, compile_note = recipe.compile_plan(cfg)
    # CUDA-Graph capture (reduce-overhead) combined with activation-
    # checkpoint recompute corrupts backward; refuse the combination
    if compile_mode == "reduce-overhead" and tc.activation_checkpointing:
        raise RuntimeError(
            f"recipe {recipe.name!r} requests reduce-overhead (CUDA Graphs) "
            f"WITH activation checkpointing — that combination breaks "
            f"backward. Return 'default' from compile_plan for training, or "
            f"disable checkpointing explicitly.")
    if getattr(recipe, "disable_ddp_optimizer", False):
        # torch._dynamo's DDPOptimizer cannot fake-split every graph; the
        # choice is recorded in config_used.yaml, never silent. Math is
        # unchanged (only the allreduce-overlap heuristic).
        torch._dynamo.config.optimize_ddp = False
        compile_note = compile_note + " | DDPOptimizer OFF"
    else:
        torch._dynamo.config.optimize_ddp = True
    cfg.run.compile_mode = compile_mode if use_cuda else ""
    cfg.run.compile_fullgraph = bool(fullgraph)
    cfg.run.compile_notes = compile_note
    if compile_mode and use_cuda:
        model = torch.compile(model, mode=compile_mode, fullgraph=fullgraph,
                              dynamic=False)
        if rank == 0:
            print(f"[train] torch.compile(mode={compile_mode!r}, "
                  f"fullgraph={fullgraph}) — {compile_note}", flush=True)
    elif rank == 0:
        print(f"[train] compile OFF ({'CPU run' if not use_cuda else compile_note})",
              flush=True)
    if is_distributed():
        model = DDP(model, device_ids=[local] if use_cuda else None,
                    static_graph=True, gradient_as_bucket_view=True,
                    find_unused_parameters=False, broadcast_buffers=False)

    optimizer, scheduler, sched_gran = recipe.build_optimizer(
        cfg, model, opt_steps_per_epoch)
    assert sched_gran in {"step", "epoch"}
    if rank == 0:
        print(f"[train] optimizer={type(optimizer).__name__} "
              f"lr={optimizer.param_groups[0]['lr']:.3g} scheduler/"
              f"{sched_gran}; eff batch {cfg.run.effective_batch} = "
              f"micro {tc.batch_size} x world {world} x accum {accum}; "
              f"{opt_steps_per_epoch} optimizer steps/epoch x "
              f"{tc.num_epochs} epochs", flush=True)

    swa_start = tc.swa_start if tc.swa_enabled else tc.num_epochs + 1
    if rank == 0 and tc.swa_enabled:
        print(f"[train] SWA window = epochs [{swa_start}, {tc.num_epochs}]",
              flush=True)

    epoch_weights_dir = ckpt_dir / "epoch_weights"
    start_epoch = 1
    if resume_path:
        last = load_full_checkpoint(
            resume_path, model, optimizer, scheduler, epoch_weights_dir,
            world_size=world, batch_size=tc.batch_size,
            shard_hash=shard_hash, recipe=recipe.name,
            profile=cfg.hardware.profile, grad_accum=accum,
            config_hash=cfg_hash)
        start_epoch = last + 1
        expected = (last * opt_steps_per_epoch) if sched_gran == "step" \
            else last
        got = int(scheduler.last_epoch)
        if got != expected:
            raise RuntimeError(
                f"resume scheduler position mismatch: last_epoch={got}, "
                f"expected {expected} ({sched_gran}-granularity, epoch "
                f"{last}) — the schedule geometry drifted; refusing")
        if rank == 0:
            print(f"[train] resumed from {resume_path} (epoch {last}; "
                  f"scheduler position verified)", flush=True)

    def autocast_ctx():
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda)

    collate_fn = recipe.collate()
    train_loader, train_sampler = build_dataloader(
        train_ds, tc.batch_size, num_workers, tc.prefetch_factor,
        seed=train_seed, shuffle=True, drop_last=True, collate_fn=collate_fn)

    if use_cuda and start_epoch <= tc.num_epochs:
        train_sampler.set_epoch(0)
        warm = next(CudaPrefetcher(train_loader, device))
        with autocast_ctx():
            loss, _ = recipe.forward_loss(model, warm, cfg)
        loss.backward()
        for p in model.parameters():
            p.grad = None
        torch.cuda.synchronize(device)
        del warm, loss
        barrier()
        if rank == 0:
            print("[train] compile warmup done", flush=True)
        # standardized peak-memory measurement starts AFTER warmup:
        # allocator pools / cudagraph buffers are live baseline by now
        torch.cuda.reset_peak_memory_stats(device)

    profiler = StepProfiler(ckpt_dir, cfg.profiling.first_window,
                            cfg.profiling.periodic_every,
                            cfg.profiling.periodic_len, enabled=(rank == 0))
    log_path = ckpt_dir / "training_log.jsonl"
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        epoch_weights_dir.mkdir(parents=True, exist_ok=True)

    dev = device if use_cuda else torch.device("cpu")
    global_opt_step = (start_epoch - 1) * opt_steps_per_epoch
    stopped_early = False
    # Peak training memory is measured once per run, after the first full
    # optimizer update (shapes are static, so any epoch gives the same
    # peak); a resumed run measures it if no number is recorded yet.
    mem_probe_pending = use_cuda and int(cfg.run.peak_train_mem_bytes) <= 0
    ckpt_writer = AsyncCheckpointWriter(depth=2, enabled=(rank == 0))
    model.train()
    n_vol_ch, n_surf_ch = len(VOL_CHANNEL_NAMES), len(SURF_CHANNEL_NAMES)

    def sync_ctx(is_boundary: bool):
        if is_distributed() and not is_boundary:
            return model.no_sync()
        return nullcontext()

    for epoch in range(start_epoch, tc.num_epochs + 1):
        t0 = time.time()
        vol_loss_sum = torch.zeros(n_vol_ch, dtype=torch.float64, device=dev)
        surf_loss_sum = torch.zeros(n_surf_ch, dtype=torch.float64, device=dev)
        n_vol = torch.zeros((), dtype=torch.float64, device=dev)
        n_surf = torch.zeros((), dtype=torch.float64, device=dev)
        gsum = torch.zeros((), device=dev)
        gmax = torch.zeros((), device=dev)

        train_sampler.set_epoch(epoch)
        it = CudaPrefetcher(train_loader, device) if use_cuda \
            else iter(train_loader)
        optimizer.zero_grad(set_to_none=True)
        for micro in range(micro_per_epoch):
            boundary = (micro + 1) % accum == 0
            active = profiler.active(global_opt_step)
            if active:
                profiler.data_wait_begin()
            batch = next(it)
            if not use_cuda:
                batch = recipe.batch_to_device(batch, device)
                batch = cpu_defloat_bf16(batch)
            if active:
                profiler.data_wait_end()
                profiler.step_begin(global_opt_step)
                profiler.mark_begin("forward")
            with sync_ctx(boundary):
                with autocast_ctx():
                    loss, br = recipe.forward_loss(model, batch, cfg)
                if active:
                    profiler.mark_end("forward")
                    profiler.mark_begin("backward")
                (loss / accum).backward()
            if active:
                profiler.mark_end("backward")
            if boundary:
                if active:
                    profiler.mark_begin("optim")
                # error_if_nonfinite: a NaN/Inf gradient aborts the run
                # BEFORE optimizer.step() — a poisoned update must never
                # reach the weights or a checkpoint
                gnorm = nn.utils.clip_grad_norm_(model.parameters(),
                                                 tc.max_grad_norm,
                                                 error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if sched_gran == "step":
                    scheduler.step()
                if active:
                    profiler.mark_end("optim")
                    profiler.step_end(global_opt_step)
                    if not profiler.active(global_opt_step + 1):
                        profiler.window_close(global_opt_step)
                gsum += gnorm.detach()
                gmax = torch.maximum(gmax, gnorm.detach())
                global_opt_step += 1
                if mem_probe_pending:
                    # The number: peak stats were reset AFTER compile
                    # warmup, so allocator pools and cudagraph buffers are
                    # baseline; by here the run has done every microstep of
                    # one FULL optimizer update AND the step() that
                    # allocates optimizer state. max_memory_allocated is a
                    # high-water mark over that whole window, so it
                    # captures the backward instant where activations and
                    # gradients coexist — not the (much lower) reading at
                    # this line, where zero_grad(set_to_none) has already
                    # freed the gradients.
                    torch.cuda.synchronize(device)
                    peak = torch.tensor(
                        float(torch.cuda.max_memory_allocated(device)),
                        device=dev)
                    if is_distributed():
                        torch.distributed.all_reduce(
                            peak, op=torch.distributed.ReduceOp.MAX)
                    cfg.run.peak_train_mem_bytes = int(peak.item())
                    cfg.run.peak_train_mem_profile = cfg.hardware.profile
                    if rank == 0:
                        print(f"[train] standardized peak train memory "
                              f"(profile {cfg.hardware.profile}, max over "
                              f"ranks): {peak.item() / 2**30:.2f} GiB",
                              flush=True)
                    mem_probe_pending = False

            vol_loss_sum += br["vol_loss_sum"].double()
            surf_loss_sum += br["surf_loss_sum"].double()
            n_vol += br["n_vol"].double()
            n_surf += br["n_surf"].double()

        if sched_gran == "epoch":
            scheduler.step()

        for t in (vol_loss_sum, surf_loss_sum, n_vol, n_surf):
            all_reduce_sum(t)

        # non-finite epoch guard on EVERY rank (consistent collective): a
        # NaN/Inf loss sum aborts BEFORE this epoch's checkpoint is written
        bad = torch.zeros((), dtype=torch.float64, device=dev)
        if not (bool(torch.isfinite(vol_loss_sum).all())
                and bool(torch.isfinite(surf_loss_sum).all())
                and bool(torch.isfinite(gmax))):
            bad += 1.0
        all_reduce_sum(bad)
        if float(bad) > 0.0:
            raise RuntimeError(
                f"epoch {epoch}: non-finite training loss detected "
                f"(vol={vol_loss_sum.tolist()}, "
                f"surf={surf_loss_sum.tolist()}, gmax={float(gmax)}) — "
                f"stopping immediately; no checkpoint of this epoch is "
                f"written and the poisoned state is never propagated")

        # rank-sync invariant: DDP must have kept parameters AND the step
        # count bitwise-identical across ranks; a drifted rank is a silent
        # corruption of every later epoch — fail here, on every rank
        if is_distributed():
            from .swa import unwrap_model as _unwrap
            fp = torch.zeros(2, dtype=torch.float64, device=dev)
            fp[0] = float(global_opt_step)
            fp[1] = sum(p.detach().double().sum()
                        for p in _unwrap(model).parameters())
            fp_min, fp_max = fp.clone(), fp.clone()
            torch.distributed.all_reduce(fp_min,
                                         op=torch.distributed.ReduceOp.MIN)
            torch.distributed.all_reduce(fp_max,
                                         op=torch.distributed.ReduceOp.MAX)
            if not torch.equal(fp_min, fp_max):
                raise RuntimeError(
                    f"epoch {epoch}: rank desynchronisation detected — "
                    f"opt_step/param fingerprint min={fp_min.tolist()} != "
                    f"max={fp_max.tolist()} across ranks; gradient "
                    f"synchronisation or step counting diverged")

        if rank == 0:
            weights_payload = epoch_weights_payload(model, epoch)
            state_payload = trainer_state_payload(
                optimizer, scheduler, epoch, world_size=world,
                batch_size=tc.batch_size, shard_hash=shard_hash,
                recipe=recipe.name, profile=cfg.hardware.profile,
                grad_accum=accum, config_hash=cfg_hash,
                global_opt_step=global_opt_step)

            vol_per_var = (vol_loss_sum / n_vol.clamp(min=1)).cpu()
            train_vol = float(vol_per_var.mean())
            train_surf = float((surf_loss_sum / n_surf.clamp(min=1))[0])
            lr_now = optimizer.param_groups[0]["lr"]
            dur = time.time() - t0
            res = EpochMonitor.snapshot()

            ckpt_writer.submit(
                write_epoch_artifacts,
                weights_payload, epoch_weights_dir / f"epoch_{epoch:04d}.pt",
                state_payload, ckpt_dir / f"full_epoch{epoch:04d}.pt",
                ckpt_dir, cfg.checkpoint.rolling_keep, epoch, swa_start)
            ckpt_blocked = ckpt_writer.take_blocked_s()

            print(f"epoch {epoch}: vol={train_vol:.6f} surf={train_surf:.6f} "
                  f"grad={float(gsum) / max(opt_steps_per_epoch, 1):.3f} "
                  f"lr={lr_now:.6g} ({dur:.1f}s)"
                  + (f" [ckpt wait {ckpt_blocked:.1f}s]"
                     if ckpt_blocked > 0.5 else ""), flush=True)
            entry = {
                "epoch": epoch, "lr": lr_now, "recipe": recipe.name,
                "loss_shape": tc.loss_shape,
                "train_loss_vol": train_vol, "train_loss_surf": train_surf,
                "train_loss_vol_per_var": {
                    name: float(vol_per_var[i])
                    for i, name in enumerate(VOL_CHANNEL_NAMES)},
                "grad_norm_avg": float(gsum) / max(opt_steps_per_epoch, 1),
                "grad_norm_max": float(gmax),
                "duration_s": dur,
                "ckpt_blocked_s": ckpt_blocked,
                "ckpt_write_s": ckpt_writer.last_write_s, **res,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

        if max_hours:
            stop_t = torch.zeros((), dtype=torch.float64, device=dev)
            if rank == 0:
                elapsed = time.time() - (t0_wall or t0)
                budget = float(max_hours) * 3600.0
                blocked = ckpt_blocked if rank == 0 else 0.0
                if elapsed + 1.25 * (dur + blocked) + 300.0 >= budget:
                    stop_t += 1.0
                    print(f"[train] walltime budget reached after epoch "
                          f"{epoch} ({elapsed / 3600:.2f}h of {max_hours}h) "
                          f"— stopping cleanly; resume with --auto-resume",
                          flush=True)
            all_reduce_sum(stop_t)
            if float(stop_t) > 0.0:
                stopped_early = True
                break

    ckpt_writer.close()
    barrier()

    del it
    del optimizer, scheduler, train_loader, train_sampler
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()

    if stopped_early:
        if rank == 0:
            (ckpt_dir / NEEDS_RESUME_MARKER).write_text(
                json.dumps({"last_epoch": epoch,
                            "num_epochs": tc.num_epochs,
                            "reason": "walltime budget (--max-hours)"}),
                encoding="utf-8")
            print(f"[train] wrote {NEEDS_RESUME_MARKER}: {epoch}/"
                  f"{tc.num_epochs} epochs done. Relaunch the SAME command "
                  f"with --auto-resume to continue.", flush=True)
        del model, train_ds
        gc.collect()
        if use_cuda:
            torch.cuda.empty_cache()
        if cleanup:
            cleanup_ddp()
        return rank

    if rank == 0:
        del model     # SWA verifies into FRESH instances; free the GPU copy
        gc.collect()
        if use_cuda:
            torch.cuda.empty_cache()

        def _recalib_batches():
            # one full rank-0 shard pass for BN-recalibration (BN models
            # only; the generator is never consumed otherwise), a fresh
            # sampling epoch tag so the stats see real training-like draws
            for i in range(micro_per_epoch):
                yield collate_fn([
                    train_ds[((i * tc.batch_size + j) % n_flat,
                              tc.num_epochs + 1)]
                    for j in range(tc.batch_size)])

        swa_path = _compute_swa_checkpoint(
            cfg, recipe, ckpt_dir, epoch_weights_dir, swa_start,
            recalib_batches=_recalib_batches(), device=device)

        import hashlib
        swa_sha = hashlib.sha256(swa_path.read_bytes()).hexdigest()

        cfg.run.train_completed = True
        cfg.run.completed_epochs = tc.num_epochs

        # the atomic completion record — the LAST line of training_log.jsonl
        record = {"event": TRAINING_COMPLETE_EVENT,
                  "recipe": recipe.name, "family": recipe.family,
                  "profile": cfg.hardware.profile, "mode": cfg.run.mode,
                  "config_hash": cfg_hash,
                  "code_digest": cfg.run.code_digest,
                  "manifest_hash": cfg.run.manifest_hash,
                  "roster_hash": cfg.run.roster_hash,
                  "epochs": tc.num_epochs,
                  "swa_window": [swa_start, tc.num_epochs],
                  "swa_sha256": swa_sha,
                  "world_size": world,
                  "effective_batch": cfg.run.effective_batch,
                  "peak_train_mem_bytes": cfg.run.peak_train_mem_bytes,
                  "train_flops_per_opt_step":
                      cfg.run.train_flops_per_opt_step,
                  "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())

        # FINAL TRAINING TRIO: after the verified SWA + complete record the
        # checkpoint area holds ONLY swa_model.pt + config_used.yaml +
        # training_log.jsonl. Everything transient goes.
        removed = []
        for pattern in ("full_epoch*.pt", "profiling_step*.json"):
            for p in ckpt_dir.glob(pattern):
                p.unlink(missing_ok=True)
                removed.append(p.name)
        for stray in (".train_flops.pt", NEEDS_RESUME_MARKER):
            p = ckpt_dir / stray
            if p.exists():
                p.unlink(missing_ok=True)
                removed.append(stray)
        print(f"[train] COMPLETE — final trio enforced (removed "
              f"{len(removed)} transient files); swa_sha256={swa_sha[:16]}",
              flush=True)
    barrier()

    if rank != 0:
        del model
    del train_ds
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()
    if cleanup:
        cleanup_ddp()
    return rank
