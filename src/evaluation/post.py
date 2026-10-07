"""Evaluation of a finished run, driven only by its config_used.yaml.

  - Weights: the identity-verified swa_model.pt, strict-loaded. A missing,
    corrupt or mismatched SWA is an error.
  - Producers that declare compiled inference are checked after warmup
    (graph capture, CUDA-Graph engagement); a silent eager fallback is an
    error.
  - Latency: every rank times its own cases, the per-rank totals are summed
    and divided by the case count, i.e. the mean per-case latency on one
    GPU. Disk reads and deserialization sit outside the timers; model-side
    online work (e.g. Erwin's per-chunk tree build) sits inside. The record
    carries the GPU name, world size, SWA sha256, config hash, code digest,
    roster hash and cache identity, and is reused only on a full match.
  - Memory: the run's own in-training peak; a run without one is probed
    once here (CUDA only).
  - post_done.json lists the run identity and every required artifact's
    relative path, size and sha256, and is written last. verify_post_done()
    re-checks all of it, so an interrupted or damaged evaluation recomputes
    exactly what is missing.

Phases: P predictions + metrics (resumable per case) -> T timed latency ->
M training memory -> R rank-0 finalize (telemetry, TeX row, post_done).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from ..config import Config, config_hash, code_digest
from ..data.estate import (SURF_CHANNEL_NAMES, VOL_CHANNEL_NAMES,
                           atomic_torch_save)
from ..manifest import load_pool_manifest, roster_sha256, test_case_names
from ..recipes.base import Recipe
from ..registry import recipe_from_config
from ..training.checkpoint import load_state_strict, verify_swa_payload
from ..training.ckpt_writer import AsyncCheckpointWriter
from ..training.ddp import barrier, cleanup_ddp, init_ddp, is_distributed
from .flops import FlopMeter
from .metrics import load_norm_stats
from .test_worker import metrics_from_predictions
from .tex import render_tex

PRED_SCHEMA = "pred_v1"
POST_DONE = "post_done.json"


# ---------------------------------------------------------------------------
# prediction persistence
# ---------------------------------------------------------------------------

def pred_path(run_dir: Path, name: str) -> Path:
    return run_dir / "predictions" / f"{name}.pt"


def metrics_path(run_dir: Path, name: str) -> Path:
    return run_dir / "predictions" / "metrics" / f"{name}.json"


def _write_case(run_dir: Path, name: str, cfg_hash: str, recipe_name: str,
                n_vol: int, n_surf: int, pred_vol_z: np.ndarray,
                pred_surf_z: np.ndarray, metrics: dict,
                save_pred: bool = True) -> None:
    if save_pred:
        atomic_torch_save({
            "schema": PRED_SCHEMA, "case_name": name, "recipe": recipe_name,
            "config_hash": cfg_hash, "n_vol": int(n_vol),
            "n_surf": int(n_surf),
            "row_order": "volume rows 0..N_vol-1, then surface rows, in "
                         "the estate pack's order and wind frame",
            "pred_vol_z": torch.from_numpy(pred_vol_z.astype(np.float32)),
            "pred_surf_z": torch.from_numpy(pred_surf_z.astype(np.float32)),
        }, pred_path(run_dir, name))
    metrics = {**metrics, "n_vol": int(n_vol), "n_surf": int(n_surf)}
    mp = metrics_path(run_dir, name)
    mp.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(mp) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(metrics, f)
    Path(tmp).replace(mp)


def case_complete(run_dir: Path, name: str, cfg_hash: str,
                  n_vol: int, n_surf: int, n_draws: int = 1,
                  need_pred: bool = True) -> bool:
    """A case is done only if both artifacts exist, load cleanly and match
    this run's config hash and shapes; anything else is redone. With
    need_pred=False (metrics-only evaluation, no prediction files) the
    metrics json alone is checked."""
    pp, mp = pred_path(run_dir, name), metrics_path(run_dir, name)
    if not need_pred:
        try:
            with open(mp, encoding="utf-8") as f:
                m = json.load(f)
        except Exception:
            return False
        return (m.get("config_hash") == cfg_hash
                and int(m.get("n_draws", 1)) == int(n_draws)
                and int(m.get("n_vol", -1)) == int(n_vol)
                and int(m.get("n_surf", -1)) == int(n_surf))
    if not (pp.exists() and mp.exists()):
        return False
    try:
        d = torch.load(str(pp), map_location="cpu", weights_only=True)
        with open(mp, encoding="utf-8") as f:
            m = json.load(f)
    except Exception:
        return False
    return (d.get("schema") == PRED_SCHEMA
            and d.get("case_name") == name
            and d.get("config_hash") == cfg_hash
            and int(d.get("n_vol", -1)) == int(n_vol)
            and int(d.get("n_surf", -1)) == int(n_surf)
            and d["pred_vol_z"].shape == (n_vol, len(VOL_CHANNEL_NAMES))
            and d["pred_surf_z"].shape == (n_surf, 1)
            and bool(torch.isfinite(d["pred_vol_z"]).all())
            and bool(torch.isfinite(d["pred_surf_z"]).all())
            and m.get("config_hash") == cfg_hash
            and int(m.get("n_draws", 1)) == int(n_draws))


# ---------------------------------------------------------------------------
# IBO FLOPs (the compiled producer is invisible to module hooks)
# ---------------------------------------------------------------------------

class _IBOFlopPlanner:
    """The encoder cost does not depend on the query count and every decode
    chunk has the same padded shape, so each is metered once on an eager
    copy; per case, total = encode + ceil(N / chunk) * decode_chunk +
    manual(N)."""

    def __init__(self, cfg: Config, recipe: Recipe, sample_case: dict,
                 device: torch.device | None = None):
        from .test_worker import decode_pool, encode_case
        use_cuda = device is not None and device.type == "cuda"
        if device is None:
            device = torch.device("cpu")
        probe = recipe.build_model(cfg).to(device)   # eager fresh instance
        probe.eval()
        chunk = int(cfg.evaluation.test_chunk_size)
        case = {k: (v[: min(len(v), chunk)] if isinstance(v, torch.Tensor)
                    and v.dim() >= 1 and v.shape[0] ==
                    sample_case["N_vol"] + sample_case["N_surf"] else v)
                for k, v in sample_case.items()}
        case["N_vol"] = min(sample_case["N_vol"], chunk)
        case["N_surf"] = max(min(sample_case["N_vol"]
                                 + sample_case["N_surf"], chunk)
                             - case["N_vol"], 0)
        from ..recipes.base import cpu_defloat_bf16
        case = cpu_defloat_bf16(case)
        with FlopMeter(probe) as m1, torch.inference_mode():
            enc = encode_case(probe, case, device, use_cuda=use_cuda)
        self.encode_flops = m1.total
        with FlopMeter(probe) as m2, torch.inference_mode():
            decode_pool(probe, enc[0], enc[1], case, device, chunk,
                        use_cuda=use_cuda)
        self.decode_chunk_flops = m2.total   # exactly one padded chunk
        self.chunk = chunk
        self.cfg = cfg
        self.recipe = recipe
        del probe

    def __call__(self, n_vol: int, n_surf: int) -> float:
        n_chunks = math.ceil((n_vol + n_surf) / self.chunk)
        manual = sum(self.recipe.manual_infer_flops(
            self.cfg, n_vol, n_surf).values())
        return self.encode_flops + n_chunks * self.decode_chunk_flops + manual


# ---------------------------------------------------------------------------
# completion manifest
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _post_identity(cfg: Config, recipe: Recipe, cfg_hash: str,
                   swa_sha: str, pool: dict) -> dict:
    return {"recipe": recipe.name, "family": recipe.family,
            "mode": cfg.run.mode, "config_hash": cfg_hash,
            "code_digest": code_digest(),
            "swa_sha256": swa_sha,
            "manifest_hash": pool["assignment_sha256"],
            "roster_hash": roster_sha256(pool),
            "cache_identity": Path(cfg.data.cache_estate_dir).name}


def write_post_done(run_dir: Path, identity: dict, all_names: list[str],
                    tex_path: Path) -> None:
    """The completion manifest: identity plus every required artifact's
    relative path, size and sha256. Written atomically, last."""
    artifacts = []
    required = []
    for name in all_names:
        required.append(pred_path(run_dir, name))
        required.append(metrics_path(run_dir, name))
    required.append(run_dir / ".post_state" / "telemetry.json")
    required.append(tex_path)
    for p in required:
        if not p.exists():
            raise RuntimeError(f"post_done: required artifact missing at "
                               f"finalize: {p}")
        artifacts.append({"path": str(p.relative_to(run_dir)).replace(
                              "\\", "/"),
                          "bytes": p.stat().st_size,
                          "sha256": _sha256_file(p)})
    done = {**identity, "n_cases": len(all_names),
            "case_names": list(all_names),
            "artifacts": artifacts,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmp = run_dir / (POST_DONE + ".tmp")
    tmp.write_text(json.dumps(done, indent=1), encoding="utf-8")
    tmp.replace(run_dir / POST_DONE)


def verify_post_done(run_dir: Path, identity: dict,
                     all_names: list[str]) -> list[str]:
    """Re-validate the completion manifest and everything it names. Returns
    the list of problems (empty = valid); never raises."""
    marker = run_dir / POST_DONE
    if not marker.exists():
        return ["post_done.json missing"]
    try:
        done = json.loads(marker.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"post_done.json unreadable: {exc}"]
    problems = []
    for key, exp in identity.items():
        if done.get(key) != exp:
            problems.append(f"identity mismatch on {key!r}: "
                            f"{done.get(key)!r} != {exp!r}")
    if done.get("case_names") != list(all_names):
        problems.append("case roster in marker differs from the manifest's "
                        "ordered test roster")
    for art in done.get("artifacts", []):
        p = run_dir / art["path"]
        if not p.exists():
            problems.append(f"missing artifact {art['path']}")
            continue
        if p.stat().st_size != art["bytes"]:
            problems.append(f"size mismatch {art['path']}")
            continue
        if _sha256_file(p) != art["sha256"]:
            problems.append(f"sha256 mismatch {art['path']}")
    if not done.get("artifacts"):
        problems.append("marker carries no artifact list")
    # predictions must load and match schema, case and shape
    for name in all_names:
        pp = pred_path(run_dir, name)
        if not pp.exists():
            problems.append(f"prediction missing for {name}")
            continue
        try:
            d = torch.load(str(pp), map_location="cpu", weights_only=True)
        except Exception as exc:
            problems.append(f"prediction unreadable for {name}: {exc}")
            continue
        if (d.get("schema") != PRED_SCHEMA or d.get("case_name") != name
                or d.get("config_hash") != identity["config_hash"]
                or d["pred_vol_z"].shape[1] != len(VOL_CHANNEL_NAMES)
                or d["pred_surf_z"].shape[1] != 1):
            problems.append(f"prediction schema/identity invalid for {name}")
    return problems


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------

SUMMARY_CHANNELS = VOL_CHANNEL_NAMES + SURF_CHANNEL_NAMES   # table order


def write_rel_l2_txt(run_dir: Path, merged: dict,
                     summary_dir: str | Path) -> Path:
    """<summary_dir>/<run dir name>.txt: the six relative-L2 numbers of the
    results table, one "channel value" per line, at full precision (for
    averaging across seeds).

    Same definition as the TeX row: per case the mean over inference draws
    (AB-UPT's five anchor draws; one draw for deterministic producers), then
    the mean over every test case."""
    missing = [ch for ch in SUMMARY_CHANNELS if ch not in merged["rel_l2"]]
    if missing:
        raise RuntimeError(f"summary: merged metrics lack {missing}")
    lines = [f"{ch} {float(merged['rel_l2'][ch]):.10g}"
             for ch in SUMMARY_CHANNELS]
    out_dir = Path(summary_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_dir.name}.txt"
    tmp = out_dir / f"{run_dir.name}.txt.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    tmp.replace(path)
    print(f"[post] table numbers -> {path}: " + "  ".join(lines), flush=True)
    return path


def write_txt_from_persisted(cfg: Config, run_dir: str | Path,
                             summary_dir: str | Path) -> Path:
    """The same TXT for a run whose evaluation already finished, from the
    persisted per-case metrics alone (no model, GPU or cache). A partial
    evaluation fails instead of averaging fewer cases."""
    run_dir = Path(run_dir)
    cfg_hash = cfg.run.config_hash or config_hash(cfg)
    pool = load_pool_manifest(cfg.data.pool_manifest_path)
    all_names = list(test_case_names(pool, 0, 1))
    merged = _merge_case_metrics(run_dir, all_names, cfg_hash)
    return write_rel_l2_txt(run_dir, merged, summary_dir)


def run_post(cfg: Config, run_dir: str | Path, retime: bool = False,
             metrics_only: bool = False,
             summary_dir: str | Path | None = None) -> int:
    """metrics_only: Phase P only (per-case metrics, resumable, no
    prediction files), then the merge; no latency, memory, TeX or
    post_done. summary_dir: also write <summary_dir>/<run dir name>.txt
    (write_rel_l2_txt), in either mode."""
    rank, world, local = init_ddp()
    run_dir = Path(run_dir)
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{local}") if use_cuda else torch.device("cpu")
    recipe = recipe_from_config(cfg)
    cfg_hash = cfg.run.config_hash or config_hash(cfg)
    state_dir = run_dir / ".post_state"
    tele_path = state_dir / "telemetry.json"

    # evaluation's world size is independent of training's; only the test
    # name list is read from the manifest
    pool = load_pool_manifest(cfg.data.pool_manifest_path)
    n_test_estates = len(pool["splits"]["test_estates"])
    if cfg.run.roster_hash and roster_sha256(pool) != cfg.run.roster_hash:
        raise SystemExit(
            "pool manifest content changed since training: the ordered "
            "test/train roster no longer hashes to the recorded "
            "run.roster_hash — restore the training-time manifest.")
    my_names = test_case_names(pool, rank, world)
    all_names = [n for r in range(world)
                 for n in test_case_names(pool, r, world)]
    if rank == 0:
        print(f"[post] recipe={recipe.name} ({recipe.tex_row_name}) "
              f"world={world} test={n_test_estates} estates / "
              f"{len(all_names)} cases; run={run_dir}", flush=True)
        (run_dir / "predictions" / "metrics").mkdir(parents=True,
                                                    exist_ok=True)
        state_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    stats = load_norm_stats(cfg.data.norm_stats_path)

    # ---- weights: the identity-verified SWA -----------------------------
    swa_path = run_dir / "swa_model.pt"
    swa_identity = {"recipe": recipe.name, "family": recipe.family,
                    "config_hash": cfg.run.config_hash,
                    "mode": cfg.run.mode,
                    "num_epochs": int(cfg.training.num_epochs),
                    "swa_window": [int(cfg.training.swa_start),
                                   int(cfg.training.num_epochs)]}
    payload = verify_swa_payload(swa_path, swa_identity)
    swa_sha = _sha256_file(swa_path)
    model = recipe.build_model(cfg).to(device)
    load_state_strict(model, payload["model_state_dict"])
    del payload
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    infer_status = recipe.prepare_infer_model(model, cfg, use_cuda)
    identity = _post_identity(cfg, recipe, cfg_hash, swa_sha, pool)
    if rank == 0:
        print(f"[post] weights=swa_model.pt (sha {swa_sha[:16]}) "
              f"params={n_params:,} inference={infer_status}", flush=True)

    n_draws = int(recipe.infer_draws(cfg))

    # a fully verified marker and no --retime: nothing to do
    if not retime:
        problems = verify_post_done(run_dir, identity, all_names) \
            if (run_dir / POST_DONE).exists() else ["no marker"]
        if not problems:
            if rank == 0:
                print("[post] post_done.json verified — complete, nothing "
                      "to redo", flush=True)
                if summary_dir is not None:
                    write_rel_l2_txt(
                        run_dir,
                        _merge_case_metrics(run_dir, all_names, cfg_hash),
                        summary_dir)
            barrier()
            cleanup_ddp()
            return rank
        if rank == 0 and (run_dir / POST_DONE).exists():
            print(f"[post] stale post_done.json ({len(problems)} problems, "
                  f"e.g. {problems[:3]}) — recomputing what is missing",
                  flush=True)
            (run_dir / POST_DONE).unlink(missing_ok=True)
    barrier()

    planner = None
    compiled_producer = bool(infer_status.get("expect_compiled", False))
    compile_checked = not compiled_producer or "health" in infer_status

    # ---- Phase P: predictions + metrics (resumable) ----------------------
    writer = AsyncCheckpointWriter(depth=4, enabled=True)
    n_done_before = 0
    n_inferred = 0
    for name, case in recipe.stream_cases(cfg, my_names):
        n_vol, n_surf = int(case["N_vol"]), int(case["N_surf"])
        if case_complete(run_dir, name, cfg_hash, n_vol, n_surf, n_draws,
                         need_pred=not metrics_only):
            n_done_before += 1
            del case
            continue
        recipe.infer_draw = 0
        # A compiled producer must not carry the meter: dynamo would trace
        # the forward hooks into the graph, and their float accumulation
        # over a symbolic row count fails under fullgraph. Compiled
        # producers own an eager FLOPs planner (recipe.infer_flops_planner);
        # eager producers are metered directly.
        if compiled_producer:
            meter = None
            with torch.inference_mode():
                pv, ps = recipe.infer_case(model, case, cfg, device, use_cuda)
        else:
            with FlopMeter(model) as meter, torch.inference_mode():
                pv, ps = recipe.infer_case(model, case, cfg, device, use_cuda)
        if not compile_checked:
            # the producer declared compiled inference but did not verify
            # it itself: check now, after its first real call
            from ..utils.compile_health import compile_health
            infer_status["health"] = compile_health(
                f"{recipe.name}-infer", expect_cudagraphs="reduce-overhead"
                in str(infer_status.get("mode", "")))
            compile_checked = True
        assert pv.shape == (n_vol, len(VOL_CHANNEL_NAMES)) \
            and ps.shape == (n_surf, len(SURF_CHANNEL_NAMES)), \
            f"{name}: producer returned {pv.shape}/{ps.shape}"
        assert np.isfinite(pv).all() and np.isfinite(ps).all(), \
            f"{name}: non-finite predictions"
        if meter is not None and meter.linear > 0:   # eager: metered pass
            flops_case = meter.total + sum(
                recipe.manual_infer_flops(cfg, n_vol, n_surf).values())
        else:                     # compiled producer: the recipe's planner
            if planner is None:
                planner = recipe.infer_flops_planner(cfg, case, device)
                if planner is None:
                    raise RuntimeError(
                        f"recipe {recipe.name!r}: compiled producer without "
                        f"an infer_flops_planner")
            flops_case = planner(n_vol, n_surf)
        metrics, _, _ = metrics_from_predictions(name, case, stats, pv, ps)
        if n_draws > 1:
            # stochastic producer: further independent draws, metrics only
            # (draw 0's predictions are the persisted ones)
            draws = [metrics["relative_l2_physical"]]
            for d in range(1, n_draws):
                recipe.infer_draw = d
                with torch.inference_mode():
                    pv_d, ps_d = recipe.infer_case(model, case, cfg, device,
                                                   use_cuda)
                m_d, _, _ = metrics_from_predictions(name, case, stats,
                                                     pv_d, ps_d)
                draws.append(m_d["relative_l2_physical"])
            recipe.infer_draw = 0
            metrics["draws"] = draws
        metrics["n_draws"] = n_draws
        metrics["config_hash"] = cfg_hash
        metrics["flops_fullcase"] = float(flops_case)
        writer.submit(_write_case, run_dir, name, cfg_hash, recipe.name,
                      n_vol, n_surf, pv, ps, metrics, not metrics_only)
        n_inferred += 1
        print(f"[post rank {rank}] {name}: "
              f"zmse={metrics['zscore_mse']['total']:.6f} "
              f"({n_done_before + n_inferred}/{len(my_names)})", flush=True)
        del case
    writer.close()
    print(f"[post rank {rank}] phase P done: {n_inferred} inferred, "
          f"{n_done_before} already complete", flush=True)
    barrier()

    if metrics_only:
        # every case is persisted and hash-matched (the merge refuses
        # otherwise); skip T/M/R
        if rank == 0:
            merged = _merge_case_metrics(run_dir, all_names, cfg_hash)
            if summary_dir is not None:
                write_rel_l2_txt(run_dir, merged, summary_dir)
            else:
                print("[post] metrics-only: " + "  ".join(
                    f"{ch}={merged['rel_l2'][ch]:.4f}"
                    for ch in SUMMARY_CHANNELS), flush=True)
        barrier()
        cleanup_ddp()
        return rank

    # ---- Phase T: timed latency over the full test set --------------------
    lat_binding = {"gpu": torch.cuda.get_device_name(device) if use_cuda
                   else "cpu",
                   "world": world,
                   "swa_sha256": swa_sha, "config_hash": cfg_hash,
                   "code_digest": code_digest(),
                   "roster_hash": identity["roster_hash"],
                   "cache_identity": identity["cache_identity"],
                   "n_cases": len(all_names)}
    have_timing = False
    if tele_path.exists() and not retime:
        try:
            with open(tele_path, encoding="utf-8") as f:
                t = json.load(f)
            old = t.get("latency", {}) or {}
            have_timing = all(old.get(k) == v for k, v in
                              lat_binding.items())
        except Exception:
            have_timing = False
    if not have_timing:
        # one uncounted warmup case (compile and caches, even if P skipped)
        for _, case in recipe.stream_cases(cfg, my_names[:1]):
            with torch.inference_mode():
                recipe.infer_case(model, case, cfg, device, use_cuda)
            del case
        if use_cuda:
            torch.cuda.synchronize(device)
        barrier()
        # Each case is fully loaded first, then only its inference is timed
        # (synchronized before and after). Every rank times its own cases
        # and the per-rank totals are summed, so total / n_cases is the
        # mean per-case latency on one GPU.
        t_inf = 0.0
        n_timed = 0
        for name, case in recipe.stream_cases(cfg, my_names):
            if use_cuda:
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                recipe.infer_case(model, case, cfg, device, use_cuda)
            if use_cuda:
                torch.cuda.synchronize(device)
            t_inf += time.perf_counter() - t0
            n_timed += 1
            del case
        wall = torch.tensor(float(t_inf), dtype=torch.float64,
                            device=device if use_cuda
                            else torch.device("cpu"))
        if is_distributed():
            torch.distributed.all_reduce(wall,
                                         op=torch.distributed.ReduceOp.SUM)
        barrier()
        if rank == 0:
            lat = {**lat_binding,
                   "wall_s": float(wall.item()),
                   "avg_s_per_case": float(wall.item())
                   / max(len(all_names), 1),
                   "inference": infer_status,
                   "measured_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            print(f"[post] latency: {lat['wall_s']:.1f} GPU-s inference "
                  f"(summed over ranks, I/O excluded) / {len(all_names)} "
                  f"cases = {lat['avg_s_per_case']:.3f} s/case "
                  f"({lat['gpu']}, world={world})", flush=True)
    else:
        lat = None
        if rank == 0:
            print("[post] latency already measured for this exact identity "
                  "(pass --retime to redo)", flush=True)

    # ---- Phase M: training memory -----------------------------------------
    # The trainer records its own peak; only a run without that number is
    # probed here (CUDA only).
    probe_mem = None
    have_train_mem = int(cfg.run.peak_train_mem_bytes) > 0
    if use_cuda and not have_train_mem:
        probe_mem = _train_memory_probe(cfg, recipe, pool, device, rank,
                                        world)

    # a run whose config carries no train FLOPs: meter them the way the
    # trainer does, at the run's own config
    train_flops = float(cfg.run.train_flops_per_opt_step)
    if train_flops <= 0.0:
        train_flops = _train_flops_recover(cfg, recipe, pool, device, rank,
                                           use_cuda)
    barrier()

    # ---- Phase R: rank-0 finalize -----------------------------------------
    if rank == 0:
        merged = _merge_case_metrics(run_dir, all_names, cfg_hash)
        tele = {}
        if tele_path.exists():
            try:
                with open(tele_path, encoding="utf-8") as f:
                    tele = json.load(f)
            except Exception:
                tele = {}
        if not have_timing:
            tele["latency"] = lat
        tele["config_hash"] = cfg_hash
        tele["recipe"] = recipe.name
        tele["mode"] = cfg.run.mode
        tele["tex_row_name"] = recipe.tex_row_name
        tele["params_trainable"] = int(n_params)
        tele["swa_sha256"] = swa_sha
        peak = int(cfg.run.peak_train_mem_bytes) if have_train_mem else 0
        mem_profile = cfg.run.peak_train_mem_profile or cfg.hardware.profile
        if probe_mem is not None:
            peak = int(probe_mem)
            mem_profile = cfg.hardware.profile
        if peak > 0:
            tele["peak_train_mem_bytes"] = peak
            tele["peak_train_mem_profile"] = mem_profile
        tele["train_flops_per_opt_step"] = float(train_flops)
        tele["test_flops_avg_fullcase"] = merged["flops_avg"]
        tele["inference"] = infer_status
        tmp = str(tele_path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(tele, f, indent=1)
        Path(tmp).replace(tele_path)

        tex_path = render_tex(run_dir, recipe.tex_row_name, merged, tele)
        write_post_done(run_dir, identity, all_names, tex_path)
        print(f"[post] COMPLETE: {tex_path}", flush=True)
        if summary_dir is not None:
            write_rel_l2_txt(run_dir, merged, summary_dir)
    barrier()
    cleanup_ddp()
    return rank


def _merge_case_metrics(run_dir: Path, all_names: list[str],
                        cfg_hash: str) -> dict:
    """Every case must be present and hash-matched: a missing case is an
    error, not a smaller mean."""
    per_case = {}
    for name in all_names:
        mp = metrics_path(run_dir, name)
        if not mp.exists():
            raise RuntimeError(f"metrics missing for {name} — rerun "
                               f"`main.py post`")
        with open(mp, encoding="utf-8") as f:
            m = json.load(f)
        if m.get("config_hash") != cfg_hash:
            raise RuntimeError(f"{mp}: config hash mismatch — a case from "
                               f"another configuration; rerun post")
        per_case[name] = m
    # per case: the draw values (stochastic producers) or the single value;
    # cell = mean over cases of the per-case mean; rel_l2_std = mean over
    # cases of the per-case std over draws
    def vals(m, side, ch):
        if m.get("draws"):
            return [d[side][ch] for d in m["draws"]]
        return [m["relative_l2_physical"][side][ch]]

    sides = [(ch, "volume_channels") for ch in VOL_CHANNEL_NAMES] \
        + [(ch, "surface_channels") for ch in SURF_CHANNEL_NAMES]
    chan, std = {}, {}
    for ch, side in sides:
        per = [vals(m, side, ch) for m in per_case.values()]
        chan[ch] = float(np.mean([np.mean(v) for v in per]))
        if all(len(v) > 1 for v in per):
            std[ch] = float(np.mean([np.std(v, ddof=1) for v in per]))
    flops_avg = float(np.mean([m["flops_fullcase"]
                               for m in per_case.values()]))
    return {"rel_l2": chan, "rel_l2_std": std or None,
            "n_cases": len(per_case), "flops_avg": flops_avg}


def _train_memory_probe(cfg: Config, recipe: Recipe, pool: dict, device,
                        rank: int, world: int) -> int:
    """Peak training memory for a run that did not record it: the
    training step rebuilt on one estate's dataset, a warmup update, then one
    measured optimizer update (all accumulation microsteps); peak allocated,
    max over ranks."""
    from torch.nn.parallel import DistributedDataParallel as DDP
    from ..manifest import dirs_by_estate

    torch._dynamo.reset()
    est = sorted(pool["splits"]["train_estates"])[0]
    ds = recipe.build_train_dataset(cfg, [est], dirs_by_estate(pool))
    collate = recipe.collate()
    batch = collate([ds[(i, 1)] for i in range(cfg.training.batch_size)])
    batch = recipe.batch_to_device(batch, device)
    model = recipe.build_model(cfg).to(device)
    mode, fullgraph, _ = recipe.compile_plan(cfg)
    if mode:
        model = torch.compile(model, mode=mode, fullgraph=fullgraph,
                              dynamic=False)
    if is_distributed():
        model = DDP(model, device_ids=[device.index],
                    static_graph=True, gradient_as_bucket_view=True,
                    broadcast_buffers=False)
    optimizer, scheduler, _ = recipe.build_optimizer(cfg, model, 1)
    accum = int(cfg.training.grad_accum)

    def one_update():
        optimizer.zero_grad(set_to_none=True)
        for i in range(accum):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = recipe.forward_loss(model, batch, cfg)
            (loss / accum).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       cfg.training.max_grad_norm,
                                       error_if_nonfinite=True)
        optimizer.step()

    one_update()                      # warmup: compile + optimizer state
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    one_update()                      # the measured update
    torch.cuda.synchronize(device)
    peak = torch.tensor(float(torch.cuda.max_memory_allocated(device)),
                        device=device)
    if is_distributed():
        torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
    if rank == 0:
        print(f"[post] training-memory probe: "
              f"{peak.item() / 2**30:.2f} GiB (max over {world} ranks)",
              flush=True)
    del model, optimizer, scheduler, ds, batch
    torch.cuda.empty_cache()
    torch._dynamo.reset()
    return int(peak.item())


def _train_flops_recover(cfg: Config, recipe: Recipe, pool: dict, device,
                         rank: int, use_cuda: bool) -> float:
    """Train FLOPs per optimizer step for a run that did not record them,
    with the trainer's definition: one eager metered forward on a real
    training sample, x3 (forward + 2x backward) x accum x world."""
    from ..manifest import dirs_by_estate
    from ..training.loop import _meter_train_flops

    est = sorted(pool["splits"]["train_estates"])[:1]
    ds = recipe.build_train_dataset(cfg, est, dirs_by_estate(pool))
    model = recipe.build_model(cfg).to(device)
    model.eval()
    flops, _ = _meter_train_flops(cfg, recipe, model, ds, device, use_cuda)
    if rank == 0:
        print(f"[post] train FLOPs/opt step = {flops / 1e12:.3f} TF",
              flush=True)
    del model, ds
    if use_cuda:
        torch.cuda.empty_cache()
    return float(flops)
