"""Command-line entry point: dataset preprocessing, training and evaluation
for every model on the menu.

  python main.py preprocess --norm-stats <norm_stats.json>
  python main.py manifest verify
  torchrun --standalone --nproc_per_node 4 main.py train --model ibo \\
      --manifest <work_dir>/manifest_400.json --auto-resume
  torchrun --standalone --nproc_per_node 4 main.py post --run-dir <run>

  --model     which recipe (src/registry.py): ibo | ibo_equal | ibo_grid |
              abupt | transolver | spider | erwin
  --manifest  which estates: a pool manifest (src/manifest.py)
  --grid      which leaf-layout cache to read: invsqrt (default) | equal |
              structured (ibo_equal and ibo_grid select their own)

A finished training run is three files: swa_model.pt, config_used.yaml and
training_log.jsonl. `post` takes no --model: it reads the run's
config_used.yaml, resolves the recipe, strict-loads swa_model.pt, predicts
every test case, measures latency, memory and FLOPs, and writes table.tex.

New cases (no dataset, no cache) go through predict_new_case.py instead.
"""
from src.utils import env  # noqa: F401  (must precede torch import)

import argparse
import os
import sys
from pathlib import Path

from src.config import (RunMeta, code_digest, config_hash,
                        config_hash_payload, load_config, save_config)
from src.registry import MENU, get_recipe
from src.utils.seed import effective_train_seed, train_seed_salt

GRIDS = ("invsqrt", "equal", "structured")


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _pool_tag(manifest_path: str | None) -> str:
    if not manifest_path:
        return "run"
    stem = Path(manifest_path).stem
    return {"manifest_400": "m400", "manifest_full": "full"}.get(stem, stem)


def _auto_run_name(args, cfg) -> str:
    parts = [_pool_tag(cfg.data.pool_manifest_path), cfg.run.recipe]
    if cfg.hardware.profile != "a100":
        parts.append(cfg.hardware.profile)
    if getattr(args, "lr", None) is not None:
        m, e = f"{args.lr:.1e}".split("e")
        parts.append(f"lr{m.rstrip('0').rstrip('.')}e{int(e)}")
    if getattr(args, "epochs", None) is not None:
        parts.append(f"ep{args.epochs}")
    if getattr(args, "abupt_geometry_features", None):
        parts.append("geo")
    # a non-base training seed gets its own default run directory
    if train_seed_salt(cfg):
        parts.append(f"seed{effective_train_seed(cfg)}")
    return "_".join(parts)


def _apply_grid(cfg, args):
    if getattr(args, "grid", None) is not None:
        cfg.preprocess.kdtree_weighting = args.grid
    if getattr(args, "pn_surf_first", None) is not None:
        cfg.preprocess.pn_surf_first = args.pn_surf_first


def _apply_work_dir(cfg, args):
    # before any resolve_paths(): the cache directories derive from it
    if getattr(args, "work_dir", None):
        cfg.data.work_dir = str(Path(args.work_dir).expanduser())


def _check_cache(cfg):
    d = Path(cfg.data.cache_estate_dir)
    missing = [n for n in ("_index.json", "norm_stats.json")
               if not (d / n).exists()]
    if missing:
        root = Path(cfg.data.work_dir) / "cache_estate"
        have = sorted(p.parent.name for p in root.glob("*/_index.json")) \
            if root.exists() else []
        raise SystemExit("\n".join([
            f"estate cache {d} is incomplete (missing "
            f"{', '.join(missing)}).",
            f"  caches built under {root}: {have or 'none'}",
            "  build one with `main.py preprocess --norm-stats <file>` or "
            "`main.py releaf --source <existing cache>`.",
        ]))


def _replica_problems(base, cfg) -> tuple[list[str], list[str]]:
    """(problems, notes) comparing a replica launch against its base run.

    Uses config_hash's own payload, so "result-changing" means exactly what
    the resume guard means by it. Ignored: the training seed (the point of
    a replica), the code digest and training.world_size (compared through
    run.world_size when the base recorded it). A base field left at 0 / ""
    is not compared. A changed input-adapter tag is reported, not refused:
    it versions code, like the digest."""
    a, b = config_hash_payload(base), config_hash_payload(cfg)
    ra, rb = a.pop("run"), b.pop("run")
    for p in (a, b):
        p["training"].pop("seed", None)
        p["training"].pop("world_size", None)
    problems = []
    for sec in sorted(set(a) | set(b)):
        va, vb = a.get(sec), b.get(sec)
        if isinstance(va, dict) and isinstance(vb, dict):
            problems += [f"{sec}.{k}: base={va.get(k)!r} replica={vb.get(k)!r}"
                         for k in sorted(set(va) | set(vb))
                         if va.get(k) != vb.get(k)]
        elif va != vb:
            problems.append(f"{sec}: base={va!r} replica={vb!r}")
    for k in ("recipe", "family", "dataset", "mode"):
        if ra.get(k) != rb.get(k):
            problems.append(f"run.{k}: base={ra.get(k)!r} "
                            f"replica={rb.get(k)!r}")
    for k in ("world_size", "effective_batch", "manifest_hash",
              "roster_hash"):
        if ra.get(k) and ra.get(k) != rb.get(k):
            problems.append(f"run.{k}: base={ra.get(k)!r} "
                            f"replica={rb.get(k)!r}")
    notes = []
    if ra.get("native_input_adapter") != rb.get("native_input_adapter"):
        notes.append(f"input adapter tag {ra.get('native_input_adapter')!r} "
                     f"-> {rb.get('native_input_adapter')!r} (code version)")
    return problems, notes


def _apply_manifest(cfg, args):
    if getattr(args, "manifest", None) is not None:
        cfg.data.pool_manifest_path = str(Path(args.manifest).expanduser())
    if not cfg.data.pool_manifest_path:
        raise SystemExit(
            "no pool manifest selected: pass --manifest "
            "<work_dir>/manifest_400.json or set data.pool_manifest_path.")


def _apply_ibo_shape_flags(cfg, args):
    """IBO architecture overrides. Each one changes which parameters exist;
    evaluation rebuilds the model from the run's config_used.yaml."""
    for name in ("latent_dim", "decoder_dh", "n_vit_layers", "n_register",
                 "head_dim", "decoder_n_heads", "idw_k", "fourier_freqs_xy",
                 "fourier_freqs_z"):
        v = getattr(args, name, None)
        if v is not None:
            setattr(cfg.model, name, v)
    if getattr(args, "rope_dims", None) is not None:
        cfg.model.rope_dims = tuple(int(x)
                                    for x in str(args.rope_dims).split(","))
    for arg_name, field in (("unet_skips", "disable_unet_skips"),
                            ("decoder_stage1", "disable_decoder_stage1"),
                            ("adaln", "disable_adaln")):
        v = getattr(args, arg_name, None)
        if v is not None:
            setattr(cfg.model, field, not v)
    if getattr(args, "decoder_register_attn", None) is not None:
        cfg.model.decoder_register_attn = args.decoder_register_attn
    if getattr(args, "n_query", None) is not None:
        cfg.training.n_query = args.n_query
    if getattr(args, "batch_size", None) is not None:
        cfg.training.batch_size = args.batch_size


def _set_recipe_lr(cfg, recipe_name: str, lr: float):
    if recipe_name in ("ibo", "ibo_equal", "ibo_grid"):
        cfg.training.lr_base = lr
    elif recipe_name == "abupt":
        cfg.abupt.lr_base = lr
    elif recipe_name == "transolver":
        cfg.transolver.lr = lr
    elif recipe_name == "spider":
        cfg.spider.lr = lr
    elif recipe_name == "erwin":
        cfg.erwin.lr = lr


def cmd_train(args) -> int:
    from src.manifest import load_pool_manifest, roster_sha256
    from src.training.loop import train

    if getattr(args, "post_metrics_only", False) \
            and not getattr(args, "post_summary_dir", None):
        raise SystemExit("--post-metrics-only needs --post-summary-dir")

    # --replicate-of: a new run of an existing experiment (a seed study).
    # Start from the base run's recorded config, not config.yaml, with a
    # fresh run section; the launch is then diffed against the base and
    # refused unless only the training seed differs.
    base_cfg = None
    base_dir = None
    if getattr(args, "replicate_of", None):
        base_dir = Path(args.replicate_of).expanduser().resolve()
        base_used = base_dir / "config_used.yaml"
        if not base_used.exists():
            raise SystemExit(f"--replicate-of {base_dir}: no config_used.yaml "
                             f"there — point it at the base run's directory")
        base_cfg = load_config(base_used, quiet=True)
        if base_cfg.run.recipe != args.model:
            raise SystemExit(f"--replicate-of {base_dir.name} is a "
                             f"{base_cfg.run.recipe!r} run but --model is "
                             f"{args.model!r}")
        cfg = load_config(base_used, quiet=True)
        cfg.run = RunMeta(mode=base_cfg.run.mode)
        print(f"[train] --replicate-of {base_dir}: starting from its "
              f"recorded config (recipe {base_cfg.run.recipe}, "
              f"{base_cfg.training.num_epochs} epochs, manifest "
              f"{base_cfg.data.pool_manifest_path}); --config is ignored",
              flush=True)
    else:
        cfg = load_config(args.config)
    _apply_work_dir(cfg, args)
    cfg.hardware.profile = args.profile
    yaml_num_epochs = cfg.training.num_epochs
    if args.epochs is not None:
        cfg.training.num_epochs = args.epochs
        if args.epochs > yaml_num_epochs \
                and args.lr_schedule_ref_epochs is None:
            cfg.training.lr_schedule_ref_epochs = yaml_num_epochs
    if args.lr_schedule_ref_epochs is not None:
        cfg.training.lr_schedule_ref_epochs = args.lr_schedule_ref_epochs
    if args.warmup_steps is not None:
        cfg.training.warmup_steps = args.warmup_steps
    if args.loss_shape is not None:
        cfg.training.loss_shape = args.loss_shape
    if getattr(args, "abupt_geometry_features", None) is not None:
        if args.model != "abupt" and args.abupt_geometry_features:
            raise SystemExit("--abupt-geometry-features requires "
                             "--model abupt")
        cfg.abupt.geometry_features = args.abupt_geometry_features
    _apply_grid(cfg, args)
    _apply_ibo_shape_flags(cfg, args)

    world = _world_size()
    if getattr(args, "check_replica", False):
        if base_cfg is None:
            raise SystemExit("--check-replica needs --replicate-of")
        # a dry run resolves at the base run's world size, or grad_accum and
        # the effective batch would differ for a bookkeeping reason
        if int(base_cfg.run.world_size) > 0:
            world = int(base_cfg.run.world_size)
    recipe = get_recipe(args.model)
    recipe.apply(cfg, world)
    # explicit CLI flags win over the recipe's defaults
    if args.lr is not None:
        _set_recipe_lr(cfg, recipe.name, args.lr)
        cfg.training.lr_base = args.lr
    if args.activation_checkpointing is not None:
        if args.activation_checkpointing \
                and not recipe.supports_activation_checkpointing:
            raise SystemExit(f"recipe {recipe.name!r} does not support "
                             f"activation checkpointing")
        cfg.training.activation_checkpointing = args.activation_checkpointing
    if getattr(args, "seed", None) is not None:
        cfg.training.seed = int(args.seed)
    if train_seed_salt(cfg) and not recipe.supports_train_seed:
        raise SystemExit(
            f"a training seed is supported for ibo and abupt only: recipe "
            f"{recipe.name!r} does not salt its per-step draws, so a seed "
            f"would change init, dropout and case order but not the sampled "
            f"points. Drop --seed (or set it equal to the config's seed "
            f"{cfg.seed}).")

    cfg.validate()
    cfg.resolve_paths()
    _apply_manifest(cfg, args)
    _check_cache(cfg)

    # the code digest and the manifest content (assignment and ordered
    # roster) are part of the experiment identity, so bind them before
    # hashing
    cfg.run.code_digest = code_digest()
    pool = load_pool_manifest(cfg.data.pool_manifest_path)
    cfg.run.manifest_hash = pool["assignment_sha256"]
    cfg.run.roster_hash = roster_sha256(pool)

    if base_cfg is not None:
        problems, notes = _replica_problems(base_cfg, cfg)
        for n in notes:
            print(f"[train] --replicate-of note: {n}", flush=True)
        if problems:
            raise SystemExit(
                f"--replicate-of {base_dir.name}: this launch differs from "
                f"the base experiment in {len(problems)} result-changing "
                f"field(s) besides the training seed:\n  "
                + "\n  ".join(problems)
                + "\nDrop the overriding flags (or fix the manifest).")
        print(f"[train] --replicate-of {base_dir.name}: identical to the base "
              f"in every result-changing field except the training seed "
              f"({effective_train_seed(base_cfg)} -> "
              f"{effective_train_seed(cfg)})", flush=True)
        if getattr(args, "check_replica", False):
            print("[train] --check-replica: OK — exiting without training",
                  flush=True)
            return 0

    run_dir = Path(args.run_dir) if args.run_dir else \
        Path(cfg.data.work_dir) / "runs" / _auto_run_name(args, cfg)
    run_dir = run_dir.resolve()
    cfg.checkpoint.save_dir = str(run_dir)
    cfg.run.config_hash = config_hash(cfg)
    used_cfg = run_dir / "config_used.yaml"

    if used_cfg.exists():
        # Read and verify first: the persisted config is the source of truth
        # for a resume, and this launch may continue it only if the
        # experiment identity matches exactly.
        persisted = load_config(used_cfg, quiet=True)
        recorded = persisted.run.config_hash
        recomputed = config_hash(persisted)
        if recorded and recomputed != recorded:
            raise SystemExit(
                f"{used_cfg}: recorded config_hash {recorded} does not match "
                f"the hash recomputed from the file itself ({recomputed}) — "
                f"the persisted config was edited. Use a fresh --run-dir.")
        if (recorded or recomputed) != cfg.run.config_hash:
            # The hash binds the whole source tree, so an edit that changes
            # no training number (e.g. TeX rendering) would also change it.
            # Re-hash with the persisted code digest: if the hashes then
            # agree, only the source tree moved.
            import copy as _copy
            probe = _copy.deepcopy(cfg)
            probe.run.code_digest = persisted.run.code_digest
            if (recorded or recomputed) != config_hash(probe):
                raise SystemExit(
                    f"{run_dir} already holds a run with config_hash "
                    f"{recorded or recomputed}, but this launch resolves to "
                    f"{cfg.run.config_hash}. Use a fresh --run-dir for the "
                    f"new configuration.")
            print(f"[train] source tree changed since this run started "
                  f"(code_digest {persisted.run.code_digest[:12]} -> "
                  f"{cfg.run.code_digest[:12]}) but every training field is "
                  f"identical — continuing from the persisted config.",
                  flush=True)
        # continue from the persisted config (it carries the telemetry);
        # only the machine-local save_dir is re-pinned
        cfg = persisted
        cfg.checkpoint.save_dir = str(run_dir)
    else:
        save_config(cfg, used_cfg)

    # --post-summary-dir: if the run is complete after this launch, evaluate
    # it in the same process group (the normal full evaluation) and also
    # write <dir>/<run dir name>.txt with the six table numbers. A walltime
    # stop defers this to the launch that completes training.
    summary_dir = getattr(args, "post_summary_dir", None)
    rank = train(cfg, recipe, resume_path=args.resume,
                 auto_resume=args.auto_resume, max_hours=args.max_hours,
                 cleanup=summary_dir is None)
    if rank == 0:
        save_config(cfg, used_cfg)   # now holds telemetry + completion meta
    if summary_dir is not None:
        import torch
        from src.evaluation.post import run_post
        from src.training.ddp import barrier, cleanup_ddp

        barrier()
        # swa_model.pt appears only at completion, and completion deletes
        # .needs_resume
        complete = (run_dir / "swa_model.pt").exists() \
            and not (run_dir / ".needs_resume").exists()
        if complete:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch._dynamo.reset()
            run_post(cfg, run_dir, summary_dir=summary_dir,
                     metrics_only=bool(getattr(args, "post_metrics_only",
                                               False)))
        else:
            if rank == 0:
                print("[train] training not complete after this launch "
                      "(walltime stop) — evaluation runs on the launch that "
                      "completes it", flush=True)
            cleanup_ddp()
    return 0


def cmd_post(args) -> int:
    from src.evaluation.post import run_post

    run_dir = Path(args.run_dir).resolve()
    used = run_dir / "config_used.yaml"
    if not used.exists():
        raise SystemExit(f"{used} missing — evaluation rebuilds the model "
                         f"from the run's own config.")
    cfg = load_config(used)
    if args.work_dir:
        _apply_work_dir(cfg, args)
        cfg.resolve_paths()
    if args.manifest:
        cfg.data.pool_manifest_path = str(Path(args.manifest).expanduser())
    if not cfg.data.pool_manifest_path:
        raise SystemExit("run config carries no pool manifest and no "
                         "--manifest was given")
    cfg.checkpoint.save_dir = str(run_dir)
    if getattr(args, "summary_only", False):
        # evaluation already finished: merge its persisted per-case metrics
        # into the TXT (no model, GPU or cache)
        if not getattr(args, "summary_dir", None):
            raise SystemExit("--summary-only needs --summary-dir")
        from src.evaluation.post import write_txt_from_persisted
        write_txt_from_persisted(cfg, run_dir, args.summary_dir)
        return 0
    _check_cache(cfg)
    run_post(cfg, run_dir, retime=args.retime,
             metrics_only=getattr(args, "metrics_only", False),
             summary_dir=getattr(args, "summary_dir", None))
    return 0


def cmd_derive(args) -> int:
    """Model-specific derived preprocessing (SpiderSolver's partitions),
    written to versioned directories next to the estate cache."""
    from src.manifest import load_pool_manifest

    cfg = load_config(args.config)
    _apply_work_dir(cfg, args)
    _apply_grid(cfg, args)
    if getattr(args, "workers", None) is not None:
        cfg.preprocess.n_workers = int(args.workers)
    cfg.validate()
    cfg.resolve_paths()
    _apply_manifest(cfg, args)
    _check_cache(cfg)
    recipe = get_recipe(args.model)
    pool = load_pool_manifest(cfg.data.pool_manifest_path)
    estates = sorted(set(pool["splits"]["train_estates"])
                     | set(pool["splits"]["test_estates"]))
    recipe.run_derive(cfg, estates)
    return 0


def cmd_preprocess(args) -> int:
    from src.preprocess import run_estate_preprocess

    cfg = load_config(args.config)
    _apply_work_dir(cfg, args)
    if args.raw_dir:
        cfg.data.raw_pt_dir = str(Path(args.raw_dir).expanduser())
    if args.no_anomaly_filter:
        # the cross-case MAD filter compares every case in the folder; with
        # a handful of cases it drops whole estates for nothing
        cfg.preprocess.anomaly_concordance_threshold = float("inf")
        cfg.preprocess.anomaly_extreme_threshold = float("inf")
    _apply_grid(cfg, args)
    cfg.validate()
    cfg.resolve_paths()
    run_estate_preprocess(cfg, Path(args.norm_stats).expanduser(),
                          force=args.force)
    return 0


def cmd_releaf(args) -> int:
    from src.manifest import read_json
    from src.preprocess import run_releaf

    cfg = load_config(args.config)
    _apply_work_dir(cfg, args)
    _apply_grid(cfg, args)
    cfg.validate()
    cfg.resolve_paths()
    only = None
    if args.manifest:
        man = read_json(Path(args.manifest).expanduser())
        only = set(man["estates"])
        print(f"[releaf] roster from {Path(args.manifest).name}: "
              f"{len(only)} estates", flush=True)
    run_releaf(cfg, args.source, force=args.force, workers=args.workers,
               out_cache=args.out, only=only)
    return 0


def cmd_manifest(args) -> int:
    from src import manifest as M
    from src.preprocess import (pool_manifest_paths, rebuild_pool_manifests,
                                retest_pool_manifest, verify_pool_manifests,
                                write_pool_manifests)

    cfg = load_config(args.config)
    _apply_work_dir(cfg, args)
    _apply_grid(cfg, args)
    cfg.validate()
    cfg.resolve_paths()

    if args.action == "retest":
        retest_pool_manifest(cfg, n_test=args.n_test, n_strata=args.strata,
                             pool=args.pool, seed=args.seed)
        return 0
    if args.action == "verify":
        n_bad = verify_pool_manifests(cfg)
        print(f"[verify] {'OK' if not n_bad else str(n_bad) + ' problems'}",
              flush=True)
        return 0 if not n_bad else 1
    if args.action == "rebuild":
        manifests = rebuild_pool_manifests(cfg, world=args.world)
    else:  # repartition
        paths = pool_manifest_paths(cfg)
        existing = {pool: M.read_json(p) for pool, p in paths.items()}
        index = M.read_json(Path(cfg.data.cache_estate_dir) / "_index.json")
        manifests = M.repartition(existing, index, world=args.world,
                                  n_leaves=cfg.model.n_leaves,
                                  n_pn_neighbors=cfg.model.n_pn_neighbors)
    write_pool_manifests(cfg, manifests)
    return 0


def _add_cache_flags(p, with_manifest: bool):
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--work-dir", default=None,
                   help="override data.work_dir (caches live under "
                        "<work-dir>/cache_estate)")
    if with_manifest:
        p.add_argument("--manifest", default=None, help="pool manifest json")
    p.add_argument("--grid", choices=GRIDS, default=None,
                   help="which leaf-layout cache to use (default: "
                        "preprocess.kdtree_weighting)")
    p.add_argument("--surf-first", dest="pn_surf_first", action="store_true",
                   default=None)
    p.add_argument("--no-surf-first", dest="pn_surf_first",
                   action="store_false", default=None)


def _add_ibo_shape_flags(p):
    p.add_argument("--latent-dim", type=int, default=None)
    p.add_argument("--decoder-dh", type=int, default=None)
    p.add_argument("--n-vit-layers", type=int, default=None)
    p.add_argument("--n-query", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--fourier-freqs-xy", type=int, default=None)
    p.add_argument("--fourier-freqs-z", type=int, default=None)
    p.add_argument("--idw-k", type=int, default=None)
    p.add_argument("--n-register", type=int, default=None)
    p.add_argument("--head-dim", type=int, default=None,
                   help="ViT per-head width (default 32, i.e. 12 heads at "
                        "latent 384); give a matching --rope-dims")
    p.add_argument("--rope-dims", default=None,
                   help="3 comma-separated even ints summing to head_dim "
                        "(default 12,12,8)")
    p.add_argument("--decoder-n-heads", type=int, default=None)
    for name, dest, on_help in (
            ("unet-skips", "unet_skips", "gated long skips in the ViT"),
            ("decoder-stage1", "decoder_stage1",
             "decoder stage reading the encoder tokens"),
            ("adaln", "adaln", "adaLN morphology conditioning (default on)"),
            ("decoder-register-attn", "decoder_register_attn",
             "decoder attends the ViT registers")):
        g = p.add_mutually_exclusive_group()
        g.add_argument(f"--{name}", dest=dest, action="store_true",
                       default=None, help=on_help)
        g.add_argument(f"--no-{name}", dest=dest, action="store_false",
                       default=None)


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="main.py",
        description="IBO / UrbanWind: preprocessing, training, evaluation")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train one model (evaluation is the "
                                     "separate `post` command)")
    t.add_argument("--model", choices=list(MENU), default="ibo")
    t.add_argument("--profile", choices=["a100", "h100"], default="a100",
                   help="hardware profile label recorded with the run; on "
                        "a100, recipes that support it train with "
                        "activation checkpointing")
    _add_cache_flags(t, with_manifest=True)
    t.add_argument("--run-dir", default=None)
    t.add_argument("--resume", default=None)
    t.add_argument("--auto-resume", action="store_true",
                   help="continue from the newest complete checkpoint in "
                        "--run-dir; fresh start if none; no-op if the run "
                        "already finished")
    t.add_argument("--max-hours", type=float, default=None,
                   help="stop cleanly at the first epoch boundary past this "
                        "walltime budget (resume with --auto-resume)")
    t.add_argument("--lr", type=float, default=None,
                   help="override the recipe's peak learning rate")
    t.add_argument("--epochs", type=int, default=None)
    t.add_argument("--seed", type=int, default=None,
                   help="training seed for a seed study (ibo / abupt): moves "
                        "weight init, dropout, case order and the per-step "
                        "query / anchor draws; the split, the cache and every "
                        "evaluation draw stay the same. Space seeds more "
                        "than num_epochs apart (case order is seeded with "
                        "seed + epoch).")
    t.add_argument("--replicate-of", default=None, metavar="BASE_RUN_DIR",
                   help="seed study: start from BASE_RUN_DIR/config_used.yaml "
                        "and refuse unless the launch equals the base in "
                        "every result-changing field except the training "
                        "seed (inherits the base's manifest)")
    t.add_argument("--check-replica", action="store_true",
                   help="with --replicate-of: resolve, diff against the "
                        "base, report and exit without training")
    t.add_argument("--post-summary-dir", default=None, metavar="DIR",
                   help="when the run is complete after this launch, evaluate "
                        "it in the same process and also write "
                        "DIR/<run dir name>.txt with the six table numbers")
    t.add_argument("--post-metrics-only", action="store_true",
                   help="with --post-summary-dir: metrics-only evaluation "
                        "(per-case metrics and the TXT; no prediction files, "
                        "latency or TeX)")
    t.add_argument("--warmup-steps", type=int, default=None)
    t.add_argument("--lr-schedule-ref-epochs", type=int, default=None)
    t.add_argument("--loss-shape", choices=["huber", "mse"], default=None)
    ac = t.add_mutually_exclusive_group()
    ac.add_argument("--activation-checkpointing",
                    dest="activation_checkpointing", action="store_true",
                    default=None)
    ac.add_argument("--no-activation-checkpointing",
                    dest="activation_checkpointing", action="store_false",
                    default=None)
    geo = t.add_mutually_exclusive_group()
    geo.add_argument("--abupt-geometry-features",
                     dest="abupt_geometry_features", action="store_true",
                     default=None)
    geo.add_argument("--no-abupt-geometry-features",
                     dest="abupt_geometry_features", action="store_false",
                     default=None)
    _add_ibo_shape_flags(t)
    t.set_defaults(fn=cmd_train)

    po = sub.add_parser("post", help="evaluate a finished run from its "
                                     "config_used.yaml: test predictions, "
                                     "metrics, cost, table.tex")
    po.add_argument("--run-dir", required=True)
    po.add_argument("--work-dir", default=None,
                    help="override data.work_dir recorded in the run config")
    po.add_argument("--manifest", default=None,
                    help="override the pool manifest path recorded in the "
                         "run config")
    po.add_argument("--retime", action="store_true",
                    help="redo the latency pass even if telemetry already "
                         "holds it")
    po.add_argument("--metrics-only", action="store_true",
                    help="per-case metrics and their merge only (no "
                         "prediction files, latency, memory, TeX or "
                         "post_done)")
    po.add_argument("--summary-dir", default=None, metavar="DIR",
                    help="also write DIR/<run dir name>.txt with the six "
                         "table numbers (relative L2: Ux Uy Uz p nut p_surf)")
    po.add_argument("--summary-only", action="store_true",
                    help="with --summary-dir: only write that TXT from the "
                         "persisted per-case metrics of a finished "
                         "evaluation (no model, no GPU)")
    po.set_defaults(fn=cmd_post)

    dv = sub.add_parser("derive", help="model-specific derived preprocessing "
                                       "(spider)")
    dv.add_argument("--model", choices=list(MENU), required=True)
    dv.add_argument("--workers", type=int, default=None,
                    help="process pool size (default: preprocess.n_workers)")
    _add_cache_flags(dv, with_manifest=True)
    dv.set_defaults(fn=cmd_derive)

    p = sub.add_parser("preprocess", help="raw CFD cases -> estate cache")
    p.add_argument("--norm-stats", required=True,
                   help="norm_stats.json (z-score statistics and detrend "
                        "constants); copied into the cache")
    p.add_argument("--raw-dir", default=None,
                   help="override data.raw_pt_dir")
    p.add_argument("--no-anomaly-filter", action="store_true",
                   help="skip the cross-case anomaly filter of the stage; it "
                        "compares statistics over every case in the folder "
                        "and needs hundreds of them to be meaningful")
    _add_cache_flags(p, with_manifest=False)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_preprocess)

    rl = sub.add_parser("releaf", help="rebuild only the leaf layout of an "
                                       "existing cache for another --grid")
    rl.add_argument("--source", required=True,
                    help="the existing estate cache directory")
    rl.add_argument("--out", default=None,
                    help="output cache directory (default: the --grid one)")
    _add_cache_flags(rl, with_manifest=True)
    rl.add_argument("--workers", type=int, default=None)
    rl.add_argument("--force", action="store_true")
    rl.set_defaults(fn=cmd_releaf)

    mf = sub.add_parser("manifest", help="pool manifests: rebuild | "
                                         "repartition | verify | retest")
    mf.add_argument("action", choices=["rebuild", "repartition", "verify",
                                       "retest"])
    _add_cache_flags(mf, with_manifest=False)
    mf.add_argument("--world", type=int, default=4)
    mf.add_argument("--n-test", type=int, default=40)
    mf.add_argument("--strata", type=int, default=8)
    mf.add_argument("--pool", choices=["400", "full"], default="400")
    mf.add_argument("--seed", type=int, default=None)
    mf.set_defaults(fn=cmd_manifest)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
