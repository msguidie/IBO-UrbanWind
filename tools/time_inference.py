"""One-GPU stage timing of full-field inference for IBO, AB-UPT and
Transolver-3 on one UrbanWind test case.

Producers, compile modes, bf16 autocast and the latency definition are those
of `main.py post` (disk read excluded; H2D, D2H and host-side gathers
included). Each model runs in its own subprocess so compile caches,
CUDA-graph pools and the peak-memory counter never mix. Needs a CUDA GPU and
evaluated runs (swa_model.pt + config_used.yaml) with their estate cache.

    python tools/time_inference.py --ibo-run work/runs/m400_ibo \
        --abupt-run work/runs/m400_abupt --transolver-run work/runs/m400_transolver
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import env  # noqa: E402,F401  (must precede torch import)

import numpy as np  # noqa: E402
import torch  # noqa: E402

MODELS = ("ibo", "abupt", "transolver")
STAGES = {
    "ibo": ("host", "h2d", "encoder", "processor", "decoder", "d2h"),
    "abupt": ("host", "h2d", "anchor_encoder", "query_decoder", "d2h"),
    "transolver": ("host", "h2d", "embed", "slice_stats", "slice_attention",
                   "deslice_mlp", "heads", "d2h"),
}
ARCH_KEYS = {
    "ibo": ("n_leaves", "latent_dim", "n_vit_layers", "head_dim",
             "n_pn_neighbors", "pn_out_dim", "decoder_dh", "decoder_n_heads",
             "idw_k", "disable_decoder_stage1", "n_register"),
    "abupt": ("hidden_dim", "num_heads", "shared_blocks",
              "num_surface_blocks", "num_volume_blocks", "n_anchors",
              "fullfield_query_chunk_per_branch", "geometry_features"),
    "transolver": ("n_layers", "n_hidden", "n_heads", "slice_num",
                   "mlp_ratio", "in_dim", "infer_chunk"),
}


class _Clock:
    """Wall time per stage. Every stage ends with a sync, so each stage is
    charged its own GPU work, not work queued by the one before."""

    def __init__(self, device, names):
        self.device = device
        self.t = dict.fromkeys(names, 0.0)

    @contextmanager
    def __call__(self, name):
        t0 = time.perf_counter()
        yield
        torch.cuda.synchronize(self.device)
        self.t[name] += time.perf_counter() - t0


def check_run(run_dir) -> Path:
    run_dir = Path(run_dir).expanduser().resolve()
    for f in ("config_used.yaml", "swa_model.pt"):
        if not (run_dir / f).exists():
            raise SystemExit(f"[timer] {run_dir} has no {f}")
    return run_dir


def load_run_config(run_dir: Path, work: Path | None):
    from src.config import load_config
    cfg = load_config(run_dir / "config_used.yaml", quiet=True)
    cfg.checkpoint.save_dir = str(run_dir)
    if work is not None:
        # the caches on this machine live under another work dir
        cfg.data.work_dir = str(work)
        cfg.resolve_paths()
    if not (Path(cfg.data.cache_estate_dir) / "_index.json").exists():
        raise SystemExit(f"[timer] {run_dir.name}: no _index.json in "
                         f"{cfg.data.cache_estate_dir}")
    return cfg


def pick_case(run_dir: Path, work: Path | None,
              manifest: str | None) -> tuple[str, str]:
    """First direction of the test estate whose point count is closest to
    the test-case mean: the table's latency is a per-case mean and latency
    is roughly linear in N, so this case should reproduce it."""
    from src.manifest import estate_of, load_pool_manifest, test_case_names
    cfg = load_run_config(run_dir, work)
    path = manifest or cfg.data.pool_manifest_path
    if not path:
        raise SystemExit("[timer] no pool manifest: pass --manifest or --case")
    pool = load_pool_manifest(Path(path).expanduser())
    names = test_case_names(pool, 0, 1)
    index = json.loads((Path(cfg.data.cache_estate_dir) / "_index.json")
                       .read_text(encoding="utf-8"))
    counts = [index[estate_of(c)] for c in names]
    n = np.array([int(c["N_vol"]) + int(c["N_surf"]) for c in counts],
                 dtype=np.float64)
    i = int(np.argmin(np.abs(n - n.mean())))
    line = (f"[timer] case {names[i]}: N_vol={int(counts[i]['N_vol']):,} "
            f"N_surf={int(counts[i]['N_surf']):,} | test set {len(names)} "
            f"cases, N mean {n.mean():,.0f} median {np.median(n):,.0f} "
            f"min {n.min():,.0f} max {n.max():,.0f}")
    return names[i], line


def recorded_telemetry(run_dir: Path) -> dict | None:
    try:
        t = json.loads((run_dir / ".post_state" / "telemetry.json")
                       .read_text(encoding="utf-8"))
    except Exception:
        return None
    lat = t.get("latency") or {}
    return {"avg_s_per_case": lat.get("avg_s_per_case"),
            "gpu": lat.get("gpu"), "world": lat.get("world"),
            "n_cases": lat.get("n_cases"),
            "measured_at": lat.get("measured_at"),
            "test_flops_avg_fullcase": t.get("test_flops_avg_fullcase"),
            "params_trainable": t.get("params_trainable")}


# ---------------------------------------------------------------------------
# stage-instrumented replicas of the three evaluation producers
# (src/evaluation/test_worker.infer_case, abupt_eval.infer_case_fullfield,
# recipes/transolver.TransolverRecipe.infer_case): same compiled callables,
# same chunking and padding, a sync at every stage boundary
# ---------------------------------------------------------------------------

@torch.inference_mode()
def stages_ibo(model, case, cfg, device, enc_fn, proc_fn):
    from src.evaluation.test_worker import (DECODE_CHUNK_KEYS, _autocast,
                                            _pad_rows, _unsq)
    clk = _Clock(device, STAGES["ibo"])
    chunk = int(cfg.evaluation.test_chunk_size)
    n_vol = int(case["N_vol"])
    n_total = n_vol + int(case["N_surf"])
    with clk("h2d"):
        leaf = {k: _unsq(case[k], device, False)
                for k in ("pn_points_feat", "pn_mask", "leaf_stats",
                          "leaf_pos_norm", "global_morph_norm")}
    # reduce-overhead outputs live in the CUDA-graph pool and are overwritten
    # by the next replay: clone before another graph reads them (as the
    # evaluation producer does)
    with _autocast(True):
        with clk("encoder"):
            h = enc_fn(leaf["pn_points_feat"], leaf["pn_mask"],
                       leaf["leaf_stats"], leaf["leaf_pos_norm"]).clone()
        with clk("processor"):
            z = proc_fn(h, leaf["leaf_pos_norm"],
                        leaf["global_morph_norm"]).clone()
    pv_parts, ps_parts = [], []
    for lo in range(0, n_total, chunk):
        n_valid = min(lo + chunk, n_total) - lo
        pad = chunk - n_valid
        with clk("host"):
            rows = {k: (_pad_rows(case[k][lo:lo + n_valid], pad) if pad
                        else case[k][lo:lo + n_valid])
                    for k in DECODE_CHUNK_KEYS}
        with clk("h2d"):
            q = {k: _unsq(v, device, False) for k, v in rows.items()}
        with clk("decoder"), _autocast(True):
            pv, ps = model.decode(
                h, z, q["query_pos_norm"], q["query_sdf_norm"],
                q["query_sdf_dir"], q["query_support_norm"],
                q["region_dim1"], q["region_dim2"], q["region_dim3"],
                q["idw_indices"], leaf["leaf_pos_norm"])
        with clk("d2h"):
            pv_parts.append(pv.squeeze(0)[:n_valid].float().cpu().numpy())
            ps_parts.append(ps.squeeze(0)[:n_valid].float().cpu().numpy())
    with clk("host"):
        pv_all = np.concatenate(pv_parts)
        ps_all = np.concatenate(ps_parts)
    n_chunks = math.ceil(n_total / chunk)
    info = {"chunk": chunk, "n_chunks": n_chunks,
            "decoded_rows_incl_padding": n_chunks * chunk}
    return clk.t, pv_all[:n_vol], ps_all[n_vol:], info


@torch.inference_mode()
def stages_abupt(recipe, case, cfg, device):
    from src.evaluation import abupt_eval as ae
    from src.evaluation.test_worker import _autocast, _unsq
    clk = _Clock(device, STAGES["abupt"])
    enc, dec = recipe._infer["encode"], recipe._infer["decode"]
    ac = cfg.abupt
    n_vol, n_surf = int(case["N_vol"]), int(case["N_surf"])
    na, chunk = int(ac.n_anchors), int(ac.fullfield_query_chunk_per_branch)
    geo, name = bool(ac.geometry_features), case["case_name"]
    with clk("host"):
        ae.check_abupt_coord_bounds(case["query_pos_norm"], name)
        va = ae._fixed_anchor_indices(name, n_vol, na, "vol", 0)
        sa = ae._fixed_anchor_indices(name, n_surf, na, "surf", 0)
        vq, sq = ae._complement(n_vol, va), ae._complement(n_surf, sa)
        s_rows = ae._branch_rows(case, sa, True, geo, n_vol)
        v_rows = ae._branch_rows(case, va, False, geo, n_vol)
        out_vol = np.empty((n_vol, 5), dtype=np.float32)
        out_surf = np.empty((n_surf, 1), dtype=np.float32)
    n_sq, n_vq = int(sq.shape[0]), int(vq.shape[0])
    with clk("h2d"):
        anchors = {k: _unsq(v, device, False) for k, v in s_rows.items()}
        anchors.update({k: _unsq(v, device, False)
                        for k, v in v_rows.items()})
    with clk("anchor_encoder"), _autocast(True):
        ps_a, pv_a, cache = enc(**anchors)
        cache = ae.clone_cache(cache)
    with clk("d2h"):
        ps_a = ps_a[0].float().cpu().numpy()
        pv_a = pv_a[0].float().cpu().numpy()
    with clk("host"):
        out_surf[sa.numpy()] = ps_a
        out_vol[va.numpy()] = pv_a
        s_pad = ae._branch_rows(case, sq[-1:] if n_sq else sa[-1:], True,
                                geo, n_vol)
        v_pad = ae._branch_rows(case, vq[-1:] if n_vq else va[-1:], False,
                                geo, n_vol)
    n_steps = max(math.ceil(n_sq / chunk), math.ceil(n_vq / chunk))
    for step in range(n_steps):
        lo = step * chunk
        with clk("host"):
            s_c, s_valid = ae._query_chunk(case, sq, s_pad, lo, chunk, True,
                                           geo, n_vol)
            v_c, v_valid = ae._query_chunk(case, vq, v_pad, lo, chunk, False,
                                           geo, n_vol)
        with clk("h2d"):
            kw = {k: _unsq(v, device, False) for k, v in s_c.items()}
            kw.update({k: _unsq(v, device, False) for k, v in v_c.items()})
        with clk("query_decoder"), _autocast(True):
            ps_q, pv_q = dec(cache, **kw)
        with clk("d2h"):
            ps_np = ps_q[0, :s_valid].float().cpu().numpy() if s_valid \
                else None
            pv_np = pv_q[0, :v_valid].float().cpu().numpy() if v_valid \
                else None
        with clk("host"):
            if s_valid:
                out_surf[sq[lo:lo + s_valid].numpy()] = ps_np
            if v_valid:
                out_vol[vq[lo:lo + v_valid].numpy()] = pv_np
    info = {"anchors_per_branch": na, "chunk_per_branch": chunk,
            "surface_queries": n_sq, "volume_queries": n_vq,
            "decode_steps": n_steps}
    return clk.t, out_vol, out_surf, info


@torch.inference_mode()
def stages_transolver(recipe, model, case, cfg, device):
    from src.training.swa import unwrap_model
    clk = _Clock(device, STAGES["transolver"])
    embed_c, stats_c, apply_c, heads_c = recipe._infer["compiled"]
    blocks = unwrap_model(model).blocks
    n_vol = int(case["N_vol"])
    chunk = int(cfg.transolver.infer_chunk)
    with clk("host"):
        x = recipe._case_x(case)
        n = x.shape[0]
        n_chunks = max(1, math.ceil(n / chunk))
        pad = n_chunks * chunk - n
        x_pad = torch.cat([x, x.new_zeros(pad, x.shape[1])]) if pad else x
        parts = [x_pad[c * chunk:(c + 1) * chunk].unsqueeze(0)
                 for c in range(n_chunks)]
    with clk("h2d"):
        xs = [p.to(device) for p in parts]
        masks = []
        for c in range(n_chunks):
            m = torch.ones(1, chunk, device=device)
            if c == n_chunks - 1 and pad:
                m[:, chunk - pad:] = 0.0
            masks.append(m)
    pv_parts, ps_parts = [], []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with clk("embed"):
            fx = [embed_c(xc).clone() for xc in xs]
        for li in range(len(stats_c)):
            with clk("slice_stats"):
                num = den = None
                for c in range(n_chunks):
                    nk, dk = stats_c[li](fx[c], masks[c])
                    num = nk.clone() if num is None else num + nk
                    den = dk.clone() if den is None else den + dk
            with clk("slice_attention"):
                out_slice = blocks[li].Attn.slice_attend(
                    num / (den[..., None] + 1e-5))
            with clk("deslice_mlp"):
                fx = [apply_c[li](fx[c], out_slice).clone()
                      for c in range(n_chunks)]
        for c in range(n_chunks):
            with clk("heads"):
                pv_c, ps_c = heads_c(fx[c])
            with clk("d2h"):
                pv_parts.append(pv_c.squeeze(0).float().cpu())
                ps_parts.append(ps_c.squeeze(0).float().cpu())
    with clk("host"):
        pv = torch.cat(pv_parts)[:n]
        ps = torch.cat(ps_parts)[:n]
        pv_np = pv[:n_vol].numpy().astype(np.float32)
        ps_np = ps[n_vol:].numpy().astype(np.float32)
    info = {"chunk": chunk, "n_chunks": n_chunks,
            "n_layers": len(stats_c)}
    return clk.t, pv_np, ps_np, info


def build_stage_fn(key, recipe, model, cfg, device):
    if key == "abupt":
        return lambda case: stages_abupt(recipe, case, cfg, device)
    if key == "transolver":
        return lambda case: stages_transolver(recipe, model, case, cfg,
                                              device)
    # evaluation compiles encoder + processor as one graph (model.encode);
    # they are compiled separately here only to time them apart
    kw = dict(mode="reduce-overhead", fullgraph=True, dynamic=False)

    def _enc(pn_points_feat, pn_mask, leaf_stats, leaf_pos_norm):
        return model.encoder(model.pointnet(pn_points_feat, pn_mask),
                             leaf_stats, leaf_pos_norm)

    def _proc(h, leaf_pos_norm, global_morph_norm):
        return model.vit(h, leaf_pos_norm, global_morph_norm)

    enc_fn, proc_fn = torch.compile(_enc, **kw), torch.compile(_proc, **kw)
    return lambda case: stages_ibo(model, case, cfg, device, enc_fn,
                                   proc_fn)


# ---------------------------------------------------------------------------
# one model per process
# ---------------------------------------------------------------------------

def child(key, run_dir, work, case_name, warmup, reps, out) -> None:
    from src.registry import recipe_from_config
    from src.training.checkpoint import load_state_strict, verify_swa_payload

    tag = f"[{key}]"
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    cfg = load_run_config(run_dir, work)
    recipe = recipe_from_config(cfg)
    payload = verify_swa_payload(run_dir / "swa_model.pt", {
        "recipe": recipe.name, "family": recipe.family,
        "config_hash": cfg.run.config_hash, "mode": cfg.run.mode,
        "num_epochs": int(cfg.training.num_epochs),
        "swa_window": [int(cfg.training.swa_start),
                       int(cfg.training.num_epochs)]})
    model = recipe.build_model(cfg).to(device)
    load_state_strict(model, payload["model_state_dict"])
    del payload
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sec = {"ibo": cfg.model, "abupt": cfg.abupt,
           "transolver": cfg.transolver}[key]
    arch = {k: getattr(sec, k) for k in ARCH_KEYS[key] if hasattr(sec, k)}
    if key == "ibo":
        arch["test_chunk_size"] = int(cfg.evaluation.test_chunk_size)
    print(f"{tag} run {run_dir} recipe={recipe.name} "
          f"params={n_params / 1e6:.2f}M gpu={torch.cuda.get_device_name(device)}"
          f"\n{tag} arch {arch}", flush=True)

    t0 = time.perf_counter()
    (name, case), = list(recipe.stream_cases(cfg, [case_name]))
    load_s = time.perf_counter() - t0
    n_vol, n_surf = int(case["N_vol"]), int(case["N_surf"])
    print(f"{tag} case {name}: N_vol={n_vol:,} N_surf={n_surf:,}; load "
          f"{load_s:.1f}s (disk read + host preprocessing, outside latency)",
          flush=True)

    t0 = time.perf_counter()
    status = recipe.prepare_infer_model(model, cfg, True)
    prepare_s = time.perf_counter() - t0

    def infer():
        with torch.inference_mode():
            return recipe.infer_case(model, case, cfg, device, True)

    t0 = time.perf_counter()
    infer()
    torch.cuda.synchronize(device)
    first_s = time.perf_counter() - t0
    if status.get("expect_compiled") and "health" not in status:
        from src.utils.compile_health import compile_health
        status["health"] = compile_health(
            f"{recipe.name}-infer",
            expect_cudagraphs="reduce-overhead" in str(status.get("mode", "")))
    for _ in range(max(warmup - 1, 0)):
        infer()
    torch.cuda.synchronize(device)
    print(f"{tag} prepare {prepare_s:.1f}s, first call {first_s:.1f}s "
          f"(compile/capture); {status.get('mode')}", flush=True)

    # ---- end-to-end, exactly the evaluation latency segment ---------------
    e2e = []
    for _ in range(reps):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        pv, ps = infer()
        torch.cuda.synchronize(device)
        e2e.append(time.perf_counter() - t0)
    if not (np.isfinite(pv).all() and np.isfinite(ps).all()):
        raise RuntimeError(f"{tag} non-finite predictions")
    torch.cuda.reset_peak_memory_stats(device)
    infer()
    torch.cuda.synchronize(device)
    peak_alloc = torch.cuda.max_memory_allocated(device) / 2 ** 30
    peak_res = torch.cuda.max_memory_reserved(device) / 2 ** 30
    tele = recorded_telemetry(run_dir)
    print(f"{tag} end-to-end median {np.median(e2e):.4f}s (min "
          f"{min(e2e):.4f} max {max(e2e):.4f}, {reps} reps); peak GPU memory "
          f"{peak_alloc:.2f} GiB allocated / {peak_res:.2f} GiB reserved\n"
          f"{tag} recorded by post: {tele}", flush=True)

    res = {"model": key, "run_dir": str(run_dir), "recipe": recipe.name,
           "case": name, "N_vol": n_vol, "N_surf": n_surf,
           "params": n_params, "arch": arch,
           "gpu": torch.cuda.get_device_name(device),
           "torch": torch.__version__, "cuda": torch.version.cuda,
           "load_s": load_s, "prepare_s": prepare_s, "first_call_s": first_s,
           "inference_status": status, "e2e_s": e2e,
           "peak_alloc_gib": peak_alloc, "peak_reserved_gib": peak_res,
           "telemetry": tele}
    path = out / f"timing_{key}.json"
    path.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")

    # ---- stage breakdown ----------------------------------------------------
    try:
        stage = build_stage_fn(key, recipe, model, cfg, device)
        stage(case)                       # warm-up (ibo: compiles the split)
        if key == "ibo":
            from src.utils.compile_health import compile_health
            try:
                compile_health("ibo-split", expect_cudagraphs=True)
            except RuntimeError as exc:
                print(f"{tag} WARNING split encoder/processor graphs: {exc}",
                      flush=True)
        runs = []
        for _ in range(reps):
            torch.cuda.synchronize(device)
            t, pv_s, ps_s, info = stage(case)
            runs.append(t)
        med = {k: float(np.median([r[k] for r in runs])) for k in STAGES[key]}
        tot = [sum(r.values()) for r in runs]
        diff = float(max(np.abs(pv_s - pv).max(), np.abs(ps_s - ps).max()))
        res.update({"stages_s": {k: [r[k] for r in runs]
                                 for k in STAGES[key]},
                    "stage_median_s": med,
                    "stage_sum_median_s": float(np.median(tot)),
                    "stage_info": info,
                    "stage_vs_evaluation_max_abs_diff": diff})
        print(f"{tag} stage medians [ms]: "
              + "  ".join(f"{k} {1e3 * v:.1f}" for k, v in med.items())
              + f"  | sum {1e3 * np.median(tot):.1f} (end-to-end "
              f"{1e3 * np.median(e2e):.1f})\n{tag} {info}; stage path vs "
              f"evaluation path max|diff| {diff:.3g}", flush=True)
    except Exception as exc:
        traceback.print_exc()
        res["stage_error"] = repr(exc)
    path.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


def print_env() -> None:
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") \
        else os.cpu_count()
    print(f"[timer] {socket.gethostname()} {time.strftime('%Y-%m-%d %H:%M:%S')}"
          f" torch {torch.__version__} cuda {torch.version.cuda} cpus {cpus}",
          flush=True)
    for q in ("--query-gpu=name,memory.total,driver_version,clocks.max.sm",
              "--query-compute-apps=pid,process_name,used_memory"):
        try:
            r = subprocess.run(["nvidia-smi", q, "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=60)
            print(f"[timer] nvidia-smi {q.split('=')[0]}: "
                  f"{r.stdout.strip() or '(none)'}", flush=True)
        except Exception as exc:
            print(f"[timer] nvidia-smi unavailable: {exc}", flush=True)


def summary(results: dict, case_line: str) -> None:
    print("\n==== summary (one GPU; medians over reps) ====\n" + case_line)
    for key, r in results.items():
        if r is None:
            print(f"{key}: FAILED (see log above)")
            continue
        rec = (r.get("telemetry") or {}).get("avg_s_per_case")
        rec = f"{rec:.4f}s" if isinstance(rec, (int, float)) else "n/a"
        print(f"{key:<11} {Path(r['run_dir']).name:<20} {r['gpu']}  params "
              f"{r['params'] / 1e6:.2f}M  end-to-end "
              f"{np.median(r['e2e_s']):.4f}s  peak "
              f"{r['peak_alloc_gib']:.2f} GiB  post-recorded {rec}")
        med = r.get("stage_median_s")
        if med:
            print("    " + "  ".join(f"{k} {1e3 * v:.1f}ms"
                                     for k, v in med.items())
                  + f"  | sum {1e3 * r['stage_sum_median_s']:.1f}ms")
        else:
            print(f"    stage breakdown failed: {r.get('stage_error')}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ibo-run", default=None, help="IBO run directory")
    ap.add_argument("--abupt-run", default=None, help="AB-UPT run directory")
    ap.add_argument("--transolver-run", default=None,
                    help="Transolver-3 run directory")
    ap.add_argument("--work-dir", default=None,
                    help="override data.work_dir recorded in the run configs")
    ap.add_argument("--manifest", default=None,
                    help="pool manifest for picking the case (default: the "
                         "one recorded in the first run's config)")
    ap.add_argument("--case", default=None,
                    help="default: the test case with the mean point count")
    ap.add_argument("--out", default="timing",
                    help="directory for timing_<model>.json")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--child", choices=list(MODELS), default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--run-dir", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    work = Path(args.work_dir).expanduser() if args.work_dir else None
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    if args.child:
        child(args.child, check_run(args.run_dir), work, args.case,
              args.warmup, args.reps, out)
        return 0

    runs = {k: check_run(getattr(args, f"{k}_run"))
            for k in MODELS if getattr(args, f"{k}_run")}
    if not runs:
        raise SystemExit("give at least one of --ibo-run, --abupt-run, "
                         "--transolver-run")
    if not torch.cuda.is_available():
        raise SystemExit("[timer] needs a CUDA GPU")
    print_env()
    if args.case:
        case, case_line = args.case, f"[timer] case {args.case} (given)"
    else:
        case, case_line = pick_case(next(iter(runs.values())), work,
                                    args.manifest)
    print(case_line, flush=True)
    results = {}
    for key, run_dir in runs.items():
        f = out / f"timing_{key}.json"
        f.unlink(missing_ok=True)
        print(f"\n[timer] ---- {key} ----", flush=True)
        cmd = [sys.executable, "-B", str(Path(__file__).resolve()),
               "--child", key, "--run-dir", str(run_dir), "--case", case,
               "--out", str(out), "--warmup", str(args.warmup),
               "--reps", str(args.reps)]
        if work is not None:
            cmd += ["--work-dir", str(work)]
        rc = subprocess.run(cmd).returncode
        if rc:
            print(f"[timer] {key} exited with code {rc}", flush=True)
        results[key] = json.loads(f.read_text(encoding="utf-8")) \
            if f.exists() else None
    summary(results, case_line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
