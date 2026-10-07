"""Dataset preprocessing: raw CFD cases -> one pack per estate.

Raw cases are `<estate>_<dir>.pt` files with dir in N/E/S/W. The wind
directions of one estate are index-aligned rigid rotations of each other, so
geometry is processed once per estate, in the frame of its canonical
direction; every other direction carries a rotation code k (counter-clockwise
quarter turns) and its own targets.

  PASS 1  parallel over estates, resumable through estate_scan.jsonl:
          detect each direction's rotation code, extract the building
          footprints once per estate, and record each direction's anomaly
          indicators and morphology conditioners.
  STAGE   drop non-finite, unrotatable and anomalous (cross-case MAD) cases,
          re-pick the canonical direction among the survivors, and write
          norm_stats.json, _index.json and _clean.json.
  PASS 2  parallel over estates, resumable through pack existence: exact
          signed distance, ray support, surface normals and region codes,
          the latent layout (features.build_latent_layout) and the
          detrended, z-scored targets of every direction.

Targets are detrended before z-scoring, Uy -= c ln(max(z, z0)) and p += g z
(volume and surface), with c, z0, g and every z-score constant read from the
normalization file passed to `main.py preprocess --norm-stats`.

`releaf` rebuilds only the latent layout of an existing cache generation
(another partition: invsqrt, equal or structured) and reuses every per-point
field. Splits are not decided here; they live in the pool manifests
(src/manifest.py).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from ..config import Config
from ..data.estate import (DIR_PRIORITY, PACK_SCHEMA, atomic_torch_save,
                           decode_sdf_dir_angles, detect_rot_code,
                           encode_sdf_dir_angles)
from ..evaluation.metrics import load_norm_stats
from ..manifest import (N_ANCHORS, POOL_FILES, apply_test_subset, atomic_json,
                        available_estates, build_pool_manifests,
                        cache_generations, estate_of, read_json,
                        stratified_test_subset)
from ..utils.env import limit_worker_threads
from .features import (build_latent_layout, classify_edges, morph_raw,
                       surface_normals_and_dims, volume_region_dim2)
from .geometry import (Extrusion25D, FootprintQCError, extract_buildings,
                       rot_poly)

ROT_FAIL_ABORT_FRACTION = 0.01  # more failed rotation detections than this
                                # fraction of all directions aborts the run

# Anomaly indicators, grouped by physical field
INDICATOR_FIELDS = {
    "Ux": ("Ux_mean", "Ux_std"),
    "Uy": ("Uy_mean", "Uy_std"),
    "Uz": ("Uz_mean", "Uz_std"),
    # p_dc_offset is the mean of the DETRENDED pressure: it isolates a
    # per-case reference-pressure offset from the case's height distribution
    "p": ("p_mean", "p_std", "p_dc_offset"),
    "nut": ("nut_mean", "nut_std", "log_nut_mean", "log_nut_std"),
    "p_surf": ("p_surf_mean", "p_surf_std", "p_surf_dc_offset"),
    "wss_x": ("wss_x_mean", "wss_x_std"),
    "wss_y": ("wss_y_mean", "wss_y_std"),
    "wss_z": ("wss_z_mean", "wss_z_std"),
}

MAD_SCALE = 1.4826


# ---------------------------------------------------------------------------
# Raw case IO
# ---------------------------------------------------------------------------

RAW_KEYS = ("volume_pos", "volume_fields", "surface_pos", "surface_fields",
            "surface_normals", "volume_sdf", "volume_sdf_grad")


@dataclass
class RawCase:
    """One raw CFD case, cropped to the learning region and in its wind
    direction's frame (wind toward -y). Fields: volume [Ux, Uy, Uz, p, nut];
    surface [p, wss_x, wss_y, wss_z]. The SDF of the raw files is negated
    on load, so it is positive outside the buildings."""
    case_name: str
    volume_pos: np.ndarray      # [N_vol, 3] f64, metres
    volume_fields: np.ndarray   # [N_vol, 5] f32
    surface_pos: np.ndarray     # [N_surf, 3] f64
    surface_fields: np.ndarray  # [N_surf, 4] f32
    surface_normals: np.ndarray  # [N_surf, 3] f64
    volume_sdf: np.ndarray      # [N_vol] f32, distance to the true geometry
    volume_sdf_grad: np.ndarray  # [N_vol, 3] f32

    @property
    def n_vol(self) -> int:
        return len(self.volume_pos)

    @property
    def n_surf(self) -> int:
        return len(self.surface_pos)


def _to_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_raw_case(path: str | Path) -> RawCase:
    pt = torch.load(str(path), map_location="cpu", weights_only=False)
    missing = [k for k in RAW_KEYS if k not in pt]
    if missing:
        raise KeyError(f"{path}: raw case missing keys {missing}")
    name = pt.get("case_name") or Path(path).stem
    return RawCase(
        case_name=str(name),
        volume_pos=_to_np(pt["volume_pos"]).astype(np.float64),
        volume_fields=_to_np(pt["volume_fields"]).astype(np.float32),
        surface_pos=_to_np(pt["surface_pos"]).astype(np.float64),
        surface_fields=_to_np(pt["surface_fields"]).astype(np.float32),
        surface_normals=_to_np(pt["surface_normals"]).astype(np.float64),
        volume_sdf=(-_to_np(pt["volume_sdf"])).astype(np.float32),
        volume_sdf_grad=(-_to_np(pt["volume_sdf_grad"])).astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _pp_dict(cfg: Config, stats: dict | None = None) -> dict:
    """Plain-dict view of the preprocessing settings for worker processes,
    plus the detrend constants of the normalization file when given."""
    pp = cfg.preprocess
    out = {
        "dbscan_eps": pp.dbscan_eps,
        "dbscan_min_samples": pp.dbscan_min_samples,
        "wall_nz_threshold": pp.wall_nz_threshold,
        "concave_hull_ratio": pp.concave_hull_ratio,
        "dp_tol": pp.dp_tol,
        "min_edge_collapse": pp.min_edge_collapse,
        "wind_direction": list(pp.wind_direction),
        "wind_face_dot_threshold": pp.wind_face_dot_threshold,
        "threshold_edge": pp.threshold_edge,
        "near_building_sdf_threshold": pp.near_building_sdf_threshold,
        "near_ground_z_threshold": pp.near_ground_z_threshold,
        "kdtree_weighting": pp.kdtree_weighting,
        "invsqrt_floor": pp.invsqrt_floor,
        "grid_dims": list(pp.grid_dims),
        "pn_surf_first": pp.pn_surf_first,
        "support_max": pp.support_max,
        "geom_chunk": pp.geom_chunk,
        "sdf_qc": pp.sdf_qc,
        "coord_divisor": cfg.data.coord_divisor,
    }
    if stats is not None:
        out.update(detrend_uy_coeff=float(stats["detrend_uy_coeff"]),
                   detrend_z_floor=float(stats["detrend_z_floor"]),
                   detrend_p_coeff=float(stats["detrend_p_coeff"]))
    return out


def _detrended_targets(raw: RawCase, pp: dict
                       ) -> tuple[np.ndarray, np.ndarray]:
    """(volume [N_vol, 5], surface pressure [N_surf, 1]) in float64 with the
    log-law profile removed from Uy and the hydrostatic term from p."""
    z_vol = raw.volume_pos[:, 2]
    tv = raw.volume_fields.astype(np.float64).copy()
    tv[:, 1] -= pp["detrend_uy_coeff"] * np.log(
        np.maximum(z_vol, pp["detrend_z_floor"]))
    tv[:, 3] += pp["detrend_p_coeff"] * z_vol
    ts = raw.surface_fields[:, 0:1].astype(np.float64).copy()
    ts[:, 0] += pp["detrend_p_coeff"] * raw.surface_pos[:, 2]
    return tv, ts


def _norm_pos_tensor(raw: RawCase, coord_divisor: float) -> torch.Tensor:
    """concat(volume, surface) positions / coord_divisor in float64, the
    layout detect_rot_code compares and the pack stores."""
    allp = np.concatenate([raw.volume_pos, raw.surface_pos], axis=0)
    return torch.from_numpy(allp / coord_divisor)


def _wind(pp: dict) -> np.ndarray:
    wind = np.asarray(pp["wind_direction"], dtype=np.float64)
    return wind / np.linalg.norm(wind)


def _footprints_of(rec: dict) -> tuple[list[np.ndarray], np.ndarray,
                                       np.ndarray]:
    verts = np.asarray(rec["footprint_verts"], dtype=np.float64).reshape(-1, 2)
    offs = np.asarray(rec["footprint_offsets"], dtype=np.int64)
    fps = [verts[offs[i]:offs[i + 1]] for i in range(len(offs) - 1)]
    return fps, np.asarray(rec["z_lo"], dtype=np.float64), \
        np.asarray(rec["z_hi"], dtype=np.float64)


# ---------------------------------------------------------------------------
# PASS 1: scan + rotation detection + footprints + indicators
# ---------------------------------------------------------------------------

def _field_indicators(raw: RawCase) -> tuple[dict, bool, bool]:
    vf = raw.volume_fields.astype(np.float64)
    sf = raw.surface_fields.astype(np.float64)
    log_nut = np.log(np.maximum(np.abs(vf[:, 4]), 1e-30))
    ind = {}
    for i, name in enumerate(("Ux", "Uy", "Uz", "p", "nut")):
        ind[f"{name}_mean"] = float(vf[:, i].mean())
        ind[f"{name}_std"] = float(vf[:, i].std())
    ind["log_nut_mean"] = float(log_nut.mean())
    ind["log_nut_std"] = float(log_nut.std())
    for i, name in enumerate(("p_surf", "wss_x", "wss_y", "wss_z")):
        ind[f"{name}_mean"] = float(sf[:, i].mean())
        ind[f"{name}_std"] = float(sf[:, i].std())
    arrays = (raw.volume_pos, raw.volume_fields, raw.surface_pos,
              raw.surface_fields, raw.volume_sdf, raw.volume_sdf_grad)
    has_nan = any(bool(np.isnan(a).any()) for a in arrays)
    has_inf = any(bool(np.isinf(a).any()) for a in arrays)
    return ind, has_nan, has_inf


def _scan_estate(args) -> dict:
    estate, dir_paths, pp = args
    limit_worker_threads()
    try:
        rec: dict = {"estate": estate, "per_dir": {}, "rot_code": {}}
        raw_c = None
        canonical = None
        # canonical = the first direction in DIR_PRIORITY that loads
        for d, path in dir_paths:
            try:
                raw_c = load_raw_case(path)
                canonical = d
                break
            except Exception as e:
                rec["per_dir"][d] = {"load_error": f"{type(e).__name__}: {e}"}
        if raw_c is None:
            return {"estate": estate, "error": "no direction loads"}

        rec["canonical"] = canonical
        rec["dirs"] = [d for d, _ in dir_paths]
        rec["n_vol"] = int(raw_c.n_vol)
        rec["n_surf"] = int(raw_c.n_surf)
        rec["rot_code"][canonical] = 0
        pos_c = _norm_pos_tensor(raw_c, pp["coord_divisor"])

        # footprint extraction, once per estate
        rec["geometry_valid"] = True
        rec["geometry_error"] = None
        footprints: list[np.ndarray] = []
        heights = np.zeros(0)
        try:
            buildings = extract_buildings(
                raw_c.surface_pos, raw_c.surface_normals,
                dbscan_eps=pp["dbscan_eps"],
                dbscan_min_samples=pp["dbscan_min_samples"],
                wall_nz_threshold=pp["wall_nz_threshold"],
                hull_ratio=pp["concave_hull_ratio"],
                dp_tol=pp["dp_tol"],
                min_edge=pp["min_edge_collapse"])
            footprints = [b.footprint for b in buildings]
            z_lo = np.array([b.z_lo for b in buildings])
            z_hi = np.array([b.z_hi for b in buildings])
            heights = z_hi - z_lo
            offsets = np.zeros(len(footprints) + 1, dtype=np.int64)
            for i, p in enumerate(footprints):
                offsets[i + 1] = offsets[i] + len(p)
            rec["n_buildings"] = len(footprints)
            rec["footprint_verts"] = np.concatenate(footprints).tolist()
            rec["footprint_offsets"] = offsets.tolist()
            rec["z_lo"] = z_lo.tolist()
            rec["z_hi"] = z_hi.tolist()
        except FootprintQCError as e:
            rec["geometry_valid"] = False
            rec["geometry_error"] = str(e)

        wind = _wind(pp)

        def _dir_record(raw: RawCase, k: int) -> dict:
            ind, has_nan, has_inf = _field_indicators(raw)
            out = {"has_nan": has_nan, "has_inf": has_inf, "indicators": ind}
            if not (has_nan or has_inf):
                tv, ts = _detrended_targets(raw, pp)
                out["indicators"]["p_dc_offset"] = float(tv[:, 3].mean())
                out["indicators"]["p_surf_dc_offset"] = float(ts[:, 0].mean())
            if rec["geometry_valid"]:
                out["morph_raw"] = morph_raw(footprints, heights, k, wind,
                                             pp["wind_face_dot_threshold"])
            return out

        rec["per_dir"][canonical] = _dir_record(raw_c, 0)

        for d, path in dir_paths:
            if d == canonical or d in rec["per_dir"]:
                continue
            try:
                raw_d = load_raw_case(path)
            except Exception as e:
                rec["per_dir"][d] = {"load_error": f"{type(e).__name__}: {e}"}
                continue
            if raw_d.n_vol != raw_c.n_vol or raw_d.n_surf != raw_c.n_surf:
                rec["per_dir"][d] = {
                    "rot_fail": f"N mismatch: vol {raw_d.n_vol} vs "
                                f"{raw_c.n_vol}, surf {raw_d.n_surf} vs "
                                f"{raw_c.n_surf}"}
                del raw_d
                continue
            try:
                k = detect_rot_code(pos_c,
                                    _norm_pos_tensor(raw_d, pp["coord_divisor"]),
                                    label=f"{estate}_{d}")
            except RuntimeError as e:
                rec["per_dir"][d] = {"rot_fail": str(e)}
                del raw_d
                continue
            rec["rot_code"][d] = int(k)
            rec["per_dir"][d] = _dir_record(raw_d, int(k))
            del raw_d
        del raw_c
        return rec
    except Exception as e:
        import traceback
        return {"estate": estate, "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(limit=6)}


def discover_estates(raw_dir: str | Path) -> dict[str, list[tuple[str, str]]]:
    """estate -> [(dir, path)] in DIR_PRIORITY order. File names without a
    _<dir> suffix are skipped."""
    grouped: dict[str, dict[str, str]] = {}
    skipped = 0
    for p in sorted(Path(raw_dir).glob("*.pt")):
        try:
            est = estate_of(p.stem)
        except ValueError:
            skipped += 1
            continue
        grouped.setdefault(est, {})[p.stem.rsplit("_", 1)[1]] = str(p)
    if skipped:
        print(f"[scan] skipped {skipped} files without a _<dir> suffix",
              flush=True)
    if not grouped:
        raise FileNotFoundError(f"no case_*_<dir>.pt files in {raw_dir}")
    return {e: [(d, ds[d]) for d in DIR_PRIORITY if d in ds]
            for e, ds in sorted(grouped.items())}


def scan_path_for(cfg: Config) -> Path:
    return Path(cfg.data.cache_estate_dir) / "estate_scan.jsonl"


def run_pass1(cfg: Config, pp: dict, force: bool = False) -> list[dict]:
    out_dir = Path(cfg.data.cache_estate_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scan_path = scan_path_for(cfg)
    if force:
        scan_path.unlink(missing_ok=True)

    done: dict[str, dict] = {}
    if scan_path.exists():
        with open(scan_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" not in rec:
                    done[rec["estate"]] = rec   # last record wins

    estates = discover_estates(cfg.data.raw_pt_dir)
    todo = [e for e in estates if e not in done]
    n_cases = sum(len(v) for v in estates.values())
    print(f"[PASS 1] {len(estates)} estates / {n_cases} raw cases; "
          f"{len(todo)} to scan ({len(done)} already in "
          f"{scan_path.name})", flush=True)
    if todo:
        args = [(e, estates[e], pp) for e in todo]
        n_err = 0
        t0 = time.time()
        with open(scan_path, "a", encoding="utf-8") as out:
            if cfg.preprocess.n_workers <= 1:
                results = map(_scan_estate, args)
                pool = None
            else:
                pool = mp.Pool(cfg.preprocess.n_workers)
                results = pool.imap_unordered(_scan_estate, args, chunksize=1)
            for i, rec in enumerate(results):
                if "error" in rec:
                    n_err += 1
                    print(f"[PASS 1] ERROR {rec['estate']}: {rec['error']}",
                          flush=True)
                else:
                    done[rec["estate"]] = rec
                out.write(json.dumps(rec) + "\n")
                out.flush()
                if (i + 1) % 100 == 0:
                    rate = (i + 1) / max(time.time() - t0, 1e-9)
                    print(f"[PASS 1] {i + 1}/{len(todo)} "
                          f"({rate:.1f} estates/s)", flush=True)
            if pool is not None:
                pool.close()
                pool.join()
        if n_err:
            print(f"[PASS 1] {n_err} estates errored (recorded, excluded)",
                  flush=True)
    return [done[e] for e in sorted(done)]


def _load_scan_records(cfg: Config) -> dict[str, dict]:
    """estate -> its estate_scan.jsonl record (last one wins, errors
    dropped)."""
    scan_path = scan_path_for(cfg)
    if not scan_path.exists():
        raise FileNotFoundError(f"{scan_path} not found — run "
                                f"`main.py preprocess` first")
    done: dict[str, dict] = {}
    with open(scan_path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" not in rec:
                done[rec["estate"]] = rec
    return done


# ---------------------------------------------------------------------------
# STAGE: filtering, canonical re-pick, sidecar files
# ---------------------------------------------------------------------------

def _mad_z(values: np.ndarray) -> np.ndarray:
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    return np.abs(values - med) / max(MAD_SCALE * mad, 1e-10)


def _mad_filter(case_recs: list[dict], concordance_threshold: float,
                extreme_threshold: float) -> set[str]:
    """Cross-case MAD over the per-direction indicators: drop a case when
    two or more physical fields exceed concordance_threshold, or any single
    indicator exceeds extreme_threshold."""
    if not case_recs:
        return set()
    names = sorted({n for group in INDICATOR_FIELDS.values() for n in group})
    mat = np.array([[r["indicators"][n] for n in names] for r in case_recs])
    z = np.stack([_mad_z(mat[:, j]) for j in range(mat.shape[1])], axis=1)
    col = {n: j for j, n in enumerate(names)}
    dropped = set()
    for i, r in enumerate(case_recs):
        field_max = {f: max(z[i, col[n]] for n in group)
                     for f, group in INDICATOR_FIELDS.items()}
        n_over = sum(1 for v in field_max.values()
                     if v > concordance_threshold)
        if n_over >= 2 or z[i].max() > extreme_threshold:
            dropped.add(r["case_name"])
    return dropped


def select_clean(cfg: Config, records: list[dict]) -> dict[str, dict]:
    """Scan records -> the clean estates, each with its surviving directions,
    canonical direction, rotation codes, footprints and morphology."""
    pp = cfg.preprocess
    records = [r for r in records if "error" not in r]
    n_dirs_total = sum(len(r.get("per_dir", {})) for r in records)
    n_rot_fail = sum(1 for r in records for d in r.get("per_dir", {}).values()
                     if "rot_fail" in d)
    if n_rot_fail:
        print(f"[stage] WARNING: {n_rot_fail}/{n_dirs_total} directions "
              f"failed rotation detection (dropped)", flush=True)
        if n_rot_fail > ROT_FAIL_ABORT_FRACTION * max(n_dirs_total, 1):
            raise RuntimeError(
                f"{n_rot_fail}/{n_dirs_total} directions failed rotation "
                f"detection, above {ROT_FAIL_ABORT_FRACTION:.0%} — this "
                f"points at the data layout, not at a few corrupt files.")

    case_recs = []
    for r in records:
        if not r.get("geometry_valid", False):
            continue
        for d, pd in r["per_dir"].items():
            if "load_error" in pd or "rot_fail" in pd:
                continue
            if pd.get("has_nan") or pd.get("has_inf"):
                continue
            case_recs.append({"case_name": f"{r['estate']}_{d}",
                              "indicators": pd["indicators"]})
    mad_dropped = _mad_filter(case_recs, pp.anomaly_concordance_threshold,
                              pp.anomaly_extreme_threshold)
    print(f"[stage] {len(case_recs)} finite direction cases, "
          f"{len(mad_dropped)} dropped as anomalous", flush=True)

    clean: dict[str, dict] = {}
    n_geo_fail = 0
    for r in records:
        if not r.get("geometry_valid", False):
            n_geo_fail += 1
            continue
        kept = [d for d in DIR_PRIORITY
                if d in r["per_dir"]
                and "load_error" not in r["per_dir"][d]
                and "rot_fail" not in r["per_dir"][d]
                and not r["per_dir"][d].get("has_nan")
                and not r["per_dir"][d].get("has_inf")
                and f"{r['estate']}_{d}" not in mad_dropped]
        if not kept:
            continue
        canonical = kept[0]
        fps, z_lo, z_hi = _footprints_of(r)
        rot = {d: int(r["rot_code"][d]) for d in kept}
        if canonical != r["canonical"]:
            # the scan-time canonical was dropped: rotate the footprints into
            # the new canonical frame and rebase the rotation codes
            # (pos_d = R_{k_d}(pos_scan), pos_new = R_{k_nc}(pos_scan)
            #  => pos_d = R_{(k_d - k_nc) % 4}(pos_new))
            k_nc = int(r["rot_code"][canonical])
            fps = [rot_poly(p, k_nc) for p in fps]
            rot = {d: (int(r["rot_code"][d]) - k_nc) % 4 for d in kept}
        assert rot[canonical] == 0
        clean[r["estate"]] = {
            "dirs": kept, "canonical": canonical, "rot_code": rot,
            "footprints": fps, "z_lo": z_lo, "z_hi": z_hi,
            "n_vol": int(r["n_vol"]), "n_surf": int(r["n_surf"]),
            "morph_raw": {d: r["per_dir"][d]["morph_raw"] for d in kept},
        }
    n_cases = sum(len(v["dirs"]) for v in clean.values())
    print(f"[stage] clean: {len(clean)} estates / {n_cases} cases "
          f"({n_geo_fail} estates failed geometry QC)", flush=True)
    if not clean:
        raise RuntimeError("stage produced zero clean estates")
    return clean


def build_stage(cfg: Config, records: list[dict], stats: dict) -> dict:
    """Clean estates + the cache generation's sidecar files: the
    normalization constants the packs are baked with (norm_stats.json),
    per-estate point counts (_index.json) and the clean roster
    (_clean.json)."""
    out_dir = Path(cfg.data.cache_estate_dir)
    clean = select_clean(cfg, records)
    atomic_json(stats, Path(cfg.data.norm_stats_path))
    index = {e: {"dirs": v["dirs"], "N_vol": v["n_vol"], "N_surf": v["n_surf"]}
             for e, v in clean.items()}
    atomic_json(index, out_dir / "_index.json")
    atomic_json({"estates": {e: {"dirs": v["dirs"],
                                 "canonical": v["canonical"]}
                             for e, v in clean.items()}},
                out_dir / "_clean.json")
    print(f"[stage] wrote {cfg.data.norm_stats_path}, _index.json and "
          f"_clean.json ({len(clean)} estates)", flush=True)
    return {"clean": clean, "norm_stats": stats}


# ---------------------------------------------------------------------------
# PASS 2: canonical geometry + per-direction targets -> one pack per estate
# ---------------------------------------------------------------------------

def _build_pack(job: dict) -> dict:
    limit_worker_threads()
    est = job["estate"]
    out_path = Path(job["out_dir"]) / f"{est}.pt"
    if out_path.exists() and not job["force"]:
        return {"estate": est, "skipped": True}
    try:
        return _build_pack_inner(job, out_path)
    except Exception as e:
        import traceback
        return {"estate": est, "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(limit=8)}


def _build_pack_inner(job: dict, out_path: Path) -> dict:
    from sklearn.cluster import DBSCAN

    t0 = time.time()
    est = job["estate"]
    dirs: list[str] = job["dirs"]
    canonical: str = job["canonical"]
    rot: dict[str, int] = job["rot_code"]
    pp = job["pp"]
    cd = float(pp["coord_divisor"])
    chunk = int(pp["geom_chunk"])
    footprints = [np.asarray(p, dtype=np.float64) for p in job["footprints"]]
    z_lo = np.asarray(job["z_lo"], dtype=np.float64)
    z_hi = np.asarray(job["z_hi"], dtype=np.float64)
    scene = Extrusion25D(footprints, z_lo, z_hi)
    wind = _wind(pp)

    raw_c = load_raw_case(job["raw_paths"][canonical])
    N_vol, N_surf = raw_c.n_vol, raw_c.n_surf
    if N_vol != job["n_vol"] or N_surf != job["n_surf"]:
        raise RuntimeError(f"{est}: raw file changed since PASS 1 "
                           f"({N_vol}/{N_surf} vs scan "
                           f"{job['n_vol']}/{job['n_surf']})")

    all_pos = np.concatenate([raw_c.volume_pos, raw_c.surface_pos], axis=0)
    pos_norm = all_pos / cd
    z_phys = all_pos[:, 2].astype(np.float32)
    pos_c_t = torch.from_numpy(pos_norm)

    # Signed distance to the 2.5D buildings. A volume point inside an
    # extruded footprint is fluid the 2.5D geometry fills (an open ground
    # floor, a stepped roof); it takes the case's own distance to the true
    # building surface and its gradient, or |sdf| where those are unusable.
    sdf_phys_vol, sdf_dir_vol = scene.sdf(raw_c.volume_pos, chunk=chunk)
    qc = {}
    if pp["sdf_qc"]:
        diff = np.abs(sdf_phys_vol - raw_c.volume_sdf.astype(np.float64))
        qc = {"sdf_qc_median": float(np.median(diff)),
              "sdf_qc_p95": float(np.percentile(diff, 95))}
    inside = np.flatnonzero(sdf_phys_vol < 0)
    if len(inside):
        r_sdf = raw_c.volume_sdf[inside].astype(np.float64)
        r_grad = raw_c.volume_sdf_grad[inside].astype(np.float64)
        gnorm = np.linalg.norm(r_grad, axis=1)
        use = (r_sdf > 0) & (gnorm > 1e-6)
        sdf_phys_vol[inside] = np.where(use, r_sdf,
                                        np.abs(sdf_phys_vol[inside]))
        sdf_dir_vol[inside[use]] = r_grad[use] / gnorm[use, None]
    qc["n_inside_footprint"] = int(len(inside))
    sdf_norm_vol = sdf_phys_vol / cd

    # surface points -> buildings: DBSCAN clusters matched to the nearest
    # footprint centroid; noise points to the nearest footprint centroid
    surf_labels = DBSCAN(eps=pp["dbscan_eps"],
                         min_samples=pp["dbscan_min_samples"]
                         ).fit_predict(raw_c.surface_pos[:, :2])
    surf_bld_ids = sorted(set(int(l) for l in surf_labels) - {-1})
    geo_centroids = np.array([fp.mean(axis=0) for fp in footprints])
    dbscan_centroids = np.array(
        [raw_c.surface_pos[surf_labels == l, :2].mean(axis=0)
         for l in surf_bld_ids])
    if len(dbscan_centroids) > 0 and len(geo_centroids) > 0:
        _, alignment = cKDTree(geo_centroids).query(dbscan_centroids)
    else:
        alignment = np.arange(len(surf_bld_ids))
    label_to_bi = {lab: int(alignment[i]) for i, lab in enumerate(surf_bld_ids)}
    surf_bld = np.array([label_to_bi.get(int(l), 0) for l in surf_labels],
                        dtype=np.int64)
    noise_mask = surf_labels == -1
    if noise_mask.any() and len(geo_centroids) > 0:
        _, nn_noise = cKDTree(geo_centroids).query(
            raw_c.surface_pos[noise_mask, :2])
        surf_bld[noise_mask] = nn_noise

    # Per-direction surface codes, computed on the canonical polygons:
    # labelling the k-rotated polygon against the fixed wind equals labelling
    # the canonical polygon against the wind rotated back (dot products are
    # invariant under a joint rotation), and edge order is preserved.
    surf_normal = None
    surf_dims = {}
    for d in dirs:
        labels = [classify_edges(rot_poly(fp, rot[d]), wind,
                                 pp["wind_face_dot_threshold"])
                  for fp in footprints]
        nrm, d2, d3 = surface_normals_and_dims(
            raw_c.surface_pos, surf_bld, footprints, z_lo, z_hi, labels,
            pp["threshold_edge"])
        if d == canonical:
            surf_normal = nrm           # geometric, direction-invariant
        surf_dims[d] = (d2, d3)

    sdf_norm = np.concatenate([sdf_norm_vol, np.zeros(N_surf)])
    sdf_dir = np.concatenate([sdf_dir_vol, surf_normal], axis=0)
    support_norm = scene.support(all_pos, support_max=pp["support_max"],
                                 chunk=chunk) / cd

    vol_dim2 = volume_region_dim2(raw_c.volume_pos[:, 2], sdf_phys_vol,
                                  pp["near_ground_z_threshold"],
                                  pp["near_building_sdf_threshold"])
    region_dim2 = np.concatenate([vol_dim2, surf_dims[canonical][0]])
    region_dim3 = np.concatenate([np.zeros(N_vol, dtype=np.int8),
                                  surf_dims[canonical][1]])

    layout = build_latent_layout(
        pos_norm[:N_vol], sdf_norm_vol, sdf_dir_vol, support_norm[:N_vol],
        pos_norm[N_vol:], surf_normal, support_norm[N_vol:],
        weighting=pp["kdtree_weighting"], invsqrt_floor=pp["invsqrt_floor"],
        grid_dims=pp["grid_dims"], n_levels=job["n_kd_levels"],
        n_pn_neighbors=job["n_pn_neighbors"],
        pn_surf_first=pp["pn_surf_first"], coord_divisor=cd,
        seed_name=f"{est}_{canonical}")

    bf16 = torch.bfloat16
    pack = {
        "schema": PACK_SCHEMA,
        "estate": est,
        "canonical": canonical,
        "dirs": list(dirs),
        "rot_code": dict(rot),
        "N_vol": N_vol,
        "N_surf": N_surf,
        "query_pos_norm": torch.from_numpy(pos_norm.astype(np.float32)),
        "query_z_phys": torch.from_numpy(z_phys),
        "query_sdf_norm": torch.from_numpy(
            sdf_norm.astype(np.float32)).to(bf16),
        "query_sdf_dir_ang": encode_sdf_dir_angles(torch.from_numpy(sdf_dir)),
        "query_support_norm": torch.from_numpy(
            support_norm.astype(np.float32)).to(bf16),
        "region_dim2": torch.from_numpy(region_dim2),
        "region_dim3": torch.from_numpy(region_dim3),
        "leaf_pos_norm": torch.from_numpy(
            layout["centroids"].astype(np.float32)),
        "leaf_stats": torch.from_numpy(
            layout["leaf_stats"].astype(np.float32)).to(bf16),
        "pn_points_feat": torch.from_numpy(layout["pn_feat"]).to(bf16),
        "pn_mask": torch.from_numpy(layout["pn_mask"]),
    }

    ns = job["norm_stats"]
    vol_mean = np.asarray(ns["vol_mean"], dtype=np.float64)
    vol_std = np.maximum(np.asarray(ns["vol_std"], dtype=np.float64), 1e-8)
    surf_mean = np.asarray(ns["surf_mean"], dtype=np.float64)
    surf_std = np.maximum(np.asarray(ns["surf_std"], dtype=np.float64), 1e-8)
    morph_mean = np.asarray(ns["morph_mean"], dtype=np.float64)
    morph_std = np.maximum(np.asarray(ns["morph_std"], dtype=np.float64), 1e-8)

    for d in dirs:
        k = rot[d]
        if d == canonical:
            raw_d = raw_c
        else:
            raw_d = load_raw_case(job["raw_paths"][d])
            if raw_d.n_vol != N_vol or raw_d.n_surf != N_surf:
                raise RuntimeError(f"{est}_{d}: N changed since PASS 1")
            k_check = detect_rot_code(
                pos_c_t, _norm_pos_tensor(raw_d, cd), label=f"{est}_{d}")
            if k_check != k:
                raise RuntimeError(f"{est}_{d}: rot_code changed between "
                                   f"passes ({k} -> {k_check})")
        tv, ts = _detrended_targets(raw_d, pp)
        pack[f"target_vol_{d}"] = torch.from_numpy(
            ((tv - vol_mean) / vol_std).astype(np.float32))
        pack[f"target_surf_{d}"] = torch.from_numpy(
            ((ts - surf_mean) / surf_std).astype(np.float32))
        pack[f"surf_dim2_{d}"] = torch.from_numpy(surf_dims[d][0])
        pack[f"surf_dim3_{d}"] = torch.from_numpy(surf_dims[d][1])
        morph = np.asarray(job["morph_raw"][d], dtype=np.float64)
        pack[f"global_morph_norm_{d}"] = torch.from_numpy(
            ((morph - morph_mean) / morph_std).astype(np.float32))
        if d != canonical:
            del raw_d

    atomic_torch_save(pack, out_path)
    return {"estate": est, "skipped": False,
            "n_empty_leaves": layout["n_empty"],
            "seconds": round(time.time() - t0, 1), **qc}


def run_pass2(cfg: Config, stage: dict, force: bool = False):
    out_dir = Path(cfg.data.cache_estate_dir)
    clean = stage["clean"]
    raw_dir = Path(cfg.data.raw_pt_dir)
    pp = _pp_dict(cfg, stage["norm_stats"])

    jobs = []
    for e, v in clean.items():
        jobs.append({
            "estate": e,
            "dirs": v["dirs"],
            "canonical": v["canonical"],
            "rot_code": v["rot_code"],
            "footprints": [p.tolist() for p in v["footprints"]],
            "z_lo": v["z_lo"].tolist(),
            "z_hi": v["z_hi"].tolist(),
            "n_vol": v["n_vol"],
            "n_surf": v["n_surf"],
            "morph_raw": v["morph_raw"],
            "raw_paths": {d: str(raw_dir / f"{e}_{d}.pt") for d in v["dirs"]},
            "pp": pp,
            "n_kd_levels": cfg.model.n_kd_levels,
            "n_pn_neighbors": cfg.model.n_pn_neighbors,
            "norm_stats": stage["norm_stats"],
            "out_dir": str(out_dir),
            "force": force,
        })

    log_path = out_dir / "pack_build.jsonl"
    n_err = n_skip = n_done = 0
    qc_med: list[float] = []
    t0 = time.time()
    print(f"[PASS 2] {len(jobs)} estates -> {out_dir}", flush=True)
    with open(log_path, "a", encoding="utf-8") as plog:
        if cfg.preprocess.n_workers <= 1:
            results = map(_build_pack, jobs)
            pool = None
        else:
            pool = mp.Pool(cfg.preprocess.n_workers)
            results = pool.imap_unordered(_build_pack, jobs, chunksize=1)
        for row in results:
            n_done += 1
            if "error" in row:
                n_err += 1
                print(f"[PASS 2] ERROR {row['estate']}: {row['error']}",
                      flush=True)
            elif row.get("skipped"):
                n_skip += 1
            elif "sdf_qc_median" in row:
                qc_med.append(row["sdf_qc_median"])
            plog.write(json.dumps(row) + "\n")
            plog.flush()
            if n_done % 50 == 0:
                rate = n_done / max(time.time() - t0, 1e-9)
                eta = (len(jobs) - n_done) / max(rate, 1e-9)
                print(f"[PASS 2] {n_done}/{len(jobs)} ({rate:.2f} est/s, "
                      f"eta {eta / 60:.0f} min, {n_skip} skipped)", flush=True)
        if pool is not None:
            pool.close()
            pool.join()
    if qc_med:
        print(f"[PASS 2] SDF check: median over estates of the per-estate "
              f"median |sdf_2.5D - sdf_raw| = {float(np.median(qc_med)):.3f} "
              f"m ({len(qc_med)} estates)", flush=True)
    if n_err:
        raise RuntimeError(f"[PASS 2] {n_err} estates failed — fix and re-run "
                           f"(finished packs are kept; the run resumes)")
    print(f"[PASS 2] done: {len(jobs)} packs ({n_skip} pre-existing) in "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)


def run_estate_preprocess(cfg: Config, norm_stats_path: str | Path,
                          force: bool = False) -> dict:
    """Raw cases -> packs of the cache generation cfg.data.cache_estate_dir,
    with the normalization constants of norm_stats_path."""
    stats = load_norm_stats(norm_stats_path)
    if abs(float(stats["coord_divisor"]) - float(cfg.data.coord_divisor)) \
            > 1e-9:
        raise SystemExit(f"{norm_stats_path}: coord_divisor "
                         f"{stats['coord_divisor']} != config "
                         f"{cfg.data.coord_divisor}")
    pp = _pp_dict(cfg, stats)
    print("=" * 60, flush=True)
    print(f"PREPROCESS -> {cfg.data.cache_estate_dir}", flush=True)
    print(f"  L={cfg.model.n_leaves} K={cfg.model.n_pn_neighbors} "
          f"partition={cfg.preprocess.kdtree_weighting} "
          f"norm stats={norm_stats_path}", flush=True)
    print("=" * 60, flush=True)
    print("PASS 1: scan + rotation detection + footprints", flush=True)
    records = run_pass1(cfg, pp, force=force)
    print("STAGE: filter + sidecar files", flush=True)
    stage = build_stage(cfg, records, stats)
    print("PASS 2: packs", flush=True)
    run_pass2(cfg, stage, force=force)
    print("Preprocessing complete. Splits are kept in the pool manifests "
          "(`main.py manifest`).", flush=True)
    return stage


# ---------------------------------------------------------------------------
# RELEAF: another partition over an existing cache generation
# ---------------------------------------------------------------------------
#
# Only four pack tensors depend on the partition (n_leaves, n_pn_neighbors,
# kdtree_weighting): leaf_pos_norm, leaf_stats, pn_points_feat and pn_mask.
# The per-point geometry and the targets are reused as stored, so a new
# generation needs neither the raw files nor PASS 2. The partition is built
# from the stored (bf16 / fp16-angle) fields rather than the float64 values
# PASS 2 starts from, which can move a near-tie split by a point or two.

def _releaf_one(job: dict) -> dict:
    limit_worker_threads()
    est = job["estate"]
    out_path = Path(job["out_dir"]) / f"{est}.pt"
    if out_path.exists() and not job["force"]:
        return {"estate": est, "skipped": True}
    try:
        pack = torch.load(job["src"], map_location="cpu", weights_only=True)
        if pack.get("schema") != PACK_SCHEMA:
            raise RuntimeError(f"source schema {pack.get('schema')!r}")
        pp = job["pp"]
        cd = pp["coord_divisor"]
        N_vol = int(pack["N_vol"])

        pos_norm = pack["query_pos_norm"].double().numpy()
        sdf_norm = pack["query_sdf_norm"].float().double().numpy()
        sdf_dir = decode_sdf_dir_angles(
            pack["query_sdf_dir_ang"]).double().numpy()
        support_norm = pack["query_support_norm"].float().double().numpy()

        layout = build_latent_layout(
            pos_norm[:N_vol], sdf_norm[:N_vol], sdf_dir[:N_vol],
            support_norm[:N_vol], pos_norm[N_vol:], sdf_dir[N_vol:],
            support_norm[N_vol:], weighting=pp["kdtree_weighting"],
            invsqrt_floor=pp["invsqrt_floor"], grid_dims=pp["grid_dims"],
            n_levels=job["n_kd_levels"], n_pn_neighbors=job["n_pn_neighbors"],
            pn_surf_first=pp["pn_surf_first"], coord_divisor=cd,
            seed_name=f"{est}_{pack['canonical']}")
        tree = layout["tree"]
        K = job["n_pn_neighbors"]

        bf16 = torch.bfloat16
        pack["leaf_pos_norm"] = torch.from_numpy(
            layout["centroids"].astype(np.float32))
        pack["leaf_stats"] = torch.from_numpy(
            layout["leaf_stats"].astype(np.float32)).to(bf16)
        pack["pn_points_feat"] = torch.from_numpy(layout["pn_feat"]).to(bf16)
        pack["pn_mask"] = torch.from_numpy(layout["pn_mask"])
        pack["releafed_from"] = {
            "src": str(job["src"]), "n_leaves": tree.n_leaves,
            "n_pn_neighbors": int(K),
            "kdtree_weighting": pp["kdtree_weighting"],
            "pn_surf_first": bool(pp["pn_surf_first"])}
        atomic_torch_save(pack, out_path)

        cnt = layout["leaf_count"]
        empty = cnt == 0
        occ = cnt[~empty]
        filled = layout["pn_mask"].sum(axis=1)
        surf_slot = layout["surf_slot"]
        placed = int((surf_slot >= 0).sum()) if surf_slot is not None else 0
        moved = int((surf_slot != layout["surf_leaf_geom"]).sum()) \
            if surf_slot is not None else 0
        bs = tree.build_stats or {}
        return {"estate": est, "n_leaves": tree.n_leaves,
                "n_empty": int(empty.sum()),
                "occ_min": int(occ.min()) if occ.size else 0,
                "occ_med": float(np.median(occ)) if occ.size else 0.0,
                "starved": int((np.minimum(cnt, K) < K).sum()),
                "tie_fixed": bs.get("tie_fixed"),
                "max_skew": bs.get("max_skew"),
                "singleton": bs.get("singleton"),
                "duplicate": bs.get("duplicate"),
                "descent_mismatch": bs.get("descent_mismatch"),
                "slots_med": float(np.median(filled)),
                "slots_full": int((filled >= K).sum()),
                "surf_placed": placed, "surf_moved": moved,
                "surf_slots": int(layout["surf_taken"].sum())}
    except Exception as e:  # noqa: BLE001 — one bad estate must not kill the pool
        return {"estate": est, "error": f"{type(e).__name__}: {e}"}


def run_releaf(cfg: Config, src_cache: str | Path, force: bool = False,
               workers: int | None = None, out_cache: str | Path | None = None,
               only: set[str] | None = None) -> None:
    """Build a cache generation from an existing one, reusing everything that
    does not depend on the partition. `out_cache` overrides the destination
    cfg.resolve_paths() derives; `only` restricts the run to a set of estates
    (normally a pool manifest's roster). norm_stats.json and _index.json are
    copied across: they do not depend on the partition."""
    src = Path(src_cache).expanduser()
    out_dir = Path(out_cache).expanduser() if out_cache \
        else Path(cfg.data.cache_estate_dir)
    if not src.is_dir():
        raise SystemExit(f"source generation not found: {src}")
    if src.resolve() == out_dir.resolve():
        raise SystemExit(f"source and destination are the same directory:\n"
                         f"  {out_dir}\nPick another --grid or n_leaves, or "
                         f"point --source at the other generation.")
    out_dir.mkdir(parents=True, exist_ok=True)
    packs = sorted(p for p in src.glob("*.pt") if not p.name.startswith("_"))
    if not packs:
        raise SystemExit(f"no packs in {src}")
    n_all = len(packs)
    if only:
        packs = [p for p in packs if p.stem in only]
        missing = sorted(only - {p.stem for p in packs})
        if missing:
            raise SystemExit(
                f"{len(missing)} estate(s) named by the manifest have no pack "
                f"in {src}, e.g. {missing[:5]}.")

    print("=" * 60, flush=True)
    print(f"RELEAF  {src.name}  ->  {out_dir.name}", flush=True)
    print(f"  src = {src}", flush=True)
    print(f"  out = {out_dir}", flush=True)
    cfg_dir = Path(cfg.data.cache_estate_dir)
    if out_dir.resolve() != cfg_dir.resolve():
        print(f"  NOTE: not the config-derived path ({cfg_dir}); `train` "
              f"resolves the latter.", flush=True)
    print(f"  L={cfg.model.n_leaves} K={cfg.model.n_pn_neighbors} "
          f"partition={cfg.preprocess.kdtree_weighting} "
          f"surf_first={cfg.preprocess.pn_surf_first}", flush=True)
    print(f"  {len(packs)} of {n_all} packs"
          f"{' (manifest roster)' if only else ''}, "
          f"{workers if workers is not None else cfg.preprocess.n_workers} "
          f"workers", flush=True)
    print("=" * 60, flush=True)

    for name in ("norm_stats.json", "_index.json"):
        s = src / name
        if not s.exists() and not (out_dir / name).exists():
            raise SystemExit(f"{s} is missing; it must be carried into the "
                             f"new generation.")
    for name in ("estate_scan.jsonl", "_index.json", "_clean.json",
                 "norm_stats.json"):
        s = src / name
        if s.exists() and not (out_dir / name).exists():
            shutil.copy2(s, out_dir / name)
            print(f"[releaf] carried over {name}", flush=True)

    pp = _pp_dict(cfg)
    jobs = [{"estate": p.stem, "src": str(p), "out_dir": str(out_dir),
             "pp": pp, "n_kd_levels": cfg.model.n_kd_levels,
             "n_pn_neighbors": cfg.model.n_pn_neighbors, "force": force}
            for p in packs]

    n_workers = workers if workers is not None else cfg.preprocess.n_workers
    log_path = out_dir / "releaf.jsonl"
    n_err = n_skip = n_done = 0
    starved_tot = empty_tot = 0
    tie_tot = dup_tot = one_tot = mism_tot = 0
    occ_meds: list[float] = []
    skews: list[float] = []
    t0 = time.time()
    with open(log_path, "a", encoding="utf-8") as plog:
        if n_workers <= 1:
            results, pool = map(_releaf_one, jobs), None
        else:
            pool = mp.Pool(n_workers)
            results = pool.imap_unordered(_releaf_one, jobs, chunksize=1)
        for row in results:
            n_done += 1
            if "error" in row:
                n_err += 1
                print(f"[releaf] ERROR {row['estate']}: {row['error']}",
                      flush=True)
            elif row.get("skipped"):
                n_skip += 1
            else:
                starved_tot += row["starved"]
                empty_tot += row["n_empty"]
                occ_meds.append(row["occ_med"])
                tie_tot += row.get("tie_fixed") or 0
                dup_tot += row.get("duplicate") or 0
                one_tot += row.get("singleton") or 0
                mism_tot += row.get("descent_mismatch") or 0
                skews.append(row.get("max_skew") or 0.0)
            plog.write(json.dumps(row) + "\n")
            plog.flush()
            if n_done % 50 == 0:
                rate = n_done / max(time.time() - t0, 1e-9)
                print(f"[releaf] {n_done}/{len(jobs)} ({rate:.2f} est/s, "
                      f"eta {(len(jobs) - n_done) / max(rate, 1e-9) / 60:.0f} "
                      f"min, {n_skip} skipped)", flush=True)
        if pool is not None:
            pool.close()
            pool.join()

    if occ_meds:
        print(f"[releaf] median leaf occupancy (median over estates): "
              f"{float(np.median(occ_meds)):.1f} points", flush=True)
        print(f"[releaf] leaves below K={cfg.model.n_pn_neighbors}: "
              f"{starved_tot} total, {empty_tot} of them empty "
              f"({starved_tot / max(len(occ_meds), 1):.0f} per estate)",
              flush=True)
        n_est = max(len(occ_meds), 1)
        structured = cfg.preprocess.kdtree_weighting == "structured"
        if not structured:
            print(f"[releaf] split: {tie_tot / n_est:.0f} tie corrections/"
                  f"estate, max weight skew {max(skews or [0.0]):.3f}, "
                  f"{one_tot} singleton + {dup_tot} coincident nodes",
                  flush=True)
        if mism_tot:
            raise RuntimeError(
                f"[releaf] {mism_tot} leaves where the descent disagrees with "
                f"the build partition — the packs written so far are not "
                f"trustworthy.")
        if structured:
            budget = n_est * cfg.model.n_leaves
            print(f"[releaf] regular grid: {empty_tot} empty cells "
                  f"({empty_tot / n_est:.0f}/estate, "
                  f"{100.0 * empty_tot / max(budget, 1):.1f}% of the leaf "
                  f"budget; expected for a data-independent grid)",
                  flush=True)
        elif empty_tot and not (one_tot or dup_tot):
            raise RuntimeError(
                f"[releaf] {empty_tot} empty leaves with no singleton or "
                f"coincident node to explain them.")
        elif empty_tot:
            print(f"[releaf] NOTE: {empty_tot} empty leaves, all attributable "
                  f"to {one_tot} singleton / {dup_tot} coincident nodes",
                  flush=True)
    if n_err:
        raise RuntimeError(f"[releaf] {n_err} estates failed — fix and re-run "
                           f"(finished packs are kept; the run resumes)")
    print(f"[releaf] done: {len(jobs)} packs ({n_skip} pre-existing) in "
          f"{(time.time() - t0) / 60:.1f} min -> {out_dir}", flush=True)


# ---------------------------------------------------------------------------
# Pool manifests: written next to work_dir, shared by every cache generation
# ---------------------------------------------------------------------------

def pool_manifest_paths(cfg: Config) -> dict[str, Path]:
    w = Path(cfg.data.work_dir)
    return {pool: w / fname for pool, fname in POOL_FILES.items()}


def write_pool_manifests(cfg: Config, manifests: dict[str, dict]) -> None:
    from ..manifest import describe
    for pool, path in pool_manifest_paths(cfg).items():
        atomic_json(manifests[pool], path)
        print(f"[manifest] wrote {path}\n{describe(manifests[pool])}",
              flush=True)


def _clean_universe(cfg: Config) -> tuple[dict, dict]:
    """(estates, index) of THIS cache generation, from the _clean.json /
    _index.json the STAGE wrote, or re-derived from estate_scan.jsonl."""
    out_dir = Path(cfg.data.cache_estate_dir)
    clean_path, index_path = out_dir / "_clean.json", out_dir / "_index.json"
    if clean_path.exists() and index_path.exists():
        return read_json(clean_path)["estates"], read_json(index_path)
    done = _load_scan_records(cfg)
    clean = select_clean(cfg, [done[e] for e in sorted(done)])
    estates = {e: {"dirs": v["dirs"], "canonical": v["canonical"]}
               for e, v in clean.items()}
    index = {e: {"dirs": v["dirs"], "N_vol": v["n_vol"], "N_surf": v["n_surf"]}
             for e, v in clean.items()}
    return estates, index


def rebuild_pool_manifests(cfg: Config, world: int = 4) -> dict[str, dict]:
    """Regenerate both pool manifests over a freshly preprocessed cache. The
    test / train_eval / 400-estate name lists are read back from the existing
    manifest_400.json and copied verbatim; only the rank assignment and the
    byte estimates are recomputed."""
    paths = pool_manifest_paths(cfg)
    if not paths["400"].exists():
        raise FileNotFoundError(
            f"{paths['400']} not found — there are no name lists to keep.")
    frozen = read_json(paths["400"])
    estates, index = _clean_universe(cfg)
    gens = cache_generations(Path(cfg.data.work_dir))
    available, _ = available_estates(gens)
    return build_pool_manifests(
        estates, index,
        test_estates=list(frozen["splits"]["test_estates"]),
        train_400=list(frozen["splits"]["train_estates"]),
        train_eval=list(frozen["splits"]["train_eval_estates"]),
        world=world, seed=cfg.seed, n_leaves=cfg.model.n_leaves,
        n_pn_neighbors=cfg.model.n_pn_neighbors,
        n_anchors=N_ANCHORS, available=available,
        source={"rebuilt_from": str(Path(cfg.data.cache_estate_dir)),
                "name_lists": f"copied from {paths['400'].name}",
                "cache_generations": [d.name for d in gens]})


def verify_pool_manifests(cfg: Config) -> int:
    """Every estate named by either pool manifest must exist as a pack in
    THIS cache generation with the same direction list. Returns the number of
    problems found (0 = consistent)."""
    out_dir = Path(cfg.data.cache_estate_dir)
    index_path = out_dir / "_index.json"
    if not index_path.exists():
        print(f"[verify] {index_path} MISSING — cannot verify {out_dir}",
              flush=True)
        return 1
    index = read_json(index_path)
    n_bad = 0
    for pool, path in pool_manifest_paths(cfg).items():
        if not path.exists():
            print(f"[verify] {path} missing", flush=True)
            n_bad += 1
            continue
        m = read_json(path)
        missing, mismatched, no_pack = [], [], []
        for e, info in m["estates"].items():
            row = index.get(e)
            if row is None:
                missing.append(e)
                continue
            if sorted(row["dirs"]) != sorted(info["dirs"]):
                mismatched.append(e)
            if not (out_dir / f"{e}.pt").exists():
                no_pack.append(e)
        n_bad += len(missing) + len(mismatched) + len(no_pack)
        print(f"[verify] {path.name} vs {out_dir.name}: "
              f"{len(m['estates'])} estates, {len(missing)} absent from "
              f"_index, {len(mismatched)} direction mismatch, "
              f"{len(no_pack)} without a pack file", flush=True)
        for label, lst in (("absent", missing), ("mismatch", mismatched),
                           ("no pack", no_pack)):
            if lst:
                print(f"[verify]   {label}: {sorted(lst)[:8]}"
                      f"{' ...' if len(lst) > 8 else ''}", flush=True)
    return n_bad


def retest_pool_manifest(cfg: Config, n_test: int, n_strata: int,
                         pool: str = "400", seed: int | None = None) -> dict:
    """Narrow ONE pool manifest's test list to a stratified subset of itself
    (stratified by building count) and write it back in place. The train
    list and the rank assignment are untouched."""
    import hashlib

    paths = pool_manifest_paths(cfg)
    if pool not in paths:
        raise SystemExit(f"unknown pool {pool!r}; expected one of "
                         f"{sorted(paths)}")
    path = paths[pool]
    m = read_json(path)
    current = list(m["splits"]["test_estates"])

    scan = _load_scan_records(cfg)
    n_buildings = {e: int(r["n_buildings"]) for e, r in scan.items()
                   if "n_buildings" in r}
    unknown = [e for e in current if e not in n_buildings]
    if unknown:
        raise SystemExit(
            f"{len(unknown)} test estates have no n_buildings in "
            f"{scan_path_for(cfg).name} (e.g. {unknown[:5]}).")

    seed = cfg.seed if seed is None else int(seed)
    keep = stratified_test_subset(current, n_buildings, n_test, n_strata, seed)
    counts = sorted(n_buildings[e] for e in keep)
    selection = {
        "method": "stratified by n_buildings, equal quota per stratum",
        "seed": seed,
        "n_strata": n_strata,
        "n_test": len(keep),
        "drawn_from": {"pool": pool, "n_candidates": len(current)},
        "n_buildings": {"min": counts[0], "median": counts[len(counts) // 2],
                        "max": counts[-1]},
        "sha256": hashlib.sha256("\n".join(keep).encode()).hexdigest(),
    }
    out = apply_test_subset(m, keep, selection)
    atomic_json(out, path)

    t = out["totals"]
    print(f"[retest] {path.name}: test {len(current)} -> {t['test_estates']} "
          f"estates / {t['test_cases']} cases", flush=True)
    print(f"[retest] n_buildings over the draw: min {counts[0]}, median "
          f"{counts[len(counts) // 2]}, max {counts[-1]}", flush=True)
    return out
