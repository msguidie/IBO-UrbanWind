#!/usr/bin/env python
"""Predict the wind field around a new estate with a trained IBO model.

    python predict_new_case.py --geometry examples/toy_estate.json \\
        --weights <weights dir> --out outputs

<weights dir> holds the three files of a trained run: swa_model.pt,
config_used.yaml and norm_stats.json. No dataset or cache is needed.

Input: a JSON file of 2.5D buildings (a footprint polygon extruded from the
ground to its height), in metres, in the model frame: origin at the centre
of the 1100 m x 1100 m domain, ground at z = 0, mean wind blowing toward -y.

    {"name": "toy_estate",
     "buildings": [{"footprint": [[x, y], [x, y], ...], "height": 45.0},
                   ...]}

For another wind direction, rotate the footprints so that the wind blows
toward -y. Footprints are simple polygons (holes are ignored); buildings
may extend beyond the domain and are cut at its edge.

Steps:
  1. mesh      Points placed like the cell centres of the UrbanWind CFD mesh:
               the blockMesh z layers (graded 50:1 up to six times the
               reference height H, the tallest building rounded up to 10 m),
               the snappyHexMesh horizontal refinement boxes (3.125 m below
               1.5 H, doubling at 2 H, 2.5 H, 3 H and 3.5 H, 100 m above),
               cells inside buildings removed, and the first cell layer
               snapped onto the building surfaces. Surface points are the
               projections of that first layer onto the buildings. Both are
               cropped to |x|, |y| <= 550 m and z <= 160 m.
  2. features  Signed distance, ray support, surface normals, region codes
               and morphology from the 2.5D geometry, with the same code and
               settings as the training data (src/preprocess).
  3. layout    Importance-weighted KD-tree, leaf statistics and PointNet
               samples (src/preprocess/features.py).
  4. model     IBO built from config_used.yaml, strict-loaded from
               swa_model.pt; CUDA when available, else CPU.
  5. output    <out>/<name>.npz and legacy VTK point clouds of the volume and
               surface predictions (<name>_volume.vtk, <name>_surface.vtk),
               in the model frame.
"""
from src.utils import env  # noqa: F401  (must precede torch import)

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree

from src.config import ModelConfig, PreprocessConfig
from src.data.estate import (SURF_CHANNEL_NAMES, VOL_CHANNEL_NAMES,
                             decode_sdf_dir_angles, encode_sdf_dir_angles,
                             knn_query)
from src.evaluation.metrics import (inverse_transform_surf,
                                    inverse_transform_vol, load_norm_stats)
from src.evaluation.test_worker import infer_case
from src.model.full_model import IBO
from src.preprocess.features import (build_latent_layout, classify_edges,
                                     morph_raw, surface_normals_and_dims,
                                     volume_region_dim2)
from src.preprocess.geometry import (Extrusion25D, canonical_footprint,
                                     point_in_polygon)
from src.training.checkpoint import load_state_strict

WEIGHT_FILES = ("swa_model.pt", "config_used.yaml", "norm_stats.json")

DOMAIN_HALF = 550.0      # output domain: |x|, |y| <= 550 m (cell centres)
DOMAIN_TOP = 160.0       # output domain: z <= 160 m (cell centres)
GRADING = 50.0           # blockMesh simpleGrading in z (last / first layer)
BACKGROUND_CELL = 100.0  # blockMesh horizontal cell size, metres
# snappyHexMesh refinement boxes: (top height in units of H, horizontal
# cell size inside); a layer belongs to the first box containing its centre
REFINEMENT_BOXES = ((1.5, 3.125), (2.0, 6.25), (2.5, 12.5), (3.0, 25.0),
                    (3.5, 50.0))
DECODE_CHUNK = 65536     # query points per decoder call (memory, not results)


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _simple_polygon(xy: np.ndarray):
    """[k, 2] vertices -> a valid shapely Polygon (largest part, exterior
    only, counter-clockwise), or None."""
    from shapely.geometry import Polygon
    from shapely.geometry.polygon import orient

    from src.preprocess.geometry import _largest_polygon

    poly = Polygon(xy)
    if not poly.is_valid:
        poly = poly.buffer(0)
    poly = _largest_polygon(poly)
    if poly is None or poly.is_empty or poly.area < 1e-6:
        return None
    return orient(Polygon(poly.exterior.coords), sign=1.0)


def read_geometry(path: Path) -> tuple[str, list[np.ndarray], np.ndarray]:
    """(case name, footprints [k, 2] counter-clockwise, heights [B])."""
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{path}: cannot read the geometry ({exc})")
    if not isinstance(spec, dict) or not spec.get("buildings"):
        raise SystemExit(f"{path}: expected {{\"buildings\": [...]}}")
    name = str(spec.get("name") or path.stem)
    polys, heights = [], []
    for i, b in enumerate(spec["buildings"]):
        try:
            xy = np.asarray(b["footprint"], dtype=np.float64)
            h = float(b["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"{path}: building {i} needs \"footprint\" "
                             f"[[x, y], ...] and \"height\" ({exc!r})")
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3 \
                or not np.isfinite(xy).all():
            raise SystemExit(f"{path}: building {i}: the footprint needs at "
                             f"least 3 finite [x, y] vertices")
        if not (math.isfinite(h) and h > 0.0):
            raise SystemExit(f"{path}: building {i}: height must be > 0")
        poly = _simple_polygon(xy)
        if poly is None:
            raise SystemExit(f"{path}: building {i}: degenerate footprint")
        polys.append(np.asarray(poly.exterior.coords, dtype=np.float64)[:-1])
        heights.append(h)
    return name, polys, np.asarray(heights, dtype=np.float64)


def read_weights(weights_dir: Path):
    """(model config, preprocessing config, norm stats) of a trained run."""
    missing = [f for f in WEIGHT_FILES if not (weights_dir / f).exists()]
    if missing:
        raise SystemExit(f"{weights_dir}: missing {', '.join(missing)} "
                         f"(expected {', '.join(WEIGHT_FILES)})")
    raw = yaml.safe_load((weights_dir / "config_used.yaml")
                         .read_text(encoding="utf-8")) or {}
    family = str((raw.get("run") or {}).get("family", ""))
    if family in ("abupt", "transolver", "spider", "erwin"):
        raise SystemExit(f"config_used.yaml describes a {family!r} model; "
                         f"this script runs IBO weights")

    def section(name, cls):
        sec = raw.get(name) or {}
        return cls(**{k: v for k, v in sec.items()
                      if k in cls.__dataclass_fields__ and v is not None})

    mcfg = section("model", ModelConfig)
    mcfg.validate()
    pp = section("preprocess", PreprocessConfig)
    if pp.kdtree_weighting not in ("invsqrt", "equal", "structured"):
        raise SystemExit(f"config_used.yaml: unknown preprocess."
                         f"kdtree_weighting {pp.kdtree_weighting!r}")
    stats = load_norm_stats(weights_dir / "norm_stats.json")
    cd = float((raw.get("data") or {}).get("coord_divisor", 550.0))
    if abs(float(stats["coord_divisor"]) - cd) > 1e-9:
        raise SystemExit(f"norm_stats.json coord_divisor "
                         f"{stats['coord_divisor']} != config_used.yaml {cd}")
    return mcfg, pp, stats


# ---------------------------------------------------------------------------
# Step 1: the CFD-like point set
# ---------------------------------------------------------------------------

def reference_height(heights: np.ndarray) -> float:
    """The tallest building rounded up to a multiple of 10 m (34.9 -> 40)."""
    return 10.0 * math.ceil(round(float(np.max(heights)), 6) / 10.0)


def z_layers(height: float) -> np.ndarray:
    """Layer boundaries [NZ + 1] of the blockMesh column: top L = 6 H,
    NZ = ceil(1 + ln 50 / ln((1 - L) / (50 - L))) layers graded 50:1, i.e.
    growth r = 50^(1 / (NZ - 1)) and a first layer L (r - 1) / (r^NZ - 1)."""
    top = math.ceil(6.0 * math.ceil(height))
    nz = math.ceil(1.0 + math.log(GRADING)
                   / math.log((1.0 - top) / (GRADING - top)))
    r = GRADING ** (1.0 / (nz - 1))
    first = top * (r - 1.0) / (r ** nz - 1.0)
    bounds = first * (r ** np.arange(nz + 1) - 1.0) / (r - 1.0)
    bounds[-1] = float(top)
    return bounds


def horizontal_cell(z_centre: float, height: float) -> float:
    for factor, size in REFINEMENT_BOXES:
        if z_centre <= factor * height:
            return size
    return BACKGROUND_CELL


def _bbox(poly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return poly.min(axis=0), poly.max(axis=0)


def inside_buildings(pts: np.ndarray, polys: list[np.ndarray],
                     heights: np.ndarray) -> np.ndarray:
    """True where a point lies inside any building."""
    out = np.zeros(len(pts), dtype=bool)
    for poly, h in zip(polys, heights):
        lo, hi = _bbox(poly)
        cand = np.flatnonzero(
            ~out & (pts[:, 2] <= h)
            & (pts[:, 0] >= lo[0]) & (pts[:, 0] <= hi[0])
            & (pts[:, 1] >= lo[1]) & (pts[:, 1] <= hi[1]))
        if len(cand):
            out[cand] = point_in_polygon(pts[cand, :2], poly)
    return out


def bbox_distance(pts: np.ndarray, polys: list[np.ndarray],
                  heights: np.ndarray) -> np.ndarray:
    """A lower bound of every point's distance to the nearest building (the
    distance to its bounding box)."""
    best = np.full(len(pts), np.inf)
    for poly, h in zip(polys, heights):
        lo, hi = _bbox(poly)
        dx = np.maximum(np.maximum(lo[0] - pts[:, 0], pts[:, 0] - hi[0]), 0.0)
        dy = np.maximum(np.maximum(lo[1] - pts[:, 1], pts[:, 1] - hi[1]), 0.0)
        dz = np.maximum(pts[:, 2] - h, 0.0)
        np.minimum(best, np.sqrt(dx * dx + dy * dy + dz * dz), out=best)
    return best


def build_mesh(polys: list[np.ndarray], heights: np.ndarray, height: float,
               chunk: int) -> dict:
    """Volume points (snapped cell centres), surface points and the index of
    the building each surface point lies on."""
    bounds = z_layers(height)
    centres, half, layers = [], [], []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if lo >= DOMAIN_TOP:
            break
        zc = 0.5 * (lo + hi)
        d = horizontal_cell(zc, height)
        # one extra cell beyond the domain edge: snapping may move it inside
        n = int(math.ceil((DOMAIN_HALF + d) / d))
        axis = (np.arange(-n, n) + 0.5) * d
        axis = axis[np.abs(axis) <= DOMAIN_HALF + d]
        gx, gy = np.meshgrid(axis, axis, indexing="ij")
        centres.append(np.stack([gx.ravel(), gy.ravel(),
                                 np.full(gx.size, zc)], axis=1))
        half.append(np.tile([0.5 * d, 0.5 * d, 0.5 * (hi - lo)],
                            (gx.size, 1)))
        layers.append((float(lo), float(hi), d))
    centres = np.concatenate(centres)
    half = np.concatenate(half)

    # castellation: cells whose centre is inside a building are removed
    keep = ~inside_buildings(centres, polys, heights)

    # snapping: a cell whose centre lies within twice its half-extent h of a
    # building surface (along the direction to that surface) is the first
    # cell layer; its near face moves onto the surface, so its centre moves
    # to distance (s + h) / 2. Its surface face is the projection of the
    # original centre onto the surface.
    scene = Extrusion25D(polys, np.zeros(len(polys)), heights)
    near = np.flatnonzero(keep & (bbox_distance(centres, polys, heights)
                                  <= 2.0 * np.linalg.norm(half, axis=1)))
    s, sdir, bld = scene.sdf(centres[near], chunk=chunk, return_building=True)
    h = (half[near] * np.abs(sdir)).sum(axis=1)
    keep[near[s <= 1e-6]] = False          # centre on a surface: degenerate
    first = (s > 1e-6) & (s <= 2.0 * h)
    idx = near[first]
    surf_pos = centres[idx] - s[first, None] * sdir[first]
    surf_bld = bld[first]
    centres[idx] += (0.5 * (h[first] - s[first]))[:, None] * sdir[first]

    vol_pos = centres[keep]
    in_domain = (np.abs(vol_pos[:, 0]) <= DOMAIN_HALF) \
        & (np.abs(vol_pos[:, 1]) <= DOMAIN_HALF) \
        & (vol_pos[:, 2] <= DOMAIN_TOP)
    vol_pos = vol_pos[in_domain]

    # one point per surface position, inside the domain
    _, uniq = np.unique(np.round(surf_pos, 6), axis=0, return_index=True)
    uniq = np.sort(uniq)
    surf_pos, surf_bld = surf_pos[uniq], surf_bld[uniq]
    in_domain = (np.abs(surf_pos[:, 0]) <= DOMAIN_HALF) \
        & (np.abs(surf_pos[:, 1]) <= DOMAIN_HALF) \
        & (surf_pos[:, 2] <= DOMAIN_TOP)
    return {"vol_pos": vol_pos, "surf_pos": surf_pos[in_domain],
            "surf_bld": surf_bld[in_domain], "layers": layers,
            "n_snapped": int(first.sum()), "scene": scene}


# ---------------------------------------------------------------------------
# Steps 2 and 3: features and latent layout
# ---------------------------------------------------------------------------

def domain_footprints(polys: list[np.ndarray], heights: np.ndarray,
                      pp: PreprocessConfig):
    """The buildings as the training data describes them: footprints cut at
    the domain edge and canonicalized (the footprint extraction's
    simplification), roofs at most at the domain top. Returns (footprints,
    z_hi, index), index[i] = the footprint of input building i or -1."""
    from shapely.geometry import Polygon, box

    domain = box(-DOMAIN_HALF, -DOMAIN_HALF, DOMAIN_HALF, DOMAIN_HALF)
    fps, z_hi = [], []
    index = np.full(len(polys), -1, dtype=np.int64)
    for i, (poly, h) in enumerate(zip(polys, heights)):
        fp = canonical_footprint(Polygon(poly).intersection(domain),
                                 pp.dp_tol, pp.min_edge_collapse)
        if fp is None:
            continue
        index[i] = len(fps)
        fps.append(fp)
        z_hi.append(min(float(h), DOMAIN_TOP))
    return fps, np.asarray(z_hi, dtype=np.float64), index


def prepare_case(name: str, polys: list[np.ndarray], heights: np.ndarray,
                 mcfg: ModelConfig, pp: PreprocessConfig, stats: dict
                 ) -> tuple[dict, dict]:
    """The model input dict (the layout and dtypes of an estate pack, as
    evaluation reads it) and the physical positions for the outputs."""
    cd = float(stats["coord_divisor"])
    chunk = int(pp.geom_chunk)
    wind = np.asarray(pp.wind_direction, dtype=np.float64)
    wind = wind / np.linalg.norm(wind)

    t0 = time.time()
    height = reference_height(heights)
    mesh = build_mesh(polys, heights, height, chunk)
    fine = [lay for lay in mesh["layers"] if lay[2] == REFINEMENT_BOXES[0][1]]
    log(f"[mesh] H = {height:.0f} m (tallest building {heights.max():.1f} m), "
        f"{len(mesh['layers'])} z layers start below {DOMAIN_TOP:.0f} m "
        f"(first {mesh['layers'][0][1]:.2f} m thick), "
        f"{len(fine)} of them at {REFINEMENT_BOXES[0][1]} m horizontally")
    log(f"[mesh] {len(mesh['vol_pos']):,} volume points "
        f"({mesh['n_snapped']:,} snapped), {len(mesh['surf_pos']):,} "
        f"surface points ({time.time() - t0:.0f} s)")

    t0 = time.time()
    fps, z_hi, index = domain_footprints(polys, heights, pp)
    if not fps:
        raise SystemExit("no building inside the 1100 m x 1100 m domain")
    z_lo = np.zeros(len(fps))
    scene = Extrusion25D(fps, z_lo, z_hi)

    surf_bld = index[mesh["surf_bld"]]
    on_domain_bld = surf_bld >= 0
    surf_pos = mesh["surf_pos"][on_domain_bld]
    surf_bld = surf_bld[on_domain_bld]
    vol_pos = mesh["vol_pos"]
    n_vol, n_surf = len(vol_pos), len(surf_pos)
    if n_surf == 0:
        raise SystemExit("no building surface inside the domain")

    # Signed distance to the 2.5D buildings. A volume point inside a
    # (simplified) footprint takes its distance to the input geometry and
    # that direction, or |sdf| where those are unusable.
    sdf_vol, dir_vol = scene.sdf(vol_pos, chunk=chunk)
    inside = np.flatnonzero(sdf_vol < 0)
    if len(inside):
        r_sdf, r_dir = mesh["scene"].sdf(vol_pos[inside], chunk=chunk)
        use = (r_sdf > 0) & (np.linalg.norm(r_dir, axis=1) > 1e-6)
        sdf_vol[inside] = np.where(use, r_sdf, np.abs(sdf_vol[inside]))
        dir_vol[inside[use]] = r_dir[use]

    labels = [classify_edges(fp, wind, pp.wind_face_dot_threshold)
              for fp in fps]
    surf_normal, surf_dim2, surf_dim3 = surface_normals_and_dims(
        surf_pos, surf_bld, fps, z_lo, z_hi, labels, pp.threshold_edge)

    all_pos = np.concatenate([vol_pos, surf_pos], axis=0)
    pos_norm = all_pos / cd
    support_norm = scene.support(all_pos, support_max=pp.support_max,
                                 chunk=chunk) / cd
    vol_dim2 = volume_region_dim2(vol_pos[:, 2], sdf_vol,
                                  pp.near_ground_z_threshold,
                                  pp.near_building_sdf_threshold)
    region_dim2 = np.concatenate([vol_dim2, surf_dim2])
    region_dim3 = np.concatenate([np.zeros(n_vol, dtype=np.int8), surf_dim3])

    morph = np.asarray(morph_raw(fps, z_hi - z_lo, 0, wind,
                                 pp.wind_face_dot_threshold))
    morph_norm = (morph - np.asarray(stats["morph_mean"], dtype=np.float64)) \
        / np.maximum(np.asarray(stats["morph_std"], dtype=np.float64), 1e-8)
    log(f"[features] {len(fps)} buildings in the domain, "
        f"{len(inside):,} volume points inside a simplified footprint "
        f"({time.time() - t0:.0f} s)")

    t0 = time.time()
    sdf_norm_vol = sdf_vol / cd
    layout = build_latent_layout(
        pos_norm[:n_vol], sdf_norm_vol, dir_vol, support_norm[:n_vol],
        pos_norm[n_vol:], surf_normal, support_norm[n_vol:],
        weighting=pp.kdtree_weighting, invsqrt_floor=pp.invsqrt_floor,
        grid_dims=pp.grid_dims, n_levels=mcfg.n_kd_levels,
        n_pn_neighbors=mcfg.n_pn_neighbors, pn_surf_first=pp.pn_surf_first,
        coord_divisor=cd, seed_name=name)
    log(f"[layout] {mcfg.n_leaves} leaves ({pp.kdtree_weighting}), "
        f"{layout['n_empty']} empty ({time.time() - t0:.0f} s)")

    # stored precision of the training packs: sdf, support, leaf statistics
    # and PointNet features in bf16, the sdf direction as fp16 angles
    bf16 = torch.bfloat16
    query_pos = torch.from_numpy(pos_norm.astype(np.float32))
    leaf_pos = torch.from_numpy(layout["centroids"].astype(np.float32))
    sdf_norm = np.concatenate([sdf_norm_vol, np.zeros(n_surf)])
    sdf_dir = np.concatenate([dir_vol, surf_normal], axis=0)
    case = {
        "case_name": name,
        "N_vol": n_vol,
        "N_surf": n_surf,
        "query_pos_norm": query_pos,
        "query_sdf_norm": torch.from_numpy(
            sdf_norm.astype(np.float32)).to(bf16).float(),
        "query_sdf_dir": decode_sdf_dir_angles(
            encode_sdf_dir_angles(torch.from_numpy(sdf_dir))),
        "query_support_norm": torch.from_numpy(
            support_norm.astype(np.float32)).to(bf16),
        "region_dim1": torch.cat([torch.zeros(n_vol, dtype=torch.int8),
                                  torch.ones(n_surf, dtype=torch.int8)]),
        "region_dim2": torch.from_numpy(region_dim2),
        "region_dim3": torch.from_numpy(region_dim3),
        "idw_indices": knn_query(cKDTree(leaf_pos.numpy()), query_pos,
                                 int(mcfg.idw_k), workers=-1),
        "leaf_pos_norm": leaf_pos,
        "leaf_stats": torch.from_numpy(
            layout["leaf_stats"].astype(np.float32)).to(bf16),
        "pn_points_feat": torch.from_numpy(layout["pn_feat"]).to(bf16),
        "pn_mask": torch.from_numpy(layout["pn_mask"]),
        "global_morph_norm": torch.from_numpy(morph_norm.astype(np.float32)),
    }
    geo = {"vol_pos": vol_pos, "surf_pos": surf_pos,
           "surf_normal": surf_normal, "surf_building": surf_bld,
           "footprints": fps, "z_hi": z_hi, "height": height,
           "layer_bounds": np.array([lay[:2] for lay in mesh["layers"]])}
    return case, geo


# ---------------------------------------------------------------------------
# Steps 4 and 5: inference and outputs
# ---------------------------------------------------------------------------

def load_model(weights_dir: Path, mcfg: ModelConfig,
               device: torch.device) -> IBO:
    payload = torch.load(str(weights_dir / "swa_model.pt"),
                         map_location="cpu", weights_only=True)
    sd = payload["model_state_dict"] \
        if isinstance(payload, dict) and "model_state_dict" in payload \
        else payload
    model = IBO(mcfg)
    try:
        load_state_strict(model, sd)
    except RuntimeError as exc:
        raise SystemExit(f"swa_model.pt does not match the architecture in "
                         f"config_used.yaml:\n{exc}")
    return model.to(device).eval()


def write_vtk_points(path: Path, points: np.ndarray, fields: dict,
                     title: str) -> None:
    """Legacy binary VTK polydata: one vertex per point, point data as
    3-vectors or scalars."""
    n = len(points)
    cells = np.empty((n, 2), dtype=">i4")
    cells[:, 0] = 1
    cells[:, 1] = np.arange(n)
    with open(path, "wb") as f:
        f.write(b"# vtk DataFile Version 3.0\n")
        f.write(title.encode("ascii", "replace")[:250] + b"\n")
        f.write(b"BINARY\nDATASET POLYDATA\n")
        f.write(f"POINTS {n} float\n".encode())
        f.write(np.ascontiguousarray(points, dtype=">f4").tobytes())
        f.write(f"\nVERTICES {n} {2 * n}\n".encode())
        f.write(cells.tobytes())
        f.write(f"\nPOINT_DATA {n}\n".encode())
        for name, arr in fields.items():
            arr = np.ascontiguousarray(arr, dtype=">f4")
            if arr.ndim == 2 and arr.shape[1] == 3:
                f.write(f"VECTORS {name} float\n".encode())
            else:
                f.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n"
                        .encode())
            f.write(arr.tobytes())
            f.write(b"\n")


def save_outputs(out_dir: Path, name: str, geo: dict, vol: np.ndarray,
                 surf: np.ndarray) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"{name}.npz"
    np.savez(npz,
             volume_pos=geo["vol_pos"].astype(np.float32),
             volume_fields=vol.astype(np.float32),
             volume_channels=np.array(VOL_CHANNEL_NAMES),
             surface_pos=geo["surf_pos"].astype(np.float32),
             surface_normal=geo["surf_normal"].astype(np.float32),
             surface_fields=surf.astype(np.float32),
             surface_channels=np.array(SURF_CHANNEL_NAMES),
             surface_building=geo["surf_building"].astype(np.int32),
             reference_height=np.float64(geo["height"]),
             layer_bounds=geo["layer_bounds"])
    vtk_vol = out_dir / f"{name}_volume.vtk"
    write_vtk_points(vtk_vol, geo["vol_pos"],
                     {"U": vol[:, 0:3], "p": vol[:, 3], "nut": vol[:, 4]},
                     f"{name} volume (IBO prediction, model frame)")
    vtk_surf = out_dir / f"{name}_surface.vtk"
    write_vtk_points(vtk_surf, geo["surf_pos"],
                     {"p": surf[:, 0], "normal": geo["surf_normal"]},
                     f"{name} surface (IBO prediction, model frame)")
    return [npz, vtk_vol, vtk_surf]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geometry", required=True,
                    help="buildings JSON (see above)")
    ap.add_argument("--weights", required=True,
                    help="directory with swa_model.pt, config_used.yaml and "
                         "norm_stats.json")
    ap.add_argument("--out", default="outputs", help="output directory")
    args = ap.parse_args()

    weights_dir = Path(args.weights).expanduser()
    mcfg, pp, stats = read_weights(weights_dir)
    name, polys, heights = read_geometry(Path(args.geometry).expanduser())
    log(f"[input] {name}: {len(polys)} buildings, heights "
        f"{heights.min():.1f}-{heights.max():.1f} m")

    case, geo = prepare_case(name, polys, heights, mcfg, pp, stats)

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # bf16 autocast as in training where the GPU supports it, fp32 otherwise
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    model = load_model(weights_dir, mcfg, device)
    pred_vol_z, pred_surf_z = infer_case(model, case, device, DECODE_CHUNK,
                                         use_bf16)
    vol = inverse_transform_vol(pred_vol_z, geo["vol_pos"][:, 2], stats)
    surf = inverse_transform_surf(pred_surf_z, geo["surf_pos"][:, 2], stats)
    log(f"[model] {device.type}{' bf16' if use_bf16 else ' fp32'}, "
        f"{case['N_vol'] + case['N_surf']:,} points "
        f"({time.time() - t0:.0f} s)")
    if not (np.isfinite(vol).all() and np.isfinite(surf).all()):
        raise SystemExit("non-finite predictions")

    speed = np.linalg.norm(vol[:, 0:3], axis=1)
    log(f"[model] |U| {speed.min():.2f}-{speed.max():.2f} m/s "
        f"(mean {speed.mean():.2f}), p_surf {surf.min():.1f}-"
        f"{surf.max():.1f}")
    for path in save_outputs(Path(args.out).expanduser(), name, geo, vol,
                             surf):
        log(f"[output] {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
