#!/usr/bin/env python
"""Predict the wind field around an estate with a trained IBO model.

    python predict_new_case.py --geometry examples/toy_estate.json \\
        --weights <weights dir> --out outputs
    python predict_new_case.py --raw-pt raw/my_estate_N.pt \\
        --weights <weights dir> --out outputs

<weights dir> holds the three files of a trained run: swa_model.pt,
config_used.yaml and norm_stats.json. No dataset or cache is needed.

Two kinds of input:

--geometry  a JSON file of 2.5D buildings (a footprint polygon extruded from
            the ground to its height), in metres, in the model frame: origin
            at the centre of the 1100 m x 1100 m domain, ground at z = 0,
            mean wind blowing toward -y.

                {"name": "toy_estate",
                 "buildings": [{"footprint": [[x, y], [x, y], ...],
                                "height": 45.0}, ...]}

            For another wind direction, rotate the footprints so that the
            wind blows toward -y. Footprints are simple polygons (holes are
            ignored); buildings may extend beyond the domain and are cut at
            its edge. The points are placed like the CFD mesh's.

--raw-pt    a raw UrbanWind case file (<estate>_<dir>.pt: the output of
            convert_cfd_case.py, or a file of the dataset). The prediction
            is made at the file's own points, with the building footprints
            extracted from its surface points as the dataset preprocessing
            does, and scored against the file's CFD fields.

Steps:
  1. points    --geometry: the vertices of the UrbanWind CFD mesh: the
               blockMesh z layers (graded 50:1 up to six times the reference
               height H, the tallest building rounded up to 10 m), the
               snappyHexMesh horizontal refinement boxes (3.125 m below
               1.5 H, doubling at 2 H, 2.5 H, 3 H and 3.5 H, 100 m above);
               the vertices of cells inside buildings are removed unless a
               fluid cell shares them, in which case they are snapped onto
               the building surface. Surface points are the wall faces of
               the first cell layer, placed by projecting its cell centres
               onto the buildings. Both are cropped to |x|, |y| <= 550 m and
               z <= 160 m.
               --raw-pt: the file's volume and surface points.
  2. features  Signed distance, ray support, surface normals, region codes
               and morphology from the 2.5D geometry, with the same code and
               settings as the training data (src/preprocess).
  3. layout    Importance-weighted KD-tree, leaf statistics and PointNet
               samples (src/preprocess/features.py).
  4. model     IBO built from config_used.yaml, strict-loaded from
               swa_model.pt; CUDA when available, else CPU.
  5. output    <out>/<name>.npz and legacy VTK point clouds of the volume and
               surface predictions (<name>_volume.vtk, <name>_surface.vtk),
               in the model frame. With --raw-pt, both also carry the CFD
               fields, and <name>_metrics.json holds the paper's metrics: the
               relative L2 error of every channel in physical units and the
               z-score MSE.
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
from src.evaluation.test_worker import infer_case, metrics_from_predictions
from src.model.full_model import IBO
from src.preprocess.build import load_raw_case
from src.preprocess.features import (build_latent_layout, classify_edges,
                                     morph_raw, surface_normals_and_dims,
                                     volume_region_dim2)
from src.preprocess.geometry import (Extrusion25D, FootprintQCError,
                                     canonical_footprint, extract_buildings,
                                     point_in_polygon)
from src.training.checkpoint import load_state_strict

WEIGHT_FILES = ("swa_model.pt", "config_used.yaml", "norm_stats.json")

DOMAIN_HALF = 550.0      # output domain: |x|, |y| <= 550 m (mesh vertices)
DOMAIN_TOP = 160.0       # output domain: z <= 160 m (mesh vertices)
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
# Step 1 (--geometry): the CFD-like point set
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
    """Volume points (the vertices of the castellated and snapped mesh),
    surface points (the wall faces of the first cell layer) and the index of
    the building each surface point lies on."""
    bounds = z_layers(height)
    layers = []                       # (lo, hi, horizontal cell) below the top
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if lo >= DOMAIN_TOP:
            break
        layers.append((float(lo), float(hi),
                       horizontal_cell(0.5 * (lo + hi), height)))
    scene = Extrusion25D(polys, np.zeros(len(polys)), heights)

    # Vertices lie on every layer boundary up to the domain top, on the
    # lattice i d of the finer of the two layers the boundary separates (the
    # hanging nodes of a refinement interface belong to the fine layer).
    # Castellation keeps a vertex outside the buildings, and a vertex inside
    # one when a cell around it has a fluid centre; snapping then moves those
    # inside vertices onto the building surface.
    vol_parts, n_snapped = [], 0
    for k in range(len(layers) + 1):
        z = float(bounds[k])
        if z > DOMAIN_TOP:
            break
        around = [lay for lay in (layers[k - 1] if k >= 1 else None,
                                  layers[k] if k < len(layers) else None)
                  if lay is not None]
        d = min(lay[2] for lay in around)
        n = int(math.floor(DOMAIN_HALF / d + 1e-9))
        axis = np.arange(-n, n + 1) * d
        gx, gy = np.meshgrid(axis, axis, indexing="ij")
        v = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)])
        inside = inside_buildings(v, polys, heights)
        keep = ~inside
        idx = np.flatnonzero(inside)
        if len(idx):
            probes = []
            for lo, hi, dl in around:
                zc = 0.5 * (lo + hi)
                for sx in (-0.5, 0.5):
                    for sy in (-0.5, 0.5):
                        probes.append(v[idx] + np.array([sx * dl, sy * dl,
                                                         zc - z]))
            probes = np.stack(probes, axis=1)                # [m, cells, 3]
            fluid = ~inside_buildings(probes.reshape(-1, 3), polys, heights)
            snap = idx[fluid.reshape(len(idx), -1).any(axis=1)]
            if len(snap):
                s, sdir = scene.sdf(v[snap], chunk=chunk)
                v[snap] -= np.abs(s)[:, None] * sdir
                keep[snap] = True
                n_snapped += len(snap)
        vol_parts.append(v[keep])
    vol_pos = np.concatenate(vol_parts)
    in_domain = (np.abs(vol_pos[:, 0]) <= DOMAIN_HALF) \
        & (np.abs(vol_pos[:, 1]) <= DOMAIN_HALF) \
        & (vol_pos[:, 2] <= DOMAIN_TOP)
    vol_pos = vol_pos[in_domain]

    # Surface points: a cell whose centre lies within twice its half-extent
    # of a building surface (along the direction to that surface) belongs to
    # the first cell layer; its wall face sits where the centre projects
    # onto the surface.
    centres, half = [], []
    for lo, hi, d in layers:
        n = int(math.ceil((DOMAIN_HALF + d) / d))
        axis = (np.arange(-n, n) + 0.5) * d
        axis = axis[np.abs(axis) <= DOMAIN_HALF + d]
        gx, gy = np.meshgrid(axis, axis, indexing="ij")
        centres.append(np.column_stack([gx.ravel(), gy.ravel(),
                                        np.full(gx.size, 0.5 * (lo + hi))]))
        half.append(np.tile([0.5 * d, 0.5 * d, 0.5 * (hi - lo)],
                            (gx.size, 1)))
    centres = np.concatenate(centres)
    half = np.concatenate(half)
    fluid = ~inside_buildings(centres, polys, heights)
    near = np.flatnonzero(fluid & (bbox_distance(centres, polys, heights)
                                   <= 2.0 * np.linalg.norm(half, axis=1)))
    s, sdir, bld = scene.sdf(centres[near], chunk=chunk, return_building=True)
    h = (half[near] * np.abs(sdir)).sum(axis=1)
    first = (s > 1e-6) & (s <= 2.0 * h)
    surf_pos = centres[near[first]] - s[first, None] * sdir[first]
    surf_bld = bld[first]

    # one point per surface position, inside the domain
    _, uniq = np.unique(np.round(surf_pos, 6), axis=0, return_index=True)
    uniq = np.sort(uniq)
    surf_pos, surf_bld = surf_pos[uniq], surf_bld[uniq]
    in_domain = (np.abs(surf_pos[:, 0]) <= DOMAIN_HALF) \
        & (np.abs(surf_pos[:, 1]) <= DOMAIN_HALF) \
        & (surf_pos[:, 2] <= DOMAIN_TOP)
    return {"vol_pos": vol_pos, "surf_pos": surf_pos[in_domain],
            "surf_bld": surf_bld[in_domain], "layers": layers,
            "n_snapped": n_snapped, "scene": scene}


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


# ---------------------------------------------------------------------------
# Steps 2 and 3: features and latent layout
# ---------------------------------------------------------------------------

def prepare_features(name: str, vol_pos: np.ndarray, surf_pos: np.ndarray,
                     surf_bld: np.ndarray, fps: list[np.ndarray],
                     z_lo: np.ndarray, z_hi: np.ndarray, inside_sdf,
                     mcfg: ModelConfig, pp: PreprocessConfig, stats: dict
                     ) -> tuple[dict, np.ndarray]:
    """The model input dict (the layout and dtypes of an estate pack, as
    evaluation reads it) and the surface normals, from physical positions
    and the 2.5D buildings (footprints with their z ranges; surf_bld is the
    footprint index of each surface point).

    A volume point inside a footprint is fluid the 2.5D geometry fills (an
    open ground floor, a stepped roof); inside_sdf(idx) returns the distance
    of those points to the true geometry and its unit direction, used where
    the distance is positive, |sdf| otherwise."""
    cd = float(stats["coord_divisor"])
    chunk = int(pp.geom_chunk)
    wind = np.asarray(pp.wind_direction, dtype=np.float64)
    wind = wind / np.linalg.norm(wind)
    n_vol, n_surf = len(vol_pos), len(surf_pos)
    if n_surf == 0:
        raise SystemExit("no building surface inside the domain")

    t0 = time.time()
    scene = Extrusion25D(fps, z_lo, z_hi)
    sdf_vol, dir_vol = scene.sdf(vol_pos, chunk=chunk)
    inside = np.flatnonzero(sdf_vol < 0)
    if len(inside):
        r_sdf, r_dir = inside_sdf(inside)
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
        f"{len(inside):,} volume points inside a footprint "
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
    return case, surf_normal


def _layer_bounds(height: float) -> np.ndarray:
    bounds = z_layers(height)
    return np.array([[lo, hi] for lo, hi in zip(bounds[:-1], bounds[1:])
                     if lo < DOMAIN_TOP])


def prepare_case(name: str, polys: list[np.ndarray], heights: np.ndarray,
                 mcfg: ModelConfig, pp: PreprocessConfig, stats: dict
                 ) -> tuple[dict, dict]:
    """--geometry: the CFD-like point set of the buildings, the model input
    and the physical positions for the outputs."""
    chunk = int(pp.geom_chunk)
    t0 = time.time()
    height = reference_height(heights)
    mesh = build_mesh(polys, heights, height, chunk)
    fine = [lay for lay in mesh["layers"] if lay[2] == REFINEMENT_BOXES[0][1]]
    log(f"[mesh] H = {height:.0f} m (tallest building {heights.max():.1f} m), "
        f"{len(mesh['layers'])} z layers start below {DOMAIN_TOP:.0f} m "
        f"(first {mesh['layers'][0][1]:.2f} m thick), "
        f"{len(fine)} of them at {REFINEMENT_BOXES[0][1]} m horizontally")
    log(f"[mesh] {len(mesh['vol_pos']):,} volume points "
        f"({mesh['n_snapped']:,} snapped onto buildings), "
        f"{len(mesh['surf_pos']):,} surface points ({time.time() - t0:.0f} s)")

    fps, z_hi, index = domain_footprints(polys, heights, pp)
    if not fps:
        raise SystemExit("no building inside the 1100 m x 1100 m domain")
    z_lo = np.zeros(len(fps))
    surf_bld = index[mesh["surf_bld"]]
    on_domain_bld = surf_bld >= 0
    surf_pos = mesh["surf_pos"][on_domain_bld]
    surf_bld = surf_bld[on_domain_bld]
    vol_pos = mesh["vol_pos"]

    def inside_sdf(idx):
        return mesh["scene"].sdf(vol_pos[idx], chunk=chunk)

    case, surf_normal = prepare_features(
        name, vol_pos, surf_pos, surf_bld, fps, z_lo, z_hi, inside_sdf,
        mcfg, pp, stats)
    geo = {"vol_pos": vol_pos, "surf_pos": surf_pos,
           "surf_normal": surf_normal, "surf_building": surf_bld,
           "footprints": fps, "z_hi": z_hi, "height": height,
           "layer_bounds": np.array([lay[:2] for lay in mesh["layers"]])}
    return case, geo


def assign_surface_buildings(surf_pos: np.ndarray, fps: list[np.ndarray],
                             pp: PreprocessConfig) -> np.ndarray:
    """Surface point -> footprint index, as the dataset preprocessing does
    it: DBSCAN clusters matched to the nearest footprint centroid, noise
    points to the nearest centroid."""
    from sklearn.cluster import DBSCAN

    labels = DBSCAN(eps=pp.dbscan_eps, min_samples=pp.dbscan_min_samples
                    ).fit_predict(surf_pos[:, :2])
    ids = sorted(set(int(l) for l in labels) - {-1})
    tree = cKDTree(np.array([fp.mean(axis=0) for fp in fps]))
    surf_bld = np.zeros(len(surf_pos), dtype=np.int64)
    if ids:
        centroids = np.array([surf_pos[labels == l, :2].mean(axis=0)
                              for l in ids])
        alignment = tree.query(centroids)[1]
        label_to_bi = {lab: int(alignment[i]) for i, lab in enumerate(ids)}
        surf_bld = np.array([label_to_bi.get(int(l), 0) for l in labels],
                            dtype=np.int64)
    noise = labels == -1
    if noise.any():
        surf_bld[noise] = tree.query(surf_pos[noise, :2])[1]
    return surf_bld


def zscored_targets(raw, stats: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """The file's CFD fields as the training targets: detrended
    (Uy -= c ln max(z, z0), p += g z) and z-scored."""
    z = raw.volume_pos[:, 2]
    tv = raw.volume_fields.astype(np.float64).copy()
    tv[:, 1] -= stats["detrend_uy_coeff"] * np.log(
        np.maximum(z, stats["detrend_z_floor"]))
    tv[:, 3] += stats["detrend_p_coeff"] * z
    ts = raw.surface_fields[:, 0:1].astype(np.float64).copy()
    ts[:, 0] += stats["detrend_p_coeff"] * raw.surface_pos[:, 2]
    vol_mean = np.asarray(stats["vol_mean"], dtype=np.float64)
    vol_std = np.maximum(np.asarray(stats["vol_std"], dtype=np.float64), 1e-8)
    surf_mean = np.asarray(stats["surf_mean"], dtype=np.float64)
    surf_std = np.maximum(np.asarray(stats["surf_std"], dtype=np.float64),
                          1e-8)
    return (torch.from_numpy(((tv - vol_mean) / vol_std).astype(np.float32)),
            torch.from_numpy(((ts - surf_mean) / surf_std).astype(np.float32)))


def prepare_raw_case(path: Path, mcfg: ModelConfig, pp: PreprocessConfig,
                     stats: dict) -> tuple[dict, dict]:
    """--raw-pt: the model input at the file's own points, its targets for
    the metrics, and the positions and CFD fields for the outputs."""
    try:
        raw = load_raw_case(path)
    except Exception as exc:  # noqa: BLE001 — any unreadable file is fatal
        raise SystemExit(f"{path}: not a raw UrbanWind case ({exc})")
    name = raw.case_name
    log(f"[input] {name}: {raw.n_vol:,} volume points, {raw.n_surf:,} "
        f"surface points")

    t0 = time.time()
    try:
        buildings = extract_buildings(
            raw.surface_pos, raw.surface_normals,
            dbscan_eps=pp.dbscan_eps, dbscan_min_samples=pp.dbscan_min_samples,
            wall_nz_threshold=pp.wall_nz_threshold,
            hull_ratio=pp.concave_hull_ratio, dp_tol=pp.dp_tol,
            min_edge=pp.min_edge_collapse)
    except FootprintQCError as exc:
        raise SystemExit(f"{name}: footprint extraction from the surface "
                         f"points failed ({exc})")
    fps = [b.footprint for b in buildings]
    z_lo = np.array([b.z_lo for b in buildings], dtype=np.float64)
    z_hi = np.array([b.z_hi for b in buildings], dtype=np.float64)
    surf_bld = assign_surface_buildings(raw.surface_pos, fps, pp)
    log(f"[footprints] {len(fps)} buildings extracted from the surface "
        f"points, roofs {z_hi.min():.1f}-{z_hi.max():.1f} m "
        f"({time.time() - t0:.0f} s)")

    def inside_sdf(idx):
        g = raw.volume_sdf_grad[idx].astype(np.float64)
        gn = np.linalg.norm(g, axis=1, keepdims=True)
        return (raw.volume_sdf[idx].astype(np.float64),
                np.where(gn > 1e-6, g / np.maximum(gn, 1e-12), 0.0))

    case, surf_normal = prepare_features(
        name, raw.volume_pos, raw.surface_pos, surf_bld, fps, z_lo, z_hi,
        inside_sdf, mcfg, pp, stats)
    case["target_vol"], case["target_surf"] = zscored_targets(raw, stats)
    case["query_z_phys"] = torch.from_numpy(np.concatenate(
        [raw.volume_pos[:, 2], raw.surface_pos[:, 2]]).astype(np.float32))
    height = reference_height(z_hi)
    geo = {"vol_pos": raw.volume_pos, "surf_pos": raw.surface_pos,
           "surf_normal": surf_normal, "surf_building": surf_bld,
           "footprints": fps, "z_hi": z_hi, "height": height,
           "layer_bounds": _layer_bounds(height),
           "vol_true": raw.volume_fields.astype(np.float64),
           "surf_true": raw.surface_fields[:, 0:1].astype(np.float64)}
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
    """<name>.npz and the two VTK point clouds; with CFD fields in geo
    (--raw-pt) those go in as well, as *_cfd."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "volume_pos": geo["vol_pos"].astype(np.float32),
        "volume_fields": vol.astype(np.float32),
        "volume_channels": np.array(VOL_CHANNEL_NAMES),
        "surface_pos": geo["surf_pos"].astype(np.float32),
        "surface_normal": geo["surf_normal"].astype(np.float32),
        "surface_fields": surf.astype(np.float32),
        "surface_channels": np.array(SURF_CHANNEL_NAMES),
        "surface_building": geo["surf_building"].astype(np.int32),
        "reference_height": np.float64(geo["height"]),
        "layer_bounds": geo["layer_bounds"],
    }
    vol_fields = {"U": vol[:, 0:3], "p": vol[:, 3], "nut": vol[:, 4]}
    surf_fields = {"p": surf[:, 0], "normal": geo["surf_normal"]}
    if geo.get("vol_true") is not None:
        vt, st = geo["vol_true"], geo["surf_true"]
        arrays["volume_fields_cfd"] = vt.astype(np.float32)
        arrays["surface_fields_cfd"] = st.astype(np.float32)
        vol_fields.update({"U_cfd": vt[:, 0:3], "p_cfd": vt[:, 3],
                           "nut_cfd": vt[:, 4]})
        surf_fields["p_cfd"] = st[:, 0]
    npz = out_dir / f"{name}.npz"
    np.savez(npz, **arrays)
    vtk_vol = out_dir / f"{name}_volume.vtk"
    write_vtk_points(vtk_vol, geo["vol_pos"], vol_fields,
                     f"{name} volume (IBO prediction, model frame)")
    vtk_surf = out_dir / f"{name}_surface.vtk"
    write_vtk_points(vtk_surf, geo["surf_pos"], surf_fields,
                     f"{name} surface (IBO prediction, model frame)")
    return [npz, vtk_vol, vtk_surf]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--geometry", help="buildings JSON (see above)")
    source.add_argument("--raw-pt",
                        help="raw UrbanWind case file: predict at its points "
                             "and score against its CFD fields")
    ap.add_argument("--weights", required=True,
                    help="directory with swa_model.pt, config_used.yaml and "
                         "norm_stats.json")
    ap.add_argument("--out", default="outputs", help="output directory")
    args = ap.parse_args()

    weights_dir = Path(args.weights).expanduser()
    mcfg, pp, stats = read_weights(weights_dir)
    if args.geometry:
        name, polys, heights = read_geometry(Path(args.geometry).expanduser())
        log(f"[input] {name}: {len(polys)} buildings, heights "
            f"{heights.min():.1f}-{heights.max():.1f} m")
        case, geo = prepare_case(name, polys, heights, mcfg, pp, stats)
    else:
        case, geo = prepare_raw_case(Path(args.raw_pt).expanduser(), mcfg, pp,
                                     stats)
        name = case["case_name"]

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # bf16 autocast as in training where the GPU supports it, fp32 otherwise
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    model = load_model(weights_dir, mcfg, device)
    pred_vol_z, pred_surf_z = infer_case(model, case, device, DECODE_CHUNK,
                                         use_bf16)
    metrics = None
    if "target_vol" in case:
        metrics, vol, surf = metrics_from_predictions(name, case, stats,
                                                      pred_vol_z, pred_surf_z)
    else:
        vol = inverse_transform_vol(pred_vol_z, geo["vol_pos"][:, 2], stats)
        surf = inverse_transform_surf(pred_surf_z, geo["surf_pos"][:, 2],
                                      stats)
    log(f"[model] {device.type}{' bf16' if use_bf16 else ' fp32'}, "
        f"{case['N_vol'] + case['N_surf']:,} points "
        f"({time.time() - t0:.0f} s)")
    if not (np.isfinite(vol).all() and np.isfinite(surf).all()):
        raise SystemExit("non-finite predictions")

    speed = np.linalg.norm(vol[:, 0:3], axis=1)
    log(f"[model] |U| {speed.min():.2f}-{speed.max():.2f} m/s "
        f"(mean {speed.mean():.2f}), p_surf {surf.min():.1f}-"
        f"{surf.max():.1f}")
    out_dir = Path(args.out).expanduser()
    paths = save_outputs(out_dir, name, geo, vol, surf)
    if metrics is not None:
        rl2 = metrics["relative_l2_physical"]
        log("[metrics] relative L2 in physical units: "
            + ", ".join(f"{k} {v:.3f}" for k, v in rl2["volume_channels"].items())
            + f", p_surf {rl2['surface_channels']['p_surf']:.3f}")
        metrics_path = out_dir / f"{name}_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2),
                                encoding="utf-8")
        paths.append(metrics_path)
    for path in paths:
        log(f"[output] {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
