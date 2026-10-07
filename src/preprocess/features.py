"""Per-point and per-leaf model inputs, computed from geometry alone.

The single implementation shared by the dataset pack builder
(preprocess/build.py) and the new-case pipeline (predict_new_case.py):

  surface normals and region codes   surface_normals_and_dims
                                     volume_region_dim2
  morphology conditioners            morph_raw
  latent layout                      build_latent_layout: importance weights
                                     -> weighted KD-tree -> leaf centroids,
                                     19-dim leaf statistics and the PointNet
                                     member samples (surface points first)

Positions entering the layout are already divided by coord_divisor (550 m).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from ..utils.seed import per_case_epoch_seed
from .geometry import (point_to_roof, point_to_segments_3d,
                       point_to_walls_dist, polygon_area, rot_poly)
from .kdtree import (KDTree, assign_points_to_leaves, build_structured_grid,
                     build_weighted_kdtree, compute_leaf_bounds)

# Wind label of a wall edge
WINDWARD, LEEWARD, SIDE_L, SIDE_R = 0, 1, 2, 3

# region_dim2 surface codes / region_dim3 edge kinds
SURF_WINDWARD, SURF_LEEWARD, SURF_TOP, SURF_SIDE = 0, 1, 2, 3
KIND_EAVE, KIND_VERT, KIND_BASE = 1, 2, 3


# ---------------------------------------------------------------------------
# Wind labels, surface normals and region codes
# ---------------------------------------------------------------------------

def classify_edges(poly: np.ndarray, wind: np.ndarray,
                   dot_threshold: float) -> np.ndarray:
    """Wind label of every edge of a CCW polygon. poly [k, 2] -> int8 [k]."""
    e = np.roll(poly, -1, axis=0) - poly
    ln = np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
    n = np.stack([e[:, 1], -e[:, 0]], axis=1) / ln  # outward for CCW
    dot = n @ wind
    labels = np.where(dot < -dot_threshold, WINDWARD,
                      np.where(dot > dot_threshold, LEEWARD,
                               np.where(n[:, 0] < 0, SIDE_L, SIDE_R)))
    return labels.astype(np.int8)


def _wind_group(edge_label: int) -> int:
    """0 = windward, 1 = leeward, 2 = side (left/right merged)."""
    if edge_label == WINDWARD:
        return 0
    if edge_label == LEEWARD:
        return 1
    return 2


def _dim2_from_edge(edge_label: int) -> int:
    if edge_label == WINDWARD:
        return SURF_WINDWARD
    if edge_label == LEEWARD:
        return SURF_LEEWARD
    return SURF_SIDE


def surface_normals_for_building(
    pts: np.ndarray, poly: np.ndarray, z_lo: float, z_hi: float,
    edge_wind: np.ndarray, threshold_edge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normals and region codes of the surface points of ONE building.

    pts [n, 3] metres; poly [k, 2] CCW; edge_wind [k] wind labels. Each point
    takes the normal of its nearest wall or of the roof; points within
    threshold_edge of a vertical edge, eave or base edge take the blended
    normal of the faces meeting there (priority: roof corner > vertical edge
    > eave > base > plain face). Returns (normals [n, 3], dim2 [n] int8,
    dim3 [n] int8). Only the codes depend on the wind.
    """
    n = len(pts)
    k = len(poly)
    v0 = poly
    v1 = np.roll(poly, -1, axis=0)
    e = v1 - v0
    ln = np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
    wall_n2 = np.stack([e[:, 1], -e[:, 0]], axis=1) / ln
    wall_n3 = np.concatenate([wall_n2, np.zeros((k, 1))], axis=1)

    dW = point_to_walls_dist(pts, v0, v1, np.full(k, z_lo), np.full(k, z_hi))
    dR, _ = point_to_roof(pts, poly, z_hi)
    j_wall = np.argmin(dW, axis=1)
    d_wall = dW[np.arange(n), j_wall]
    on_roof = dR < d_wall

    normals = wall_n3[j_wall].copy()
    normals[on_roof] = (0.0, 0.0, 1.0)
    dim2 = np.array([_dim2_from_edge(int(edge_wind[j])) for j in j_wall],
                    dtype=np.int8)
    dim2[on_roof] = SURF_TOP
    dim3 = np.zeros(n, dtype=np.int8)

    lo3 = np.concatenate([poly, np.full((k, 1), z_lo)], axis=1)
    hi3 = np.concatenate([poly, np.full((k, 1), z_hi)], axis=1)
    d_vert = point_to_segments_3d(pts, lo3, hi3)
    d_eave = point_to_segments_3d(pts, hi3, np.roll(hi3, -1, axis=0))
    d_base = point_to_segments_3d(pts, lo3, np.roll(lo3, -1, axis=0))
    d_topc = np.linalg.norm(pts[:, None, :] - hi3[None, :, :], axis=-1)

    jv, dv = np.argmin(d_vert, axis=1), d_vert.min(axis=1)
    je, de = np.argmin(d_eave, axis=1), d_eave.min(axis=1)
    jb, db = np.argmin(d_base, axis=1), d_base.min(axis=1)
    jt, dt = np.argmin(d_topc, axis=1), d_topc.min(axis=1)

    prev = lambda i: (i - 1) % k  # noqa: E731

    def _norm(v):
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)

    up = np.array([0.0, 0.0, 1.0])
    down = np.array([0.0, 0.0, -1.0])

    sel = db < threshold_edge
    if sel.any():
        i = jb[sel]
        normals[sel] = _norm(wall_n3[i] + down)
        dim3[sel] = (np.array([_wind_group(int(edge_wind[x])) for x in i]) * 3
                     + KIND_BASE).astype(np.int8)
    sel = de < threshold_edge
    if sel.any():
        i = je[sel]
        normals[sel] = _norm(wall_n3[i] + up)
        dim3[sel] = (np.array([_wind_group(int(edge_wind[x])) for x in i]) * 3
                     + KIND_EAVE).astype(np.int8)
    sel = dv < threshold_edge
    if sel.any():
        i = jv[sel]
        pi = prev(i)
        normals[sel] = _norm(wall_n3[i] + wall_n3[pi])
        lab = np.where(j_wall[sel] == pi, pi, i)
        dim3[sel] = (np.array([_wind_group(int(edge_wind[x])) for x in lab]) * 3
                     + KIND_VERT).astype(np.int8)
    sel = dt < threshold_edge
    if sel.any():
        i = jt[sel]
        pi = prev(i)
        normals[sel] = _norm(wall_n3[i] + wall_n3[pi] + up)
        lab = np.where(j_wall[sel] == pi, pi, i)
        dim3[sel] = (np.array([_wind_group(int(edge_wind[x])) for x in lab]) * 3
                     + KIND_VERT).astype(np.int8)
    return normals, dim2, dim3


def surface_normals_and_dims(surf_pos: np.ndarray, surf_bld: np.ndarray,
                             footprints: list[np.ndarray], z_lo: np.ndarray,
                             z_hi: np.ndarray, edge_labels: list[np.ndarray],
                             threshold_edge: float
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """surface_normals_for_building over every building. surf_bld [n] is the
    building index of each surface point; edge_labels[b] the wind labels of
    building b's edges. Returns (normals [n, 3], dim2 [n], dim3 [n])."""
    n = len(surf_pos)
    normals = np.zeros((n, 3), dtype=np.float64)
    dim2 = np.zeros(n, dtype=np.int8)
    dim3 = np.zeros(n, dtype=np.int8)
    for bi in range(len(footprints)):
        idx = np.where(surf_bld == bi)[0]
        if len(idx) == 0:
            continue
        nrm, d2, d3 = surface_normals_for_building(
            surf_pos[idx], footprints[bi], float(z_lo[bi]), float(z_hi[bi]),
            edge_labels[bi], threshold_edge)
        normals[idx] = nrm
        dim2[idx] = d2
        dim3[idx] = d3
    return normals, dim2, dim3


def volume_region_dim2(z: np.ndarray, sdf_phys: np.ndarray,
                       near_ground_z: float, near_building_sdf: float
                       ) -> np.ndarray:
    """Volume region code: 0 near a building (sdf < near_building_sdf),
    1 near the ground (z < near_ground_z), 2 elsewhere."""
    dim2 = np.full(len(z), 2, dtype=np.int8)
    dim2[z < near_ground_z] = 1
    dim2[sdf_phys < near_building_sdf] = 0
    return dim2


# ---------------------------------------------------------------------------
# Morphology conditioners
# ---------------------------------------------------------------------------

MORPH_NAMES = ("log_n_buildings", "lambda_p_raw", "lambda_f_raw", "h_mean",
               "h_max")


def morph_raw(footprints: list[np.ndarray], heights: np.ndarray, k: int,
              wind: np.ndarray, dot_threshold: float) -> list[float]:
    """[log(n_buildings), sum of footprint areas, sum over windward edges of
    length x height, mean height, max height] for the geometry rotated by k
    counter-clockwise quarter turns, against the fixed wind direction."""
    lam_p = 0.0
    lam_f = 0.0
    for bi, poly in enumerate(footprints):
        lam_p += polygon_area(poly)
        pk = rot_poly(np.asarray(poly, dtype=np.float64), k)
        labels = classify_edges(pk, wind, dot_threshold)
        edge_len = np.linalg.norm(np.roll(pk, -1, axis=0) - pk, axis=1)
        lam_f += float(edge_len[labels == WINDWARD].sum()) * float(heights[bi])
    return [math.log(len(footprints)), lam_p, lam_f,
            float(heights.mean()), float(heights.max())]


# ---------------------------------------------------------------------------
# Latent layout
# ---------------------------------------------------------------------------

def leaf_mass(sdf_norm_vol: np.ndarray, weighting: str, invsqrt_floor: float,
              coord_divisor: float) -> np.ndarray:
    """Importance weight of every volume point for the weighted-median split.

    "invsqrt": w = 1 / sqrt(max(d, invsqrt_floor)), d the distance to the
    nearest building surface in metres. Every leaf then carries the same
    total weight, so leaves are smaller near buildings.
    "equal" (and "structured", whose grid ignores weights): w = 1, i.e. a
    plain median split by point count.
    """
    if weighting == "invsqrt":
        d = np.abs(sdf_norm_vol) * coord_divisor
        return 1.0 / np.sqrt(np.maximum(d, invsqrt_floor))
    return np.ones(len(sdf_norm_vol), dtype=np.float64)


def _bincount(lid: np.ndarray, w: np.ndarray | None, L: int) -> np.ndarray:
    return np.bincount(lid, weights=w, minlength=L).astype(np.float64)


def _reduceat_minmax(op, arr: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    idx = np.minimum(offsets[:-1], max(len(arr) - 1, 0))
    return op.reduceat(arr, idx) if len(arr) else np.zeros(len(offsets) - 1)


def compute_leaf_stats(tree: KDTree, pos_norm_vol: np.ndarray,
                       sdf_norm_vol: np.ndarray, centroids: np.ndarray,
                       bounds: np.ndarray, n_surf_per_leaf: np.ndarray,
                       coord_divisor: float) -> np.ndarray:
    """19 statistics per leaf: log point count, log point density, surface
    fraction, 6 trace-normalized covariance entries, mean / std / skewness of
    the distance to the leaf centroid, sdf min / max / mean / std, and the
    three box extents. Empty leaves are all zeros."""
    L = tree.n_leaves
    eps = 1e-12
    lid = tree.leaf_ids(len(pos_norm_vol))
    cnt = np.bincount(lid, minlength=L).astype(np.float64)
    nonempty = cnt > 0
    cnt_safe = np.maximum(cnt, 1.0)

    rel = pos_norm_vol - centroids[lid]
    d = np.linalg.norm(rel, axis=1)

    stats = np.zeros((L, 19), dtype=np.float64)
    ext = bounds[:, 1] - bounds[:, 0]
    ext_phys = ext * coord_divisor
    vol_phys = ext_phys[:, 0] * ext_phys[:, 1] * ext_phys[:, 2]

    stats[:, 0] = np.log1p(cnt)
    stats[:, 1] = np.where(nonempty, np.log(cnt_safe / (vol_phys + eps)), 0.0)
    stats[:, 2] = n_surf_per_leaf / (cnt + n_surf_per_leaf + eps)

    rel_mean = np.stack([_bincount(lid, rel[:, a], L) / cnt_safe
                         for a in range(3)], axis=1)
    cov = {}
    for a, b, name in ((0, 0, "xx"), (1, 1, "yy"), (2, 2, "zz"),
                       (0, 1, "xy"), (0, 2, "xz"), (1, 2, "yz")):
        cov[name] = (_bincount(lid, rel[:, a] * rel[:, b], L) / cnt_safe
                     - rel_mean[:, a] * rel_mean[:, b])
    trace = np.maximum(cov["xx"] + cov["yy"] + cov["zz"], eps)
    for i, name in enumerate(("xx", "yy", "zz", "xy", "xz", "yz")):
        stats[:, 3 + i] = cov[name] / trace
    d1 = _bincount(lid, d, L) / cnt_safe
    d2 = _bincount(lid, d * d, L) / cnt_safe
    d3 = _bincount(lid, d ** 3, L) / cnt_safe
    var = np.maximum(d2 - d1 ** 2, 0.0)
    std = np.sqrt(var)
    m3 = d3 - 3 * d1 * d2 + 2 * d1 ** 3
    stats[:, 9] = d1
    stats[:, 10] = std
    stats[:, 11] = m3 / np.maximum(std ** 3, eps)

    sdf_perm = sdf_norm_vol[tree.perm]
    stats[:, 12] = _reduceat_minmax(np.minimum, sdf_perm, tree.leaf_offsets)
    stats[:, 13] = _reduceat_minmax(np.maximum, sdf_perm, tree.leaf_offsets)
    stats[:, 14] = _bincount(lid, sdf_norm_vol, L) / cnt_safe
    s2 = _bincount(lid, sdf_norm_vol ** 2, L) / cnt_safe
    stats[:, 15] = np.sqrt(np.maximum(s2 - stats[:, 14] ** 2, 0.0))

    stats[:, 16:19] = ext
    stats[~nonempty] = 0.0
    return stats


def assign_surface_capacitated(surf_pos: np.ndarray, centroids: np.ndarray,
                               capacity: int, k_cand: int = 16
                               ) -> tuple[np.ndarray, np.ndarray]:
    """Assign every surface point to a leaf, at most `capacity` per leaf,
    each point as close to its leaf centroid as the capacities allow.

    All (point, leaf) candidate pairs are consumed in increasing distance
    order, which yields a stable matching independent of visiting order;
    points whose candidates are all full retry with a wider candidate set.

    Returns (leaf_of_point [N_surf] int64, n_taken [L] int32).
    """
    L = len(centroids)
    n_surf = len(surf_pos)
    n_taken = np.zeros(L, dtype=np.int32)
    out = np.full(n_surf, -1, dtype=np.int64)
    if n_surf == 0 or capacity <= 0:
        return out, n_taken

    todo = np.arange(n_surf)
    while todo.size:
        free = np.flatnonzero(n_taken < capacity)
        if free.size == 0:
            break
        k = int(min(k_cand, free.size))
        dist, j = cKDTree(centroids[free]).query(surf_pos[todo], k=k,
                                                 workers=-1)
        if k == 1:
            dist, j = dist[:, None], j[:, None]
        order = np.argsort(dist, axis=None, kind="stable")
        pts = np.repeat(todo, k)[order]
        lvs = free[j.ravel()[order]]
        for p, l in zip(pts.tolist(), lvs.tolist()):
            if out[p] < 0 and n_taken[l] < capacity:
                out[p] = l
                n_taken[l] += 1
        nxt = np.flatnonzero(out < 0)
        if nxt.size == todo.size:
            k_cand = min(k_cand * 4, L)
        todo = nxt
    return out, n_taken


def sample_pn_neighbors(tree: KDTree, pos_norm_vol, centroids, sdf_norm_vol,
                        sdf_dir_vol, support_norm_vol, rng,
                        n_neighbors: int,
                        surf_leaf: np.ndarray | None = None,
                        surf_pos_norm: np.ndarray | None = None,
                        surf_sdf_norm: np.ndarray | None = None,
                        surf_sdf_dir: np.ndarray | None = None,
                        surf_support_norm: np.ndarray | None = None
                        ) -> tuple[np.ndarray, np.ndarray]:
    """The n_neighbors member points the PointNet encodes, per leaf.

    Surface points assigned to a leaf (surf_leaf) take its slots first;
    volume members, drawn uniformly without replacement, fill the rest.
    Unused slots stay zero and masked. Both point types share one 13-dim
    layout [rel_pos (3), sdf (1), sdf_dir (3), support (6)]; a surface point
    is the zero-distance limit of a volume point (sdf 0, sdf_dir = normal).

    Returns (feat [L, n, 13] float32, mask [L, n] bool).
    """
    L = tree.n_leaves
    feat = np.zeros((L, n_neighbors, 13), dtype=np.float32)
    mask = np.zeros((L, n_neighbors), dtype=bool)
    use_surf = surf_leaf is not None and surf_pos_norm is not None \
        and len(surf_leaf) > 0
    if use_surf:
        order = np.argsort(surf_leaf, kind="stable")
        starts = np.searchsorted(surf_leaf[order], np.arange(L), "left")
        ends = np.searchsorted(surf_leaf[order], np.arange(L), "right")

    def _write(leaf, cells, pos, sdf, sdir, sup):
        feat[leaf, cells, 0:3] = (pos - centroids[leaf]).astype(np.float32)
        feat[leaf, cells, 3] = sdf.astype(np.float32)
        feat[leaf, cells, 4:7] = sdir.astype(np.float32)
        feat[leaf, cells, 7:13] = sup.astype(np.float32)
        mask[leaf, cells] = True

    for leaf in range(L):
        used = 0
        if use_surf and ends[leaf] > starts[leaf]:
            si = order[starts[leaf]:ends[leaf]]
            if len(si) > n_neighbors:
                si = si[:n_neighbors]
            used = len(si)
            _write(leaf, np.arange(used), surf_pos_norm[si],
                   surf_sdf_norm[si], surf_sdf_dir[si], surf_support_norm[si])
        room = n_neighbors - used
        if room <= 0:
            continue
        members = tree.leaf_members(leaf)
        n = len(members)
        if n == 0:
            continue
        if n <= room:
            picks, cells = members, np.arange(used, used + n)
        else:
            picks = rng.choice(members, size=room, replace=False)
            cells = np.arange(used, used + room)
        _write(leaf, cells, pos_norm_vol[picks], sdf_norm_vol[picks],
               sdf_dir_vol[picks], support_norm_vol[picks])
    return feat, mask


def build_latent_layout(pos_norm_vol: np.ndarray, sdf_norm_vol: np.ndarray,
                        sdf_dir_vol: np.ndarray, support_norm_vol: np.ndarray,
                        pos_norm_surf: np.ndarray, sdf_dir_surf: np.ndarray,
                        support_norm_surf: np.ndarray, *, weighting: str,
                        invsqrt_floor: float, grid_dims, n_levels: int,
                        n_pn_neighbors: int, pn_surf_first: bool,
                        coord_divisor: float, seed_name: str) -> dict:
    """Importance weights -> partition -> latent points, leaf statistics and
    PointNet member samples of one case.

    weighting "invsqrt" / "equal" builds the weighted KD-tree of n_levels
    levels; "structured" the regular grid_dims grid. A latent point is the
    weighted centroid of its leaf (cell centre for the regular grid and for
    empty leaves). The PointNet draw is seeded by seed_name.
    """
    if weighting == "structured":
        tree = build_structured_grid(pos_norm_vol, grid_dims)
        weights = np.ones(len(pos_norm_vol), dtype=np.float64)
    else:
        weights = leaf_mass(sdf_norm_vol, weighting, invsqrt_floor,
                            coord_divisor)
        tree = build_weighted_kdtree(pos_norm_vol, weights, n_levels)
    L = tree.n_leaves

    # geometric containment of the surface points (leaf_stats' surface
    # fraction); the PointNet slot assignment below is a separate thing
    surf_leaf_geom = assign_points_to_leaves(pos_norm_surf, tree)
    n_surf_per_leaf = np.bincount(surf_leaf_geom,
                                  minlength=L).astype(np.float64)
    lid = tree.leaf_ids(len(pos_norm_vol))
    wsum = np.maximum(np.bincount(lid, weights=weights, minlength=L), 1e-300)
    centroids = np.stack([
        np.bincount(lid, weights=weights * pos_norm_vol[:, a], minlength=L)
        / wsum for a in range(3)], axis=1)
    bounds = compute_leaf_bounds(tree, pos_norm_vol.min(axis=0),
                                 pos_norm_vol.max(axis=0))
    cnt = np.bincount(lid, minlength=L)
    empty = cnt == 0
    if empty.any():
        centroids[empty] = 0.5 * (bounds[empty, 0] + bounds[empty, 1])
    if weighting == "structured":
        centroids = 0.5 * (bounds[:, 0] + bounds[:, 1])

    rng = np.random.default_rng(per_case_epoch_seed(seed_name, 0))
    if pn_surf_first:
        surf_slot, surf_taken = assign_surface_capacitated(
            pos_norm_surf, centroids, n_pn_neighbors)
    else:
        surf_slot, surf_taken = None, np.zeros(L, dtype=np.int32)
    pn_feat, pn_mask = sample_pn_neighbors(
        tree, pos_norm_vol, centroids, sdf_norm_vol, sdf_dir_vol,
        support_norm_vol, rng, n_neighbors=n_pn_neighbors,
        surf_leaf=surf_slot, surf_pos_norm=pos_norm_surf,
        surf_sdf_norm=np.zeros(len(pos_norm_surf), dtype=np.float64),
        surf_sdf_dir=sdf_dir_surf, surf_support_norm=support_norm_surf)
    leaf_stats = compute_leaf_stats(tree, pos_norm_vol, sdf_norm_vol,
                                    centroids, bounds, n_surf_per_leaf,
                                    coord_divisor)
    return {"tree": tree, "centroids": centroids, "leaf_stats": leaf_stats,
            "pn_feat": pn_feat, "pn_mask": pn_mask, "leaf_count": cnt,
            "n_empty": int(empty.sum()), "surf_slot": surf_slot,
            "surf_taken": surf_taken, "surf_leaf_geom": surf_leaf_geom}
