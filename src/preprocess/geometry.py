"""Exact 2.5D building geometry.

A building is a vertical extrusion of a CCW footprint polygon (metres) over
[z_lo, z_hi], with a flat roof at z_hi and no bottom face. Every distance in
this module is exact and computed in float64:

  Extrusion25D.sdf      signed distance to the nearest wall or roof (positive
                        outside) and the unit direction from that nearest
                        surface point to the query point
  Extrusion25D.support  distance along +x, -x, +y, -y, +z, -z to the nearest
                        building face, capped at support_max

Footprints come either from CFD surface points (extract_buildings: DBSCAN over
the surface points, concave hull of the wall points, simplification) or from
given polygons (canonical_footprint). Both end in the same canonical form:
CCW, short edges collapsed, start vertex at minimum y (ties: minimum x).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Polygon utilities
# ---------------------------------------------------------------------------

def polygon_area(poly: np.ndarray) -> float:
    """Signed area (positive = CCW). poly [k, 2]."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def ensure_ccw(poly: np.ndarray) -> np.ndarray:
    return poly if polygon_area(poly) >= 0 else poly[::-1].copy()


def normalize_start_vertex(poly: np.ndarray) -> np.ndarray:
    """Rotate so the first vertex has minimum y, tie-broken by minimum x."""
    y_min = poly[:, 1].min()
    cand = np.where(np.abs(poly[:, 1] - y_min) < 1e-9)[0]
    start = int(cand[np.argmin(poly[cand, 0])])
    return np.roll(poly, -start, axis=0)


def rot_poly(poly: np.ndarray, k: int) -> np.ndarray:
    """k counter-clockwise quarter turns of an [n, 2] array (signed
    permutation, exact in float64; same convention as data.estate)."""
    if k == 1:
        return np.stack([-poly[:, 1], poly[:, 0]], axis=1)
    if k == 2:
        return -poly
    if k == 3:
        return np.stack([poly[:, 1], -poly[:, 0]], axis=1)
    return poly


def point_in_polygon(pts_xy: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorized crossing-number test. pts_xy [N, 2], poly [k, 2] -> bool [N].

    Half-open edge rule (v0.y <= p.y < v1.y), so shared vertices are not
    counted twice. Points exactly on the boundary may land on either side;
    every caller treats the boundary as distance 0.
    """
    x = pts_xy[:, 0][:, None]
    y = pts_xy[:, 1][:, None]
    v0 = poly
    v1 = np.roll(poly, -1, axis=0)
    y0, y1 = v0[:, 1][None, :], v1[:, 1][None, :]
    x0, x1 = v0[:, 0][None, :], v1[:, 0][None, :]
    straddle = (y0 <= y) != (y1 <= y)
    dy = y1 - y0
    dy_safe = np.where(np.abs(dy) < 1e-300, 1.0, dy)
    x_int = x0 + (y - y0) * (x1 - x0) / dy_safe
    crossings = np.sum(straddle & (x < x_int), axis=1)
    return (crossings % 2) == 1


# ---------------------------------------------------------------------------
# Exact distances
# ---------------------------------------------------------------------------

def point_to_segments_2d(pts_xy: np.ndarray, a: np.ndarray, b: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Distance from N points to E 2D segments.

    pts_xy [N, 2], a/b [E, 2] -> (dist [N, E], nearest [N, E, 2])
    """
    e = b - a
    len2 = np.sum(e * e, axis=-1)
    len2_safe = np.maximum(len2, 1e-300)
    dp = pts_xy[:, None, :] - a[None, :, :]
    t = np.clip(np.sum(dp * e[None, :, :], axis=-1) / len2_safe, 0.0, 1.0)
    nearest = a[None, :, :] + t[..., None] * e[None, :, :]
    diff = pts_xy[:, None, :] - nearest
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    return dist, nearest


def point_to_walls_dist(pts: np.ndarray, v0: np.ndarray, v1: np.ndarray,
                        z_lo: np.ndarray, z_hi: np.ndarray) -> np.ndarray:
    """Exact distance from N points to E vertical rectangles
    [v0, v1] x [z_lo, z_hi]. pts [N, 3]; v0/v1 [E, 2]; z_lo/z_hi [E]
    -> [N, E]. The nearest point on a rectangle is the segment-clamped xy
    projection at the z-clamped height."""
    e = v1 - v0
    len2 = np.maximum(np.sum(e * e, axis=-1), 1e-300)
    px = pts[:, 0][:, None]
    py = pts[:, 1][:, None]
    t = ((px - v0[:, 0][None, :]) * e[:, 0][None, :] +
         (py - v0[:, 1][None, :]) * e[:, 1][None, :]) / len2[None, :]
    np.clip(t, 0.0, 1.0, out=t)
    dx = px - (v0[:, 0][None, :] + t * e[:, 0][None, :])
    dy = py - (v0[:, 1][None, :] + t * e[:, 1][None, :])
    d2 = dx * dx
    d2 += dy * dy
    dz = pts[:, 2][:, None] - np.clip(pts[:, 2][:, None], z_lo[None, :],
                                      z_hi[None, :])
    d2 += dz * dz
    return np.sqrt(d2)


def nearest_on_walls(pts: np.ndarray, v0: np.ndarray, v1: np.ndarray,
                     z_lo: np.ndarray, z_hi: np.ndarray) -> np.ndarray:
    """Nearest point on a per-point wall: pts [N, 3], v0/v1 [N, 2],
    z_lo/z_hi [N] -> nearest [N, 3]."""
    e = v1 - v0
    len2 = np.maximum(np.sum(e * e, axis=-1), 1e-300)
    t = np.clip(np.sum((pts[:, :2] - v0) * e, axis=-1) / len2, 0.0, 1.0)
    near_xy = v0 + t[:, None] * e
    z = np.clip(pts[:, 2], z_lo, z_hi)
    return np.concatenate([near_xy, z[:, None]], axis=-1)


def point_to_roof(pts: np.ndarray, poly: np.ndarray, z_top: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Exact distance from N points to a horizontal polygon at z = z_top.

    Inside the footprint the nearest point is (x, y, z_top); outside it is
    the closest boundary point at z_top. pts [N, 3] -> (dist [N],
    nearest [N, 3])
    """
    inside = point_in_polygon(pts[:, :2], poly)
    a = poly
    b = np.roll(poly, -1, axis=0)
    dist_b, near_b = point_to_segments_2d(pts[:, :2], a, b)
    j = np.argmin(dist_b, axis=1)
    rows = np.arange(len(pts))
    bnd_xy = near_b[rows, j]
    bnd_d2 = dist_b[rows, j]

    dz = pts[:, 2] - z_top
    nearest = np.empty((len(pts), 3), dtype=np.float64)
    nearest[:, 2] = z_top
    nearest[inside, 0] = pts[inside, 0]
    nearest[inside, 1] = pts[inside, 1]
    nearest[~inside, :2] = bnd_xy[~inside]

    dist = np.where(inside, np.abs(dz), np.sqrt(bnd_d2 * bnd_d2 + dz * dz))
    return dist, nearest


def point_to_segments_3d(pts: np.ndarray, a: np.ndarray, b: np.ndarray
                         ) -> np.ndarray:
    """Distance from N points to E 3D segments. pts [N, 3], a/b [E, 3]
    -> [N, E]."""
    e = b - a
    len2 = np.maximum(np.sum(e * e, axis=-1), 1e-300)
    dp = pts[:, None, :] - a[None, :, :]
    t = np.clip(np.sum(dp * e[None, :, :], axis=-1) / len2, 0.0, 1.0)
    nearest = a[None, :, :] + t[..., None] * e[None, :, :]
    diff = pts[:, None, :] - nearest
    return np.sqrt(np.sum(diff * diff, axis=-1))


# ---------------------------------------------------------------------------
# Scene: all buildings of one case as flat face arrays
# ---------------------------------------------------------------------------

class Extrusion25D:
    """All wall rectangles and roof polygons of one case, in metres."""

    def __init__(self, footprints: list[np.ndarray], z_lo: np.ndarray,
                 z_hi: np.ndarray):
        assert len(footprints) == len(z_lo) == len(z_hi)
        self.footprints = [np.asarray(p, dtype=np.float64) for p in footprints]
        self.z_lo = np.asarray(z_lo, dtype=np.float64)
        self.z_hi = np.asarray(z_hi, dtype=np.float64)
        self.n_buildings = len(footprints)

        v0, v1, wzlo, wzhi, wbld, wedge, wnrm = [], [], [], [], [], [], []
        for bi, poly in enumerate(self.footprints):
            k = len(poly)
            for ei in range(k):
                a = poly[ei]
                b = poly[(ei + 1) % k]
                e = b - a
                ln = np.linalg.norm(e)
                if ln < 1e-9:
                    continue
                v0.append(a)
                v1.append(b)
                wzlo.append(self.z_lo[bi])
                wzhi.append(self.z_hi[bi])
                wbld.append(bi)
                wedge.append(ei)
                wnrm.append(np.array([e[1], -e[0]]) / ln)  # outward for CCW
        self.wall_v0 = np.asarray(v0, dtype=np.float64).reshape(-1, 2)
        self.wall_v1 = np.asarray(v1, dtype=np.float64).reshape(-1, 2)
        self.wall_zlo = np.asarray(wzlo, dtype=np.float64)
        self.wall_zhi = np.asarray(wzhi, dtype=np.float64)
        self.wall_bld = np.asarray(wbld, dtype=np.int64)
        self.wall_edge = np.asarray(wedge, dtype=np.int64)
        self.wall_normal = np.asarray(wnrm, dtype=np.float64).reshape(-1, 2)
        self.n_walls = len(self.wall_v0)

    def inside(self, pts: np.ndarray) -> np.ndarray:
        """True where a point lies inside any building volume."""
        out = np.zeros(len(pts), dtype=bool)
        for bi, poly in enumerate(self.footprints):
            zin = (pts[:, 2] >= self.z_lo[bi]) & (pts[:, 2] <= self.z_hi[bi])
            cand = np.where(zin & ~out)[0]
            if len(cand):
                out[cand] = point_in_polygon(pts[cand, :2], poly)
        return out

    def sdf(self, pts: np.ndarray, chunk: int = 16384,
            return_building: bool = False):
        """Signed distance (positive outside) and the unit direction from the
        nearest surface point to the query point, brute force over every
        face. With return_building=True also the index of the building that
        owns the nearest face.

        pts [N, 3] -> (sdf [N], sdf_dir [N, 3][, building [N]])
        """
        pts = np.asarray(pts, dtype=np.float64)
        N = len(pts)
        sdf = np.empty(N, dtype=np.float64)
        sdir = np.empty((N, 3), dtype=np.float64)
        bld = np.empty(N, dtype=np.int64)
        for lo in range(0, N, chunk):
            p = pts[lo:lo + chunk]
            d, near, b = self._min_dist_all_faces(p)
            diff = p - near
            nrm = np.linalg.norm(diff, axis=-1, keepdims=True)
            direction = np.where(nrm > 1e-12, diff / np.maximum(nrm, 1e-12),
                                 0.0)
            sgn = np.where(self.inside(p), -1.0, 1.0)
            sdf[lo:lo + chunk] = d * sgn
            sdir[lo:lo + chunk] = direction
            bld[lo:lo + chunk] = b
        if return_building:
            return sdf, sdir, bld
        return sdf, sdir

    def _min_dist_all_faces(self, p: np.ndarray
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        best = np.full(len(p), np.inf)
        near = np.zeros((len(p), 3), dtype=np.float64)
        bld = np.zeros(len(p), dtype=np.int64)
        if self.n_walls:
            dw = point_to_walls_dist(p, self.wall_v0, self.wall_v1,
                                     self.wall_zlo, self.wall_zhi)
            j = np.argmin(dw, axis=1)
            rows = np.arange(len(p))
            best = dw[rows, j]
            near = nearest_on_walls(p, self.wall_v0[j], self.wall_v1[j],
                                    self.wall_zlo[j], self.wall_zhi[j])
            bld = self.wall_bld[j]
        for bi, poly in enumerate(self.footprints):
            dr, nr = point_to_roof(p, poly, float(self.z_hi[bi]))
            closer = dr < best
            best[closer] = dr[closer]
            near[closer] = nr[closer]
            bld[closer] = bi
        return best, near, bld

    def support(self, pts: np.ndarray, support_max: float = 550.0,
                chunk: int = 16384) -> np.ndarray:
        """Distance along +x, -x, +y, -y, +z, -z to the nearest building
        face, capped at support_max when nothing is hit. Walls are the hits
        for the horizontal rays and roofs for the vertical ones (grazing
        contacts have measure zero).

        pts [N, 3] -> [N, 6]
        """
        pts = np.asarray(pts, dtype=np.float64)
        N = len(pts)
        out = np.full((N, 6), support_max, dtype=np.float64)
        for lo in range(0, N, chunk):
            p = pts[lo:lo + chunk]
            out[lo:lo + chunk, 0] = self._ray_walls(p, 0, +1, support_max)
            out[lo:lo + chunk, 1] = self._ray_walls(p, 0, -1, support_max)
            out[lo:lo + chunk, 2] = self._ray_walls(p, 1, +1, support_max)
            out[lo:lo + chunk, 3] = self._ray_walls(p, 1, -1, support_max)
            up, dn = self._ray_roofs(p, cap=support_max)
            out[lo:lo + chunk, 4] = up
            out[lo:lo + chunk, 5] = dn
        return out

    def _ray_walls(self, p: np.ndarray, axis: int, sign: int, cap: float
                   ) -> np.ndarray:
        if self.n_walls == 0:
            return np.full(len(p), cap)
        other = 1 - axis
        a_o = self.wall_v0[:, other][None, :]
        e_o = (self.wall_v1[:, other] - self.wall_v0[:, other])[None, :]
        a_x = self.wall_v0[:, axis][None, :]
        e_x = (self.wall_v1[:, axis] - self.wall_v0[:, axis])[None, :]
        p_o = p[:, other][:, None]
        p_x = p[:, axis][:, None]
        p_z = p[:, 2][:, None]

        e_o_safe = np.where(np.abs(e_o) < 1e-300, 1.0, e_o)
        s = (p_o - a_o) / e_o_safe
        valid = (np.abs(e_o) > 1e-12) & (s >= 0.0) & (s <= 1.0)
        valid &= (p_z >= self.wall_zlo[None, :]) & (p_z <= self.wall_zhi[None, :])
        x_int = a_x + s * e_x
        t = (x_int - p_x) * sign
        valid &= t > 1e-12
        t = np.where(valid, t, np.inf)
        return np.minimum(t.min(axis=1), cap)

    def _ray_roofs(self, p: np.ndarray, cap: float
                   ) -> tuple[np.ndarray, np.ndarray]:
        up = np.full(len(p), cap)
        dn = np.full(len(p), cap)
        for bi, poly in enumerate(self.footprints):
            inp = point_in_polygon(p[:, :2], poly)
            if not inp.any():
                continue
            dz = self.z_hi[bi] - p[:, 2]
            hit_up = inp & (dz > 1e-12)
            hit_dn = inp & (-dz > 1e-12)
            up[hit_up] = np.minimum(up[hit_up], dz[hit_up])
            dn[hit_dn] = np.minimum(dn[hit_dn], -dz[hit_dn])
        return up, dn


# ---------------------------------------------------------------------------
# Footprints
# ---------------------------------------------------------------------------

@dataclass
class Building:
    footprint: np.ndarray   # [k, 2] canonical CCW polygon, metres
    z_lo: float
    z_hi: float


class FootprintQCError(Exception):
    pass


def _collapse_short_edges(pts: np.ndarray, min_len: float) -> np.ndarray:
    pts = pts.copy()
    for _ in range(500):
        if len(pts) < 3:
            break
        edges = np.roll(pts, -1, axis=0) - pts
        lengths = np.sqrt((edges ** 2).sum(axis=1))
        short = np.where(lengths < min_len)[0]
        if len(short) == 0:
            break
        w = short[np.argmin(lengths[short])]
        n = len(pts)
        i, j = w, (w + 1) % n
        prev_len = lengths[(i - 1) % n]
        next_len = lengths[j] if j != 0 else lengths[0]
        if prev_len >= next_len:
            pts[i] = (pts[i] + pts[j]) / 2
            pts = np.delete(pts, j, axis=0)
        else:
            pts[j] = (pts[i] + pts[j]) / 2
            pts = np.delete(pts, i, axis=0)
    return pts


def _largest_polygon(geom):
    """The single Polygon with the largest area of any shapely geometry
    (multi-part geometries and collections searched recursively), or None."""
    from shapely.geometry import Polygon
    if isinstance(geom, Polygon):
        return geom
    parts = [_largest_polygon(g) for g in getattr(geom, "geoms", ())]
    parts = [p for p in parts if p is not None and not p.is_empty]
    return max(parts, key=lambda g: g.area) if parts else None


def canonical_footprint(poly, dp_tol: float, min_edge: float
                        ) -> np.ndarray | None:
    """Shapely polygon -> canonical footprint [k, 2], or None if nothing of
    it survives. Repairs an invalid polygon, keeps the largest part, drops
    holes, simplifies (Douglas-Peucker, dp_tol metres), orients CCW,
    collapses edges shorter than min_edge and normalizes the start vertex."""
    from shapely.geometry.polygon import orient

    if poly is None or poly.is_empty:
        return None
    if not poly.is_valid:
        poly = poly.buffer(0)
    poly = _largest_polygon(poly)
    if poly is None or poly.is_empty:
        return None
    poly_s = poly.simplify(dp_tol, preserve_topology=True)
    if poly_s is None or poly_s.is_empty:
        return None
    poly_s = _largest_polygon(poly_s)
    if poly_s is None or poly_s.is_empty:
        return None
    poly_s = orient(poly_s, sign=1.0)
    ext = np.asarray(poly_s.exterior.coords, dtype=np.float64)[:-1]
    if len(ext) < 3:
        return None
    ext = _collapse_short_edges(ext, min_edge)
    if len(ext) < 3:
        return None
    return normalize_start_vertex(ensure_ccw(ext))


def _extract_one_building(wall_xy: np.ndarray, all_z: np.ndarray,
                          hull_ratio: float, dp_tol: float,
                          min_edge: float) -> Building | None:
    """One DBSCAN cluster -> Building, or None when it cannot be extracted."""
    from shapely import concave_hull
    from shapely.geometry import MultiPoint

    if len(wall_xy) < 5:
        return None
    hull = concave_hull(MultiPoint(wall_xy), ratio=hull_ratio)
    footprint = canonical_footprint(hull, dp_tol, min_edge)
    if footprint is None:
        return None
    return Building(footprint=footprint, z_lo=float(all_z.min()),
                    z_hi=float(all_z.max()))


def extract_buildings(surface_pos: np.ndarray, surface_normals: np.ndarray,
                      dbscan_eps: float, dbscan_min_samples: int,
                      wall_nz_threshold: float, hull_ratio: float,
                      dp_tol: float, min_edge: float) -> list[Building]:
    """All buildings of one CFD case from its building-surface points.

    Clusters the surface points in xy (DBSCAN), takes the concave hull of
    each cluster's wall points (|n_z| < wall_nz_threshold) as the footprint
    and the cluster's z range as [z_lo, z_hi]. Clusters that cannot be
    extracted are skipped; zero buildings raises FootprintQCError.
    """
    from sklearn.cluster import DBSCAN

    labels = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples
                    ).fit_predict(surface_pos[:, :2])
    bld_ids = sorted(set(int(l) for l in labels) - {-1})
    if not bld_ids:
        raise FootprintQCError("DBSCAN found zero building clusters")

    buildings = []
    for bid in bld_ids:
        mask = labels == bid
        cluster_pos = surface_pos[mask]
        cluster_nrm = surface_normals[mask]
        wall_mask = np.abs(cluster_nrm[:, 2]) < wall_nz_threshold
        bld = _extract_one_building(
            cluster_pos[wall_mask, :2], cluster_pos[:, 2],
            hull_ratio=hull_ratio, dp_tol=dp_tol, min_edge=min_edge)
        if bld is not None:
            buildings.append(bld)

    if not buildings:
        raise FootprintQCError("all building clusters failed extraction")
    return buildings
