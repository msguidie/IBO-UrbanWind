#!/usr/bin/env python
"""Convert one CFD case into a raw UrbanWind case file.

    python convert_cfd_case.py --volume case/internal.vtu \\
        --surface case/boundary/buildings.vtp --stl case/buildings.stl \\
        --estate my_estate --direction N --out raw

The output, <out>/<estate>_<direction>.pt, has the layout of the files the
dataset was built from (schema hdb-3d-v3-physical). A folder of them is the
input of `main.py preprocess`, and `predict_new_case.py --raw-pt` predicts
one and scores the prediction against its CFD fields.
docs/cfd_case_format.md describes the inputs, the output and every
convention below.

Inputs:
  --volume    the internal mesh as a VTK unstructured grid: the cells with
              their connectivity (polyhedra not decomposed) and CELL data
              U [m/s], p and nut [m2/s]
  --surface   the building wall patches as VTK polygonal data, one or more
              files: the wall faces with CELL data p and wallShearStress
  --stl       the buildings as a watertight triangle mesh, in metres, in
              the frame of the CFD

Steps:
  1. points     cell values averaged onto the mesh vertices (the training
                data's convention) and the vertices cropped to
                |x|, |y| <= 550 m, 0 <= z <= 160 m
  2. frame      quarter-turn rotation so that the wind blows toward -y
  3. units      p and the wall shear stress to the training convention:
                kinematic (divided by the density) with the hydrostatic
                term -g z included; nut floored at 2e-12 m2/s
  4. surface    face centres, areas and normals (oriented into the fluid
                with the STL), p and wall shear stress
  5. distance   signed distance of every volume vertex to the STL and its
                gradient, negative outside the buildings
  6. checks     units, conventions and the point set; then the .pt file
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import igl
    import pyvista as pv
    import trimesh
except ImportError as exc:
    raise SystemExit(f"convert_cfd_case.py needs pyvista, trimesh and libigl "
                     f"({exc}); pip install -r requirements.txt")
import torch

SCHEMA = "hdb-3d-v3-physical"
DIRECTIONS = ("N", "E", "S", "W")   # src/manifest.DIRECTION_SUFFIXES
DOMAIN_HALF = 550.0      # m, learning region |x|, |y| <= DOMAIN_HALF
DOMAIN_TOP = 160.0       # m, learning region 0 <= z <= DOMAIN_TOP
CROP_TOL = 1e-3          # m, vertices on the crop faces are kept
G = 9.81                 # m/s2
U_REF = 2.0              # m/s, the training inflow at 15 m (informational)
NUT_FLOOR = 2e-12        # m2/s, the training data's floor on nut
MAX_SPEED = 50.0         # m/s, a larger speed means U is not in m/s
P_SLOPE_TOL = 1.0        # m/s2, window around a hydrostatic slope of p
SURF_ON_STL_TOL = 0.5    # m, median distance of the wall faces to the STL
MIN_GROUND_POINTS = 100  # vertices expected on the ground plane
STL_MAX_SPAN = 5000.0    # m, a larger STL is not in metres
ORIENT_SAMPLE = 20000    # points certainly outside, for the orientation test
ORIENT_MAX_INSIDE = 0.02  # fraction of them the sign may call inside
VOL_NAMES = ["Ux", "Uy", "Uz", "p", "nut"]
VOL_UNITS = ["m/s", "m/s", "m/s", "m2/s2", "m2/s"]
SURF_NAMES = ["p", "wss_x", "wss_y", "wss_z"]
SURF_UNITS = ["m2/s2", "m2/s2", "m2/s2", "m2/s2"]
# counter-clockwise quarter turns that bring the CFD's wind direction to -y
ROT_FOR_WIND = {"-y": 0, "-x": 1, "+y": 2, "+x": 3}
P_CONVENTIONS = ("auto", "kinematic-g", "pa-g", "kinematic", "pa")
# VTK cell types
VTK_TETRA, VTK_HEXAHEDRON, VTK_WEDGE, VTK_PYRAMID, VTK_POLYHEDRON = 10, 12, 13, 14, 42


def log(msg: str) -> None:
    print(msg, flush=True)


def fail(msg: str) -> None:
    raise SystemExit(f"error: {msg}")


def rotate(a: np.ndarray, k: int) -> np.ndarray:
    """[N, 3] positions or vectors under k counter-clockwise quarter turns
    about z: sign flips and swaps only (src/data/estate.rot_points)."""
    a = np.asarray(a, dtype=np.float64)
    if k == 0:
        return a
    x, y, z = a[:, 0], a[:, 1], a[:, 2]
    if k == 1:
        x, y = -y, x
    elif k == 2:
        x, y = -x, -y
    else:
        x, y = y, -x
    return np.column_stack([x, y, z])


def f32(a) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


# ---------------------------------------------------------------------------
# VTK inputs
# ---------------------------------------------------------------------------

def read_vtk(path: Path):
    if not path.exists():
        fail(f"{path}: not found")
    try:
        ds = pv.read(str(path))
    except Exception as exc:  # noqa: BLE001 — any reader failure is fatal
        fail(f"{path}: cannot read ({exc})")
    if isinstance(ds, pv.MultiBlock):
        ds = ds.combine(merge_points=False)
    return ds


def cell_array(ds, name: str, path: Path, ncomp: int) -> np.ndarray:
    """A finite cell array with ncomp components, as float64."""
    if name not in ds.cell_data:
        hint = (" (it is point data; the converter needs the cell values)"
                if name in ds.point_data else "")
        fail(f"{path}: no cell array {name!r}{hint}; cell arrays: "
             f"{list(ds.cell_data.keys())}")
    a = np.asarray(ds.cell_data[name], dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2 or a.shape[1] != ncomp:
        fail(f"{path}: {name!r} has shape {a.shape}, expected "
             f"({ds.n_cells}, {ncomp})")
    if not np.isfinite(a).all():
        fail(f"{path}: {name!r} has non-finite values")
    return a if ncomp > 1 else a[:, 0]


def read_volume(path: Path, names: dict) -> tuple[np.ndarray, np.ndarray,
                                                  np.ndarray, np.ndarray]:
    """Mesh vertices and the cell values of U, p, nut averaged onto them:
    (points [N, 3], U [N, 3], p [N], nut [N]) in the file's frame and
    units."""
    ds = read_vtk(path)
    if not isinstance(ds, pv.UnstructuredGrid):
        ds = ds.cast_to_unstructured_grid()
    if ds.n_cells == 0 or ds.n_points == 0:
        fail(f"{path}: the volume mesh has no cells")
    types = set(int(t) for t in np.unique(ds.celltypes))
    decomposed = types & {VTK_TETRA, VTK_WEDGE, VTK_PYRAMID}
    if decomposed and VTK_HEXAHEDRON in types and VTK_POLYHEDRON not in types:
        fail(f"{path}: the mesh mixes hexahedra with tetrahedra/pyramids "
             f"and has no polyhedral cells: its polyhedra were decomposed on "
             f"export, which adds points that are not mesh vertices. Export "
             f"the polyhedra as polyhedra (docs/cfd_case_format.md).")
    if decomposed:
        log(f"[volume] WARNING: cell types {sorted(decomposed)} (tetrahedra/"
            f"wedges/pyramids) present; an UrbanWind mesh has hexahedra and "
            f"polyhedra only")

    # the three cell arrays only, then the unweighted mean over the cells
    # sharing each vertex (vtkCellDataToPointData, as the dataset was made)
    keep = {names["U"], names["p"], names["nut"]}
    for n in (names["U"], names["p"], names["nut"]):
        cell_array(ds, n, path, 3 if n == names["U"] else 1)
    for n in list(ds.cell_data.keys()):
        if n not in keep:
            ds.cell_data.remove(n)
    ds.point_data.clear()
    pts = ds.cell_data_to_point_data(pass_cell_data=False)
    points = np.asarray(ds.points, dtype=np.float64)
    U = np.asarray(pts.point_data[names["U"]], dtype=np.float64)
    p = np.asarray(pts.point_data[names["p"]], dtype=np.float64).reshape(-1)
    nut = np.asarray(pts.point_data[names["nut"]], dtype=np.float64).reshape(-1)
    if U.shape != (len(points), 3) or len(p) != len(points) \
            or len(nut) != len(points):
        fail(f"{path}: averaged point arrays do not match the {len(points)} "
             f"points")
    if not (np.isfinite(U).all() and np.isfinite(p).all()
            and np.isfinite(nut).all()):
        fail(f"{path}: non-finite point values after averaging")
    log(f"[volume] {ds.n_cells:,} cells, {len(points):,} vertices, cell "
        f"types {sorted(types)}")
    return points, U, p, nut


def face_geometry(poly: pv.PolyData, path: Path
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Face centres, unit normals (winding order) and areas of every
    polygon, with OpenFOAM's definitions: the area vector is the sum over
    the fan triangles about the vertex mean, and the centre is their
    area-weighted centroid."""
    if poly.n_lines or poly.n_verts or poly.n_strips:
        fail(f"{path}: the surface file must contain polygons only")
    pts = np.asarray(poly.points, dtype=np.float64)
    flat = np.asarray(poly.faces, dtype=np.int64)
    n = poly.n_cells
    centres = np.empty((n, 3))
    normals = np.zeros((n, 3))
    areas = np.zeros(n)
    pos = 0
    for i in range(n):
        k = int(flat[pos])
        v = pts[flat[pos + 1:pos + 1 + k]]
        pos += 1 + k
        if k < 3:
            fail(f"{path}: face {i} has {k} vertices")
        c0 = v.mean(axis=0)
        nxt = np.roll(v, -1, axis=0)
        tri_n = 0.5 * np.cross(v - c0, nxt - c0)
        mags = np.linalg.norm(tri_n, axis=1)
        sf = tri_n.sum(axis=0)
        area = float(np.linalg.norm(sf))
        tot = float(mags.sum())
        centres[i] = ((c0 + v + nxt) / 3.0 * mags[:, None]).sum(axis=0) / tot \
            if tot > 0 else c0
        areas[i] = area
        if area > 0:
            normals[i] = sf / area
    if pos != len(flat):
        fail(f"{path}: malformed face connectivity")
    return centres, normals, areas


def read_surface(paths: list[Path], names: dict) -> dict:
    """Wall faces of the building patches: centres, winding normals, areas,
    p and wall shear stress, concatenated over the files."""
    parts = {"centres": [], "normals": [], "areas": [], "p": [], "wss": []}
    for path in paths:
        ds = read_vtk(path)
        if not isinstance(ds, pv.PolyData):
            ds = ds.extract_surface()
        if ds.n_cells == 0:
            fail(f"{path}: no faces")
        p = cell_array(ds, names["p"], path, 1)
        if names["wss"]:
            wss = cell_array(ds, names["wss"], path, 3)
        else:
            wss = np.zeros((ds.n_cells, 3))
        c, nrm, a = face_geometry(ds, path)
        ok = a > 1e-10
        if not ok.all():
            log(f"[surface] {path.name}: {int((~ok).sum())} zero-area faces "
                f"dropped")
        parts["centres"].append(c[ok])
        parts["normals"].append(nrm[ok])
        parts["areas"].append(a[ok])
        parts["p"].append(p[ok])
        parts["wss"].append(wss[ok])
        log(f"[surface] {path.name}: {int(ok.sum()):,} faces")
    return {k: np.concatenate(v, axis=0) for k, v in parts.items()}


# ---------------------------------------------------------------------------
# STL and the signed distance
# ---------------------------------------------------------------------------

def read_stl(path: Path) -> tuple[trimesh.Trimesh, bool]:
    """The building mesh with consistent, outward winding, and whether it is
    watertight (then the orientation is repaired by body; an open mesh keeps
    the file's orientation, checked later against points that are certainly
    outside)."""
    if not path.exists():
        fail(f"{path}: not found")
    try:
        mesh = trimesh.load(str(path), force="mesh", process=True)
    except Exception as exc:  # noqa: BLE001
        fail(f"{path}: cannot read ({exc})")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        fail(f"{path}: no triangles")
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    span = mesh.bounds[1] - mesh.bounds[0]
    if span.max() > STL_MAX_SPAN:
        fail(f"{path}: the STL spans {span.max():.0f} units; it must be in "
             f"metres")
    watertight = bool(mesh.is_watertight)
    if watertight:
        trimesh.repair.fix_normals(mesh, multibody=True)
    else:
        trimesh.repair.fix_winding(mesh)
        log("[stl] WARNING: the STL is not watertight (an edge with other "
            "than two triangles, e.g. open bottoms or buildings sharing a "
            "wall). The sign of the distance field then relies on the "
            "file's own orientation; it is checked below.")
    if mesh.bounds[0][2] < -1.0:
        log(f"[stl] WARNING: the lowest STL point is at z = "
            f"{mesh.bounds[0][2]:.2f} m; the ground is z = 0")
    log(f"[stl] {len(mesh.vertices):,} vertices, {len(mesh.faces):,} "
        f"triangles, tallest point {mesh.bounds[1][2]:.2f} m"
        f"{'' if watertight else ', open'}")
    return mesh, watertight


def signed_distance(mesh: trimesh.Trimesh, pts: np.ndarray, chunk: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    """(sdf [N] positive outside, grad [N, 3] unit): exact distance to the
    triangles (libigl AABB tree) with the sign of the angle-weighted
    pseudonormal of the closest feature (face, edge or vertex), which is
    right where the normal of the closest face alone is not. grad is the
    gradient of the sdf; on the surface the pseudonormal."""
    V = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    F = np.ascontiguousarray(mesh.faces, dtype=np.int32)
    pts = np.ascontiguousarray(pts, dtype=np.float64)
    n = len(pts)
    sdf = np.empty(n)
    grad = np.empty((n, 3))
    for lo in range(0, n, chunk):
        p = pts[lo:lo + chunk]
        s, _, c, nrm = igl.signed_distance(p, V, F, return_normals=True)
        s = np.asarray(s, dtype=np.float64).reshape(-1)
        c = np.asarray(c, dtype=np.float64).reshape(-1, 3)
        nrm = np.asarray(nrm, dtype=np.float64).reshape(-1, 3)
        diff = p - c
        d = np.linalg.norm(diff, axis=1)
        on = d < 1e-9
        sign = np.where(s >= 0.0, 1.0, -1.0)
        g = sign[:, None] * diff / np.where(on, 1.0, d)[:, None]
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-20)
        g[on] = nrm[on]
        sdf[lo:lo + chunk] = s
        grad[lo:lo + chunk] = g
    return sdf, grad


def check_orientation(mesh: trimesh.Trimesh, watertight: bool,
                      points: np.ndarray, chunk: int) -> trimesh.Trimesh:
    """Volume vertices above the STL or beyond its horizontal extent are
    outside every building; the sign must say so. An open mesh with the
    orientation inverted is flipped; a mixed orientation is fatal."""
    lo, hi = mesh.bounds
    sure = (points[:, 2] > hi[2] + 0.5) \
        | (points[:, 0] < lo[0] - 0.5) | (points[:, 0] > hi[0] + 0.5) \
        | (points[:, 1] < lo[1] - 0.5) | (points[:, 1] > hi[1] + 0.5)
    idx = np.flatnonzero(sure)
    if len(idx) < 100:
        log("[stl] WARNING: too few vertices certainly outside the buildings "
            "to test the orientation of the STL")
        return mesh
    rng = np.random.default_rng(0)
    sample = points[rng.choice(idx, min(ORIENT_SAMPLE, len(idx)),
                               replace=False)]
    inside = float((signed_distance(mesh, sample, chunk)[0] < 0.0).mean())
    if inside > 0.5 and not watertight:
        mesh.invert()
        log("[stl] the STL was oriented inward; all triangles flipped")
        inside = float((signed_distance(mesh, sample, chunk)[0] < 0.0).mean())
    if inside > ORIENT_MAX_INSIDE:
        fail(f"{100 * inside:.1f}% of the vertices that are certainly "
             f"outside the buildings get a negative distance: parts of the "
             f"STL are oriented inward. Make the STL watertight so that the "
             f"orientation can be repaired per building.")
    return mesh


# ---------------------------------------------------------------------------
# Conventions and checks
# ---------------------------------------------------------------------------

def p_convention(p: np.ndarray, z: np.ndarray, requested: str, rho: float
                 ) -> tuple[str, float]:
    """The pressure convention of the input from the slope of p against z
    (hydrostatics dominate it): about -g for kinematic pressure with
    gravity, -rho g for Pa with gravity, 0 without gravity."""
    slope = float(np.polyfit(z, p, 1)[0])
    expected = {"kinematic-g": -G, "pa-g": -G * rho, "kinematic": 0.0,
                "pa": 0.0}
    if requested == "auto":
        if abs(slope - expected["kinematic-g"]) <= P_SLOPE_TOL:
            return "kinematic-g", slope
        if abs(slope - expected["pa-g"]) <= P_SLOPE_TOL:
            return "pa-g", slope
        if abs(slope) <= P_SLOPE_TOL:
            fail(f"p has no hydrostatic trend (slope {slope:.2f} per metre of "
                 f"height): the solver ran without gravity, so the unit is "
                 f"not decidable from the data. Pass --p kinematic (p in "
                 f"m2/s2) or --p pa (p in Pa).")
        fail(f"the slope of p against z is {slope:.2f} per metre, which "
             f"matches no known convention ({-G:.2f} kinematic with gravity, "
             f"{-G * rho:.2f} Pa with gravity, about 0 without gravity)")
    if abs(slope - expected[requested]) > 2.0 * P_SLOPE_TOL:
        fail(f"--p {requested} expects a slope of p against z of about "
             f"{expected[requested]:.2f} per metre, the data has {slope:.2f}")
    return requested, slope


def to_training_pressure(p: np.ndarray, z: np.ndarray, conv: str, rho: float
                         ) -> np.ndarray:
    """Kinematic pressure with the hydrostatic term -g z included, the
    convention of the training data (its detrended target is p + g z)."""
    if conv == "kinematic-g":
        return p
    if conv == "pa-g":
        return p / rho
    if conv == "pa":
        return p / rho - G * z
    return p - G * z            # kinematic, no gravity


def check_wind(U: np.ndarray, pos: np.ndarray, wind_toward: str) -> None:
    """After the rotation the mean wind of the upper layers must blow
    toward -y."""
    z = pos[:, 2]
    top = z >= 0.8 * z.max()
    ux, uy = float(U[top, 0].mean()), float(U[top, 1].mean())
    if not (uy < 0.0 and abs(uy) >= abs(ux)):
        fail(f"after the rotation for --wind-toward {wind_toward} the mean "
             f"wind of the upper layers is ({ux:.2f}, {uy:.2f}) m/s; the model "
             f"frame needs it toward -y. Check --wind-toward.")
    inflow = (pos[:, 1] > DOMAIN_HALF - 50.0) & (np.abs(pos[:, 0]) < 500.0) \
        & (z > 13.0) & (z < 17.0)
    if inflow.sum() >= 20:
        speed = float(np.median(np.linalg.norm(U[inflow], axis=1)))
        note = "" if 1.5 <= speed <= 2.5 else \
            f" — WARNING: the model was trained at {U_REF:.1f} m/s"
        log(f"[check] inflow speed at 13-17 m: {speed:.2f} m/s{note}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--volume", required=True,
                    help="internal mesh (.vtu/.vtk) with cell data U, p, nut")
    ap.add_argument("--surface", required=True, nargs="+",
                    help="building wall patch(es) (.vtp/.vtk) with cell data "
                         "p and wallShearStress")
    ap.add_argument("--stl", required=True, help="watertight building STL, m")
    ap.add_argument("--estate", required=True, help="estate name")
    ap.add_argument("--direction", required=True, choices=DIRECTIONS,
                    help="wind direction label of this case")
    ap.add_argument("--out", default="raw", help="output directory")
    ap.add_argument("--wind-toward", default="-y", choices=sorted(ROT_FOR_WIND),
                    help="axis the mean wind blows toward in the CFD frame "
                         "(default -y, the model frame)")
    ap.add_argument("--p", default="auto", choices=P_CONVENTIONS,
                    help="pressure convention of the input: kinematic-g "
                         "(m2/s2 with gravity), pa-g (Pa with gravity), "
                         "kinematic, pa (without gravity); auto detects the "
                         "first two")
    ap.add_argument("--rho", type=float, default=1.225, help="air density")
    ap.add_argument("--u-name", default="U")
    ap.add_argument("--p-name", default="p")
    ap.add_argument("--nut-name", default="nut")
    ap.add_argument("--wss-name", default="wallShearStress",
                    help="wall shear stress array; 'none' writes zeros (the "
                         "model does not use it, but such a file must not "
                         "join a training set: the anomaly filter compares "
                         "wall shear statistics across cases)")
    ap.add_argument("--chunk", type=int, default=1000000,
                    help="points per distance query")
    args = ap.parse_args()

    if not args.estate or any(c in args.estate for c in "/\\ \t"):
        fail("--estate must be a plain name without spaces or slashes")
    case_name = f"{args.estate}_{args.direction}"
    k = ROT_FOR_WIND[args.wind_toward]
    names = {"U": args.u_name, "p": args.p_name, "nut": args.nut_name,
             "wss": None if args.wss_name.lower() == "none" else args.wss_name}
    if names["wss"] is None:
        log("[surface] WARNING: no wall shear stress; zeros are written. "
            "Score the model on this file, do not train on it.")
    t_start = time.time()

    # ---- inputs, rotated into the model frame -----------------------------
    points, U, p, nut = read_volume(Path(args.volume), names)
    surf = read_surface([Path(s) for s in args.surface], names)
    mesh, watertight = read_stl(Path(args.stl))
    if k:
        points, U = rotate(points, k), rotate(U, k)
        for key in ("centres", "normals", "wss"):
            surf[key] = rotate(surf[key], k)
        mesh.vertices = rotate(np.asarray(mesh.vertices), k)
        log(f"[frame] rotated by {k} quarter turn(s): wind toward "
            f"{args.wind_toward} -> toward -y")

    # ---- crop ----------------------------------------------------------------
    keep = (np.abs(points[:, 0]) <= DOMAIN_HALF + CROP_TOL) \
        & (np.abs(points[:, 1]) <= DOMAIN_HALF + CROP_TOL) \
        & (points[:, 2] >= -CROP_TOL) & (points[:, 2] <= DOMAIN_TOP + CROP_TOL)
    points, U, p, nut = points[keep], U[keep], p[keep], nut[keep]
    if len(points) < 1000:
        fail(f"only {len(points)} vertices inside |x|, |y| <= {DOMAIN_HALF:.0f}"
             f" m, 0 <= z <= {DOMAIN_TOP:.0f} m; is the mesh in metres and "
             f"centred on the estate?")
    points[:, 2] = np.where(np.abs(points[:, 2]) < CROP_TOL, 0.0, points[:, 2])
    n_ground = int((points[:, 2] == 0.0).sum())
    if n_ground < MIN_GROUND_POINTS:
        fail(f"{n_ground} vertices on the ground plane z = 0: the points are "
             f"not the mesh vertices (cell centres?) or the ground is not at "
             f"z = 0")
    s_keep = (np.abs(surf["centres"][:, 0]) <= DOMAIN_HALF + CROP_TOL) \
        & (np.abs(surf["centres"][:, 1]) <= DOMAIN_HALF + CROP_TOL) \
        & (surf["centres"][:, 2] <= DOMAIN_TOP + CROP_TOL)
    if not s_keep.all():
        log(f"[crop] {int((~s_keep).sum())} wall faces outside the domain "
            f"dropped")
    surf = {key: v[s_keep] for key, v in surf.items()}
    if len(surf["centres"]) == 0:
        fail("no wall face inside the domain")
    log(f"[crop] {len(points):,} volume vertices ({n_ground:,} on the "
        f"ground), {len(surf['centres']):,} wall faces")

    # ---- units ---------------------------------------------------------------
    speed = np.linalg.norm(U, axis=1)
    if speed.max() > MAX_SPEED:
        fail(f"|U| reaches {speed.max():.1f}; U must be in m/s")
    if nut.min() < -1e-6:
        fail(f"nut has negative values down to {nut.min():.3g}")
    nut = np.maximum(nut, NUT_FLOOR)
    conv, slope = p_convention(p, points[:, 2], args.p, args.rho)
    p = to_training_pressure(p, points[:, 2], conv, args.rho)
    surf_p = to_training_pressure(surf["p"], surf["centres"][:, 2], conv,
                                  args.rho)
    wss = surf["wss"] / args.rho if conv.startswith("pa") else surf["wss"]
    log(f"[units] p: {conv} (slope of p against z {slope:.2f} per metre) -> "
        f"kinematic with the hydrostatic term; |U| max {speed.max():.2f} m/s; "
        f"nut {nut.min():.3g}-{nut.max():.3g} m2/s")
    check_wind(U, points, args.wind_toward)

    # ---- the STL's orientation, the surface normals ---------------------------
    t0 = time.time()
    mesh = check_orientation(mesh, watertight, points, args.chunk)
    sdf_c, grad_c = signed_distance(mesh, surf["centres"], args.chunk)
    med = float(np.median(np.abs(sdf_c)))
    if med > SURF_ON_STL_TOL:
        fail(f"the wall faces are a median {med:.2f} m away from the STL; the "
             f"patches and the STL must describe the same buildings in the "
             f"same frame and unit")
    flip = np.einsum("ij,ij->i", surf["normals"], grad_c) < 0.0
    normals = np.where(flip[:, None], -surf["normals"], surf["normals"])
    ground_like = (np.abs(normals[:, 2]) > 0.9) \
        & (surf["centres"][:, 2] < 0.05)
    if ground_like.mean() > 0.01:
        log(f"[surface] WARNING: {100 * ground_like.mean():.1f}% of the faces "
            f"are horizontal at z = 0; the patches should hold building "
            f"walls only, not the ground")
    log(f"[surface] faces within {med:.3f} m (median) of the STL, "
        f"{int(flip.sum()):,} normals flipped to point into the fluid "
        f"({time.time() - t0:.0f} s)")

    # ---- signed distance of the volume vertices ------------------------------
    t0 = time.time()
    sdf, grad = signed_distance(mesh, points, args.chunk)
    n_on = int((np.abs(sdf) < 1e-6).sum())
    n_in = int((sdf < -1e-6).sum())
    log(f"[distance] {len(points):,} vertices: {n_on:,} on the surface, "
        f"{n_in:,} inside the STL ({time.time() - t0:.0f} s)")

    # ---- record ----------------------------------------------------------------
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    bbox_min, bbox_max = vertices.min(axis=0), vertices.max(axis=0)
    span = bbox_max - bbox_min
    z_max_building = float(bbox_max[2])
    if z_max_building > DOMAIN_TOP:
        log(f"[check] WARNING: the tallest building ({z_max_building:.1f} m) "
            f"is above the {DOMAIN_TOP:.0f} m domain top the model was "
            f"trained with")
    vol_fields = np.column_stack([U, p, nut])
    surf_fields = np.column_stack([surf_p, wss])
    record = {
        "schema": SCHEMA,
        "case_name": case_name,
        "L_curator": float(span.max()),
        "z_max_building": z_max_building,
        "U_REF": U_REF,
        "RHO": float(args.rho),
        "Q": float(args.rho * U_REF ** 2),
        "G_GRAVITY": G,
        "stl_vertices": f32(vertices),
        "stl_faces": torch.from_numpy(np.ascontiguousarray(mesh.faces,
                                                           dtype=np.int64)),
        "stl_centers": f32(mesh.triangles_center),
        "stl_areas": f32(mesh.area_faces),
        "stl_face_normals": f32(mesh.face_normals),
        "stl_centroid": f32(vertices.mean(axis=0)),
        "stl_bbox_min": f32(bbox_min),
        "stl_bbox_max": f32(bbox_max),
        "stl_bbox_span": f32(span),
        "volume_pos": f32(points),
        "volume_fields": f32(vol_fields),
        "volume_field_names": list(VOL_NAMES),
        "volume_field_units": list(VOL_UNITS),
        # the dataset's sign: negative outside the buildings
        "volume_sdf": f32(-sdf),
        "volume_sdf_grad": f32(-grad),
        "volume_bbox_min": f32([-DOMAIN_HALF, -DOMAIN_HALF, 0.0]),
        "volume_bbox_max": f32([DOMAIN_HALF, DOMAIN_HALF, DOMAIN_TOP]),
        "surface_pos": f32(surf["centres"]),
        "surface_normals": f32(normals),
        "surface_areas": f32(surf["areas"]),
        "surface_fields": f32(surf_fields),
        "surface_field_names": list(SURF_NAMES),
        "surface_field_units": list(SURF_UNITS),
        "source": {
            "converter": "convert_cfd_case.py",
            "volume": str(args.volume),
            "surface": [str(s) for s in args.surface],
            "stl": str(args.stl),
            "stl_watertight": watertight,
            "wind_toward_input": args.wind_toward,
            "rotation_quarter_turns": k,
            "p_convention_input": conv,
            "p_slope_per_m": slope,
            "rho": float(args.rho),
            "wall_shear_stress": names["wss"] or "not in the input: zeros",
            "n_normals_flipped": int(flip.sum()),
            "surface_to_stl_median_m": med,
            "utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        },
    }
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{case_name}.pt"
    torch.save(record, str(out_path))
    log(f"[output] {out_path}: {len(points):,} volume points, "
        f"{len(surf['centres']):,} surface points, tallest building "
        f"{z_max_building:.2f} m, L_curator {span.max():.1f} m "
        f"({(time.time() - t_start) / 60:.1f} min)")
    log(f"[next] python predict_new_case.py --raw-pt {out_path} --weights "
        f"<weights dir> --out outputs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
