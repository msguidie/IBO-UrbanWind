#!/usr/bin/env python
"""CFD cases in the dataset's NPZ format -> raw case files for the
preprocessing (`main.py preprocess`) and for `predict_new_case.py --raw-pt`.

    python npz_to_raw.py --npz-dir path/to/npz --out data/raw --workers 8

One `<estate>_<dir>.pt` per NPZ, named after the NPZ stem. The NPZ format is
described in README.md under "Preprocessing your own cases". The recipe is
the one the dataset was made with:

  volume points   the NPZ "volume_mesh_centers" (they are mesh vertices)
                  inside |x|, |y| <= 550 m and 0 <= z <= 160 m, bounds
                  included, in file order; the surfaces are not cropped
  fields          U = U_nondim * U_REF, p = p_nondim * Q (Q = rho U_REF^2),
                  nut = exp(log(max(nut_nondim * L_curator, 1e-12))) * U_REF,
                  surface p and wall shear stress * Q; float32
  normals         the NPZ surface normals point into the buildings; they
                  are negated
  distance        Open3D closest points on the STL, sign from the dot
                  product with the cross-product face normal, stored as
                  computed. The STL triangles of the dataset are wound so
                  that these normals point into the buildings, which is why
                  the stored distance is negative outside the buildings and
                  its gradient points toward them; the code reading the
                  files flips the sign
  constants       L_curator = largest side of the STL bounding box,
                  z_max_building = top of the STL, both from float32
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "TBB_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

import numpy as np

SCHEMA = "hdb-3d-v3-physical"
U_REF = 2.0                 # m/s, free-stream reference (global_params_values[0])
RHO = 1.225                 # kg/m3 (global_params_values[1])
Q = RHO * U_REF ** 2        # 4.9, pressure / wall-shear normalisation
G_GRAVITY = 9.81
EPS_NUT = 1e-12             # floor before the log of nut
CUT_X = (-550.0, 550.0)
CUT_Y = (-550.0, 550.0)
CUT_Z = (0.0, 160.0)
VOLUME_FIELD_NAMES = ["Ux", "Uy", "Uz", "p", "nut"]
VOLUME_FIELD_UNITS = ["m/s", "m/s", "m/s", "Pa", "m2/s"]
SURFACE_FIELD_NAMES = ["p", "wss_x", "wss_y", "wss_z"]
SURFACE_FIELD_UNITS = ["Pa", "Pa", "Pa", "Pa"]
REQUIRED_KEYS = {
    "stl_coordinates": 2, "stl_faces": None, "stl_centers": 2, "stl_areas": 1,
    "surface_mesh_centers": 2, "surface_normals": 2, "surface_areas": 1,
    "surface_fields": 2, "volume_mesh_centers": 2, "volume_fields": 2,
}
_EPS_ON_SURFACE = 1e-6


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# NPZ extraction
# ---------------------------------------------------------------------------

def faces_to_2d(faces: np.ndarray) -> np.ndarray:
    if faces.ndim == 1:
        if faces.size % 3 != 0:
            raise ValueError(f"flat stl_faces of size {faces.size} is not a multiple of 3")
        return faces.reshape(-1, 3)
    if faces.ndim == 2 and faces.shape[1] == 3:
        return faces
    raise ValueError(f"unsupported stl_faces shape {faces.shape}")


def bbox_mask(points: np.ndarray) -> np.ndarray:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    return ((x >= CUT_X[0]) & (x <= CUT_X[1])
            & (y >= CUT_Y[0]) & (y <= CUT_Y[1])
            & (z >= CUT_Z[0]) & (z <= CUT_Z[1]))


def check_schema(d: dict, name: str) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in d]
    if missing:
        raise ValueError(f"missing keys {missing}; keys present: {sorted(d)}")
    for k, ndim in REQUIRED_KEYS.items():
        if ndim is not None and d[k].ndim != ndim:
            raise ValueError(f"{k} has {d[k].ndim} dimensions, expected {ndim}")
    n_vol, n_surf = len(d["volume_mesh_centers"]), len(d["surface_mesh_centers"])
    shapes = {"volume_mesh_centers": (n_vol, 3), "volume_fields": (n_vol, 5),
              "surface_mesh_centers": (n_surf, 3), "surface_normals": (n_surf, 3),
              "surface_fields": (n_surf, 4), "surface_areas": (n_surf,),
              "stl_coordinates": (len(d["stl_coordinates"]), 3),
              "stl_centers": (len(d["stl_areas"]), 3)}
    for k, shape in shapes.items():
        if tuple(d[k].shape) != shape:
            raise ValueError(f"{k} has shape {tuple(d[k].shape)}, expected {shape}")
    for k in ("volume_fields", "surface_fields", "volume_mesh_centers",
              "surface_mesh_centers", "stl_coordinates"):
        if not np.isfinite(d[k]).all():
            raise ValueError(f"{k} has non-finite values")
    gp = d.get("global_params_values")
    if gp is not None and np.asarray(gp).size >= 2:
        gp = np.asarray(gp, dtype=np.float64).ravel()
        if abs(gp[0] - U_REF) > 1e-6 or abs(gp[1] - RHO) > 1e-6:
            raise ValueError(f"global_params_values {gp[:2].tolist()} differ from the "
                             f"dataset's U_REF = {U_REF}, rho = {RHO}")
    if np.abs(d["volume_mesh_centers"][:, :2]).max() > 50000:
        raise ValueError("volume coordinates exceed 50 km; the NPZ must be in metres")


def load_and_extract(npz_path: Path) -> dict:
    with np.load(npz_path, allow_pickle=False) as raw:
        d = {k: np.array(raw[k]) for k in raw.files}
    check_schema(d, npz_path.name)
    stl_vertices = np.ascontiguousarray(d["stl_coordinates"], dtype=np.float32)
    stl_faces = faces_to_2d(np.ascontiguousarray(d["stl_faces"])).astype(np.int64)
    if stl_faces.min() < 0 or stl_faces.max() >= len(stl_vertices):
        raise ValueError("stl_faces index outside stl_coordinates")
    out = {
        "stl_vertices": stl_vertices,
        "stl_faces": stl_faces,
        "stl_centers": np.ascontiguousarray(d["stl_centers"], dtype=np.float32),
        "stl_areas": np.ascontiguousarray(d["stl_areas"], dtype=np.float32),
        "surface_pos": np.ascontiguousarray(d["surface_mesh_centers"], dtype=np.float32),
        # the NPZ normals point into the buildings; the dataset holds outward normals
        "surface_normals": -np.ascontiguousarray(d["surface_normals"], dtype=np.float32),
        "surface_areas": np.ascontiguousarray(d["surface_areas"], dtype=np.float32),
        "surface_fields": np.ascontiguousarray(d["surface_fields"], dtype=np.float32),
        "volume_pos": np.ascontiguousarray(d["volume_mesh_centers"], dtype=np.float32),
        "volume_fields": np.ascontiguousarray(d["volume_fields"], dtype=np.float32),
    }
    bbox_min = stl_vertices.min(axis=0)
    bbox_max = stl_vertices.max(axis=0)
    span = (bbox_max - bbox_min).astype(np.float32)
    out["stl_bbox_min"] = bbox_min
    out["stl_bbox_max"] = bbox_max
    out["stl_bbox_span"] = span
    out["L_curator"] = float(span.max())
    out["z_max_building"] = float(bbox_max[2])
    return out


# ---------------------------------------------------------------------------
# Signed distance (Open3D)
# ---------------------------------------------------------------------------

class SDFComputer:
    """Open3D closest points on the STL; the sign from the dot product of
    (query - closest) with the cross-product normal of the closest face.
    Positive on the side the triangle winding calls outside, which for the
    dataset's STLs is the inside of the buildings."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        import open3d as o3d
        self._vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        self._faces = np.ascontiguousarray(faces, dtype=np.int32)
        mesh = o3d.t.geometry.TriangleMesh()
        mesh.vertex.positions = o3d.core.Tensor(self._vertices)
        mesh.triangle.indices = o3d.core.Tensor(self._faces)
        self._scene = o3d.t.geometry.RaycastingScene()
        self._scene.add_triangles(mesh)
        v0 = self._vertices[self._faces[:, 0]]
        v1 = self._vertices[self._faces[:, 1]]
        v2 = self._vertices[self._faces[:, 2]]
        n = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(n, axis=1, keepdims=True)
        self.face_normals = (n / np.maximum(norms, 1e-20)).astype(np.float32)
        self._o3d = o3d

    def sdf_and_grad(self, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        query = np.ascontiguousarray(query, dtype=np.float32)
        res = self._scene.compute_closest_points(self._o3d.core.Tensor(query))
        closest = res["points"].numpy()
        prim = res["primitive_ids"].numpy()
        diff = query - closest
        dist = np.linalg.norm(diff, axis=1)
        on_surface = dist < _EPS_ON_SURFACE
        face_n = self.face_normals[prim]
        dot = (diff * face_n).sum(axis=1)
        sign = np.where(dot >= 0, 1.0, -1.0)
        sign[on_surface] = 1.0
        sdf = sign * dist
        safe = np.maximum(dist, _EPS_ON_SURFACE)
        grad = sign[:, None] * (diff / safe[:, None])
        if on_surface.any():
            grad[on_surface] = face_n[on_surface]
        grad = grad / np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-20)
        return sdf.astype(np.float32), grad.astype(np.float32)


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def convert_one(args: tuple[str, str]) -> tuple[str, str, str]:
    npz_path, out_dir = Path(args[0]), Path(args[1])
    name = npz_path.stem
    try:
        import torch
        torch.set_num_threads(1)
        t0 = time.time()
        d = load_and_extract(npz_path)
        L = d["L_curator"]

        mask = bbox_mask(d["volume_pos"])
        vol_pos = d["volume_pos"][mask]
        vol = d["volume_fields"][mask]
        if len(vol_pos) == 0:
            return "failed", name, "no volume points inside the learning region"

        # physical fields: float32 arithmetic with Python-float factors
        U = vol[:, 0:3] * U_REF
        p = vol[:, 3] * Q
        log_nut = np.log(np.maximum(vol[:, 4] * L, EPS_NUT))
        nut = (np.exp(log_nut) * U_REF).astype(np.float32)
        vol_fields = np.column_stack([U, p, nut]).astype(np.float32)
        surf = d["surface_fields"]
        surf_fields = np.column_stack([surf[:, 0] * Q, surf[:, 1:4] * Q]).astype(np.float32)

        sdf = SDFComputer(d["stl_vertices"], d["stl_faces"])
        vol_sdf, vol_grad = sdf.sdf_and_grad(vol_pos)

        def f32(a):
            return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))

        record = {
            "schema": SCHEMA,
            "case_name": name,
            "L_curator": L,
            "z_max_building": d["z_max_building"],
            "U_REF": U_REF,
            "RHO": RHO,
            "Q": Q,
            "G_GRAVITY": G_GRAVITY,
            "stl_vertices": f32(d["stl_vertices"]),
            "stl_faces": torch.from_numpy(d["stl_faces"]).long(),
            "stl_centers": f32(d["stl_centers"]),
            "stl_areas": f32(d["stl_areas"]),
            "stl_face_normals": f32(sdf.face_normals),
            "stl_centroid": f32(d["stl_vertices"].mean(axis=0)),
            "stl_bbox_min": f32(d["stl_bbox_min"]),
            "stl_bbox_max": f32(d["stl_bbox_max"]),
            "stl_bbox_span": f32(d["stl_bbox_span"]),
            "volume_pos": f32(vol_pos),
            "volume_fields": f32(vol_fields),
            "volume_field_names": list(VOLUME_FIELD_NAMES),
            "volume_field_units": list(VOLUME_FIELD_UNITS),
            # as computed: negative outside the buildings for the dataset's STLs
            "volume_sdf": f32(vol_sdf),
            "volume_sdf_grad": f32(vol_grad),
            "volume_bbox_min": torch.tensor([CUT_X[0], CUT_Y[0], CUT_Z[0]]).float(),
            "volume_bbox_max": torch.tensor([CUT_X[1], CUT_Y[1], CUT_Z[1]]).float(),
            "surface_pos": f32(d["surface_pos"]),
            "surface_normals": f32(d["surface_normals"]),
            "surface_areas": f32(d["surface_areas"]),
            "surface_fields": f32(surf_fields),
            "surface_field_names": list(SURFACE_FIELD_NAMES),
            "surface_field_units": list(SURFACE_FIELD_UNITS),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{name}.pt"
        tmp = out_path.with_suffix(".pt.tmp")
        torch.save(record, str(tmp))
        os.replace(tmp, out_path)

        z = vol_pos[:, 2].astype(np.float64)
        slope = float(np.polyfit(z, vol_fields[:, 3].astype(np.float64), 1)[0])
        msg = (f"{len(vol_pos):,} volume points ({int((vol_pos[:, 2] == 0).sum()):,} on the "
               f"ground), {len(d['surface_pos']):,} surface points, L_curator {L:.3f} m, "
               f"z_max {d['z_max_building']:.3f} m, p slope {slope:.3f}/m, sdf<0 "
               f"{float((vol_sdf < 0).mean()):.4f} (outside), sdf>0 "
               f"{float((vol_sdf > 0).mean()):.4f}, sdf==0 {float((vol_sdf == 0).mean()):.4f}, "
               f"{time.time() - t0:.0f} s")
        return "ok", name, msg
    except Exception as exc:  # noqa: BLE001 — reported per case
        return "failed", name, f"{exc}\n{traceback.format_exc()}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz-dir", required=True, help="folder of <estate>_<dir>.npz")
    ap.add_argument("--out", default="data/raw", help="output folder of raw case files")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    files = sorted(Path(args.npz_dir).expanduser().glob("*.npz"))
    if not files:
        raise SystemExit(f"no .npz files in {args.npz_dir}")
    out_dir = Path(args.out).expanduser()
    log(f"[npz_to_raw] {len(files)} files, {args.workers} workers -> {out_dir}")
    t0 = time.time()
    jobs = [(str(f), str(out_dir)) for f in files]
    n_fail = 0
    if args.workers <= 1:
        results = map(convert_one, jobs)
        pool = None
    else:
        pool = mp.Pool(args.workers)
        results = pool.imap_unordered(convert_one, jobs, chunksize=1)
    for status, name, msg in results:
        if status == "ok":
            log(f"[ok] {name}: {msg}")
        else:
            n_fail += 1
            log(f"[FAILED] {name}: {msg}")
    if pool is not None:
        pool.close()
        pool.join()
    log(f"[npz_to_raw] done in {(time.time() - t0) / 60:.1f} min, {n_fail} failed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    mp.set_start_method("spawn")
    sys.exit(main())
