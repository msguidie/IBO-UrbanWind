"""Estate packs: one file per estate, written by preprocess/build.py.

Geometry is stored ONCE, in the frame of the estate's canonical direction
(the first present in E > W > S > N); targets are stored per direction. The
wind directions of an estate are exact, index-aligned rigid rotations of
each other, so every other direction carries a rotation code k in {1, 2, 3}
(counter-clockwise quarter turns from the canonical frame). The code is
detected by elementwise comparison of the raw coordinates, never derived
from the file name, which also proves the row correspondence that lets
targets be stored per direction without matching.

The canonical partition (latent points, leaf statistics, PointNet samples)
holds for every direction: a non-canonical direction's latent tensors are
the canonical ones transformed. Every rotation here is a signed permutation
(sign flips plus column gathers), exact in any float dtype; k = 0 hands back
the pack tensors by reference.

Mirror order: rotate into direction D's frame FIRST, then mirror (the x
mirror is a symmetry of the flow only in the wind-aligned frame, and
M_x R_k = R_{-k} M_x).

Pack fields: query_* hold every volume row followed by every surface row;
sdf_dir is stored as fp16 (azimuth, elevation) angles, sdf / support /
leaf_stats / PointNet features as bf16, positions and targets as fp32.
region_dim1 (volume / surface) and the decoder k-NN are not stored: dim1 is
`row >= N_vol`, and the k-NN is recomputed from the leaf centroids at load
time, so the runtime neighbour count is free.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import Config
from ..manifest import estate_of
from ..utils.seed import per_case_epoch_seed
from .dataset import load_ux_norm_stats, mirror_slim_sample

DIR_PRIORITY = ("E", "W", "S", "N")
PACK_SCHEMA = "estate_v2"

# ---------------------------------------------------------------------------
# Transform tables. Rotation k = number of 90° CCW steps,
# (x, y) -> k=1: (-y, x); k=2: (-x, -y); k=3: (y, -x). Every entry is a
# signed permutation.
# ---------------------------------------------------------------------------

# query_support_norm columns are [+x, -x, +y, -y, +z, -z]; a rotated frame's
# +x ray is the old frame's R^{-1}(+x) ray, so rotation is a pure column
# permutation (z columns never move).
SUPPORT_PERM = {
    0: (0, 1, 2, 3, 4, 5),
    1: (3, 2, 0, 1, 4, 5),
    2: (1, 0, 3, 2, 4, 5),
    3: (2, 3, 1, 0, 4, 5),
}

# pn_points_feat columns: [0:2]=rel_pos xy, [2]=rel z, [3]=sdf,
# [4:6]=sdf_dir xy, [6]=dir z, [7:13]=support (same layout as above).
# PN_IDX[k][i] = source column of output column i; PN_NEG[k] = output
# columns to negate after the gather.
PN_IDX = {
    1: (1, 0, 2, 3, 5, 4, 6, 10, 9, 7, 8, 11, 12),
    2: (0, 1, 2, 3, 4, 5, 6, 8, 7, 10, 9, 11, 12),
    3: (1, 0, 2, 3, 5, 4, 6, 9, 10, 8, 7, 11, 12),
}
PN_NEG = {1: (0, 4), 2: (0, 1, 4, 5), 3: (1, 5)}

# leaf_stats channels: 3..8 = covariance xx,yy,zz,xy,xz,yz (trace-normalised
# — trace is a rotation invariant, so the normalisation never moves);
# 16,17,18 = box extents x,y,z. Everything else (occupancy/density/scalar
# distance moments/SDF stats) carries no xy anisotropy and copies through.
def _leaf_tables():
    tables = {}
    for k in (1, 2, 3):
        idx = list(range(19))
        neg: tuple[int, ...]
        if k == 1:
            idx[3], idx[4] = 4, 3
            idx[7], idx[8] = 8, 7
            idx[16], idx[17] = 17, 16
            neg = (6, 7)          # new_xy = -old_xy; new_xz = -old_yz
        elif k == 2:
            neg = (7, 8)          # xz, yz negate; xy invariant under 180°
        else:
            idx[3], idx[4] = 4, 3
            idx[7], idx[8] = 8, 7
            idx[16], idx[17] = 17, 16
            neg = (6, 8)          # new_xy = -old_xy; new_yz = -old_xz
        tables[k] = (tuple(idx), neg)
    return tables


LEAF_TABLES = _leaf_tables()


def rot_xy_pair(x: torch.Tensor, y: torch.Tensor, k: int
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """One (x, y) column pair under k CCW quarter turns. Sign flip + swap
    only — bitwise exact in any float dtype."""
    if k == 1:
        return -y, x
    if k == 2:
        return -x, -y
    if k == 3:
        return y, -x
    return x, y


def rot_points(t: torch.Tensor, k: int) -> torch.Tensor:
    """[..., 3] positions/vectors: rotate the xy columns, z untouched."""
    if k == 0:
        return t
    x, y = rot_xy_pair(t[..., 0], t[..., 1], k)
    return torch.stack([x, y, t[..., 2]], dim=-1)


def rot_support(t: torch.Tensor, k: int) -> torch.Tensor:
    """[..., 6] axis-ray support distances: pure column permutation."""
    if k == 0:
        return t
    return t[..., list(SUPPORT_PERM[k])]


def rot_pn_feat(t: torch.Tensor, k: int) -> torch.Tensor:
    """[..., 13] PointNet per-neighbour features: gather + negate."""
    if k == 0:
        return t
    out = t[..., list(PN_IDX[k])]
    for c in PN_NEG[k]:
        out[..., c] = -out[..., c]
    return out


def rot_leaf_stats(t: torch.Tensor, k: int) -> torch.Tensor:
    """[..., 19] leaf statistics: gather + negate (bf16-exact — negation is
    a sign-bit flip, never arithmetic)."""
    if k == 0:
        return t
    idx, neg = LEAF_TABLES[k]
    out = t[..., list(idx)]
    for c in neg:
        out[..., c] = -out[..., c]
    return out


# detect_rot_code tolerance in normalized (/550) units, i.e. 5.5e-7 m. A
# 90° rotation produced through a cos/sin matrix is not bit-exact
# (cos(pi/2) = 6e-17 in float64), so the comparison needs a tolerance; this
# one sits orders of magnitude above that rounding and below any mesh
# spacing.
ROT_DETECT_ATOL = 1e-9


def detect_rot_code(pos_canonical: torch.Tensor, pos_d: torch.Tensor,
                    label: str = "") -> int:
    """The k in {1, 2, 3} with max|rot_points(canonical, k) - pos_d| below
    ROT_DETECT_ATOL (compared in float64). Raises unless exactly one
    candidate passes; the error lists the residual of every candidate.
    Passing also proves the row correspondence between the two cases."""
    if pos_canonical.shape != pos_d.shape:
        raise RuntimeError(
            f"rot detect{f' [{label}]' if label else ''}: shape mismatch "
            f"{tuple(pos_canonical.shape)} vs {tuple(pos_d.shape)} — "
            f"N_vol/N_surf differ across directions")
    pc = pos_canonical.double()
    pd = pos_d.double()
    resid = {k: float((rot_points(pc, k) - pd).abs().max())
             for k in (1, 2, 3)}
    hits = [k for k, r in resid.items() if r <= ROT_DETECT_ATOL]
    if len(hits) != 1:
        detail = ", ".join(f"k={k}: max|Δ|={r:.3e}"
                           for k, r in sorted(resid.items()))
        raise RuntimeError(
            f"rot detect{f' [{label}]' if label else ''}: {len(hits)} of 3 "
            f"candidate rotations within atol={ROT_DETECT_ATOL:.0e} "
            f"(expected exactly 1) — {detail}")
    return hits[0]


# ---------------------------------------------------------------------------
# sdf_dir angle codec: [N, 3] unit vectors stored as [N, 2] fp16
# (azimuth, elevation), 4 bytes per point instead of 12, one rounding step
# (~1e-3 rad) from the float64 source.
# ---------------------------------------------------------------------------

def encode_sdf_dir_angles(sdf_dir: torch.Tensor) -> torch.Tensor:
    """[N, 3] unit vectors -> [N, 2] fp16 (theta = atan2(y, x),
    phi = asin(z)), computed in float64."""
    d = sdf_dir.double()
    theta = torch.atan2(d[:, 1], d[:, 0])
    phi = torch.asin(d[:, 2].clamp(-1.0, 1.0))
    return torch.stack([theta, phi], dim=1).to(torch.float16)


def decode_sdf_dir_angles(ang: torch.Tensor) -> torch.Tensor:
    """[..., 2] fp16 -> [..., 3] f32 exactly-unit vectors."""
    a = ang.float()
    ct, st = torch.cos(a[..., 0]), torch.sin(a[..., 0])
    cp, sp = torch.cos(a[..., 1]), torch.sin(a[..., 1])
    return torch.stack([cp * ct, cp * st, sp], dim=-1)


# ---------------------------------------------------------------------------
# Pack loading / materialisation
# ---------------------------------------------------------------------------

def knn_query(tree, pos: torch.Tensor, k: int, workers: int) -> torch.Tensor:
    """Indices of the k nearest leaf centroids (nearest first) as int32.
    tree is a scipy cKDTree over the centroids; k is the runtime
    model.idw_k."""
    _, idx = tree.query(pos.numpy(), k=k, workers=workers)
    if k == 1:
        idx = idx[:, None]
    return torch.from_numpy(idx.astype(np.int32))


def load_estate(cfg: Config, estate: str, resident_mode: bool = False,
                build_idw_tree: bool = True) -> dict:
    """Read one pack. With build_idw_tree, attach a cKDTree over the leaf
    centroids ("_idw_tree"); training and evaluation both recompute the
    decoder k-NN from it, in the canonical frame (rotation preserves
    distances and row order). resident_mode=True (training) drops the
    evaluation-only query_z_phys."""
    from scipy.spatial import cKDTree

    path = Path(cfg.data.cache_estate_dir) / f"{estate}.pt"
    pack = torch.load(str(path), map_location="cpu", weights_only=True)
    if pack.get("schema") != PACK_SCHEMA:
        raise RuntimeError(
            f"{path}: schema {pack.get('schema')!r} != {PACK_SCHEMA!r} — "
            f"rebuild this cache generation with `python main.py "
            f"preprocess`")
    if build_idw_tree:
        # only IBO consumes the leaf k-NN; the point-cloud baselines skip it
        pack["_idw_tree"] = cKDTree(pack["leaf_pos_norm"].numpy())
        pack["_idw_k"] = int(cfg.model.idw_k)
    if resident_mode:
        del pack["query_z_phys"]
    return pack


def load_resident_pack(cfg: Config, estate: str,
                       keep_dirs: list[str] | None = None) -> dict:
    """load_estate(resident_mode=True) plus the fp16 target downcast, done
    pack by pack: the training dataset loads every pack through here, so
    the fp32 targets of one pack are released before the next is read
    (deferring the downcast would hold all fp32 targets at once)."""
    pack = load_estate(cfg, estate, resident_mode=True)
    if keep_dirs is not None:
        # the pool manifest decides which directions exist; drop the
        # per-direction tensors of any other direction the pack holds
        for d in list(pack["dirs"]):
            if d in keep_dirs:
                continue
            for key in (f"target_vol_{d}", f"target_surf_{d}",
                        f"surf_dim2_{d}", f"surf_dim3_{d}",
                        f"global_morph_norm_{d}"):
                pack.pop(key, None)
        pack["dirs"] = [d for d in pack["dirs"] if d in keep_dirs]
    for d in pack["dirs"]:
        pack[f"target_vol_{d}"] = pack[f"target_vol_{d}"].half()
        pack[f"target_surf_{d}"] = pack[f"target_surf_{d}"].half()
    return pack


def materialize(pack: dict, direction: str, keep_z_phys: bool = True,
                with_idw: bool = True) -> dict:
    """The full case dict of one direction (every point, rotated into that
    direction's frame). k = 0 hands back pack tensors by reference; callers
    must treat the result as read-only."""
    if direction not in pack["dirs"]:
        raise KeyError(f"{pack['estate']}: direction {direction!r} not in "
                       f"{pack['dirs']}")
    if "query_z_phys" not in pack:
        raise RuntimeError(
            "materialize needs an eval-mode pack (load_estate with "
            "resident_mode=False) — the resident-mode pack dropped the "
            "eval-only fields")
    k = int(pack["rot_code"][direction])
    n_vol = int(pack["N_vol"])
    n_surf = int(pack["N_surf"])

    # k-NN of every point, computed once per pack and shared by all its
    # directions (rotation preserves distances and row order); the
    # point-cloud baselines skip it (with_idw=False)
    if with_idw and "_idw_full" not in pack:
        pack["_idw_full"] = knn_query(pack["_idw_tree"],
                                      pack["query_pos_norm"],
                                      pack["_idw_k"], workers=-1)

    if k == 0:
        rd2, rd3 = pack["region_dim2"], pack["region_dim3"]
    else:
        rd2 = pack["region_dim2"].clone()
        rd2[n_vol:] = pack[f"surf_dim2_{direction}"]
        rd3 = pack["region_dim3"].clone()
        rd3[n_vol:] = pack[f"surf_dim3_{direction}"]

    case = {
        "case_name": f"{pack['estate']}_{direction}",
        "N_vol": n_vol,
        "N_surf": int(pack["N_surf"]),
        "query_pos_norm": rot_points(pack["query_pos_norm"], k),
        "query_sdf_norm": pack["query_sdf_norm"].float(),
        "query_sdf_dir": rot_points(
            decode_sdf_dir_angles(pack["query_sdf_dir_ang"]), k),
        "query_support_norm": rot_support(pack["query_support_norm"], k),
        "region_dim1": torch.cat([
            torch.zeros(n_vol, dtype=torch.int8),
            torch.ones(n_surf, dtype=torch.int8)]),
        "region_dim2": rd2,
        "region_dim3": rd3,
        "idw_indices": pack.get("_idw_full"),
        "target_vol": pack[f"target_vol_{direction}"],
        "target_surf": pack[f"target_surf_{direction}"],
        "leaf_pos_norm": rot_points(pack["leaf_pos_norm"], k),
        "leaf_stats": rot_leaf_stats(pack["leaf_stats"], k),
        "pn_points_feat": rot_pn_feat(pack["pn_points_feat"], k),
        "pn_mask": pack["pn_mask"],
        "global_morph_norm": pack[f"global_morph_norm_{direction}"],
    }
    if keep_z_phys:
        case["query_z_phys"] = pack["query_z_phys"]
    return case


def sample_estate_query(pack: dict, direction: str, mirror: bool,
                        n_query: int, epoch: int, cfg: Config,
                        vol_mean_ux: float | None = None,
                        vol_std_ux: float | None = None,
                        seed_salt: str = "") -> dict:
    """One training sample of (direction, mirror): n_query rows drawn
    without replacement over the volume + surface pool, seeded by
    (`<estate>_<dir>` [+ "_xmirror"] + seed_salt, epoch). The rows are
    gathered from the canonical arrays, then rotated into the direction's
    frame, then (mirror twin only) mirrored. Returns exactly
    data.dataloader.BATCH_KEYS."""
    k = int(pack["rot_code"][direction])
    n_vol = int(pack["N_vol"])
    n_total = n_vol + int(pack["N_surf"])
    if n_total < n_query:
        raise RuntimeError(f"{pack['estate']}_{direction}: N={n_total} < "
                           f"n_query={n_query}")
    name = f"{pack['estate']}_{direction}"
    seed_name = name + ("_xmirror" if mirror else "")

    # seed_salt: the training seed's stream (utils.seed.train_seed_salt);
    # "" is the base stream
    g = torch.Generator()
    g.manual_seed(per_case_epoch_seed(seed_name + seed_salt, epoch))
    sel = torch.randperm(n_total, generator=g)[:n_query].sort().values

    is_volume = sel < n_vol
    tv_pack = pack[f"target_vol_{direction}"]
    ts_pack = pack[f"target_surf_{direction}"]
    target_vol = torch.zeros(n_query, 5, dtype=tv_pack.dtype)
    target_surf = torch.zeros(n_query, 1, dtype=ts_pack.dtype)
    target_vol[is_volume] = tv_pack[sel[is_volume]]
    target_surf[~is_volume] = ts_pack[sel[~is_volume] - n_vol]

    # the k-NN is queried in the canonical frame, where the tree was built
    # (rotation preserves distances, so the neighbour indices carry over);
    # workers=1 because this runs inside a single-threaded loader worker
    pos_rows = pack["query_pos_norm"][sel]
    idw = knn_query(pack["_idw_tree"], pos_rows, pack["_idw_k"], workers=1)
    rd1 = (sel >= n_vol).to(torch.int8)
    rd2 = pack["region_dim2"][sel]
    rd3 = pack["region_dim3"][sel]
    if k != 0:
        surf_rows = ~is_volume
        rd2[surf_rows] = pack[f"surf_dim2_{direction}"][sel[surf_rows] - n_vol]
        rd3[surf_rows] = pack[f"surf_dim3_{direction}"][sel[surf_rows] - n_vol]

    slim = {
        "query_pos_norm": rot_points(pos_rows, k),
        "query_sdf_norm": pack["query_sdf_norm"][sel].float(),
        "query_sdf_dir": rot_points(
            decode_sdf_dir_angles(pack["query_sdf_dir_ang"][sel]), k),
        "query_support_norm": rot_support(pack["query_support_norm"][sel], k),
        "region_dim1": rd1,
        "region_dim2": rd2,
        "region_dim3": rd3,
        "idw_indices": idw,
        "target_vol": target_vol,
        "target_surf": target_surf,
        "is_volume": is_volume,
        "pn_points_feat": rot_pn_feat(pack["pn_points_feat"], k),
        "pn_mask": pack["pn_mask"],
        "leaf_stats": rot_leaf_stats(pack["leaf_stats"], k),
        "leaf_pos_norm": rot_points(pack["leaf_pos_norm"], k),
        "global_morph_norm": pack[f"global_morph_norm_{direction}"],
    }
    if not mirror:
        return slim
    if vol_mean_ux is None or vol_std_ux is None:
        raise RuntimeError("mirror sample requested without the Ux "
                           "normalization constants (load_ux_norm_stats)")
    return mirror_slim_sample(slim, vol_mean_ux, vol_std_ux)


class RamEstateDataset(Dataset):
    """RAM-resident training dataset over estate packs: a flat index over
    (estate, direction, mirror). Targets are held as fp16."""

    def __init__(self, cfg: Config, estate_names: list[str], n_query: int,
                 mirror_augment: bool, only_canonical: bool = False,
                 dirs_by_estate: dict[str, list[str]] | None = None,
                 seed_salt: str = ""):
        from .dataloader import BATCH_KEYS
        self.cfg = cfg
        self.n_query = int(n_query)
        self.mirror_augment = bool(mirror_augment)
        self.seed_salt = str(seed_salt)
        # the manifest's direction lists decide which cases exist (they also
        # fix the per-rank step count, which must match across ranks)
        self.dirs_by_estate = dirs_by_estate
        print(f"[data] loading {len(estate_names)} estate packs into RAM...",
              flush=True)
        self.packs = [
            load_resident_pack(
                cfg, e, keep_dirs=None if dirs_by_estate is None
                else dirs_by_estate.get(e))
            for e in estate_names]
        # (pack index, direction) in a stable order: the same flat index is
        # the same case on every rank and across restarts
        self.cases = [(pi, d)
                      for pi, p in enumerate(self.packs)
                      for d in ([p["canonical"]] if only_canonical
                                else p["dirs"])]
        self.case_names = [f"{self.packs[pi]['estate']}_{d}"
                           for pi, d in self.cases]
        if self.mirror_augment:
            self.vol_mean_ux, self.vol_std_ux = load_ux_norm_stats(cfg)
        else:
            self.vol_mean_ux = self.vol_std_ux = None
        gb = sum(sum(v.numel() * v.element_size() for v in p.values()
                     if isinstance(v, torch.Tensor)) for p in self.packs) / 1024 ** 3
        print(f"[data] resident: {gb:.1f} GiB ({len(self.cases)} cases, "
              f"mirror {'ON' if self.mirror_augment else 'off'})", flush=True)
        probe = self._sample(0, epoch=0)
        assert set(probe) == set(BATCH_KEYS), \
            (f"estate sample keys {sorted(set(probe) ^ set(BATCH_KEYS))} "
             f"diverge from dataloader.BATCH_KEYS")

    @property
    def samples_per_case(self) -> int:
        return 2 if self.mirror_augment else 1

    def __len__(self) -> int:
        return len(self.cases) * self.samples_per_case

    def _sample(self, flat_idx: int, epoch: int) -> dict:
        if self.mirror_augment:
            case_idx, is_mirror = divmod(flat_idx, 2)
        else:
            case_idx, is_mirror = flat_idx, 0
        pi, d = self.cases[case_idx]
        return sample_estate_query(
            self.packs[pi], d, bool(is_mirror), self.n_query, epoch, self.cfg,
            vol_mean_ux=self.vol_mean_ux, vol_std_ux=self.vol_std_ux,
            seed_salt=self.seed_salt)

    def __getitem__(self, item) -> dict:
        flat_idx, epoch = item
        return self._sample(flat_idx, epoch)


# ---------------------------------------------------------------------------
# Evaluation-side loading
# ---------------------------------------------------------------------------

def _group_by_estate(case_names: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for n in case_names:
        grouped.setdefault(estate_of(n), []).append(n.rsplit("_", 1)[1])
    return grouped


def stream_full_estate_cases(cfg: Config, case_names: list[str],
                             with_idw: bool = True):
    """Yield (case name, full case dict). Each estate's pack is read once and
    serves all its requested directions; the next pack is read ahead on a
    background thread."""
    from concurrent.futures import ThreadPoolExecutor

    grouped = list(_group_by_estate(case_names).items())
    if not grouped:
        return
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(load_estate, cfg, grouped[0][0], False, with_idw)
        for i, (est, dirs) in enumerate(grouped):
            pack = fut.result()
            if i + 1 < len(grouped):
                fut = pool.submit(load_estate, cfg, grouped[i + 1][0],
                                  False, with_idw)
            for d in dirs:
                yield f"{est}_{d}", materialize(pack, d, keep_z_phys=True,
                                                with_idw=with_idw)
            del pack


# Target channels, in the column order of the packs
VOL_CHANNEL_NAMES = ("Ux", "Uy", "Uz", "p", "nut")
SURF_CHANNEL_NAMES = ("p_surf",)


def atomic_torch_save(obj, path: str | Path):
    """Write to a temporary file, then rename, so an interrupted write never
    leaves a truncated .pt behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    Path(tmp).replace(path)
