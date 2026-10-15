"""Point-cloud data path over the same estate packs, for the Transolver-3,
SpiderSolver and Erwin baselines.

Packs are read with data.estate.load_estate and everything the recipe does
not consume is dropped inside the per-pack load. Native inputs:

  Transolver / Spider   [pos (3, /550), sdf (1, /550), sdf_dir (3, unit)]
  Erwin                 coordinates only, through its random-Fourier encoder

Sampling (deterministic through per_case_epoch_seed, no mirror):
  uniform    the same mixed n_query draw over volume + surface rows that IBO
             uses (Transolver, Erwin); targets zero-padded per side plus an
             is_volume mask.
  spider     stratified, volume rows first: (n_query - surf_quota) volume
             rows + surf_quota surface rows, plus SpiderSolver's sector
             auxiliaries (per-row sector id = onion shell x surface region,
             the innermost volume participants, the sampled surface rows'
             region ids). Fixed counts, so shapes are static.

SpiderSolver regions on UrbanWind: a surface point's region is the dominant
signed axis of its (rotated) outward normal, 0:+X 1:-X 2:+Y 3:-Y 4:+Z 5:-Z;
a volume point inherits the region of its nearest surface point. That index
does not depend on the direction, so it is computed once per estate in the
canonical frame and cached under <cache_estate_dir>/derived/spider_v1/
(`main.py derive --model spider`). Onion shells are within-sample sdf-rank
bins and the innermost participants the n_inner smallest-|sdf| sampled
volume rows, as in the reference.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import Config
from ..utils.seed import per_case_epoch_seed
from .estate import (SURF_CHANNEL_NAMES, VOL_CHANNEL_NAMES,
                     atomic_torch_save, decode_sdf_dir_angles, load_estate,
                     rot_points)

SPIDER_DERIVED_VERSION = "spider_v1"

# pack keys the point-cloud recipes never consume (the IBO latent layout,
# support and region codes; the morphology conditioners are IBO-only)
_LATENT_KEYS = ("leaf_pos_norm", "leaf_stats", "pn_points_feat", "pn_mask",
                "_idw_k")
_SUPPORT_REGION_KEYS = ("query_support_norm", "region_dim2", "region_dim3")


def load_resident_pack_pointcloud(cfg: Config, estate: str,
                                  keep_dirs: list[str] | None,
                                  want_geometry: bool) -> dict:
    """Resident pack for the point-cloud recipes: the same reader, then the
    drop-set and the fp16 target downcast inside the per-pack load."""
    pack = load_estate(cfg, estate, resident_mode=True, build_idw_tree=False)
    all_dirs = list(pack["dirs"])
    kept = all_dirs if keep_dirs is None \
        else [d for d in all_dirs if d in keep_dirs]
    for d in all_dirs:
        pack.pop(f"global_morph_norm_{d}", None)
        if d not in kept:
            for key in (f"target_vol_{d}", f"target_surf_{d}",
                        f"surf_dim2_{d}", f"surf_dim3_{d}"):
                pack.pop(key, None)
    pack["dirs"] = kept
    for key in _LATENT_KEYS + _SUPPORT_REGION_KEYS:
        pack.pop(key, None)
    for d in kept:
        pack.pop(f"surf_dim2_{d}", None)
        pack.pop(f"surf_dim3_{d}", None)
    if not want_geometry:
        pack.pop("query_sdf_norm", None)
        pack.pop("query_sdf_dir_ang", None)
    for d in kept:
        pack[f"target_vol_{d}"] = pack[f"target_vol_{d}"].half()
        pack[f"target_surf_{d}"] = pack[f"target_surf_{d}"].half()
    return pack


# ---------------------------------------------------------------------------
# native input assembly
# ---------------------------------------------------------------------------

def native7_rows(pack: dict, rows: torch.Tensor, k: int) -> torch.Tensor:
    """[len(rows), 7] = [pos(3) rotated, sdf(1), sdf_dir(3) rotated] for
    GLOBAL row indices, in direction-frame k."""
    pos = rot_points(pack["query_pos_norm"][rows], k)
    sdf = pack["query_sdf_norm"][rows].float().unsqueeze(-1)
    sdf_dir = rot_points(decode_sdf_dir_angles(pack["query_sdf_dir_ang"][rows]), k)
    return torch.cat([pos, sdf, sdf_dir], dim=-1)


# ---------------------------------------------------------------------------
# uniform mixed sampling (Transolver / Erwin), the same draw IBO uses
# ---------------------------------------------------------------------------

def sample_uniform(pack: dict, direction: str, n_query: int, epoch: int,
                   want_geometry: bool) -> dict:
    k = int(pack["rot_code"][direction])
    n_vol = int(pack["N_vol"])
    n_total = n_vol + int(pack["N_surf"])
    if n_total < n_query:
        raise RuntimeError(f"{pack['estate']}_{direction}: N={n_total} < "
                           f"n_query={n_query}")
    name = f"{pack['estate']}_{direction}"
    g = torch.Generator()
    g.manual_seed(per_case_epoch_seed(name, epoch))
    sel = torch.randperm(n_total, generator=g)[:n_query].sort().values

    is_volume = sel < n_vol
    tv_pack = pack[f"target_vol_{direction}"]
    ts_pack = pack[f"target_surf_{direction}"]
    target_vol = torch.zeros(n_query, len(VOL_CHANNEL_NAMES),
                             dtype=tv_pack.dtype)
    target_surf = torch.zeros(n_query, len(SURF_CHANNEL_NAMES),
                              dtype=ts_pack.dtype)
    target_vol[is_volume] = tv_pack[sel[is_volume]]
    target_surf[~is_volume] = ts_pack[sel[~is_volume] - n_vol]

    out = {"target_vol": target_vol, "target_surf": target_surf,
           "is_volume": is_volume}
    if want_geometry:
        out["x"] = native7_rows(pack, sel, k)
        out["pos"] = out["x"][:, :3]
    else:
        out["pos"] = rot_points(pack["query_pos_norm"][sel], k)
    return out


# ---------------------------------------------------------------------------
# spider stratified sampling
# ---------------------------------------------------------------------------

def surf_region_classes(pack: dict, direction: str) -> torch.Tensor:
    """[N_surf] int64 region class of EVERY surface point in direction-frame
    k: dominant axis + sign of the rotated outward normal
    (0:+X 1:-X 2:+Y 3:-Y 4:+Z 5:-Z). Cached on the pack per direction —
    a pure function of stored angles, recomputed lazily per worker."""
    key = f"_spider_surfclass_{direction}"
    if key in pack:
        return pack[key]
    k = int(pack["rot_code"][direction])
    n_vol = int(pack["N_vol"])
    normals = rot_points(
        decode_sdf_dir_angles(pack["query_sdf_dir_ang"][n_vol:]), k)
    axis = normals.abs().argmax(dim=-1)                       # [Ns]
    sign_neg = torch.gather(normals, 1, axis.unsqueeze(1)).squeeze(1) < 0
    cls = axis * 2 + sign_neg.long()
    pack[key] = cls
    return cls


def onion_shell_ids(sdf_rows: torch.Tensor, onion_num: int) -> torch.Tensor:
    """Reference onion construction, per SAMPLE: sort the sampled volume
    rows by |sdf| ascending, first floor(n/onion) rows are shell 0, ...,
    the remainder joins the last shell (the reference's exact bin sizes)."""
    n = sdf_rows.shape[0]
    per = n // onion_num
    order = torch.argsort(sdf_rows.abs(), stable=True)
    shell = torch.empty(n, dtype=torch.long)
    src = torch.clamp(torch.arange(n) // max(per, 1), max=onion_num - 1)
    shell[order] = src
    return shell


def sample_spider(pack: dict, direction: str, n_vol_sample: int,
                  n_surf_sample: int, n_inner: int, onion_num: int,
                  n_regions: int, epoch: int) -> dict:
    k = int(pack["rot_code"][direction])
    n_vol = int(pack["N_vol"])
    n_surf = int(pack["N_surf"])
    if n_vol < n_vol_sample or n_surf < n_surf_sample:
        raise RuntimeError(
            f"{pack['estate']}_{direction}: N_vol={n_vol} N_surf={n_surf} < "
            f"spider sample ({n_vol_sample}+{n_surf_sample}) — this estate "
            f"is not eligible for the stratified draw")
    name = f"{pack['estate']}_{direction}"
    g_vol = torch.Generator()
    g_vol.manual_seed(per_case_epoch_seed(name + "__spider_vol", epoch))
    vol_local = torch.randperm(n_vol, generator=g_vol)[:n_vol_sample] \
        .sort().values
    g_surf = torch.Generator()
    g_surf.manual_seed(per_case_epoch_seed(name + "__spider_surf", epoch))
    surf_local = torch.randperm(n_surf, generator=g_surf)[:n_surf_sample] \
        .sort().values
    surf_global = surf_local + n_vol

    x_vol = native7_rows(pack, vol_local, k)
    x_surf = native7_rows(pack, surf_global, k)

    # sector auxiliaries (see the module docstring)
    sdf_vol = x_vol[:, 3]
    shell = onion_shell_ids(sdf_vol, onion_num)               # [Nv]
    surf_cls_full = surf_region_classes(pack, direction)     # [N_surf]
    nearest = pack["_spider_nearest_surf"]                    # [N_vol] int32
    vol_cls = surf_cls_full[nearest[vol_local].long()]        # [Nv]
    vol_gid = shell * n_regions + vol_cls
    inner_idx = torch.topk(sdf_vol.abs(), n_inner, largest=False).indices \
        .sort().values
    surf_region_id = surf_cls_full[surf_local]

    return {
        "x": torch.cat([x_vol, x_surf], dim=0),               # vol-first
        "target_vol": pack[f"target_vol_{direction}"][vol_local],
        "target_surf": pack[f"target_surf_{direction}"][surf_local],
        "vol_gid": vol_gid,
        "inner_idx": inner_idx,
        "surf_region_id": surf_region_id,
    }


# ---------------------------------------------------------------------------
# spider derived cache (nearest-surface index, canonical frame)
# ---------------------------------------------------------------------------

def spider_derived_dir(cfg: Config) -> Path:
    return Path(cfg.data.cache_estate_dir) / "derived" / SPIDER_DERIVED_VERSION


def _spider_derived_meta(cfg: Config) -> dict:
    return {"version": SPIDER_DERIVED_VERSION,
            "source_cache": Path(cfg.data.cache_estate_dir).name,
            "definition": "vol_nearest_surf = argmin_j |pos_vol - pos_surf_j|"
                          " (canonical frame, rotation-invariant)"}


def build_spider_derived(cfg: Config, estates: list[str],
                         workers: int = 8) -> None:
    """One int32 [N_vol] nearest-surface-point index per estate. Reads the
    packs and writes only under derived/spider_v1/."""
    from concurrent.futures import ProcessPoolExecutor

    out_dir = spider_derived_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "_meta.json"
    if not meta_path.exists():
        tmp = str(meta_path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_spider_derived_meta(cfg), f, indent=1)
        Path(tmp).replace(meta_path)
    todo = [e for e in estates if not (out_dir / f"{e}.pt").exists()]
    print(f"[spider-derive] {len(todo)}/{len(estates)} estates to build "
          f"-> {out_dir}", flush=True)
    if not todo:
        return
    args = [(cfg.data.cache_estate_dir, str(out_dir), e) for e in todo]
    if workers <= 1:
        for a in args:
            _spider_derive_one(a)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, _ in enumerate(pool.map(_spider_derive_one, args)):
                if (i + 1) % 25 == 0:
                    print(f"[spider-derive] {i + 1}/{len(args)}", flush=True)
    print("[spider-derive] done", flush=True)


def _spider_derive_one(arg: tuple[str, str, str]) -> None:
    from scipy.spatial import cKDTree

    cache_dir, out_dir, estate = arg
    pack = torch.load(str(Path(cache_dir) / f"{estate}.pt"),
                      map_location="cpu", weights_only=True)
    n_vol = int(pack["N_vol"])
    pos = pack["query_pos_norm"].float().numpy()
    tree = cKDTree(pos[n_vol:])
    _, idx = tree.query(pos[:n_vol], k=1, workers=1)
    payload = {"version": SPIDER_DERIVED_VERSION, "estate": estate,
               "n_vol": n_vol,
               "vol_nearest_surf": torch.from_numpy(idx.astype(np.int32)),
               "pos_sha8": hashlib.sha256(
                   pos[: min(4096, len(pos))].tobytes()).hexdigest()[:8]}
    atomic_torch_save(payload, Path(out_dir) / f"{estate}.pt")


def attach_spider_derived(cfg: Config, pack: dict) -> None:
    """Load this estate's derived nearest-surface index into the resident
    pack; version-mismatch or absence raises with build instructions."""
    path = spider_derived_dir(cfg) / f"{pack['estate']}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — the Spider recipe needs its derived cache. "
            f"Build it once on a CPU node:  python main.py derive --model "
            f"spider --manifest <pool manifest>")
    d = torch.load(str(path), map_location="cpu", weights_only=True)
    if d.get("version") != SPIDER_DERIVED_VERSION or \
            int(d.get("n_vol", -1)) != int(pack["N_vol"]):
        raise RuntimeError(
            f"{path}: derived-cache version/shape mismatch "
            f"({d.get('version')!r}, n_vol {d.get('n_vol')} vs "
            f"{pack['N_vol']}) — rebuild with `main.py derive --model "
            f"spider`")
    pack["_spider_nearest_surf"] = d["vol_nearest_surf"]


# ---------------------------------------------------------------------------
# resident datasets — the (idx, epoch) contract of data/dataloader.py
# ---------------------------------------------------------------------------

class PointcloudDataset(Dataset):
    """RAM-resident dataset for the point-cloud recipes. mode:
    'transolver' (uniform + 7ch), 'erwin' (uniform + coords), 'spider'
    (stratified + auxiliaries). Mirror augmentation is NOT supported here —
    the baseline protocol runs without it."""

    def __init__(self, cfg: Config, estate_names: list[str], mode: str,
                 dirs_by_estate: dict[str, list[str]] | None = None):
        assert mode in {"transolver", "erwin", "spider"}
        assert not cfg.training.mirror_augment, \
            "baseline recipes run mirror-free by protocol"
        self.cfg = cfg
        self.mode = mode
        self.n_query = int(cfg.training.n_query)
        want_geo = mode in {"transolver", "spider"}
        print(f"[data] loading {len(estate_names)} estate packs into RAM "
              f"({mode} point-cloud view)...", flush=True)
        self.packs = []
        for e in estate_names:
            pack = load_resident_pack_pointcloud(
                cfg, e, keep_dirs=None if dirs_by_estate is None
                else dirs_by_estate.get(e), want_geometry=want_geo)
            if mode == "spider":
                attach_spider_derived(cfg, pack)
            self.packs.append(pack)
        self.cases = [(pi, d) for pi, p in enumerate(self.packs)
                      for d in p["dirs"]]
        self.case_names = [f"{self.packs[pi]['estate']}_{d}"
                           for pi, d in self.cases]
        gb = sum(sum(v.numel() * v.element_size() for v in p.values()
                     if isinstance(v, torch.Tensor))
                 for p in self.packs) / 1024 ** 3
        print(f"[data] resident: {gb:.1f} GiB ({len(self.cases)} cases)",
              flush=True)

    @property
    def samples_per_case(self) -> int:
        return 1

    def __len__(self) -> int:
        return len(self.cases)

    def _sample(self, flat_idx: int, epoch: int) -> dict:
        pi, d = self.cases[flat_idx]
        pack = self.packs[pi]
        cfg = self.cfg
        if self.mode == "spider":
            sc = cfg.spider
            return sample_spider(
                pack, d, self.n_query - int(sc.surf_quota),
                int(sc.surf_quota), int(sc.n_inner), int(sc.onion_num),
                int(sc.n_regions), epoch)
        sample = sample_uniform(pack, d, self.n_query, epoch,
                                want_geometry=(self.mode == "transolver"))
        if self.mode == "erwin":
            from ..model.erwin.balltree import (build_tree, knn_edges,
                                                restore_index)
            ec = cfg.erwin
            ti, tm, rot = build_tree(sample["pos"], ec.strides,
                                     ec.ball_sizes, ec.rotate)
            sample["tree_idx"], sample["tree_mask"] = ti, tm
            sample["tree_idx_rot"] = rot
            # STATIC unpad-and-reorder index, built HERE in the loader
            # worker so the model's compiled tail never needs a boolean
            # mask (aten.nonzero would break fullgraph compilation)
            sample["restore_idx"] = restore_index(ti, tm)
            if ec.mp_steps > 0:
                sample["edge_index"] = knn_edges(sample["pos"], int(ec.knn))
        return sample

    def __getitem__(self, item) -> dict:
        flat_idx, epoch = item
        return self._sample(flat_idx, epoch)


def collate_single(samples: list[dict]) -> dict:
    """Point-cloud recipes are microbatch-1 by protocol: pass the single
    sample through untouched (recipes add batch dims where their forward
    wants them)."""
    assert len(samples) == 1, "point-cloud recipes are microbatch-1"
    return samples[0]
