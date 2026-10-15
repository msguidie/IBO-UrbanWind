"""AB-UPT data path over the same estate packs IBO trains on.

Packs are read with data.estate.load_estate; everything the AB-UPT variant
does not consume is dropped inside the per-pack load: always the latent
layout (leaf_pos_norm / leaf_stats / pn_points_feat / pn_mask) and the
morphology conditioners, and for the coordinate-only variant also every
geometry feature (sdf, sdf_dir, support, region codes).

One sample = one estate x one wind direction: two independent draws without
replacement, n_anchors volume rows and n_anchors surface rows, seeded by
("<estate>_<dir>[_xmirror]__abupt_vol" / "...__abupt_surf", epoch). Both
pools must hold at least n_anchors rows, which the AB-UPT-eligible pool
manifest guarantees. Rows are gathered in the canonical frame, rotated into
the direction's frame, then (mirror twin only) mirrored. Positions stay in
the /550 frame; the affine to AB-UPT's non-negative coordinates is part of
the model stem (model/abupt.py).
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from ..config import Config
from ..utils.seed import per_case_epoch_seed
from .dataset import load_ux_norm_stats
from .estate import (decode_sdf_dir_angles, load_estate, rot_points,
                     rot_support)

ABUPT_BATCH_KEYS_VANILLA = ("surface_position", "volume_position",
                            "target_surf", "target_vol")
ABUPT_BATCH_KEYS_GEOMETRY = ABUPT_BATCH_KEYS_VANILLA + (
    "surface_sdf", "surface_sdf_dir", "surface_support",
    "surface_region_dim2", "surface_region_dim3",
    "volume_sdf", "volume_sdf_dir", "volume_support",
    "volume_region_dim2", "volume_region_dim3")

# geometry-feature keys of the pack, kept only with geometry features on
_PACK_GEO_KEYS = ("query_sdf_norm", "query_sdf_dir_ang",
                  "query_support_norm", "region_dim2", "region_dim3")
# the IBO latent layout, consumed by no AB-UPT variant
_PACK_LATENT_KEYS = ("leaf_pos_norm", "leaf_stats", "pn_points_feat",
                     "pn_mask", "_idw_k")


def abupt_batch_keys(geometry_features: bool) -> tuple[str, ...]:
    return ABUPT_BATCH_KEYS_GEOMETRY if geometry_features \
        else ABUPT_BATCH_KEYS_VANILLA


def check_abupt_coord_bounds(pos_norm: torch.Tensor, label: str) -> None:
    """The domain premise behind the fixed affine (model/abupt.py): xy in
    [-550, 550] m and z >= 0, i.e. /550-normalized xy in [-1, 1], so
    (pos + [1, 1, 0]) * 500 is non-negative (the reference asserts
    coords >= 0 inside its forward; here it is checked outside the compiled
    graph)."""
    xy = pos_norm[..., :2]
    z = pos_norm[..., 2]
    if bool((xy.amin() < -1.0) or (xy.amax() > 1.0) or (z.amin() < 0.0)):
        raise RuntimeError(
            f"{label}: query_pos_norm outside the UrbanWind domain "
            f"(xy in [-1, 1], z >= 0 after /550): xy range "
            f"[{float(xy.amin()):.6f}, {float(xy.amax()):.6f}], "
            f"z min {float(z.amin()):.6f} — the AB-UPT fixed affine "
            f"(pos + [1, 1, 0]) * 500 would produce negative coordinates. "
            f"Fix the domain bounds in cfg.abupt, do not clamp.")


def load_resident_pack_abupt(cfg: Config, estate: str,
                             keep_dirs: list[str] | None,
                             geometry_features: bool) -> dict:
    """The AB-UPT counterpart of data.estate.load_resident_pack: the same
    reader, then this variant's drop-set and the fp16 target downcast, all
    inside the per-pack load."""
    pack = load_estate(cfg, estate, resident_mode=True, build_idw_tree=False)
    all_dirs = list(pack["dirs"])
    # the manifest's direction lists decide which cases exist
    kept = all_dirs if keep_dirs is None \
        else [d for d in all_dirs if d in keep_dirs]
    for d in all_dirs:
        pack.pop(f"global_morph_norm_{d}", None)
        if d not in kept:
            for key in (f"target_vol_{d}", f"target_surf_{d}",
                        f"surf_dim2_{d}", f"surf_dim3_{d}"):
                pack.pop(key, None)
    pack["dirs"] = kept
    for key in _PACK_LATENT_KEYS:
        pack.pop(key, None)
    if not geometry_features:
        for key in _PACK_GEO_KEYS:
            pack.pop(key, None)
        for d in kept:
            pack.pop(f"surf_dim2_{d}", None)
            pack.pop(f"surf_dim3_{d}", None)
    check_abupt_coord_bounds(pack["query_pos_norm"], estate)
    for d in kept:
        pack[f"target_vol_{d}"] = pack[f"target_vol_{d}"].half()
        pack[f"target_surf_{d}"] = pack[f"target_surf_{d}"].half()
    return pack


def _mirror_abupt_sample_(sample: dict, vol_mean_ux: float,
                          vol_std_ux: float) -> dict:
    """x-mirror of a freshly built AB-UPT sample, in place (its tensors are
    fresh gathers, not pack storage). Same rules as
    data.dataset.mirror_slim_sample: positions and sdf_dir flip x, support
    swaps its +x / -x columns, the z-scored Ux maps z -> -z - 2 mu / sigma;
    sdf magnitudes, region codes and p_surf are invariant."""
    for key in ("surface_position", "volume_position"):
        sample[key][:, 0] = -sample[key][:, 0]
    tv = sample["target_vol"]
    tv[:, 0] = -tv[:, 0] - 2.0 * (vol_mean_ux / vol_std_ux)
    if "surface_sdf_dir" in sample:
        for key in ("surface_sdf_dir", "volume_sdf_dir"):
            sample[key][:, 0] = -sample[key][:, 0]
        for key in ("surface_support", "volume_support"):
            sup = sample[key]
            sample[key] = sup[:, (1, 0, 2, 3, 4, 5)]
    return sample


def sample_estate_anchors(pack: dict, direction: str, mirror: bool,
                          n_anchors: int, epoch: int,
                          geometry_features: bool,
                          vol_mean_ux: float | None = None,
                          vol_std_ux: float | None = None,
                          seed_salt: str = "") -> dict:
    """One AB-UPT sample of (estate, direction, mirror): n_anchors volume and
    n_anchors surface rows, independently drawn without replacement, gathered
    in the canonical frame, rotated into the direction's frame, then (mirror
    twin only) mirrored."""
    k = int(pack["rot_code"][direction])
    n_vol = int(pack["N_vol"])
    n_surf = int(pack["N_surf"])
    if n_vol < n_anchors or n_surf < n_anchors:
        raise RuntimeError(
            f"{pack['estate']}_{direction}: N_vol={n_vol} N_surf={n_surf} "
            f"< n_anchors={n_anchors} — this estate is not AB-UPT-eligible; "
            f"train AB-UPT on an AB-UPT-eligible pool manifest.")
    # seed_salt: the training seed's stream (training dataset only; the
    # inference anchors in evaluation/abupt_eval.py never see it)
    seed_base = (f"{pack['estate']}_{direction}"
                 + ("_xmirror" if mirror else "") + seed_salt)

    g_vol = torch.Generator()
    g_vol.manual_seed(per_case_epoch_seed(seed_base + "__abupt_vol", epoch))
    vol_local = torch.randperm(n_vol, generator=g_vol)[:n_anchors].sort().values
    g_surf = torch.Generator()
    g_surf.manual_seed(per_case_epoch_seed(seed_base + "__abupt_surf", epoch))
    surf_local = torch.randperm(n_surf,
                                generator=g_surf)[:n_anchors].sort().values
    surf_global = surf_local + n_vol

    sample = {
        "surface_position": rot_points(pack["query_pos_norm"][surf_global], k),
        "volume_position": rot_points(pack["query_pos_norm"][vol_local], k),
        # per-direction targets are indexed by local rows
        "target_surf": pack[f"target_surf_{direction}"][surf_local],
        "target_vol": pack[f"target_vol_{direction}"][vol_local],
    }
    if geometry_features:
        # combined per-point arrays are indexed by global rows; surface rows
        # carry sdf = 0 and the surface normal as sdf_dir
        sample["surface_sdf"] = pack["query_sdf_norm"][surf_global].float()
        sample["volume_sdf"] = pack["query_sdf_norm"][vol_local].float()
        sample["surface_sdf_dir"] = rot_points(
            decode_sdf_dir_angles(pack["query_sdf_dir_ang"][surf_global]), k)
        sample["volume_sdf_dir"] = rot_points(
            decode_sdf_dir_angles(pack["query_sdf_dir_ang"][vol_local]), k)
        sample["surface_support"] = rot_support(
            pack["query_support_norm"][surf_global], k).float()
        sample["volume_support"] = rot_support(
            pack["query_support_norm"][vol_local], k).float()
        # volume region codes hold for every direction; surface rows take
        # the direction's own labels for k != 0
        sample["volume_region_dim2"] = pack["region_dim2"][vol_local]
        sample["volume_region_dim3"] = pack["region_dim3"][vol_local]
        if k == 0:
            sample["surface_region_dim2"] = pack["region_dim2"][surf_global]
            sample["surface_region_dim3"] = pack["region_dim3"][surf_global]
        else:
            sample["surface_region_dim2"] = \
                pack[f"surf_dim2_{direction}"][surf_local]
            sample["surface_region_dim3"] = \
                pack[f"surf_dim3_{direction}"][surf_local]
    if not mirror:
        return sample
    if vol_mean_ux is None or vol_std_ux is None:
        raise RuntimeError("mirror sample requested without the Ux "
                           "normalization constants (load_ux_norm_stats)")
    return _mirror_abupt_sample_(sample, vol_mean_ux, vol_std_ux)


class RamEstateDatasetABUPT(Dataset):
    """RAM-resident AB-UPT training dataset: the flat (estate, direction,
    mirror) index of data.estate.RamEstateDataset with AB-UPT samples."""

    def __init__(self, cfg: Config, estate_names: list[str],
                 mirror_augment: bool,
                 dirs_by_estate: dict[str, list[str]] | None = None,
                 seed_salt: str = ""):
        self.cfg = cfg
        self.seed_salt = str(seed_salt)
        self.n_anchors = int(cfg.abupt.n_anchors)
        self.geometry_features = bool(cfg.abupt.geometry_features)
        self.mirror_augment = bool(mirror_augment)
        self.dirs_by_estate = dirs_by_estate
        print(f"[data] loading {len(estate_names)} estate packs into RAM "
              f"(AB-UPT{', geometry features' if self.geometry_features else ', vanilla'})...",
              flush=True)
        self.packs = [
            load_resident_pack_abupt(
                cfg, e, keep_dirs=None if dirs_by_estate is None
                else dirs_by_estate.get(e),
                geometry_features=self.geometry_features)
            for e in estate_names]
        self.cases = [(pi, d)
                      for pi, p in enumerate(self.packs)
                      for d in p["dirs"]]
        self.case_names = [f"{self.packs[pi]['estate']}_{d}"
                           for pi, d in self.cases]
        if self.mirror_augment:
            self.vol_mean_ux, self.vol_std_ux = load_ux_norm_stats(cfg)
        else:
            self.vol_mean_ux = self.vol_std_ux = None
        gb = sum(sum(v.numel() * v.element_size() for v in p.values()
                     if isinstance(v, torch.Tensor))
                 for p in self.packs) / 1024 ** 3
        print(f"[data] resident: {gb:.1f} GiB ({len(self.cases)} cases, "
              f"mirror {'ON' if self.mirror_augment else 'off'})", flush=True)
        probe = self._sample(0, epoch=0)
        expected = set(abupt_batch_keys(self.geometry_features))
        assert set(probe) == expected, \
            (f"AB-UPT sample keys {sorted(set(probe) ^ expected)} diverge "
             f"from the batch contract")

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
        return sample_estate_anchors(
            self.packs[pi], d, bool(is_mirror), self.n_anchors, epoch,
            self.geometry_features, vol_mean_ux=self.vol_mean_ux,
            vol_std_ux=self.vol_std_ux, seed_salt=self.seed_salt)

    def __getitem__(self, item) -> dict:
        flat_idx, epoch = item
        return self._sample(flat_idx, epoch)
