"""Normalization constants, inverse transforms and per-case metrics.

Every constant of the target transform lives in norm_stats.json: the z-score
mean/std of the five volume channels, the surface pressure and the five
morphology conditioners, the coordinate divisor, and the detrend terms
(Uy -= c ln(max(z, z0)), p += g z).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..data.estate import SURF_CHANNEL_NAMES, VOL_CHANNEL_NAMES

NORM_STATS_LENGTHS = {"vol_mean": 5, "vol_std": 5, "surf_mean": 1,
                      "surf_std": 1, "morph_mean": 5, "morph_std": 5}
NORM_STATS_SCALARS = ("coord_divisor", "detrend_uy_coeff", "detrend_z_floor",
                      "detrend_p_coeff")


def load_norm_stats(path: str | Path) -> dict:
    """Read norm_stats.json and check that every constant is present."""
    with open(path, encoding="utf-8") as f:
        stats = json.load(f)
    problems = [f"{k} (expected {n} values)"
                for k, n in NORM_STATS_LENGTHS.items()
                if len(stats.get(k) or []) != n]
    problems += [k for k in NORM_STATS_SCALARS if k not in stats]
    if problems:
        raise KeyError(f"{path}: missing or malformed entries: {problems}")
    return stats


def inverse_transform_vol(pred_z: np.ndarray, z_phys_vol: np.ndarray,
                          stats: dict) -> np.ndarray:
    """z-scored [N_vol, 5] -> physical: inverse z-score, then inverse detrend
    (Uy += c*log(max(z, floor)); p -= g*z)."""
    mean = np.asarray(stats["vol_mean"], dtype=np.float64)
    std = np.asarray(stats["vol_std"], dtype=np.float64)
    out = pred_z.astype(np.float64) * std + mean
    z = z_phys_vol.astype(np.float64)
    out[:, 1] += stats["detrend_uy_coeff"] * np.log(
        np.maximum(z, stats["detrend_z_floor"]))
    out[:, 3] -= stats["detrend_p_coeff"] * z
    return out


def inverse_transform_surf(pred_z: np.ndarray, z_phys_surf: np.ndarray,
                           stats: dict) -> np.ndarray:
    """z-scored [N_surf, 1] -> physical p_surf."""
    mean = np.asarray(stats["surf_mean"], dtype=np.float64)
    std = np.asarray(stats["surf_std"], dtype=np.float64)
    out = pred_z.astype(np.float64) * std + mean
    out[:, 0] -= stats["detrend_p_coeff"] * z_phys_surf.astype(np.float64)
    return out


def relative_l2(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-channel ||pred - true|| / ||true|| over points. [N, C] -> [C]."""
    num = np.linalg.norm(pred - true, axis=0)
    den = np.maximum(np.linalg.norm(true, axis=0), 1e-12)
    return num / den


def zscore_mse_case(pred_vol_z, true_vol_z, pred_surf_z, true_surf_z) -> dict:
    """Per-channel MSE in z-score units. A side with zero rows contributes
    zeros instead of a NaN mean."""
    vol_mse = ((pred_vol_z - true_vol_z) ** 2).mean(axis=0) \
        if len(pred_vol_z) else np.zeros(len(VOL_CHANNEL_NAMES))
    surf_mse = ((pred_surf_z - true_surf_z) ** 2).mean(axis=0) \
        if len(pred_surf_z) else np.zeros(len(SURF_CHANNEL_NAMES))
    return {
        "total": float(vol_mse.mean() + surf_mse.mean()),
        "volume": float(vol_mse.mean()),
        "surface": float(surf_mse.mean()),
        "volume_channels": {n: float(vol_mse[i])
                            for i, n in enumerate(VOL_CHANNEL_NAMES)},
        "surface_channels": {n: float(surf_mse[i])
                             for i, n in enumerate(SURF_CHANNEL_NAMES)},
    }


def case_metrics(pred_vol_z, true_vol_z, pred_surf_z, true_surf_z,
                 pred_vol_phys, true_vol_phys, pred_surf_phys, true_surf_phys
                 ) -> dict:
    rl2_vol = relative_l2(pred_vol_phys, true_vol_phys)
    rl2_surf = relative_l2(pred_surf_phys, true_surf_phys)
    return {
        "zscore_mse": zscore_mse_case(pred_vol_z, true_vol_z,
                                      pred_surf_z, true_surf_z),
        "relative_l2_physical": {
            "volume_channels": {n: float(rl2_vol[i])
                                for i, n in enumerate(VOL_CHANNEL_NAMES)},
            "surface_channels": {n: float(rl2_surf[i])
                                 for i, n in enumerate(SURF_CHANNEL_NAMES)},
        },
    }
