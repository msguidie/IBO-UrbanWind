"""Deterministic seeding helpers."""
from __future__ import annotations

import hashlib
import random

import numpy as np
import torch


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def per_case_epoch_seed(case_name: str, epoch: int) -> int:
    """Stable 32-bit seed per (case, epoch) — deterministic query subsampling."""
    digest = hashlib.blake2b(f"{case_name}:{epoch}".encode(), digest_size=4).digest()
    return int.from_bytes(digest, "little")


def effective_train_seed(cfg) -> int:
    """The seed a training run uses: training.seed when set, else the
    config's top-level `seed`.

    The top-level `seed` also drives the split (test selection, rank
    assignment), so a seed study never moves it. training.seed (`train
    --seed`) moves only the training randomness: weight init, dropout,
    per-epoch case order and the per-step spatial draws (train_seed_salt)."""
    ts = getattr(cfg.training, "seed", None)
    return int(cfg.seed) if ts is None else int(ts)


def train_seed_salt(cfg) -> str:
    """Suffix mixed into every per-step training draw's seed string (IBO's
    query subsample, AB-UPT's anchors, the point-cloud baselines' draws).

    per_case_epoch_seed keys on (case, epoch) alone, so without this every
    seed of a study would see the same query points and anchors.

    Empty when the run uses the config's own seed (that run is the base
    member of a seed study); ":s<seed>" otherwise.

    Never applied to evaluation draws (CURVE_SAMPLE_MARKER, AB-UPT
    inference anchors) or to anything stored in a cache (the PointNet
    neighbour draw): every seed is measured on identical samples."""
    s = effective_train_seed(cfg)
    return "" if s == int(cfg.seed) else f":s{s}"
