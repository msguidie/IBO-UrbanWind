"""Learning-rate schedules, each stepped at the recipe's own granularity
("step" = once per optimizer update, "epoch" = once per epoch). The trainer
calls scheduler.step() only at the declared granularity, and resuming
asserts scheduler.last_epoch matches the resumed position.

build_ibo_scheduler     IBO: linear warmup + cosine to a floor (or
                        warmup-stable-decay), with an optional second cosine
                        for training past a reference length.
build_onecycle          torch OneCycleLR (pct 0.3, div 25, cosine anneal),
                        per optimizer step (Transolver, SpiderSolver).
build_warmup_cosine     linear warmup + cosine to an absolute floor lr
                        (AB-UPT; Erwin with no warmup).
"""
from __future__ import annotations

import math

import torch


def build_ibo_scheduler(optimizer, total_steps: int, warmup_fraction: float,
                        min_lr_fraction: float, schedule: str = "cosine",
                        wsd_stable_fraction: float = 0.70,
                        ref_total_steps: int | None = None,
                        warmup_steps: int | None = None):
    """Multiplier on the peak lr. warmup_steps (else warmup_fraction of the
    reference length) of linear warmup, then cosine down to min_lr_fraction
    over the reference length (ref_total_steps, default total_steps). Steps
    beyond a given reference length follow a second cosine from 0.3 down to
    min_lr_fraction. schedule "wsd": warmup, constant, then linear decay;
    "constant": no schedule."""
    if schedule == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

    ref_steps = ref_total_steps if ref_total_steps is not None else total_steps
    warmup = max(int(warmup_steps if warmup_steps is not None
                     else warmup_fraction * ref_steps), 1)
    warmup = min(warmup, max(ref_steps - 1, 1))
    restart_frac = 0.3

    def lr_lambda(step: int) -> float:
        if step >= ref_steps:
            if ref_total_steps is None:
                return min_lr_fraction
            progress = (step - ref_steps) / max(total_steps - ref_steps, 1)
            progress = min(max(progress, 0.0), 1.0)
            return min_lr_fraction + 0.5 * (restart_frac - min_lr_fraction) \
                * (1 + math.cos(math.pi * progress))
        if step < warmup:
            return max(step, 1) / warmup
        progress = (step - warmup) / max(ref_steps - warmup, 1)
        return min_lr_fraction + (1 - min_lr_fraction) * 0.5 \
            * (1 + math.cos(math.pi * progress))

    if schedule == "wsd":
        stable = max(int(wsd_stable_fraction * ref_steps), 0)
        decay_start = min(warmup + stable, max(ref_steps - 1, warmup))

        def lr_lambda(step: int) -> float:  # noqa: F811
            if step >= ref_steps:
                return min_lr_fraction
            if step < warmup:
                return max(step, 1) / warmup
            if step < decay_start:
                return 1.0
            progress = (step - decay_start) / max(ref_steps - decay_start, 1)
            return 1.0 - (1.0 - min_lr_fraction) * min(max(progress, 0.0), 1.0)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_onecycle(optimizer, total_opt_steps: int, max_lr: float,
                   pct_start: float = 0.3, div_factor: float = 25.0,
                   final_div_factor: float = 1e4):
    """OneCycleLR over optimizer steps, as in the reference Transolver /
    SpiderSolver code. Stepping past total_steps raises, which catches a
    resumed run whose step count drifted."""
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lr, total_steps=int(total_opt_steps),
        pct_start=pct_start, div_factor=div_factor,
        final_div_factor=final_div_factor, anneal_strategy="cos")


def build_warmup_cosine(optimizer, total_units: int, peak_lr: float,
                        end_lr: float, warmup_fraction: float = 0.0):
    """Linear warmup over warmup_fraction of total_units, then cosine from
    peak_lr to the absolute end_lr at the final unit. warmup_fraction 0
    reproduces a plain CosineAnnealingLR(T_max=total, eta_min=end_lr)."""
    total_units = int(total_units)
    warmup = int(round(warmup_fraction * total_units))
    warmup = min(warmup, max(total_units - 1, 0))
    floor = end_lr / peak_lr

    def lr_lambda(u: int) -> float:
        if warmup > 0 and u < warmup:
            return max(u, 1) / warmup
        progress = (u - warmup) / max(total_units - warmup, 1)
        progress = min(max(progress, 0.0), 1.0)
        return floor + (1.0 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
