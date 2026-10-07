"""The losses every recipe reports through.

Both compute, in fp32, a weighted per-channel volume mean plus the surface
mean (loss = vol + surf), with the same breakdown keys (vol_loss_sum
[C_vol], surf_loss_sum [C_surf], n_vol, n_surf) for the epoch accumulators.

compute_loss         mixed rows: zero-padded per-side targets plus an
                     is_volume mask (ibo, transolver, erwin).
compute_loss_branch  separate volume and surface branches (abupt, spider);
                     the same normalization and weighting.

loss_shape "mse" is the default for every model; "huber" is optional.
"""
from __future__ import annotations

import torch


def _elementwise(pred, target, loss_shape: str, huber_delta: float):
    r = pred.float() - target.float()
    if loss_shape == "huber":
        abs_r = r.abs()
        return torch.where(abs_r <= huber_delta, 0.5 * r ** 2,
                           huber_delta * (abs_r - 0.5 * huber_delta))
    return r ** 2


def _weighted_vol(vol_per: torch.Tensor, vol_weights) -> torch.Tensor:
    if vol_weights is None:
        return vol_per.mean()
    w = torch.as_tensor(vol_weights, dtype=vol_per.dtype,
                        device=vol_per.device)
    return (vol_per * w).sum() / w.sum().clamp(min=1e-12)


def compute_loss(pred_vol, pred_surf, target_vol, target_surf, is_volume,
                 vol_weights=None, loss_shape: str = "mse",
                 huber_delta: float = 1.75):
    """Masked per-channel loss: weighted mean_C(vol) + surf."""
    vol_mask = is_volume[..., None].float()
    surf_mask = 1.0 - vol_mask
    dv = _elementwise(pred_vol, target_vol, loss_shape, huber_delta) * vol_mask
    ds = _elementwise(pred_surf, target_surf, loss_shape, huber_delta) * surf_mask
    vol_loss_sum = dv.sum(dim=(0, 1))
    surf_loss_sum = ds.sum(dim=(0, 1))
    n_vol = vol_mask.sum()
    n_surf = surf_mask.sum()
    vol_loss = _weighted_vol(vol_loss_sum / n_vol.clamp(min=1), vol_weights)
    surf_loss = (surf_loss_sum / n_surf.clamp(min=1)).mean()
    loss = vol_loss + surf_loss
    breakdown = {"vol_loss_sum": vol_loss_sum.detach(),
                 "surf_loss_sum": surf_loss_sum.detach(),
                 "n_vol": n_vol.detach(), "n_surf": n_surf.detach()}
    return loss, breakdown


def compute_loss_branch(pred_vol, pred_surf, target_vol, target_surf,
                        vol_weights=None, loss_shape: str = "mse",
                        huber_delta: float = 1.75):
    """Branch-separated loss: no mask, no padded rows (the sampler only
    admits cases whose pools cover the draw). Static per-point counts —
    shapes are compile-time constants."""
    dv = _elementwise(pred_vol, target_vol, loss_shape, huber_delta)
    ds = _elementwise(pred_surf, target_surf, loss_shape, huber_delta)
    vol_loss_sum = dv.sum(dim=(0, 1))
    surf_loss_sum = ds.sum(dim=(0, 1))
    n_vol = dv.new_tensor(float(dv.shape[0] * dv.shape[1]))
    n_surf = ds.new_tensor(float(ds.shape[0] * ds.shape[1]))
    vol_loss = _weighted_vol(vol_loss_sum / n_vol.clamp(min=1), vol_weights)
    surf_loss = (surf_loss_sum / n_surf.clamp(min=1)).mean()
    loss = vol_loss + surf_loss
    breakdown = {"vol_loss_sum": vol_loss_sum.detach(),
                 "surf_loss_sum": surf_loss_sum.detach(),
                 "n_vol": n_vol.detach(), "n_surf": n_surf.detach()}
    return loss, breakdown
