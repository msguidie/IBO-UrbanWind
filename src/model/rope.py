"""3D rotary position embeddings.

theta_axis = pos_norm[axis] * scale_axis * inv_freq with global, fixed
scales, so a physical location gets the same phase in every case. The head
dimension is split across the axes (rope_dims, e.g. 12/12/8 of 32); each
axis has its own frequency ladder (rope_base per axis). Register tokens get
the identity rotation (cos = 1, sin = 0). Computed on the GPU inside the
processor from the latent positions; real arithmetic only.
"""
from __future__ import annotations

import torch


def precompute_rope_3d(
    pos_norm: torch.Tensor,
    head_dim: int = 32,
    rope_dims: tuple[int, int, int] = (12, 12, 8),
    scale_xy: float = 500.0,
    scale_z: float = 500.0,
    rope_base: float | tuple[float, float, float] = (450.0, 450.0, 64.0),
    n_register: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """pos_norm [..., L, 3] (already /550) -> (cos, sin), each
    [..., L + n_register, head_dim // 2] in fp32 on pos_norm's device,
    whatever the ambient autocast dtype (the caller casts)."""
    assert sum(rope_dims) == head_dim, f"rope_dims {rope_dims} must sum to {head_dim}"
    assert all(d % 2 == 0 for d in rope_dims)
    pos = pos_norm.float()
    dev = pos.device
    scales = (scale_xy, scale_xy, scale_z)
    # Per-axis base: scale fixes the finest wavelength on an axis
    # (2*pi*550/scale metres), base how far the ladder stretches from there
    # (lambda_max = lambda_min * base^((dim-2)/dim)). The domain is 1100 m
    # wide and 160 m tall, so the axes need different bases. A scalar base
    # applies to all three axes.
    bases = tuple(rope_base) if isinstance(rope_base, (list, tuple)) \
        else (rope_base,) * 3

    cos_parts, sin_parts = [], []
    for axis in range(3):
        dim = rope_dims[axis]
        half = dim // 2
        idx = torch.arange(half, dtype=torch.float32, device=dev)
        inv_freq = bases[axis] ** (-2.0 * idx / dim)  # [half]
        theta = pos[..., axis: axis + 1] * scales[axis] * inv_freq  # [..., L, half]
        cos_parts.append(torch.cos(theta))
        sin_parts.append(torch.sin(theta))
    cos = torch.cat(cos_parts, dim=-1)  # [..., L, head_dim//2]
    sin = torch.cat(sin_parts, dim=-1)

    if n_register > 0:
        reg_shape = (*cos.shape[:-2], n_register, head_dim // 2)
        reg_cos = torch.ones(reg_shape, dtype=cos.dtype, device=dev)
        reg_sin = torch.zeros(reg_shape, dtype=sin.dtype, device=dev)
        cos = torch.cat([cos, reg_cos], dim=-2)
        sin = torch.cat([sin, reg_sin], dim=-2)
    return cos, sin


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate q and k. q/k [B, H, T, D]; cos/sin [B, T, D//2] (or [T, D//2])."""
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    c = cos.unsqueeze(1).to(q.dtype)  # [B, 1, T, D//2]
    s = sin.unsqueeze(1).to(q.dtype)

    def rot(x: torch.Tensor) -> torch.Tensor:
        xp = x.reshape(*x.shape[:-1], -1, 2)
        x0, x1 = xp[..., 0], xp[..., 1]
        out0 = x0 * c - x1 * s
        out1 = x0 * s + x1 * c
        return torch.stack([out0, out1], dim=-1).flatten(-2)

    return rot(q), rot(k)
