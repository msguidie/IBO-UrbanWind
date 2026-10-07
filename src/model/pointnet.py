"""PointNet leaf encoder.

A shared MLP [13 -> 64 -> 256] over each leaf's member slots, then masked
max / mean / std pooling over the valid members. The three pooled vectors are
concatenated (3 x 256 = 768) and passed on unprojected; the leaf token
projection (encoder.py) consumes them directly. The reductions run in fp32.
A leaf with no valid member yields exactly zero.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PointNetEncoder(nn.Module):
    def __init__(self, in_dim: int = 13, hidden: int = 64, out_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, pn_feat: torch.Tensor, pn_mask: torch.Tensor) -> torch.Tensor:
        """pn_feat [B, L, k, 13]; pn_mask [B, L, k] bool -> [B, L, 3*out_dim]."""
        m = pn_mask[..., None]
        safe = torch.where(m, pn_feat, torch.zeros_like(pn_feat))
        h = self.mlp(safe)  # [B, L, k, out_dim]

        # forced-fp32 reductions; cast back to the ambient dtype at the end
        with torch.autocast("cuda", enabled=False):
            hf = h.float()
            neg = torch.finfo(hf.dtype).min
            h_max = torch.where(m, hf, torch.full_like(hf, neg)).max(dim=-2).values

            cnt = m.sum(dim=-2).clamp(min=1).to(hf.dtype)  # [B, L, 1] valid count
            h_valid = torch.where(m, hf, torch.zeros_like(hf))
            mu = h_valid.sum(dim=-2) / cnt
            dev = torch.where(m, hf - mu.unsqueeze(-2), torch.zeros_like(hf))
            var = dev.pow(2).sum(dim=-2) / cnt
            std = torch.sqrt(var + 1e-6)

        out = torch.cat([h_max, mu, std], dim=-1).to(h.dtype)  # [B, L, 3*out_dim]
        # fully-empty leaves -> exactly 0 across the whole concat (h_max is a
        # huge negative sentinel there, mu 0, std ~1e-3 — all zeroed here,
        # BEFORE Encoder.proj ever sees the vector)
        any_valid = pn_mask.any(dim=-1, keepdim=True)
        return torch.where(any_valid, out, torch.zeros_like(out))
