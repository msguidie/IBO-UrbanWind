"""RMSNorm and the leaf token projection."""
from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm computed in fp32, eps 1e-6."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        rms = xf.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (xf * rms).to(x.dtype) * self.weight.to(x.dtype)


class Encoder(nn.Module):
    """Leaf token: Linear(cat(PointNet max|mean|std, leaf_stats (19),
    latent position (3)) -> D)."""

    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, latent_dim)

    def forward(self, pn_feat: torch.Tensor, leaf_stats: torch.Tensor,
                leaf_pos_norm: torch.Tensor) -> torch.Tensor:
        x = torch.cat([pn_feat, leaf_stats, leaf_pos_norm], dim=-1)
        return self.proj(x)
