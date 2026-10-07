"""SpiderSolver — compile-safe rewrite of the reference attention (Qi et
al., 2025), adapted to UrbanWind (surface regions by normal direction, see
data/pointcloud.py).

Faithful to the reference (SpiderSolver-main, NeurIPS 2025) in MATH:
  - coarse attention: full self-attention between the innermost volume
    participants and every sampled surface point (reference
    `_process_coarse_attention`), per-head q/k/v (no bias), softmax scale
    dim_head^-0.5;
  - fine attention: sector tokens = per-(onion shell x surface cluster)
    aggregation of the volume rows, where — exactly as the reference dense
    one-hot computes it — a sector token is sum(members) / N_vol_total,
    NOT a per-member mean (the reference's `torch.mean(axis=2)` over the
    zero-masked full row axis divides by the total row count); surface
    token(s): 2D mode = ONE true mean over all surface rows, 3D mode = one
    token per surface region, sum(members) / N_surf_total, then the
    reference's LayerNorm across the token axis; plain q/k/v attention over
    the token sequence;
  - T2P: per-point weights (weight_surf / weight_velo_velo /
    weight_velo_car, orthogonally initialised) matmul the attended tokens
    back to points; the reference fusion formulas verbatim
    (surf: (w_surf@tok_surf + w_car@tok_velo)/2 + coarse_update;
    volume: w_velo@tok_velo, coarse update ADDED on the innermost rows);
  - block layout (pre-LN attn residual + pre-LN MLP residual), MLP ratio 2,
    trunc_normal(0.02) init, learnable placeholder on the embedding.

Form-only rewrites (tests/test_gates.py checks them against
model/spider_reference.py, a near-verbatim dense port, forward and
backward):
  - the dense [B, onion, clusters, N, C] one-hot intermediates become
    segment sums over a per-row group id (identical values, a [G, N]
    one-hot matmul instead of O(onion*clusters*N*C) memory);
  - explicit q@k^T softmax attention becomes F.scaled_dot_product_attention
    (dropout is 0 in the protocol, so the two are the same math);
  - boolean-mask row splits become STATIC slices: a sample is laid out
    [volume rows (fixed count) ; surface rows (fixed count)], so every
    shape is a compile-time constant (torch.compile / CUDA-Graph safe);
  - `.cuda()` calls and hardcoded point counts (3586/28504/500/28004...)
    are gone; counts come from tensor shapes;
  - the reference 2D block's `mlp_new` and the attention `temperature` are
    NOT ported: both are constructed but never used in any reference
    forward (dead parameters that would only distort the Params column).

Batch contract: micro-batch 1 (the protocol of every model in the
comparison); the segment / index ops rely on it and forward asserts B == 1.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class Mlp(nn.Module):
    """Reference MLP (n_layers=0, res=False): Linear -> GELU -> Linear."""

    def __init__(self, n_input: int, n_hidden: int, n_output: int):
        super().__init__()
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden),
                                        nn.GELU())
        self.linear_post = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_post(self.linear_pre(x))


def _segment_sum(src: torch.Tensor, gid: torch.Tensor,
                 n_groups: int) -> torch.Tensor:
    """[N, C] rows summed into [G, C] by gid — the segment form of the
    reference's dense one-hot einsum, as a [G, N] one-hot matmul. The old
    index_add of ~126k rows into 60 groups serialised on bf16 atomics:
    494 -> 83 ms per compiled step, max |diff| 1.8e-6 in fp32."""
    return F.one_hot(gid.long(), n_groups).to(src.dtype).t() @ src


class SpiderAttention(nn.Module):
    """The rewritten SpiderSolver attention (coarse + fine + T2P).

    surf_token_mode: "single" (2D reference — one mean surface token,
    weight_surf: dim -> 1) or "per_region" (3D reference — n_regions
    surface tokens, weight_surf: dim -> n_regions, token-axis LayerNorm).
    """

    def __init__(self, dim: int, heads: int, dropout: float, onion_num: int,
                 n_clusters: int, surf_token_mode: str = "per_region"):
        super().__init__()
        assert dim % heads == 0
        assert surf_token_mode in {"single", "per_region"}
        self.dim = dim
        self.heads = heads
        self.dim_head = dim // heads
        self.onion_num = onion_num
        self.n_clusters = n_clusters
        self.surf_token_mode = surf_token_mode
        self.n_vol_tokens = onion_num * n_clusters
        self.n_surf_tokens = 1 if surf_token_mode == "single" else n_clusters

        self.in_project_x = nn.Linear(dim, dim)
        self.in_project_fx = nn.Linear(dim, dim)

        self.weight_surf = nn.Linear(dim, self.n_surf_tokens)
        self.weight_velo_car = nn.Linear(dim, self.n_vol_tokens)
        self.weight_velo_velo = nn.Linear(dim, self.n_vol_tokens)
        for lin in (self.weight_surf, self.weight_velo_velo,
                    self.weight_velo_car):
            torch.nn.init.orthogonal_(lin.weight)

        if surf_token_mode == "per_region":
            self.norm = nn.LayerNorm(self.n_vol_tokens + self.n_surf_tokens)

        self.dropout_p = float(dropout)
        self.to_q_first_lap = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_k_first_lap = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_v_first_lap = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_q = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_k = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_v = nn.Linear(self.dim_head, self.dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def _heads(self, t: torch.Tensor) -> torch.Tensor:
        B, T, _ = t.shape
        return t.view(B, T, self.heads, self.dim_head).permute(0, 2, 1, 3)

    def _unheads(self, t: torch.Tensor) -> torch.Tensor:
        B, H, T, D = t.shape
        return t.permute(0, 2, 1, 3).reshape(B, T, H * D)

    def forward(self, x: torch.Tensor, n_vol: int, vol_gid: torch.Tensor,
                inner_idx: torch.Tensor,
                surf_region_id: torch.Tensor | None) -> torch.Tensor:
        """x [1, Nv+Ns, C] with volume rows FIRST (static layout);
        vol_gid [Nv] long = shell * n_clusters + cluster of each volume row;
        inner_idx [n_inner] long = innermost volume rows (indices into the
        volume range); surf_region_id [Ns] long (per_region mode only)."""
        B, N, C = x.shape
        assert B == 1, "SpiderAttention is microbatch-1 by protocol"
        n_surf = N - n_vol

        fx_mid = self.in_project_fx(x)                     # [1, N, C]
        vol_fx = fx_mid[:, :n_vol]
        surf_fx = fx_mid[:, n_vol:]

        # ---- coarse attention: innermost volume rows + ALL surface rows --
        inner_fx = vol_fx.index_select(1, inner_idx)       # [1, n_inner, C]
        lap = torch.cat([inner_fx, surf_fx], dim=1)
        lap_h = self._heads(lap)
        q1 = self.to_q_first_lap(lap_h)
        k1 = self.to_k_first_lap(lap_h)
        v1 = self.to_v_first_lap(lap_h)
        out1 = F.scaled_dot_product_attention(
            q1, k1, v1,
            dropout_p=self.dropout_p if self.training else 0.0)
        out1 = self._unheads(out1)
        n_inner = inner_fx.shape[1]
        coarse_inner = out1[:, :n_inner]                   # [1, n_inner, C]
        coarse_surf = out1[:, n_inner:]                    # [1, Ns, C]

        # ---- fine attention: sector tokens ------------------------------
        # volume sector tokens: sum(members) / N_vol_total — the reference's
        # dense zero-masked mean over the FULL row axis, in segment form.
        vol_tok = _segment_sum(vol_fx[0], vol_gid,
                               self.n_vol_tokens) / float(n_vol)
        if self.surf_token_mode == "single":
            surf_tok = surf_fx[0].mean(dim=0, keepdim=True)   # true mean
        else:
            surf_tok = _segment_sum(surf_fx[0], surf_region_id,
                                    self.n_surf_tokens) / float(n_surf)
        tokens = torch.cat([surf_tok, vol_tok], dim=0).unsqueeze(0)
        if self.surf_token_mode == "per_region":
            # reference 3D: LayerNorm ACROSS the token axis, per channel
            tokens = self.norm(tokens.permute(0, 2, 1)).permute(0, 2, 1)
        tok_h = self._heads(tokens)
        q2 = self.to_q(tok_h)
        k2 = self.to_k(tok_h)
        v2 = self.to_v(tok_h)
        out_tok = F.scaled_dot_product_attention(
            q2, k2, v2,
            dropout_p=self.dropout_p if self.training else 0.0)
        out_tok = self._unheads(out_tok)[0]                # [T, C]
        tok_surf = out_tok[:self.n_surf_tokens]            # [Ts, C]
        tok_vol = out_tok[self.n_surf_tokens:]             # [G, C]

        # ---- T2P ---------------------------------------------------------
        x_mid = self.in_project_x(x)
        vol_x = x_mid[:, :n_vol]
        surf_x = x_mid[:, n_vol:]

        feature_surf = self.weight_surf(surf_x) @ tok_surf.unsqueeze(0)
        feature_velo = self.weight_velo_velo(vol_x) @ tok_vol.unsqueeze(0)
        feature_velo_car = self.weight_velo_car(surf_x) @ tok_vol.unsqueeze(0)

        feature_surf = (feature_surf + feature_velo_car) / 2.0 + coarse_surf
        feature_velo = feature_velo.index_add(1, inner_idx, coarse_inner)

        return self.to_out(torch.cat([feature_velo, feature_surf], dim=1))


class SpiderBlock(nn.Module):
    def __init__(self, num_heads: int, hidden_dim: int, dropout: float,
                 mlp_ratio: int, onion_num: int, n_clusters: int,
                 surf_token_mode: str):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = SpiderAttention(hidden_dim, num_heads, dropout,
                                    onion_num, n_clusters, surf_token_mode)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = Mlp(hidden_dim, hidden_dim * mlp_ratio, hidden_dim)

    def forward(self, fx, n_vol, vol_gid, inner_idx, surf_region_id):
        fx = self.Attn(self.ln_1(fx), n_vol, vol_gid, inner_idx,
                       surf_region_id) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        return fx


class SpiderSolver(nn.Module):
    """Shared backbone + volume/surface heads. Rows are laid out
    [volume (n_vol); surface (n_surf)] — the heads split statically."""

    def __init__(self, in_dim: int, n_hidden: int = 256, n_layers: int = 8,
                 n_heads: int = 8, mlp_ratio: int = 2, dropout: float = 0.0,
                 onion_num: int = 10, n_clusters: int = 6,
                 surf_token_mode: str = "per_region",
                 out_dim_vol: int = 5, out_dim_surf: int = 1,
                 use_checkpoint: bool = False):
        super().__init__()
        self.__name__ = "SpiderSolver"
        self.use_checkpoint = bool(use_checkpoint)
        self.onion_num = onion_num
        self.n_clusters = n_clusters
        self.preprocess = Mlp(in_dim, n_hidden * 2, n_hidden)
        self.blocks = nn.ModuleList([
            SpiderBlock(n_heads, n_hidden, dropout, mlp_ratio, onion_num,
                        n_clusters, surf_token_mode)
            for _ in range(n_layers)])
        self.ln_out = nn.LayerNorm(n_hidden)
        self.vol_head = nn.Linear(n_hidden, out_dim_vol)
        self.surf_head = nn.Linear(n_hidden, out_dim_surf)
        self.initialize_weights()
        self.placeholder = nn.Parameter(
            (1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float))

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        # NOTE: initialize_weights runs BEFORE the orthogonal T2P inits would
        # be overwritten — so re-apply them here, matching the reference
        # (which constructs modules, orthogonal-inits, then trunc_normal
        # never touches them because its init happens in __init__ order).
        if isinstance(m, SpiderAttention):
            for lin in (m.weight_surf, m.weight_velo_velo, m.weight_velo_car):
                torch.nn.init.orthogonal_(lin.weight)
                if lin.bias is not None:
                    nn.init.constant_(lin.bias, 0)

    def forward(self, x: torch.Tensor, n_vol: int, vol_gid: torch.Tensor,
                inner_idx: torch.Tensor,
                surf_region_id: torch.Tensor | None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        fx = self.preprocess(x) + self.placeholder[None, None, :]
        for block in self.blocks:
            if self.use_checkpoint and torch.is_grad_enabled():
                fx = checkpoint(block, fx, n_vol, vol_gid, inner_idx,
                                surf_region_id, use_reentrant=False)
            else:
                fx = block(fx, n_vol, vol_gid, inner_idx, surf_region_id)
        h = self.ln_out(fx)
        return self.vol_head(h[:, :n_vol]), self.surf_head(h[:, n_vol:])


def build_spider_model(scfg, out_dim_vol: int, surf_token_mode: str,
                       in_dim: int | None = None,
                       use_checkpoint: bool = False) -> SpiderSolver:
    return SpiderSolver(
        in_dim=int(in_dim if in_dim is not None else scfg.in_dim),
        n_hidden=int(scfg.n_hidden), n_layers=int(scfg.n_layers),
        n_heads=int(scfg.n_heads), mlp_ratio=int(scfg.mlp_ratio),
        dropout=float(scfg.dropout), onion_num=int(scfg.onion_num),
        n_clusters=int(getattr(scfg, "n_regions", None)
                       or getattr(scfg, "n_clusters")),
        surf_token_mode=surf_token_mode, out_dim_vol=int(out_dim_vol),
        out_dim_surf=1, use_checkpoint=use_checkpoint)
