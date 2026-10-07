"""Local decoder: cross-attention of each query to its k nearest latent
points.

Each query point gets a self-embedding (Fourier features of its position,
signed distance and wall direction with separate xy / z frequency ladders,
plus ray support, region codes and the local latent spacing), routed through
a volume or a surface first layer. Per neighbour j, a pair embedding encodes
the normalized offset to the latent point. A decoder stage is multi-head
attention over the k neighbours:

  score_j = <RMSNorm(W_Q (h + p_j)), RMSNorm(W_K t_j)> / sqrt(d)
            - lambda_h * log(d_j / dbar)
  out     = sum_j softmax_j(score) (W_V t_j + gamma * p_j)

followed by a SwiGLU feed-forward layer. lambda_h starts at 1, so the
distance term starts as an inverse-distance prior. With two stages the first
reads the encoder tokens and the second the processor tokens; the UrbanWind
setting uses the processor stage only. Optional learned registers are read
through a separate softmax and added with a gate. Separate output heads give
the five volume channels and the surface pressure.

Neighbours form a tensor axis [B, N, K, ...]: one gather per table through
F.embedding (whose backward is a segmented reduction instead of an atomic
scatter), scores accumulated in fp32, and the value aggregate split as
sum_j a_j v_j + gamma * sum_j a_j p_j so v + gamma * p is never
materialized. tests/test_decoder_equivalence.py checks this form against a
per-neighbour loop.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import RMSNorm


def _init_linear(module: nn.Linear):
    nn.init.trunc_normal_(module.weight, std=0.02)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class FourierEncoding(nn.Module):
    """Query self-descriptor: Fourier features with separate xy (K = 8) and
    z (K = 6) ladders, bands 2^k * pi.

    xy branch: [x, y, dir_x, dir_y]; with xy_fourier_abs=False the absolute
    x and y stay raw channels and only the direction components are banded.
    z branch:  [log(max(z_norm * 550, 0.1)), dir_z, sdf_norm].
    The signed distance and the wall direction are independent channels
    (surface rows carry sdf = 0 and their normal as direction).
    """

    def __init__(self, freqs_xy: int = 8, freqs_z: int = 6,
                 xy_fourier_abs: bool = True):
        super().__init__()
        self.xy_fourier_abs = bool(xy_fourier_abs)
        self.register_buffer("freq_xy", (2.0 ** torch.arange(freqs_xy).float()) * math.pi)
        self.register_buffer("freq_z", (2.0 ** torch.arange(freqs_z).float()) * math.pi)
        # banded channels contribute 1 + 2K each; x and y drop to 1 when the
        # ladder is withheld from them
        n_banded_xy = 4 if self.xy_fourier_abs else 2
        self.out_dim = (n_banded_xy * (1 + 2 * freqs_xy)
                        + (4 - n_banded_xy)
                        + 3 * (1 + 2 * freqs_z))

    def forward(self, pos_norm: torch.Tensor, sdf_norm: torch.Tensor,
                sdf_dir: torch.Tensor) -> torch.Tensor:
        x = pos_norm[..., 0:1].float()
        y = pos_norm[..., 1:2].float()
        z = pos_norm[..., 2:3].float()
        s = sdf_norm[..., None].float()   # independent scalar channel
        d = sdf_dir.float()               # raw unit direction, NOT scaled by s
        z_log = torch.log(torch.clamp(z * 550.0, min=0.1))
        z_in = torch.cat([z_log, d[..., 2:3], s], dim=-1)
        if self.xy_fourier_abs:
            xy_enc = _encode(torch.cat([x, y, d[..., 0:1], d[..., 1:2]],
                                       dim=-1), self.freq_xy)
        else:
            xy_enc = torch.cat(
                [x, y, _encode(torch.cat([d[..., 0:1], d[..., 1:2]], dim=-1),
                               self.freq_xy)], dim=-1)
        return torch.cat([xy_enc, _encode(z_in, self.freq_z)], dim=-1)


class FourierEncodingXYZ(nn.Module):
    """Fourier features of the query-to-latent offset: the xy ladder on x
    and y, the z ladder on z."""

    def __init__(self, freqs_xy: int = 8, freqs_z: int = 6):
        super().__init__()
        self.register_buffer("freq_xy", (2.0 ** torch.arange(freqs_xy).float()) * math.pi)
        self.register_buffer("freq_z", (2.0 ** torch.arange(freqs_z).float()) * math.pi)
        self.out_dim = 2 * (1 + 2 * freqs_xy) + 1 * (1 + 2 * freqs_z)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        v = v.float()
        return torch.cat([_encode(v[..., 0:2], self.freq_xy),
                          _encode(v[..., 2:3], self.freq_z)], dim=-1)


def _encode(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    theta = x[..., None] * freqs  # [..., D, F]
    sc = torch.cat([torch.sin(theta), torch.cos(theta)], dim=-1)
    return torch.cat([x, sc.reshape(*x.shape[:-1], -1)], dim=-1)


class RegionEmbedding(nn.Module):
    """Region codes -> 12 dims: point type (volume / surface), sub-category
    (volume: near building / near ground / other; surface: windward /
    leeward / roof / side) and edge detail (surface edge kind x wind
    group)."""

    def __init__(self):
        super().__init__()
        self.emb1 = nn.Embedding(2, 4)    # point_type: volume / surface
        self.emb2 = nn.Embedding(5, 4)    # sub_category
        self.emb3 = nn.Embedding(10, 4)   # edge_detail

    def forward(self, d1: torch.Tensor, d2: torch.Tensor, d3: torch.Tensor
                ) -> torch.Tensor:
        return torch.cat([self.emb1(d1.long()), self.emb2(d2.long()),
                          self.emb3(d3.long())], dim=-1)


class _DecoderFFN(nn.Module):
    """SwiGLU feed-forward layer of a decoder stage (bias-free, no
    dropout)."""

    def __init__(self, dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.gate_up = nn.Linear(dim, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.gate_up.weight, std=0.02)
        nn.init.trunc_normal_(self.down.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.drop(self.down(F.silu(gate) * up))


class TwoStageCrossAttentionDecoder(nn.Module):
    def __init__(self, latent_dim: int = 384, d_h: int = 384, idw_k: int = 16,
                 fourier_freqs_xy: int = 8, fourier_freqs_z: int = 6,
                 n_heads: int = 6, n_register: int = 8,
                 use_register_attn: bool = True, use_stage1: bool = True,
                 xy_fourier_abs: bool = True):
        super().__init__()
        assert d_h % n_heads == 0, "d_h must divide evenly into n_heads"
        self.idw_k = idw_k
        self.d_h = d_h
        self.n_heads = n_heads
        self.head_dim = d_h // n_heads
        self.n_register = n_register
        self.use_register_attn = bool(use_register_attn)
        # use_stage1=False: no stage reading the encoder tokens (its
        # parameters are not constructed); the processor stage then runs
        # directly on the query self-embedding
        self.use_stage1 = bool(use_stage1)

        self.fourier = FourierEncoding(fourier_freqs_xy, fourier_freqs_z,
                                       xy_fourier_abs=xy_fourier_abs)
        # the pair encoder always bands its input: an offset carries no
        # absolute position
        self.fourier_pair = FourierEncodingXYZ(fourier_freqs_xy, fourier_freqs_z)
        self.region_emb = RegionEmbedding()
        # pair: Fourier(delta/dbar) + unit dir(3) + log(d/dbar)(1) + log_dbar(1)
        d_pair_raw = self.fourier_pair.out_dim + 3 + 1 + 1
        # self: Fourier(pos + sdf + dir) + support(6) + region(12) + log_dbar(1)
        d_self_raw = self.fourier.out_dim + 6 + 12 + 1

        self.pair_proj = nn.Linear(d_pair_raw, d_h)
        # separate first layers for volume and surface queries; region_dim1
        # selects between their outputs with an exact 0/1 mask (static
        # shapes)
        self.self_proj1_vol = nn.Linear(d_self_raw, d_h)
        self.self_proj1_surf = nn.Linear(d_self_raw, d_h)
        self.self_proj2 = nn.Linear(d_h, d_h)
        # query-type embedding added to the self-embedding
        self.type_emb = nn.Embedding(2, d_h)
        nn.init.trunc_normal_(self.type_emb.weight, std=0.02)

        # key / value projections of the latent tokens, computed once per
        # forward over the L tokens
        if self.use_stage1:
            self.k_enc = nn.Linear(latent_dim, d_h)
            self.v_enc = nn.Linear(latent_dim, d_h)
            self.q_proj1 = nn.Linear(d_h, d_h)
        self.k_vit = nn.Linear(latent_dim, d_h)
        self.v_vit = nn.Linear(latent_dim, d_h)
        self.q_proj2 = nn.Linear(d_h, d_h)

        # pre-norm per stage: RMSNorm on the query state and on the token
        # source, head-wise RMSNorm on Q and K
        if self.use_stage1:
            self.h_norm1 = RMSNorm(d_h)
            self.src_norm1 = RMSNorm(latent_dim)
            self.q_head_norm1 = RMSNorm(self.head_dim)
            self.k_head_norm1 = RMSNorm(self.head_dim)
            self.ffn_norm1 = RMSNorm(d_h)
        self.h_norm2 = RMSNorm(d_h)
        self.src_norm2 = RMSNorm(latent_dim)
        self.q_head_norm2 = RMSNorm(self.head_dim)
        self.k_head_norm2 = RMSNorm(self.head_dim)
        self.ffn_norm2 = RMSNorm(d_h)

        # zero-initialized output projections: at initialization the
        # attention adds nothing and the prediction is query-local
        self.out_proj2 = nn.Linear(d_h, d_h)
        zero_init = [self.out_proj2]
        if self.use_stage1:
            self.out_proj1 = nn.Linear(d_h, d_h)
            zero_init.append(self.out_proj1)
        for layer in zero_init:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        # learnable distance-prior exponent per head, initialized to 1
        # (an inverse-distance prior)
        if self.use_stage1:
            self.lambda1 = nn.Parameter(torch.ones(n_heads))
        self.lambda2 = nn.Parameter(torch.ones(n_heads))
        # the pair embedding also enters the values: v_j = V t_j + gamma p_j
        self.gamma2 = nn.Parameter(torch.empty(d_h))
        nn.init.trunc_normal_(self.gamma2, std=0.02)
        if self.use_stage1:
            self.gamma1 = nn.Parameter(torch.empty(d_h))
            nn.init.trunc_normal_(self.gamma1, std=0.02)
        # register path (last stage only): a separate softmax over the
        # registers, added with a small gate; built only when it can run
        self._has_reg_gate = n_register > 0 and self.use_register_attn
        if self._has_reg_gate:
            self.gate_reg = nn.Parameter(torch.full((n_heads,), 1e-2))

        # SwiGLU feed-forward layer after each stage (4x expansion)
        self.ffn2 = _DecoderFFN(d_h, 4 * d_h)
        if self.use_stage1:
            self.ffn1 = _DecoderFFN(d_h, 4 * d_h)

        self.head = nn.Linear(d_h, 256)
        self.volume_head = nn.Linear(256, 5)
        self.surface_head = nn.Linear(256, 1)
        to_init = [self.pair_proj, self.self_proj1_vol, self.self_proj1_surf,
                   self.self_proj2, self.q_proj2, self.head,
                   self.volume_head, self.surface_head,
                   self.k_vit, self.v_vit]
        if self.use_stage1:
            to_init += [self.q_proj1, self.k_enc, self.v_enc]
        for layer in to_init:
            _init_linear(layer)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[..., d_h] -> [..., n_heads, head_dim]"""
        return x.view(*x.shape[:-1], self.n_heads, self.head_dim)

    @staticmethod
    def _gather_neighbours(table: torch.Tensor, flat_idx: torch.Tensor
                           ) -> torch.Tensor:
        """[B, L, *tail] -> [B, N, K, *tail]: row flat_idx[b, n, k] of the
        batch-flattened table (flat_idx already carries the b * L offset).
        F.embedding instead of advanced indexing: same forward, but its
        backward is a sorted segmented reduction rather than an atomic
        scatter-add over ~N K / L queries per row."""
        tail = table.shape[2:]
        flat = table.reshape(table.shape[0] * table.shape[1], -1)
        return F.embedding(flat_idx, flat).view(*flat_idx.shape, *tail)

    def forward(
        self,
        encoder_out: torch.Tensor,        # [B, L, D] encoder output (pre-ViT)
        vit_features: torch.Tensor,       # [B, L+R, D] ViT output, registers INCLUDED
        query_pos_norm: torch.Tensor,     # [B, N, 3]
        query_sdf_norm: torch.Tensor,     # [B, N]
        query_sdf_dir: torch.Tensor,      # [B, N, 3]
        query_support_norm: torch.Tensor,  # [B, N, 6]
        region_dim1: torch.Tensor,        # [B, N] int (0=volume, 1=surface)
        region_dim2: torch.Tensor,
        region_dim3: torch.Tensor,
        idw_indices: torch.Tensor,        # [B, N, K] int
        leaf_pos_norm: torch.Tensor,      # [B, L, 3]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        idw_indices = idw_indices.long()
        B, N, K = idw_indices.shape
        L = encoder_out.shape[1]
        dtype = encoder_out.dtype
        # one flat row index for the [B, N, K] neighbourhood, shared by all
        # gathers (the b * L offset folds the batch axis in)
        if B > 1:
            offset = torch.arange(B, device=idw_indices.device)[:, None, None] * L
            flat_idx = idw_indices + offset
        else:
            flat_idx = idw_indices
        R = self.n_register
        hd = self.head_dim
        H = self.n_heads

        vit_cells = vit_features[:, :-R, :] if R > 0 else vit_features
        vit_regs = vit_features[:, -R:, :] if R > 0 else None

        # --- token keys / values, once over the L latent tokens ---------------
        if self.use_stage1:
            enc_src = self.src_norm1(encoder_out)
            k1 = self.k_enc(enc_src); v1 = self.v_enc(enc_src)
        vit_src = self.src_norm2(vit_cells)
        k2 = self.k_vit(vit_src); v2 = self.v_vit(vit_src)
        if R > 0 and self.use_register_attn:
            reg_src = self.src_norm2(vit_regs)   # SAME norm module as cells
            k_reg = self.k_head_norm2(self._split_heads(self.k_vit(reg_src)))  # [B,R,H,hd]
            v_reg = self.v_vit(reg_src)                                         # [B,R,d_h]
        else:
            k_reg = v_reg = None

        # --- per-query geometry in fp32, one gather over the K neighbours --
        qpg = query_pos_norm.float()
        cpg = leaf_pos_norm.float()
        cj = self._gather_neighbours(cpg, flat_idx)                # [B,N,K,3]
        delta = qpg.unsqueeze(2) - cj                             # [B,N,K,3]
        d = delta.norm(dim=-1, keepdim=True).clamp(min=1e-6)      # [B,N,K,1]
        dbar = d.mean(dim=2, keepdim=True).clamp(min=1e-6)        # [B,N,1,1]
        log_dbar = torch.log(dbar)                                # [B,N,1,1]
        # shared by the pair embedding and every stage's distance prior
        log_ratio = torch.log(d / dbar)                           # [B,N,K,1] fp32

        # --- query self-embedding --------------------------------------------
        fourier = self.fourier(query_pos_norm, query_sdf_norm, query_sdf_dir)
        region = self.region_emb(region_dim1, region_dim2, region_dim3)
        q_self = torch.cat([
            fourier.to(dtype),
            query_support_norm.to(dtype),
            region.to(dtype),
            log_dbar[:, :, 0, :].to(dtype),                       # [B,N,1]
        ], dim=-1)
        mask = (region_dim1 == 0).to(dtype)[..., None]   # 1 volume, 0 surface
        h_pre = mask * F.silu(self.self_proj1_vol(q_self)) \
            + (1.0 - mask) * F.silu(self.self_proj1_surf(q_self))
        h0 = self.self_proj2(h_pre)
        h0 = h0 + self.type_emb(region_dim1.long())

        # --- pair embedding, shared by every stage ----------------------------
        p_raw = torch.cat([
            self.fourier_pair(delta / dbar).to(dtype),
            (delta / d).to(dtype),
            log_ratio.to(dtype),
            log_dbar.expand(B, N, K, 1).to(dtype),
        ], dim=-1)                                                # [B,N,K,d_pair_raw]
        p = F.silu(self.pair_proj(p_raw))                         # [B,N,K,d_h]

        # --- cross-attention stages (sequential) --------------------------
        h_cur = h0
        stages = []
        if self.use_stage1:
            stages.append(
                (k1, v1, None, None, self.q_proj1, self.h_norm1, self.out_proj1,
                 self.lambda1, self.gamma1, self.ffn1, self.ffn_norm1,
                 self.q_head_norm1, self.k_head_norm1))
        stages.append(
            (k2, v2, k_reg, v_reg, self.q_proj2, self.h_norm2, self.out_proj2,
             self.lambda2, self.gamma2, self.ffn2, self.ffn_norm2,
             self.q_head_norm2, self.k_head_norm2))
        for (k_tab, v_tab, kr, vr, q_proj, h_norm, out_proj, lam, gamma,
             ffn, ffn_norm, q_hn, k_hn) in stages:
            h_normed = h_norm(h_cur)
            # the head-wise K norm acts row by row, so it commutes with the
            # gather: normalize the L token rows once, then gather
            k_tab_normed = k_hn(self._split_heads(k_tab))
            # one q GEMM over the K axis; one gather per table
            q = q_hn(self._split_heads(q_proj(h_normed.unsqueeze(2) + p)))  # [B,N,K,H,hd]
            kj = self._gather_neighbours(k_tab_normed, flat_idx)             # [B,N,K,H,hd]
            vj = self._split_heads(
                self._gather_neighbours(v_tab, flat_idx))                    # [B,N,K,H,hd]
            pj = self._split_heads(p)                                       # view
            # multiply in the ambient dtype, accumulate in fp32
            score = (q * kj).sum(dim=-1, dtype=torch.float32) / (hd ** 0.5)  # [B,N,K,H]
            score = score - lam.view(1, 1, 1, H) * log_ratio
            alpha_local = F.softmax(score, dim=2).to(dtype)        # softmax over K
            # sum_j a_j (v_j + gamma p_j) = sum_j a_j v_j + gamma sum_j a_j p_j
            agg_local = torch.einsum('bnkh,bnkhd->bnhd', alpha_local, vj) \
                + gamma.view(H, hd) * torch.einsum('bnkh,bnkhd->bnhd',
                                                   alpha_local, pj)

            if kr is not None:
                # separate softmax over the registers, added with a gate;
                # registers have no position, so no pair embedding
                q_reg = q_hn(self._split_heads(q_proj(h_normed)))
                reg_scores = torch.einsum(
                    'bnhd,brhd->bnrh', q_reg.float(), kr.float()) / (hd ** 0.5)
                alpha_reg = F.softmax(reg_scores, dim=2).to(dtype)     # softmax over R
                agg_reg = torch.einsum('bnrh,brhd->bnhd', alpha_reg,
                                       self._split_heads(vr))
                agg = agg_local + self.gate_reg[None, None, :, None] * agg_reg
            else:
                agg = agg_local

            h_cur = h_cur + out_proj(agg.reshape(B, N, self.d_h))
            h_cur = h_cur + ffn(ffn_norm(h_cur))   # stage-specific Pre-Norm FFN

        h = F.silu(self.head(h_cur))
        return self.volume_head(h), self.surface_head(h)
