"""Dense reference for the SpiderSolver attention — the equivalence ORACLE.

This is the ORIGINAL SpiderSolver computation (SpiderSolver-main, 2D and 3D
variants) expressed over the SAME parameter set as model/spider.py's
SpiderAttention: dense one-hot sector aggregation (zero-masked mean over the
FULL row axis, i.e. sum/​N_total — the reference's `torch.mean(axis=2)`
semantics), explicit q@k^T softmax attention (scale = dim_head^-0.5, the
default SDPA scale), boolean-style row handling. It is used only by
tests/test_gates.py to check the production rewrite's forward and backward;
training and evaluation never call it.
"""
from __future__ import annotations

import torch


def _explicit_attention(q, k, v, scale):
    dots = torch.matmul(q, k.transpose(-1, -2)) * scale
    attn = dots.softmax(dim=-1)
    return torch.matmul(attn, v)


def dense_spider_attention(attn, x: torch.Tensor, n_vol: int,
                           vol_gid: torch.Tensor, inner_idx: torch.Tensor,
                           surf_region_id: torch.Tensor | None
                           ) -> torch.Tensor:
    """The original dense computation, on spider.SpiderAttention's weights."""
    B, N, C = x.shape
    assert B == 1
    H, D = attn.heads, attn.dim_head
    scale = D ** -0.5
    n_surf = N - n_vol
    G = attn.n_vol_tokens

    def heads(t):
        b, T, _ = t.shape
        return t.view(b, T, H, D).permute(0, 2, 1, 3)

    def unheads(t):
        b, h, T, d = t.shape
        return t.permute(0, 2, 1, 3).reshape(b, T, h * d)

    fx_mid = attn.in_project_fx(x)
    vol_fx = fx_mid[:, :n_vol]
    surf_fx = fx_mid[:, n_vol:]

    # ---- coarse: dense gather + explicit attention -----------------------
    inner_onehot = torch.zeros(inner_idx.shape[0], n_vol, dtype=x.dtype,
                               device=x.device)
    inner_onehot[torch.arange(inner_idx.shape[0]), inner_idx] = 1.0
    inner_fx = torch.matmul(inner_onehot, vol_fx[0]).unsqueeze(0)
    lap = torch.cat([inner_fx, surf_fx], dim=1)
    lap_h = heads(lap)
    out1 = _explicit_attention(attn.to_q_first_lap(lap_h),
                               attn.to_k_first_lap(lap_h),
                               attn.to_v_first_lap(lap_h), scale)
    out1 = unheads(out1)
    n_inner = inner_fx.shape[1]
    coarse_inner = out1[:, :n_inner]
    coarse_surf = out1[:, n_inner:]

    # ---- fine: DENSE one-hot sector tokens (sum / N_total) ---------------
    onehot_vol = torch.zeros(G, n_vol, dtype=x.dtype, device=x.device)
    onehot_vol[vol_gid, torch.arange(n_vol)] = 1.0
    vol_tok = torch.matmul(onehot_vol, vol_fx[0]) / float(n_vol)
    if attn.surf_token_mode == "single":
        surf_tok = surf_fx[0].mean(dim=0, keepdim=True)
    else:
        onehot_surf = torch.zeros(attn.n_surf_tokens, n_surf, dtype=x.dtype,
                                  device=x.device)
        onehot_surf[surf_region_id, torch.arange(n_surf)] = 1.0
        surf_tok = torch.matmul(onehot_surf, surf_fx[0]) / float(n_surf)
    tokens = torch.cat([surf_tok, vol_tok], dim=0).unsqueeze(0)
    if attn.surf_token_mode == "per_region":
        tokens = attn.norm(tokens.permute(0, 2, 1)).permute(0, 2, 1)
    tok_h = heads(tokens)
    out_tok = _explicit_attention(attn.to_q(tok_h), attn.to_k(tok_h),
                                  attn.to_v(tok_h), scale)
    out_tok = unheads(out_tok)[0]
    tok_surf = out_tok[:attn.n_surf_tokens]
    tok_vol = out_tok[attn.n_surf_tokens:]

    # ---- T2P --------------------------------------------------------------
    x_mid = attn.in_project_x(x)
    vol_x = x_mid[:, :n_vol]
    surf_x = x_mid[:, n_vol:]
    feature_surf = attn.weight_surf(surf_x) @ tok_surf.unsqueeze(0)
    feature_velo = attn.weight_velo_velo(vol_x) @ tok_vol.unsqueeze(0)
    feature_velo_car = attn.weight_velo_car(surf_x) @ tok_vol.unsqueeze(0)
    feature_surf = (feature_surf + feature_velo_car) / 2.0 + coarse_surf
    scatter = torch.matmul(inner_onehot.transpose(0, 1),
                           coarse_inner[0]).unsqueeze(0)
    feature_velo = feature_velo + scatter
    return attn.to_out(torch.cat([feature_velo, feature_surf], dim=1))
