"""Equivalence test for the vectorized decoder.

`_reference_forward` is a plain per-neighbour loop implementation of the
same math (K-loop gathers, fp32 score operands, chained-add value
aggregate), driven by the same module instance and weights. With autocast
off the two differ only in floating-point reduction order:

  fp32:  max|new - ref| / max|ref| < 1e-5   (CPU, always run)
  bf16:  < 5e-3 under autocast on CUDA      (skipped without CUDA)

Plus a gradient check: every decoder parameter receives a finite gradient.
K / d_h / H / R are the model's; N and L are shrunk so this runs on CPU in
seconds. B=2 exercises the batched gather path.
"""
import math

import torch
import torch.nn.functional as F

from src.model.decoder import TwoStageCrossAttentionDecoder

B, N, L, K = 2, 4096, 512, 16
D_LATENT, D_H, H, R = 384, 384, 6, 8
HD = D_H // H


def _reference_forward(dec, encoder_out, vit_features, query_pos_norm,
                       query_sdf_norm, query_sdf_dir, query_support_norm,
                       region_dim1, region_dim2, region_dim3, idw_indices,
                       leaf_pos_norm):
    """Pre-rewrite forward, copied verbatim (loop form). Do not modernize."""
    idw_indices = idw_indices.long()
    B_, N_, K_ = idw_indices.shape
    dtype = encoder_out.dtype
    batch_idx = torch.arange(B_, device=idw_indices.device)[:, None]
    R_ = dec.n_register
    hd = dec.head_dim

    vit_cells = vit_features[:, :-R_, :] if R_ > 0 else vit_features
    vit_regs = vit_features[:, -R_:, :] if R_ > 0 else None

    enc_src = dec.src_norm1(encoder_out)
    k1 = dec.k_enc(enc_src); v1 = dec.v_enc(enc_src)
    vit_src = dec.src_norm2(vit_cells)
    k2 = dec.k_vit(vit_src); v2 = dec.v_vit(vit_src)
    if R_ > 0 and dec.use_register_attn:
        reg_src = dec.src_norm2(vit_regs)
        k_reg = dec.k_head_norm2(dec._split_heads(dec.k_vit(reg_src)))
        v_reg = dec.v_vit(reg_src)
    else:
        k_reg = v_reg = None

    qpg = query_pos_norm.float()
    cpg = leaf_pos_norm.float()
    deltas, dists = [], []
    for j in range(K_):
        ij = idw_indices[:, :, j]
        cj = cpg[batch_idx, ij]
        delta = qpg - cj
        d = delta.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        deltas.append(delta); dists.append(d)
    dbar = torch.stack(dists, dim=2).mean(dim=2).clamp(min=1e-6)
    log_dbar = torch.log(dbar)

    fourier = dec.fourier(query_pos_norm, query_sdf_norm, query_sdf_dir)
    region = dec.region_emb(region_dim1, region_dim2, region_dim3)
    q_self = torch.cat([fourier.to(dtype), query_support_norm.to(dtype),
                        region.to(dtype), log_dbar.to(dtype)], dim=-1)
    mask = (region_dim1 == 0).to(dtype)[..., None]
    h_pre = mask * F.silu(dec.self_proj1_vol(q_self)) \
        + (1.0 - mask) * F.silu(dec.self_proj1_surf(q_self))
    h0 = dec.self_proj2(h_pre)
    h0 = h0 + dec.type_emb(region_dim1.long())

    p_list = []
    for j in range(K_):
        delta, d = deltas[j], dists[j]
        p_raw = torch.cat([
            dec.fourier_pair(delta / dbar).to(dtype),
            (delta / d).to(dtype),
            torch.log(d / dbar).to(dtype),
            log_dbar.to(dtype),
        ], dim=-1)
        p_list.append(F.silu(dec.pair_proj(p_raw)))

    h_cur = h0
    stages = [
        (k1, v1, None, None, dec.q_proj1, dec.h_norm1, dec.out_proj1,
         dec.lambda1, dec.gamma1, dec.ffn1, dec.ffn_norm1,
         dec.q_head_norm1, dec.k_head_norm1),
        (k2, v2, k_reg, v_reg, dec.q_proj2, dec.h_norm2, dec.out_proj2,
         dec.lambda2, dec.gamma2, dec.ffn2, dec.ffn_norm2,
         dec.q_head_norm2, dec.k_head_norm2),
    ]
    for (k_tab, v_tab, kr, vr, q_proj, h_norm, out_proj, lam, gamma,
         ffn, ffn_norm, q_hn, k_hn) in stages:
        h_normed = h_norm(h_cur)
        k_tab_normed = k_hn(dec._split_heads(k_tab))
        local_scores, values_h = [], []
        for j in range(K_):
            ij = idw_indices[:, :, j]
            q_ij = q_hn(dec._split_heads(q_proj(h_normed + p_list[j])))
            kj = k_tab_normed[batch_idx, ij]
            vj = v_tab[batch_idx, ij] + gamma * p_list[j]
            score = (q_ij.float() * kj.float()).sum(-1) / (hd ** 0.5)
            score = score - lam[None, None, :] * torch.log(dists[j] / dbar)
            local_scores.append(score)
            values_h.append(dec._split_heads(vj))
        local_scores = torch.stack(local_scores, dim=2)
        alpha_local = F.softmax(local_scores, dim=2).to(dtype)
        agg_local = sum(alpha_local[:, :, j, :, None] * values_h[j]
                        for j in range(K_))

        if kr is not None:
            q_reg = q_hn(dec._split_heads(q_proj(h_normed)))
            reg_scores = torch.einsum(
                'bnhd,brhd->bnrh', q_reg.float(), kr.float()) / (hd ** 0.5)
            alpha_reg = F.softmax(reg_scores, dim=2).to(dtype)
            agg_reg = torch.einsum('bnrh,brhd->bnhd', alpha_reg,
                                   dec._split_heads(vr))
            agg = agg_local + dec.gate_reg[None, None, :, None] * agg_reg
        else:
            agg = agg_local

        h_cur = h_cur + out_proj(agg.reshape(B_, N_, dec.d_h))
        h_cur = h_cur + ffn(ffn_norm(h_cur))

    h = F.silu(dec.head(h_cur))
    return dec.volume_head(h), dec.surface_head(h)


def _make_inputs(device=torch.device("cpu"), dtype=torch.float32):
    g = torch.Generator().manual_seed(7)

    def rand(*shape):
        return torch.randn(*shape, generator=g).to(device=device, dtype=dtype)

    pos = rand(B, N, 3) * 0.3
    sdf_dir = rand(B, N, 3)
    sdf_dir = sdf_dir / sdf_dir.norm(dim=-1, keepdim=True)
    inputs = {
        "encoder_out": rand(B, L, D_LATENT),
        "vit_features": rand(B, L + R, D_LATENT),
        "query_pos_norm": pos,
        "query_sdf_norm": rand(B, N).abs() * 0.1,
        "query_sdf_dir": sdf_dir,
        "query_support_norm": rand(B, N, 6),
        # both volume AND surface rows -> both self_proj1_* copies and both
        # type_emb rows are exercised
        "region_dim1": (torch.arange(N)[None, :].expand(B, N) % 2).to(device),
        "region_dim2": torch.randint(0, 5, (B, N), generator=g).to(device),
        "region_dim3": torch.randint(0, 10, (B, N), generator=g).to(device),
        "idw_indices": torch.randint(0, L, (B, N, K), generator=g).to(device),
        "leaf_pos_norm": rand(B, L, 3) * 0.3,
    }
    return inputs


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max()
                 / b.float().abs().max().clamp(min=1e-12))


def test_vectorized_forward_matches_loop_reference_fp32():
    torch.manual_seed(0)
    dec = TwoStageCrossAttentionDecoder(D_LATENT, D_H, K, n_heads=H,
                                        n_register=R, use_register_attn=True)
    dec.eval()
    inputs = _make_inputs()
    with torch.no_grad():
        ref_vol, ref_surf = _reference_forward(dec, **inputs)
        new_vol, new_surf = dec(**inputs)
    ev, es = _rel_err(new_vol, ref_vol), _rel_err(new_surf, ref_surf)
    assert ev < 1e-5, f"fp32 gate failed (volume): rel={ev:.3e}"
    assert es < 1e-5, f"fp32 gate failed (surface): rel={es:.3e}"
    assert math.isfinite(float(new_vol.sum())) and math.isfinite(float(new_surf.sum()))


def test_vectorized_forward_matches_loop_reference_no_register_path():
    """use_register_attn=False exercises the kr-is-None branch."""
    torch.manual_seed(1)
    dec = TwoStageCrossAttentionDecoder(D_LATENT, D_H, K, n_heads=H,
                                        n_register=R, use_register_attn=False)
    dec.eval()
    inputs = _make_inputs()
    with torch.no_grad():
        ref_vol, ref_surf = _reference_forward(dec, **inputs)
        new_vol, new_surf = dec(**inputs)
    assert _rel_err(new_vol, ref_vol) < 1e-5
    assert _rel_err(new_surf, ref_surf) < 1e-5


def test_vectorized_backward_all_params_get_finite_grads():
    torch.manual_seed(2)
    dec = TwoStageCrossAttentionDecoder(D_LATENT, D_H, K, n_heads=H,
                                        n_register=R, use_register_attn=True)
    dec.train()
    inputs = _make_inputs()
    pv, ps = dec(**inputs)
    loss = pv.float().pow(2).mean() + ps.float().pow(2).mean()
    loss.backward()
    missing = [n for n, p in dec.named_parameters() if p.grad is None]
    assert not missing, f"params without grad: {missing[:5]}"
    bad = [n for n, p in dec.named_parameters()
           if not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite grads: {bad[:5]}"


def test_vectorized_forward_matches_loop_reference_bf16_cuda():
    """bf16 gate under autocast — CUDA only; silently skipped without it."""
    if not torch.cuda.is_available():
        print("  [skip] bf16 gate needs CUDA")
        return
    device = torch.device("cuda:0")
    torch.manual_seed(3)
    dec = TwoStageCrossAttentionDecoder(D_LATENT, D_H, K, n_heads=H,
                                        n_register=R,
                                        use_register_attn=True).to(device)
    dec.eval()
    inputs = _make_inputs(device=device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ref_vol, ref_surf = _reference_forward(dec, **inputs)
        new_vol, new_surf = dec(**inputs)
    ev, es = _rel_err(new_vol, ref_vol), _rel_err(new_surf, ref_surf)
    assert ev < 5e-3, f"bf16 gate failed (volume): rel={ev:.3e}"
    assert es < 5e-3, f"bf16 gate failed (surface): rel={es:.3e}"
