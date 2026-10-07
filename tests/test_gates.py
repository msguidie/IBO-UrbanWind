"""Numerical equivalence tests: each optimized implementation (AB-UPT's
anchor K/V cache, chunked Transolver, Erwin's static restore index and
pooling, SpiderSolver's attention) against its reference at fp64 precision,
forward and backward where gradients exist.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.abupt import AnchoredBranchedUPT  # noqa: E402
from src.model.erwin.balltree import (build_tree, knn_edges,
                                      restore_index)
from src.model.erwin.erwin import grouped_mean, scatter_mean  # noqa: E402
from src.model.spider import SpiderAttention  # noqa: E402
from src.model.spider_reference import dense_spider_attention  # noqa: E402
from src.model.transolver import Transolver  # noqa: E402

TOL = 1e-10


def test_abupt_kv_cache():
    """Cached anchor-K/V decode == forward_with_queries (the oracle), with
    TRUE unequal branch lengths — anchors and queries both."""
    torch.manual_seed(0)
    m = AnchoredBranchedUPT(use_geometry_features=False, dim=24, num_heads=3,
                            shared_blocks="sscscs", num_surface_blocks=2,
                            num_volume_blocks=2, output_dim_surface=1,
                            output_dim_volume=5).double().eval()
    ns_a, nv_a, ns_q, nv_q = 16, 20, 7, 11
    sp = torch.rand(1, ns_a + ns_q, 3, dtype=torch.float64)
    vp = torch.rand(1, nv_a + nv_q, 3, dtype=torch.float64)
    with torch.no_grad():
        ps_ref, pv_ref = m.forward_with_queries(
            sp, vp, num_surface_anchors=ns_a, num_volume_anchors=nv_a)
        ps_a, pv_a, cache = m.encode_anchor_cache(sp[:, :ns_a], vp[:, :nv_a])
        ps_q, pv_q = m.decode_queries_cached(cache, sp[:, ns_a:],
                                             vp[:, nv_a:])
        ps_o, pv_o = m(sp[:, :ns_a], vp[:, :nv_a])
    assert (ps_ref[:, :ns_a] - ps_a).abs().max() < TOL
    assert (pv_ref[:, :nv_a] - pv_a).abs().max() < TOL
    assert (ps_ref[:, ns_a:] - ps_q).abs().max() < TOL
    assert (pv_ref[:, nv_a:] - pv_q).abs().max() < TOL
    assert (ps_o - ps_a).abs().max() == 0 and (pv_o - pv_a).abs().max() == 0
    print("gate abupt_kv_cache: PASS")


def test_transolver_chunks():
    """fused single-tensor == plain-block reference; multi-chunk global
    aggregation == single chunk (fwd and grads) — the exactness claim behind
    'chunked full-field inference IS the global model'."""
    torch.manual_seed(1)
    m = Transolver(in_dim=7, n_hidden=32, n_layers=3, n_heads=4, slice_num=8,
                   mlp_ratio=2, out_dim_vol=5).double().eval()
    x = torch.rand(1, 300, 7, dtype=torch.float64)
    with torch.no_grad():
        fx = m.embed(x)
        for b in m.blocks:
            fx = b(fx)
        h = m.ln_out(fx)
        pv_ref, ps_ref = m.vol_head(h), m.surf_head(h)
        pv1, ps1 = m(x)
        outs = m.forward_chunks([x[:, :101], x[:, 101:207], x[:, 207:]])
        pv_m = torch.cat([o[0] for o in outs], dim=1)
        ps_m = torch.cat([o[1] for o in outs], dim=1)
    assert (pv_ref - pv1).abs().max() < TOL
    assert (pv_m - pv1).abs().max() < TOL and (ps_m - ps1).abs().max() < TOL

    m2 = Transolver(in_dim=7, n_hidden=32, n_layers=2, n_heads=4, slice_num=8,
                    mlp_ratio=2, out_dim_vol=5).double()
    x2 = torch.rand(1, 200, 7, dtype=torch.float64)
    pv, ps = m2(x2)
    (pv.square().mean() + ps.square().mean()).backward()
    g1 = {n: p.grad.clone() for n, p in m2.named_parameters()
          if p.grad is not None}
    m2.zero_grad()
    outs = m2.forward_chunks([x2[:, :77], x2[:, 77:]])
    pv2 = torch.cat([o[0] for o in outs], 1)
    ps2 = torch.cat([o[1] for o in outs], 1)
    (pv2.square().mean() + ps2.square().mean()).backward()
    gd = max((g1[n] - p.grad).abs().max().item()
             for n, p in m2.named_parameters() if p.grad is not None)
    assert gd < 1e-9
    # cached pass-1 slice weights (the training path) must reproduce the
    # recompute EXACTLY — same deterministic function, same input
    m3 = Transolver(in_dim=7, n_hidden=32, n_layers=3, n_heads=4,
                    slice_num=8, mlp_ratio=2, out_dim_vol=3).double().eval()
    with torch.no_grad():
        for p_ in m3.parameters():          # move biases off zero, or the
            p_.add_(0.01 * torch.randn_like(p_))   # eps placement hides
        x3 = torch.rand(1, 257, 7, dtype=torch.float64)
        (pv_c, ps_c), = m3.forward_chunks([x3], cache_weights=True)
        (pv_r, ps_r), = m3.forward_chunks([x3], cache_weights=False)
    assert (pv_c - pv_r).abs().max().item() == 0.0
    assert (ps_c - ps_r).abs().max().item() == 0.0
    print("gate transolver_chunks: PASS (incl. cached-weight identity)")


def test_spider_rewrite():
    """segment/index_add + SDPA rewrite == the dense one-hot reference, both
    surface-token modes, forward + input grads + weight grads."""
    torch.manual_seed(3)
    for mode, K in (("per_region", 6), ("single", 4)):
        attn = SpiderAttention(dim=32, heads=4, dropout=0.0, onion_num=5,
                               n_clusters=K,
                               surf_token_mode=mode).double().eval()
        n_vol, n_surf, n_inner = 400, 60, 40
        x = torch.rand(1, n_vol + n_surf, 32, dtype=torch.float64,
                       requires_grad=True)
        gid = torch.randint(0, 5 * K, (n_vol,))
        inner = torch.randperm(n_vol)[:n_inner]
        reg = torch.randint(0, K, (n_surf,)) if mode == "per_region" else None
        y1 = attn(x, n_vol, gid, inner, reg)
        y2 = dense_spider_attention(attn, x, n_vol, gid, inner, reg)
        assert (y1 - y2).abs().max() < TOL
        g1 = torch.autograd.grad(y1.square().mean(), x)[0]
        x2 = x.detach().clone().requires_grad_(True)
        g2 = torch.autograd.grad(dense_spider_attention(
            attn, x2, n_vol, gid, inner, reg).square().mean(), x2)[0]
        assert (g1 - g2).abs().max() < TOL
        attn.zero_grad()
        attn(x.detach(), n_vol, gid, inner, reg).square().mean().backward()
        gw1 = {n: p.grad.clone() for n, p in attn.named_parameters()
               if p.grad is not None}
        attn.zero_grad()
        dense_spider_attention(attn, x.detach(), n_vol, gid, inner,
                               reg).square().mean().backward()
        gwd = max((gw1[n] - p.grad).abs().max().item()
                  for n, p in attn.named_parameters()
                  if p.grad is not None and n in gw1)
        assert gwd < TOL
    print("gate spider_rewrite: PASS")


def test_balltree_contract():
    """tree_idx/tree_mask/tree_idx_rot contract on awkward N (padding), and
    the exact-inverse row-restoration path erwin's decode relies on."""
    torch.manual_seed(5)
    for N in (1000, 1024, 777):
        pos = torch.rand(N, 3)
        ti, tm, rot = build_tree(pos, strides=[2, 2],
                                 ball_sizes=[64, 64, 64], rotate=45)
        P = ti.shape[0]
        assert P % 64 == 0 and P >= N
        assert tm.sum().item() == N
        assert torch.equal(ti[tm].sort().values, torch.arange(N))
        scr = torch.arange(N, dtype=torch.float32).unsqueeze(1)
        out = scr[ti][tm][torch.argsort(ti[tm])]
        assert torch.equal(out.squeeze(1).long(), torch.arange(N))
        for r, sz in zip(rot, [P, P // 2, P // 4]):
            assert r.shape[0] == sz
            assert torch.equal(r.sort().values, torch.arange(sz))
    # 2D trees (the tandem sibling uses the same builder)
    pos2 = torch.rand(500, 2)
    ti, tm, rot = build_tree(pos2, strides=[1], ball_sizes=[32, 32],
                             rotate=45)
    assert torch.equal(ti[tm].sort().values, torch.arange(500))
    # kNN edges: self-loop present, targets aggregated
    ei = knn_edges(torch.rand(50, 3), 4)
    assert ei.shape == (2, 200)
    assert (ei[1].bincount(minlength=50) == 4).all()
    print("gate balltree_contract: PASS")


def test_restore_index_contract():
    # The STATIC unpad-and-reorder index that replaces the decoder tail's
    # boolean mask (aten.nonzero -> dynamic shape -> uncompilable):
    # elementwise identical on awkward N/padding, plus the dtype / length /
    # permutation contract.
    torch.manual_seed(11)
    for N, dim in ((1000, 2), (1024, 2), (777, 2), (513, 2), (257, 3),
                   (999, 3)):
        pos = torch.rand(N, dim)
        ti, tm, rot = build_tree(pos, strides=[2, 2],
                                 ball_sizes=[64, 64, 64], rotate=45)
        ri = restore_index(ti, tm)
        P = int(ti.shape[0])
        assert ri.dtype == torch.int64, ri.dtype
        assert ri.shape == (N,), (ri.shape, N)
        assert int(ri.min()) >= 0 and int(ri.max()) < P
        assert torch.equal(ri.sort().values, torch.nonzero(tm).squeeze(1)),             "restore_idx must be a permutation of the real (unpadded) rows"
        assert torch.equal(ti[ri], torch.arange(N)),             "row i of the output must be original row i"
        x = torch.randn(P, 5, dtype=torch.float64)
        reference = x[tm][torch.argsort(ti[tm])]
        static = x.index_select(0, ri)
        assert torch.equal(reference, static), (N, dim)
    pos = torch.rand(500, 2)
    ti, tm, rot = build_tree(pos, strides=[1], ball_sizes=[32, 32],
                             rotate=45)
    ri = restore_index(ti, tm)
    x = torch.randn(int(ti.shape[0]), 3, dtype=torch.float64)
    assert torch.equal(x[tm][torch.argsort(ti[tm])], x.index_select(0, ri))
    print("gate restore_index_contract: PASS")


def test_mpnn_grouped_mean():
    # grouped_mean (used by the model, scatter-free, CUDA-Graph capturable)
    # == the author's scatter_mean reference, in fp64, forward AND
    # gradients, on REAL knn_edges output; plus the edge-target layout
    # contract that makes the two identical.
    torch.manual_seed(17)
    worst = 0.0
    for n, k, dim, c in ((1000, 8, 2, 16), (777, 8, 3, 8), (256, 4, 2, 5),
                         (32, 16, 3, 3)):
        pos = torch.rand(n, dim)
        ei = knn_edges(pos, k)
        col = ei[1]
        # contract: every receiver gets exactly k edges, contiguous, in
        # receiver order -> edge_index[1] == repeat(arange(n), k)
        assert ei.shape == (2, n * k), ei.shape
        assert torch.equal(col, torch.arange(n).repeat_interleave(k)),             "edge target layout contract violated"
        assert torch.equal(col.bincount(minlength=n),
                           torch.full((n,), k, dtype=torch.long))
        src = torch.randn(n * k, c, dtype=torch.float64, requires_grad=True)
        ref = scatter_mean(src, col, n)
        new = grouped_mean(src, n)
        # exact in exact arithmetic; the two differ only by fp SUMMATION
        # ORDER (index_add walks edges, mean uses a vectorised reduction),
        # so the fp64 gate is machine precision, not bitwise
        dev = (ref - new).abs().max().item()
        assert dev < 1e-13, (n, k, dev)
        # backward compared in ISOLATION: identical upstream gradient, so
        # any difference is the reduction's own backward, not the ulp-level
        # difference of the forward values
        gout = torch.randn_like(ref)
        g_ref = torch.autograd.grad(ref, src, grad_outputs=gout,
                                    retain_graph=True)[0]
        g_new = torch.autograd.grad(new, src, grad_outputs=gout)[0]
        gdev = (g_ref - g_new).abs().max().item()
        assert gdev < 1e-13, (n, k, gdev)
        worst = max(worst, dev, gdev)
    print(f"gate mpnn_grouped_mean: PASS (max fp64 dev {worst:.2e})")


def test_tree_threading_bitwise():
    # 16-thread tree construction must be BIT-IDENTICAL to the serial path
    # (threads only decide who computes first; every build owns its arrays)
    torch.manual_seed(23)
    for n, dim, strides, balls in ((3000, 2, [1], [256, 256]),
                                   (5000, 3, [2, 2], [64, 64, 64]),
                                   (777, 2, [2], [128, 128])):
        pos = torch.rand(n, dim)
        a = build_tree(pos, strides, balls, 45, workers=1)
        b = build_tree(pos, strides, balls, 45, workers=16)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        assert len(a[2]) == len(b[2])
        for ra, rb in zip(a[2], b[2]):
            assert torch.equal(ra, rb)
        e1 = knn_edges(pos, 8, workers=1)
        e16 = knn_edges(pos, 8, workers=16)
        assert torch.equal(e1, e16)
    print("gate tree_threading_bitwise: PASS")


if __name__ == "__main__":
    test_abupt_kv_cache()
    test_transolver_chunks()
    test_spider_rewrite()
    test_balltree_contract()
    test_restore_index_contract()
    test_mpnn_grouped_mean()
    test_tree_threading_bitwise()
    print("ALL GATES PASS")
