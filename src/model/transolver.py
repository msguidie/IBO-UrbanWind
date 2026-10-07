"""Transolver — port of the reference chunked Physics-Attention implementation
(Transolver-3, `models/Transolver_chunk_opt_matrix_mul.py`).

What is verbatim from the reference:
  - Physics-Attention: per-head slice assignment softmax(logits / T) with an
    orthogonally-initialised slice projection, slice-token aggregation as a
    weighted mean, plain q/k/v attention among the G slice tokens, deslice
    back to points, `to_out` projection; the fused-weight re-association
    (`_get_fused_weight_slice`, project-in-slice-domain-first) that makes the
    per-point cost O(N*G) with no [N, H*D] intermediates;
  - the two-pass chunked block (`forward_chunks`): pass 1 accumulates the
    UNNORMALISED slice numerator/denominator over every chunk, the slice
    attention runs ONCE on the globally-normalised tokens, pass 2 deslices
    per chunk. This is the reference TRAINING forward — mathematically the
    attention over the union of all chunks, exactly (softmax-weighted means
    are chunk-additive) — and therefore also the EXACT full-field inference:
    feeding the whole case as a chunk list IS the global model, never an
    approximation.
  - MLP shape (Linear -> GELU -> Linear, ratio 2), pre-LN residual block
    layout, trunc_normal(0.02) init, the learnable `placeholder` added to
    every embedded point, the 0.5-initialised per-head temperature.

Deviations (form-only or required by the shared protocol):
  - einops removed (reshape/permute/einsum only);
  - ONE shared backbone with TWO output heads (volume C_vol, surface 1)
    instead of the reference's separate surface/volume models (joint
    training, as for every model in the comparison). The reference's last-block `ln_3 + mlp2` head
    becomes a final LayerNorm + two Linear heads: same math, split output.
  - `unified_pos` distance features are computed by the DATA adapter over
    the dataset's true physical domain (the reference hardcodes the
    AirfRANS/ShapeNet domain inside the model); the model just sees a wider
    input. Pure relocation of a deterministic function of position.
  - dropout exists but is 0 in the protocol.
  - activation checkpointing is a constructor flag applied around the
    reference's own checkpoint call sites.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class PhysicsAttention(nn.Module):
    """Reference Physics_Attention_Irregular_Mesh (chunk-opt form)."""

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64,
                 dropout: float = 0.0, slice_num: int = 64):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.slice_num = slice_num

        self.in_project = nn.Linear(dim, 2 * heads * dim_head)  # fx and x
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice.weight)
        self.to_out_linear = nn.Linear(heads * dim_head, dim)

        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

    # ---- reference helpers (verbatim math) ------------------------------

    def _get_fused_weight_slice(self):
        H, D = self.heads, self.dim_head
        w_in = self.in_project.weight[H * D:].view(H, D, -1)
        w_slice = self.in_project_slice.weight                # (G, D)
        fused_w = torch.matmul(w_slice, w_in)                 # (H, G, C)
        b_in = self.in_project.bias[H * D:].view(H, D)
        fused_b = torch.matmul(w_slice, b_in.unsqueeze(-1)).squeeze(-1)
        fused_b = fused_b + self.in_project_slice.bias
        return fused_w, fused_b

    def chunk_weights(self, x: torch.Tensor, fused_w=None, fused_b=None):
        if fused_w is None:
            fused_w, fused_b = self._get_fused_weight_slice()
        logits = torch.einsum("bnc, hgc -> bhng", x, fused_w) \
            + fused_b.view(1, self.heads, 1, self.slice_num)
        return F.softmax(logits / self.temperature, dim=-1)

    def chunk_stats(self, x: torch.Tensor,
                    row_mask: torch.Tensor | None = None,
                    return_weights: bool = False):
        """(num, den) — the UNNORMALISED slice-token numerator/denominator
        of this chunk, additive across chunks. row_mask [B, N] (1 real /
        0 pad) zeroes padded rows' slice weights, so their contribution to
        the GLOBAL slice statistics is EXACTLY removed — fixed-shape padded
        chunks stay mathematically identical to the unpadded case."""
        H, D = self.heads, self.dim_head
        f_w, f_b = self._get_fused_weight_slice()
        slice_weights = self.chunk_weights(x, f_w, f_b)       # (B, H, N, G)
        if row_mask is not None:
            slice_weights = slice_weights * row_mask[:, None, :, None]
        den = slice_weights.sum(dim=2)                        # (B, H, G)
        x_agg = torch.einsum("bnc, bhng -> bhgc", x, slice_weights)
        w_fx = self.in_project.weight[:H * D].view(H, D, -1)
        b_fx = self.in_project.bias[:H * D].view(H, D)
        num = torch.einsum("bhgc, hdc -> bhgd", x_agg, w_fx)
        num = num + den.unsqueeze(-1) * b_fx.unsqueeze(1)
        if return_weights:
            return num, den, slice_weights
        return num, den

    def slice_attend(self, slice_token: torch.Tensor) -> torch.Tensor:
        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)
        return F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=False)

    def chunk_deslice_to_out(self, x: torch.Tensor,
                             out_slice_token: torch.Tensor,
                             slice_weights=None) -> torch.Tensor:
        H, D = self.heads, self.dim_head
        if slice_weights is None:
            slice_weights = self.chunk_weights(x)
        w_out = self.to_out_linear.weight.view(-1, H, D).permute(1, 2, 0)
        projected = torch.einsum("bhgd, hdc -> bhgc", out_slice_token, w_out)
        out_x = torch.einsum("bhng, bhgc -> bnc", slice_weights, projected)
        return self.dropout(out_x + self.to_out_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single-tensor reference path (equals forward_chunks on 1 chunk)."""
        H, D = self.heads, self.dim_head
        f_w, f_b = self._get_fused_weight_slice()
        slice_weights = self.chunk_weights(x, f_w, f_b)
        slice_norm = slice_weights.sum(dim=2, keepdim=True) + 1e-5
        raw_states = torch.einsum("bnc, bhng -> bhgc", x, slice_weights)
        raw_states = raw_states / slice_norm.transpose(-1, -2)
        w_fx = self.in_project.weight[:H * D].view(H, D, -1)
        b_fx = self.in_project.bias[:H * D].view(H, D)
        slice_token = torch.einsum("bhgc, hdc -> bhgd", raw_states, w_fx) \
            + b_fx.view(1, H, 1, D)
        out_slice_token = self.slice_attend(slice_token)
        return self.chunk_deslice_to_out(x, out_slice_token, slice_weights)


class Mlp(nn.Module):
    """Reference MLP (n_layers=0, res=False): Linear -> GELU -> Linear."""

    def __init__(self, n_input: int, n_hidden: int, n_output: int):
        super().__init__()
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden),
                                        nn.GELU())
        self.linear_post = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_post(self.linear_pre(x))


class TransolverBlock(nn.Module):
    """Pre-LN residual block. `last_layer` heads are NOT here — the model
    applies the final LayerNorm + the two output heads itself (same math as
    the reference's in-block ln_3+mlp2, split into vol/surf heads)."""

    def __init__(self, num_heads: int, hidden_dim: int, dropout: float,
                 mlp_ratio: int = 2, slice_num: int = 32):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = PhysicsAttention(hidden_dim, heads=num_heads,
                                     dim_head=hidden_dim // num_heads,
                                     dropout=dropout, slice_num=slice_num)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = Mlp(hidden_dim, hidden_dim * mlp_ratio, hidden_dim)

    def forward(self, fx: torch.Tensor) -> torch.Tensor:
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        return fx

    def forward_chunks(self, fx_list: list[torch.Tensor], eps: float = 1e-5,
                       use_checkpoint: bool = False,
                       cache_weights: bool = False) -> list[torch.Tensor]:
        """Two-pass global aggregation over a chunk LIST (reference
        Transolver_block.forward_chunks): mathematically the block applied
        to the concatenation of all chunks."""
        global_num, global_den = None, None
        cached_w: list = []
        for fxk in fx_list:
            uk = self.ln_1(fxk)
            if cache_weights:
                # keep pass 1's slice weights instead of recomputing the
                # same fused einsum+softmax in pass 2. BITWISE identical
                # (same deterministic function of the same input) and free
                # in memory: autograd already retains them for backward.
                num_k, den_k, w_k = self.Attn.chunk_stats(
                    uk, return_weights=True)
                cached_w.append(w_k)
            elif use_checkpoint and torch.is_grad_enabled():
                num_k, den_k = checkpoint(self.Attn.chunk_stats, uk,
                                          use_reentrant=False)
            else:
                num_k, den_k = self.Attn.chunk_stats(uk)
            global_num = num_k if global_num is None else global_num + num_k
            global_den = den_k if global_den is None else global_den + den_k

        slice_token = global_num / (global_den[..., None] + eps)
        out_slice = self.Attn.slice_attend(slice_token)

        out_list = []
        for ci, fxk in enumerate(fx_list):
            def chunk_compute(f_k, o_slice, _w=None):
                # with cached weights the deslice needs neither ln_1(f_k)
                # nor the weight recompute (chunk_deslice_to_out ignores x
                # when slice_weights is given)
                uk = None if _w is not None else self.ln_1(f_k)
                a_out = self.Attn.chunk_deslice_to_out(uk, o_slice,
                                                       slice_weights=_w)
                res_fx = a_out + f_k
                return self.mlp(self.ln_2(res_fx)) + res_fx

            w = cached_w[ci] if cache_weights else None
            if use_checkpoint and torch.is_grad_enabled():
                out_list.append(checkpoint(chunk_compute, fxk, out_slice, w,
                                           use_reentrant=False))
            else:
                out_list.append(chunk_compute(fxk, out_slice, w))
        return out_list

    # ---- static-shape compiled-producer entry points --------------------
    # (inference only — one fixed chunk shape, padded rows exactly masked
    # out of the global statistics; the recipe compiles these per block)

    def stats_masked(self, fx: torch.Tensor, row_mask: torch.Tensor):
        return self.Attn.chunk_stats(self.ln_1(fx), row_mask)

    def apply_chunk(self, fx: torch.Tensor,
                    out_slice: torch.Tensor) -> torch.Tensor:
        uk = self.ln_1(fx)
        a_out = self.Attn.chunk_deslice_to_out(uk, out_slice)
        res_fx = a_out + fx
        return self.mlp(self.ln_2(res_fx)) + res_fx


class Transolver(nn.Module):
    """Shared backbone + volume/surface heads.

    in_dim is the ADAPTER's channel count (native geometry channels, plus
    the 64 unified-position distance features when the dataset recipe uses
    them). The reference preprocess MLP is Linear(in, 2*hidden) -> GELU ->
    Linear(2*hidden, hidden).
    """

    def __init__(self, in_dim: int, n_hidden: int = 256, n_layers: int = 8,
                 n_heads: int = 8, slice_num: int = 32, mlp_ratio: int = 2,
                 dropout: float = 0.0, out_dim_vol: int = 5,
                 out_dim_surf: int = 1, use_checkpoint: bool = False):
        super().__init__()
        self.__name__ = "Transolver"
        self.use_checkpoint = bool(use_checkpoint)
        self.preprocess = Mlp(in_dim, n_hidden * 2, n_hidden)
        self.blocks = nn.ModuleList([
            TransolverBlock(num_heads=n_heads, hidden_dim=n_hidden,
                            dropout=dropout, mlp_ratio=mlp_ratio,
                            slice_num=slice_num)
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

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.preprocess(x) + self.placeholder[None, None, :]

    def forward_chunks(self, x_list: list[torch.Tensor],
                       cache_weights: bool = False
                       ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Global-semantics forward over a chunk list. Returns one
        (pred_vol, pred_surf) pair per chunk. Feeding a full case split
        into chunks IS the model on the full case (see module docstring)."""
        fx_list = [self.embed(x) for x in x_list]
        for block in self.blocks:
            fx_list = block.forward_chunks(
                fx_list, use_checkpoint=self.use_checkpoint,
                cache_weights=cache_weights)
        out = []
        for fx in fx_list:
            h = self.ln_out(fx)
            out.append((self.vol_head(h), self.surf_head(h)))
        return out

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Training path: ONE chunk (static shape for torch.compile),
        with the pass-1 slice weights REUSED in pass 2 — bitwise identical
        to the recompute, ~15% off the forward. The recompute exists so
        multi-chunk INFERENCE never has to hold [B, H, N, G] per chunk;
        with a single chunk there is nothing to save."""
        (pv, ps), = self.forward_chunks([x], cache_weights=True)
        return pv, ps


def build_transolver_model(tcfg, out_dim_vol: int, in_dim: int | None = None,
                           use_checkpoint: bool = False) -> Transolver:
    """From a TransolverConfig section."""
    return Transolver(
        in_dim=int(in_dim if in_dim is not None else tcfg.in_dim),
        n_hidden=int(tcfg.n_hidden), n_layers=int(tcfg.n_layers),
        n_heads=int(tcfg.n_heads), slice_num=int(tcfg.slice_num),
        mlp_ratio=int(tcfg.mlp_ratio), dropout=float(tcfg.dropout),
        out_dim_vol=int(out_dim_vol), out_dim_surf=1,
        use_checkpoint=use_checkpoint)
