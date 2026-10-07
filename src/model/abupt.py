"""AB-UPT: dual-branch (surface / volume) Anchored-Branched UPT.

Port of the reference implementation (Alkin et al., 2025) WITHOUT the
geometry branch: no supernode pooling, no
geometry transformer, no Perceiver cross-attention to geometry tokens. The
shared trunk is "sscscs" — the reference's default "pscscs" with the
geometry-perceiver block replaced by a third split-attention block — so
every surface/volume token sees exactly:

    s s c s c s   +   6 branch-specific AnchorAttention blocks

Effective depth per token is 12; the module count is 18 (6 shared trunk +
6 surface-specific + 6 volume-specific). The trunk blocks are weight-shared
between the branches (one parameter set applied to both token groups);
the branch-specific blocks are independent per branch.

Architecture (cfg.abupt): dim 192, 3 heads (head_dim 64), pre-norm LayerNorm(eps 1e-6) blocks, MLP 192 -> 768 -> 192 with GELU,
no dropout (the reference has none), trunc_normal(0.02) + zero-bias init on
every Linear (the reference's init_weights; LayerNorm default init;
nn.Embedding default init, matching model/decoder.py's RegionEmbedding).

Deviations from the reference file, all form-only (values identical):
  - no einops (not a repo dependency): rearranges are reshape/permute.
  - RoPE is computed with real arithmetic instead of torch.polar /
    view_as_complex: (x0*cos - x1*sin, x0*sin + x1*cos) IS the complex
    multiply, and complex dtypes are avoided inside compiled graphs. Frequency layout (max_wavelength 10000, per-axis split with
    zero padding -> identity rotation on the padded dims) is unchanged.
  - the reference's `assert torch.all(coords >= 0)` (ContinuousSincosEmbed
    / RopeFrequency) cannot live inside a fullgraph-compiled forward; the
    check runs in the data path instead (data/abupt_estate.py
    check_abupt_coord_bounds), once per pack load / eval case.
  - positions enter in the cache's /550 frame (query_pos_norm) and the
    fixed global affine to AB-UPT's non-negative scale happens HERE, in the
    input stem: abupt_pos = (pos_norm + pos_offset) * pos_scale. With the
    UrbanWind domain (xy in [-550, 550] m, z in [0, 160] m -> pos_norm xy in
    [-1, 1], z in [0, ~0.291]) the defaults (offset [1,1,0], scale 500) map
    xy to [0, 1000] and z to [0, ~145] — uniform per-axis scale, same
    construction as the reference's PositionNormalizationPreprocessor.
    No per-estate or per-direction normalization anywhere.

Input variants (chosen at CONSTRUCTION time — use_geometry_features is a
python constant inside forward, so each configured instance traces a fully
static graph with no optional-tensor branching):
  A (vanilla): coordinates only. sincos embed R^3 -> R^192 (fp32 math),
     then per-branch MLP 192 -> 192 -> 192 with GELU (the reference's
     surface_bias / volume_bias — independently parameterized). No extra
     type embedding (the reference has none; branch identity comes from
     the separate input MLPs).
  B (--abupt-geometry-features): additionally g_b = GeometryFeatureMLP_b(
     concat(sdf 1, sdf_dir 3, support 6, RegionEmbedding 12)) with
     22 -> 192 -> 192 GELU, independent per branch; z0 = p_b + g_b.
     Injected ONCE in the stem; hidden dim stays 192. region_dim1 is NOT a
     model input: by definition it is the branch identity (volume=0,
     surface=1) and is synthesized per branch inside the stem — anchors,
     queries and padded query rows alike (padded rows live inside a branch
     stream, inherit its constant, and their outputs are discarded by the
     caller before any metric).

Precision: sincos/RoPE angle math runs in fp32 via elementwise ops
(broadcast multiply + sin/cos are not autocast-cast ops, so no autocast
context wrangling is needed); rope rotation upcasts q/k to fp32 and casts
back to the ambient dtype (the reference's x.float()/type_as(x)). All
Linears/SDPA run under the ambient bf16 autocast.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# geometry feature layout: sdf(1) + sdf_dir(3) + support(6) + region emb(12)
GEOMETRY_FEATURE_DIM = 1 + 3 + 6 + 12


def _init_linear_weights(module: nn.Module) -> None:
    """The reference's init_weights: trunc_normal(0.02) + zero bias on every
    nn.Linear; LayerNorm keeps its default init; nn.Embedding keeps its
    default init (same treatment model/decoder.py gives RegionEmbedding)."""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ContinuousSincosEmbed(nn.Module):
    """Reference modules/continuous_sincos_embed.py, assert moved to the
    data path. dim=192 / ndim=3 divides cleanly (no padding): 32 wavelengths
    per axis, sin||cos = 64 dims per axis, 192 total."""

    def __init__(self, dim: int, ndim: int, max_wavelength: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.ndim = ndim
        self.ndim_padding = dim % ndim
        dim_per_ndim = (dim - self.ndim_padding) // ndim
        self.sincos_padding = dim_per_ndim % 2
        self.padding = self.ndim_padding + self.sincos_padding * ndim
        effective_dim_per_wave = (dim - self.padding) // ndim
        assert effective_dim_per_wave > 0
        arange = torch.arange(0, effective_dim_per_wave, 2, dtype=torch.float32)
        self.register_buffer(
            "omega", 1.0 / max_wavelength ** (arange / effective_dim_per_wave))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # fp32 angle math regardless of ambient autocast: broadcast multiply
        # and sin/cos are not autocast-cast ops, so no context needed. The
        # result is returned fp32; downstream Linears cast at their input.
        theta = coords.float().unsqueeze(-1) * self.omega    # [B, N, ndim, E/2]
        emb = torch.cat([theta.sin(), theta.cos()], dim=-1)  # [B, N, ndim, E]
        emb = emb.flatten(-2)                                # [B, N, ndim*E]
        if self.padding > 0:
            pad = emb.new_zeros(*emb.shape[:-1], self.padding)
            emb = torch.cat([emb, pad], dim=-1)
        return emb


class RopeFrequency(nn.Module):
    """Reference modules/rope_frequency.py, returning (cos, sin) instead of
    torch.polar complex — identical rotation, real dtypes only. For
    head_dim=64 / ndim=3: 10 wavelengths per axis -> 30 phases + 2 zero-pad
    (cos=1, sin=0 == polar(1, 0), i.e. identity on the padded pairs) = 32
    = head_dim // 2."""

    def __init__(self, dim: int, ndim: int, max_wavelength: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.ndim = ndim
        self.ndim_padding = dim % ndim
        dim_per_ndim = (dim - self.ndim_padding) // ndim
        self.sincos_padding = dim_per_ndim % 2
        self.padding = self.ndim_padding + self.sincos_padding * ndim
        assert self.padding % 2 == 0
        effective_dim_per_wave = (dim - self.padding) // ndim
        assert effective_dim_per_wave > 0
        arange = torch.arange(0, effective_dim_per_wave, 2, dtype=torch.float32)
        self.register_buffer(
            "omega", 1.0 / max_wavelength ** (arange / effective_dim_per_wave))

    def forward(self, coords: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """coords [B, N, ndim] -> (cos, sin) each [B, N, dim // 2] fp32."""
        theta = coords.float().unsqueeze(-1) * self.omega    # [B, N, ndim, E/2]
        theta = theta.flatten(-2)                            # [B, N, ndim*E/2]
        pad = theta.new_zeros(*theta.shape[:-1], self.padding // 2)
        theta = torch.cat([theta, pad], dim=-1)              # [B, N, dim//2]
        return theta.cos(), theta.sin()


def apply_rope(x: torch.Tensor, freqs: tuple[torch.Tensor, torch.Tensor]
               ) -> torch.Tensor:
    """Real-arithmetic RoPE, elementwise-identical (in fp32) to the
    reference's complex multiply: pair j = (x[..., 2j], x[..., 2j+1])
    rotated by angle theta_j. x [B, H, T, hd]; cos/sin [B, T, hd//2]."""
    cos, sin = freqs
    xf = x.float()
    xp = xf.reshape(*xf.shape[:-1], -1, 2)
    x0, x1 = xp[..., 0], xp[..., 1]
    c = cos.unsqueeze(1)   # [B, 1, T, hd//2] — broadcast over heads
    s = sin.unsqueeze(1)
    out0 = x0 * c - x1 * s
    out1 = x0 * s + x1 * c
    return torch.stack([out0, out1], dim=-1).flatten(-2).type_as(x)


def _slice_freqs(freqs: tuple[torch.Tensor, torch.Tensor], n: int
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = freqs
    return cos[:, :n], sin[:, :n]


class DotProductAttention(nn.Module):
    """Reference modules/attention/dot_product_attention.py (qkv and proj
    keep the reference's default bias=True)."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def _qkv(self, x: torch.Tensor
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)   # each [B, H, T, hd]
        return q, k, v

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, _, T, _ = x.shape
        return x.transpose(1, 2).reshape(B, T, self.dim)

    def forward(self, x: torch.Tensor,
                freqs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        q, k, v = self._qkv(x)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(self._merge_heads(out))

    # ---- anchor-K/V caching (full-field inference) ----------------------
    # Anchors never read query tokens anywhere in this architecture (s/c
    # trunk splits and AnchorAttention alike), so their per-block K/V are
    # chunk-invariant and can be computed ONCE per case. kv_from/query_attend
    # are the two primitives the cached decode path is assembled from; both
    # reuse this module's own qkv/proj weights, so the math is the joint
    # forward's, re-associated (forward_with_queries is the reference
    # form).

    def kv_from(self, x_normed: torch.Tensor,
                freqs: tuple[torch.Tensor, torch.Tensor]
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """(k_roped, v) of these rows, from the qkv projection's k/v slices."""
        B, T, _ = x_normed.shape
        w = self.qkv.weight
        b = self.qkv.bias
        k = F.linear(x_normed, w[self.dim:2 * self.dim],
                     None if b is None else b[self.dim:2 * self.dim])
        v = F.linear(x_normed, w[2 * self.dim:],
                     None if b is None else b[2 * self.dim:])
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        return apply_rope(k, freqs), v

    def query_attend(self, x_normed: torch.Tensor, k_cached: torch.Tensor,
                     v_cached: torch.Tensor,
                     freqs_q: tuple[torch.Tensor, torch.Tensor]
                     ) -> torch.Tensor:
        """Query rows read cached anchor K/V — the AnchorAttention query
        path, factored out so the trunk s/c blocks can use it too."""
        B, T_q, _ = x_normed.shape
        b = self.qkv.bias
        q = F.linear(x_normed, self.qkv.weight[:self.dim],
                     None if b is None else b[:self.dim])
        q = q.view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, freqs_q)
        out = F.scaled_dot_product_attention(q, k_cached, v_cached)
        return self.proj(self._merge_heads(out))


class SharedweightsSplitattnAttention(DotProductAttention):
    """`s` block attention: ONE parameter set applied to both branches,
    attention restricted WITHIN each branch. split_size len 2 =
    (surface_anchors, volume_anchors); len 4 = (surface_anchors,
    surface_queries, volume_anchors, volume_queries) — queries attend only
    their own branch's ANCHORS. The equal-size 2-way path folds the two
    branches into the batch dim (reference's optimization) so it can never
    degrade into unrestricted full attention over the concatenated
    sequence."""

    def forward(self, x: torch.Tensor, split_size: list[int],
                freqs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        q, k, v = self._qkv(x)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)
        qs = q.split(split_size, dim=2)
        ks = k.split(split_size, dim=2)
        vs = v.split(split_size, dim=2)
        if len(split_size) == 4:
            # anchors + queries: q = branch anchors||queries, kv = branch
            # anchors only
            out1 = F.scaled_dot_product_attention(
                torch.cat([qs[0], qs[1]], dim=2), ks[0], vs[0])
            out2 = F.scaled_dot_product_attention(
                torch.cat([qs[2], qs[3]], dim=2), ks[2], vs[2])
            out = torch.cat([out1, out2], dim=2)
        elif len(split_size) == 2:
            if split_size[0] == split_size[1]:
                # equal splits: fold branch axis into batch, one SDPA
                o = F.scaled_dot_product_attention(
                    torch.cat(qs, dim=0), torch.cat(ks, dim=0),
                    torch.cat(vs, dim=0))
                out = torch.cat(o.chunk(2, dim=0), dim=2)
            else:
                out = torch.cat(
                    [F.scaled_dot_product_attention(qs[i], ks[i], vs[i])
                     for i in range(len(qs))], dim=2)
        else:
            raise NotImplementedError
        return self.proj(self._merge_heads(out))


class SharedweightsCrossattnAttention(DotProductAttention):
    """`c` block attention: ONE parameter set, bidirectional cross between
    the branches — surface queries read volume anchors' K/V and volume
    queries read surface anchors' K/V; neither side reads its own branch's
    K/V in this block."""

    def forward(self, x: torch.Tensor, split_size: list[int],
                freqs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        q, k, v = self._qkv(x)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)
        ks = k.split(split_size, dim=2)
        vs = v.split(split_size, dim=2)
        if len(split_size) == 4:
            qs = q.split([split_size[0] + split_size[1],
                          split_size[2] + split_size[3]], dim=2)
            out1 = F.scaled_dot_product_attention(qs[0], ks[2], vs[2])
            out2 = F.scaled_dot_product_attention(qs[1], ks[0], vs[0])
            out = torch.cat([out1, out2], dim=2)
        elif len(split_size) == 2:
            if split_size[0] == split_size[1]:
                # equal splits: fold branch axis into batch with SWAPPED kv
                B, H, _, hd = q.shape
                n = split_size[0]
                q2 = q.view(B, H, 2, n, hd).permute(2, 0, 1, 3, 4) \
                     .reshape(2 * B, H, n, hd)
                k2 = torch.cat([ks[1], ks[0]], dim=0)
                v2 = torch.cat([vs[1], vs[0]], dim=0)
                o = F.scaled_dot_product_attention(q2, k2, v2)
                out = o.view(2, B, H, n, hd).permute(1, 2, 0, 3, 4) \
                      .reshape(B, H, 2 * n, hd)
            else:
                qs = q.split(split_size, dim=2)
                out1 = F.scaled_dot_product_attention(qs[0], ks[1], vs[1])
                out2 = F.scaled_dot_product_attention(qs[1], ks[0], vs[0])
                out = torch.cat([out1, out2], dim=2)
        else:
            raise NotImplementedError
        return self.proj(self._merge_heads(out))


class AnchorAttention(DotProductAttention):
    """Branch-specific block attention: anchors self-attend; query tokens
    (everything past num_anchor_tokens) get a Q-only projection and
    cross-attend the anchors' K/V — anchor outputs are therefore identical
    with or without queries present. num_anchor_tokens=None (anchors-only
    training) is plain full self-attention over the branch's 16K anchors —
    the AnchorAttention-compatible degenerate case, not a different
    module."""

    def forward(self, x: torch.Tensor,
                freqs: tuple[torch.Tensor, torch.Tensor],
                num_anchor_tokens: int | None = None) -> torch.Tensor:
        if num_anchor_tokens is None:
            return super().forward(x, freqs)
        n_a = num_anchor_tokens
        anchors, queries = x.split([n_a, x.size(1) - n_a], dim=1)
        q, k, v = self._qkv(anchors)
        B, T_q, _ = queries.shape
        q_query = F.linear(
            queries, weight=self.qkv.weight[:self.dim],
            bias=None if self.qkv.bias is None else self.qkv.bias[:self.dim])
        q_query = q_query.view(B, T_q, self.num_heads, self.head_dim) \
                         .transpose(1, 2)                # [B, H, T_q, hd]
        q = torch.cat([q, q_query], dim=2)
        q = apply_rope(q, freqs)
        k = apply_rope(k, _slice_freqs(freqs, n_a))
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(self._merge_heads(out))


class Mlp(nn.Module):
    """Reference modules/mlp.py: Linear(dim, 4*dim) -> GELU -> Linear."""

    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    """Reference modules/blocks/transformer_block.py: pre-norm
    LayerNorm(eps 1e-6) + attn residual, pre-norm + MLP residual."""

    def __init__(self, dim: int, num_heads: int,
                 attn_ctor: type[nn.Module] = DotProductAttention):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = attn_ctor(dim=dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim)

    def forward(self, x: torch.Tensor, attn_kwargs: dict | None = None
                ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), **(attn_kwargs or {}))
        x = x + self.mlp(self.norm2(x))
        return x


class RegionEmbedding(nn.Module):
    """Same tables/semantics as model/decoder.py's RegionEmbedding
    (cardinalities 2/5/10, 4 dims each, 12 total; default nn.Embedding
    init), instantiated fresh for AB-UPT. dim1 is not a tensor input: it is
    definitionally the branch identity (volume=0, surface=1), passed as a
    python constant and broadcast — covering anchors, queries and padded
    query rows uniformly."""

    def __init__(self):
        super().__init__()
        self.emb1 = nn.Embedding(2, 4)    # point_type: volume / surface
        self.emb2 = nn.Embedding(5, 4)    # sub_category
        self.emb3 = nn.Embedding(10, 4)   # edge_detail

    def forward(self, dim1_const: int, d2: torch.Tensor, d3: torch.Tensor
                ) -> torch.Tensor:
        e1 = self.emb1.weight[dim1_const].view(1, 1, -1) \
                 .expand(*d2.shape, -1)   # [B, N, 4], constant per branch
        return torch.cat([e1, self.emb2(d2.long()), self.emb3(d3.long())],
                         dim=-1)


def _branch_mlp(dim: int) -> nn.Sequential:
    """The reference's surface_bias/volume_bias shape: 192 -> 192 -> 192."""
    return nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))


def _geometry_mlp(dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(GEOMETRY_FEATURE_DIM, dim), nn.GELU(),
                         nn.Linear(dim, dim))


class AnchoredBranchedUPT(nn.Module):
    def __init__(
        self,
        use_geometry_features: bool,
        dim: int = 192,
        num_heads: int = 3,
        shared_blocks: str = "sscscs",
        num_surface_blocks: int = 6,
        num_volume_blocks: int = 6,
        output_dim_surface: int = 1,
        output_dim_volume: int = 5,
        pos_offset: tuple[float, float, float] = (1.0, 1.0, 0.0),
        pos_scale: float = 500.0,
    ):
        super().__init__()
        self.use_geometry_features = bool(use_geometry_features)
        self.dim = dim
        self.rope = RopeFrequency(dim=dim // num_heads, ndim=3)
        self.pos_embed = ContinuousSincosEmbed(dim=dim, ndim=3)
        # reference names: the per-branch position input MLPs
        self.surface_bias = _branch_mlp(dim)
        self.volume_bias = _branch_mlp(dim)
        if self.use_geometry_features:
            # ONE region-code table (branch identity flows through dim1);
            # the projection MLPs are independent per branch.
            self.region_embed = RegionEmbedding()
            self.surface_geometry_mlp = _geometry_mlp(dim)
            self.volume_geometry_mlp = _geometry_mlp(dim)

        ctors = {"s": SharedweightsSplitattnAttention,
                 "c": SharedweightsCrossattnAttention}
        self.shared_pattern = str(shared_blocks)
        self.blocks = nn.ModuleList()
        for ch in shared_blocks:
            if ch not in ctors:
                raise NotImplementedError(
                    f"shared block {ch!r} not supported (the geometry "
                    f"branch and its 'p' perceiver block are deliberately "
                    f"not ported)")
            self.blocks.append(
                TransformerBlock(dim=dim, num_heads=num_heads,
                                 attn_ctor=ctors[ch]))
        self.surface_blocks = nn.ModuleList([
            TransformerBlock(dim=dim, num_heads=num_heads,
                             attn_ctor=AnchorAttention)
            for _ in range(num_surface_blocks)])
        self.volume_blocks = nn.ModuleList([
            TransformerBlock(dim=dim, num_heads=num_heads,
                             attn_ctor=AnchorAttention)
            for _ in range(num_volume_blocks)])
        # per-point linear projections — the ONLY decoders
        self.surface_decoder = nn.Linear(dim, output_dim_surface)
        self.volume_decoder = nn.Linear(dim, output_dim_volume)

        # fixed global affine from the cache's /550 frame to the AB-UPT
        # non-negative scale; a derived constant, not a learned/persisted
        # state (config_used.yaml records it), hence persistent=False.
        self.register_buffer(
            "pos_offset", torch.tensor(pos_offset, dtype=torch.float32),
            persistent=False)
        self.pos_scale = float(pos_scale)

        self.apply(_init_linear_weights)

    # ---- input stem -----------------------------------------------------

    def _abupt_coords(self, pos_norm: torch.Tensor) -> torch.Tensor:
        return (pos_norm.float() + self.pos_offset) * self.pos_scale

    def _branch_tokens(self, pos_abupt: torch.Tensor, is_surface: bool,
                       sdf: torch.Tensor | None,
                       sdf_dir: torch.Tensor | None,
                       support: torch.Tensor | None,
                       region_dim2: torch.Tensor | None,
                       region_dim3: torch.Tensor | None) -> torch.Tensor:
        bias_mlp = self.surface_bias if is_surface else self.volume_bias
        p = bias_mlp(self.pos_embed(pos_abupt))
        if not self.use_geometry_features:
            return p
        geo_mlp = self.surface_geometry_mlp if is_surface \
            else self.volume_geometry_mlp
        e = self.region_embed(1 if is_surface else 0, region_dim2, region_dim3)
        g_in = torch.cat([sdf.float().unsqueeze(-1), sdf_dir.float(),
                          support.float(), e], dim=-1)
        return p + geo_mlp(g_in)

    # ---- anchors-only forward (training + sampled validation) ----------

    def forward(
        self,
        surface_position: torch.Tensor,    # [B, Ns, 3] cache /550 frame
        volume_position: torch.Tensor,     # [B, Nv, 3]
        surface_sdf: torch.Tensor | None = None,          # [B, Ns]
        surface_sdf_dir: torch.Tensor | None = None,      # [B, Ns, 3]
        surface_support: torch.Tensor | None = None,      # [B, Ns, 6]
        surface_region_dim2: torch.Tensor | None = None,  # [B, Ns] int
        surface_region_dim3: torch.Tensor | None = None,  # [B, Ns] int
        volume_sdf: torch.Tensor | None = None,
        volume_sdf_dir: torch.Tensor | None = None,
        volume_support: torch.Tensor | None = None,
        volume_region_dim2: torch.Tensor | None = None,
        volume_region_dim3: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Anchors-only. Returns (pred_surface [B, Ns, 1],
        pred_volume [B, Nv, 5]) in z-score space."""
        ns = surface_position.size(1)
        nv = volume_position.size(1)
        pos_s = self._abupt_coords(surface_position)
        pos_v = self._abupt_coords(volume_position)
        freqs_s = self.rope(pos_s)
        freqs_v = self.rope(pos_v)
        freqs_all = (torch.cat([freqs_s[0], freqs_v[0]], dim=1),
                     torch.cat([freqs_s[1], freqs_v[1]], dim=1))

        z_s = self._branch_tokens(pos_s, True, surface_sdf, surface_sdf_dir,
                                  surface_support, surface_region_dim2,
                                  surface_region_dim3)
        z_v = self._branch_tokens(pos_v, False, volume_sdf, volume_sdf_dir,
                                  volume_support, volume_region_dim2,
                                  volume_region_dim3)

        x = torch.cat([z_s, z_v], dim=1)
        split_size = [ns, nv]
        for block in self.blocks:
            x = block(x, attn_kwargs={"split_size": split_size,
                                      "freqs": freqs_all})
        x_s, x_v = x.split(split_size, dim=1)
        for block in self.surface_blocks:
            x_s = block(x_s, attn_kwargs={"freqs": freqs_s})
        for block in self.volume_blocks:
            x_v = block(x_v, attn_kwargs={"freqs": freqs_v})
        return self.surface_decoder(x_s), self.volume_decoder(x_v)

    # ---- anchors + queries forward (full-field inference only) ---------

    def forward_with_queries(
        self,
        surface_position: torch.Tensor,    # [B, Ns_a + Ns_q, 3], anchors first
        volume_position: torch.Tensor,     # [B, Nv_a + Nv_q, 3], anchors first
        num_surface_anchors: int,
        num_volume_anchors: int,
        surface_sdf: torch.Tensor | None = None,
        surface_sdf_dir: torch.Tensor | None = None,
        surface_support: torch.Tensor | None = None,
        surface_region_dim2: torch.Tensor | None = None,
        surface_region_dim3: torch.Tensor | None = None,
        volume_sdf: torch.Tensor | None = None,
        volume_sdf_dir: torch.Tensor | None = None,
        volume_support: torch.Tensor | None = None,
        volume_region_dim2: torch.Tensor | None = None,
        volume_region_dim3: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The native anchor/query mechanism for full-field inference: the
        first num_*_anchors rows of each branch are anchors (identical
        compute path to the anchors-only forward — queries never influence
        them), the rest are query tokens that only read anchor K/V. Returns
        the FULL per-branch sequences (anchors first, then queries); the
        caller slices and strips its own chunk padding."""
        ns_a = int(num_surface_anchors)
        nv_a = int(num_volume_anchors)
        ns_q = surface_position.size(1) - ns_a
        nv_q = volume_position.size(1) - nv_a
        pos_s = self._abupt_coords(surface_position)
        pos_v = self._abupt_coords(volume_position)
        freqs_s = self.rope(pos_s)
        freqs_v = self.rope(pos_v)
        freqs_all = (torch.cat([freqs_s[0], freqs_v[0]], dim=1),
                     torch.cat([freqs_s[1], freqs_v[1]], dim=1))

        z_s = self._branch_tokens(pos_s, True, surface_sdf, surface_sdf_dir,
                                  surface_support, surface_region_dim2,
                                  surface_region_dim3)
        z_v = self._branch_tokens(pos_v, False, volume_sdf, volume_sdf_dir,
                                  volume_support, volume_region_dim2,
                                  volume_region_dim3)

        x = torch.cat([z_s, z_v], dim=1)
        split_size = [ns_a, ns_q, nv_a, nv_q]
        for block in self.blocks:
            x = block(x, attn_kwargs={"split_size": split_size,
                                      "freqs": freqs_all})
        x_s, x_v = x.split([ns_a + ns_q, nv_a + nv_q], dim=1)
        for block in self.surface_blocks:
            x_s = block(x_s, attn_kwargs={"freqs": freqs_s,
                                          "num_anchor_tokens": ns_a})
        for block in self.volume_blocks:
            x_v = block(x_v, attn_kwargs={"freqs": freqs_v,
                                          "num_anchor_tokens": nv_a})
        return self.surface_decoder(x_s), self.volume_decoder(x_v)


    # ---- anchor-K/V-cached full-field inference -------------------------
    # encode_anchor_cache runs the anchors ONCE (identical math to the
    # anchors-only forward) while capturing every block's post-RoPE anchor
    # K/V; decode_queries_cached then runs ONLY the query rows of a chunk
    # against those caches. Equivalent to forward_with_queries (queries only
    # ever read anchor K/V — see AnchorAttention), without re-encoding the
    # 2 x n_anchors anchor tokens on every chunk. forward_with_queries is
    # KEPT as the equivalence oracle; the generic post uses this path.

    def encode_anchor_cache(
        self,
        surface_position: torch.Tensor,    # [B, Ns, 3] cache /550 frame
        volume_position: torch.Tensor,     # [B, Nv, 3]
        surface_sdf: torch.Tensor | None = None,
        surface_sdf_dir: torch.Tensor | None = None,
        surface_support: torch.Tensor | None = None,
        surface_region_dim2: torch.Tensor | None = None,
        surface_region_dim3: torch.Tensor | None = None,
        volume_sdf: torch.Tensor | None = None,
        volume_sdf_dir: torch.Tensor | None = None,
        volume_support: torch.Tensor | None = None,
        volume_region_dim2: torch.Tensor | None = None,
        volume_region_dim3: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Returns (pred_surface_anchors, pred_volume_anchors, cache)."""
        ns = surface_position.size(1)
        nv = volume_position.size(1)
        pos_s = self._abupt_coords(surface_position)
        pos_v = self._abupt_coords(volume_position)
        freqs_s = self.rope(pos_s)
        freqs_v = self.rope(pos_v)
        freqs_all = (torch.cat([freqs_s[0], freqs_v[0]], dim=1),
                     torch.cat([freqs_s[1], freqs_v[1]], dim=1))

        z_s = self._branch_tokens(pos_s, True, surface_sdf, surface_sdf_dir,
                                  surface_support, surface_region_dim2,
                                  surface_region_dim3)
        z_v = self._branch_tokens(pos_v, False, volume_sdf, volume_sdf_dir,
                                  volume_support, volume_region_dim2,
                                  volume_region_dim3)

        x = torch.cat([z_s, z_v], dim=1)
        split_size = [ns, nv]
        trunk_kv = []
        for block in self.blocks:
            normed = block.norm1(x)
            k_s, v_s = block.attn.kv_from(normed[:, :ns], freqs_s)
            k_v, v_v = block.attn.kv_from(normed[:, ns:], freqs_v)
            trunk_kv.append((k_s, v_s, k_v, v_v))
            x = block(x, attn_kwargs={"split_size": split_size,
                                      "freqs": freqs_all})
        x_s, x_v = x.split(split_size, dim=1)
        surf_kv, vol_kv = [], []
        for block in self.surface_blocks:
            surf_kv.append(block.attn.kv_from(block.norm1(x_s), freqs_s))
            x_s = block(x_s, attn_kwargs={"freqs": freqs_s})
        for block in self.volume_blocks:
            vol_kv.append(block.attn.kv_from(block.norm1(x_v), freqs_v))
            x_v = block(x_v, attn_kwargs={"freqs": freqs_v})
        cache = {"trunk": trunk_kv, "surface": surf_kv, "volume": vol_kv}
        return self.surface_decoder(x_s), self.volume_decoder(x_v), cache

    def decode_queries_cached(
        self,
        cache: dict,
        surface_position: torch.Tensor,    # [B, Sq, 3] QUERY rows only
        volume_position: torch.Tensor,     # [B, Vq, 3]
        surface_sdf: torch.Tensor | None = None,
        surface_sdf_dir: torch.Tensor | None = None,
        surface_support: torch.Tensor | None = None,
        surface_region_dim2: torch.Tensor | None = None,
        surface_region_dim3: torch.Tensor | None = None,
        volume_sdf: torch.Tensor | None = None,
        volume_sdf_dir: torch.Tensor | None = None,
        volume_support: torch.Tensor | None = None,
        volume_region_dim2: torch.Tensor | None = None,
        volume_region_dim3: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos_s = self._abupt_coords(surface_position)
        pos_v = self._abupt_coords(volume_position)
        freqs_qs = self.rope(pos_s)
        freqs_qv = self.rope(pos_v)
        x_s = self._branch_tokens(pos_s, True, surface_sdf, surface_sdf_dir,
                                  surface_support, surface_region_dim2,
                                  surface_region_dim3)
        x_v = self._branch_tokens(pos_v, False, volume_sdf, volume_sdf_dir,
                                  volume_support, volume_region_dim2,
                                  volume_region_dim3)
        for ch, block, kv in zip(self.shared_pattern, self.blocks,
                                 cache["trunk"]):
            k_s, v_s, k_v, v_v = kv
            if ch == "s":       # queries read their OWN branch's anchors
                kv_for_s, kv_for_v = (k_s, v_s), (k_v, v_v)
            else:               # 'c': queries read the OTHER branch's anchors
                kv_for_s, kv_for_v = (k_v, v_v), (k_s, v_s)
            x_s = x_s + block.attn.query_attend(block.norm1(x_s),
                                                *kv_for_s, freqs_qs)
            x_s = x_s + block.mlp(block.norm2(x_s))
            x_v = x_v + block.attn.query_attend(block.norm1(x_v),
                                                *kv_for_v, freqs_qv)
            x_v = x_v + block.mlp(block.norm2(x_v))
        for block, (k, v) in zip(self.surface_blocks, cache["surface"]):
            x_s = x_s + block.attn.query_attend(block.norm1(x_s), k, v,
                                                freqs_qs)
            x_s = x_s + block.mlp(block.norm2(x_s))
        for block, (k, v) in zip(self.volume_blocks, cache["volume"]):
            x_v = x_v + block.attn.query_attend(block.norm1(x_v), k, v,
                                                freqs_qv)
            x_v = x_v + block.mlp(block.norm2(x_v))
        return self.surface_decoder(x_s), self.volume_decoder(x_v)


def build_abupt_model(cfg) -> AnchoredBranchedUPT:
    """Construct the statically-configured model instance from cfg.abupt.
    Output channel counts come from the UrbanWind targets (5 volume
    channels, surface pressure)."""
    from ..data.estate import SURF_CHANNEL_NAMES, VOL_CHANNEL_NAMES

    ac = cfg.abupt
    return AnchoredBranchedUPT(
        use_geometry_features=ac.geometry_features,
        dim=ac.hidden_dim,
        num_heads=ac.num_heads,
        shared_blocks=ac.shared_blocks,
        num_surface_blocks=ac.num_surface_blocks,
        num_volume_blocks=ac.num_volume_blocks,
        output_dim_surface=len(SURF_CHANNEL_NAMES),
        output_dim_volume=len(VOL_CHANNEL_NAMES),
        pos_offset=tuple(ac.pos_offset),
        pos_scale=ac.pos_scale,
    )
