"""Erwin transformer — port of the author implementation (erwin-main
`models/erwin.py`, ICML 2025), integrated for this repo.

Verbatim from the author code: SwiGLU, MPNN message passing (scatter-mean
aggregation), ErwinEmbedding, Node dataclass, BallPooling / BallUnpooling
(BatchNorm1d), BallMSA (distance attention bias sigma_att + relative-
position embedding), ErwinTransformerBlock (RMSNorm), BasicLayer with the
alternating rotated-tree blocks, the U-Net encoder/bottleneck/decoder
assembly, trunc_normal(0.02) init.

Integration deviations, disclosed:
  - torch_cluster / balltree-erwin dependencies removed: the kNN graph and
    the ball tree (+ rotated variants) arrive as INPUTS, built on CPU in
    loader workers by model/erwin/balltree.py — construction never sits in
    the compiled graph (and never silently rebuilds in-forward: forward
    asserts the tree inputs are provided);
  - the MPNN's scatter_mean is replaced by the equivalent
    scatter-free grouped mean (grouped_mean) — same math under
    the kNN edge-layout contract, fp64-gated forward AND
    gradients, and free of index_put(accumulate) so the graph
    stays CUDA-Graph capturable; the author's version is kept
    as the equivalence oracle;
  - everything else (heads, depths, strides, ball sizes, rotate, mlp_ratio)
    comes from the config.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def scatter_mean(src: torch.Tensor, idx: torch.Tensor, num_receivers: int):
    """REFERENCE implementation (the author's): scatter-add into receivers,
    divide by the per-receiver count. Kept ONLY as the equivalence oracle
    for grouped_mean — the model itself never calls it, because both
    index_add_ calls are index_put(accumulate) ops and inductor refuses
    CUDA-Graph capture for any graph containing them."""
    result = torch.zeros(num_receivers, src.size(1), dtype=src.dtype,
                         device=src.device)
    count = torch.zeros(num_receivers, dtype=torch.long, device=src.device)
    result.index_add_(0, idx, src)
    count.index_add_(0, idx, torch.ones_like(idx, dtype=torch.long))
    return result / count.unsqueeze(1).clamp(min=1)


def grouped_mean(src: torch.Tensor, num_receivers: int) -> torch.Tensor:
    """The scatter-free equivalent used by the model in BOTH training and
    inference (one implementation — no train/test operator divergence).

    Valid under the kNN edge-layout contract that balltree.knn_edges
    constructs and asserts: every receiver gets EXACTLY k = E // N edges,
    contiguous, in receiver order, i.e. edge_index[1] == repeat(arange(N),
    k). Then the scatter-add of src into receiver i sums exactly rows
    [i*k, (i+1)*k) and every count is k, so

        scatter_mean(src, idx, N) == src.reshape(N, k, C).mean(1)

    elementwise (gated in fp64, forward AND gradients, in
    tests/test_gates.py). Being a plain reshape+mean it contains no
    index_put(accumulate), so the graph stays CUDA-Graph capturable."""
    e, c = src.shape[0], src.shape[1]
    assert e % num_receivers == 0, (
        f"grouped_mean: {e} edges is not a whole multiple of "
        f"{num_receivers} receivers — the kNN layout contract is broken")
    return src.reshape(num_receivers, e // num_receivers, c).mean(dim=1)


class SwiGLU(nn.Module):
    """W_3 SiLU(W_1 x) * W_2 x"""

    def __init__(self, in_dim: int, dim: int):
        super().__init__()
        self.w1 = nn.Linear(in_dim, dim)
        self.w2 = nn.Linear(in_dim, dim)
        self.w3 = nn.Linear(dim, in_dim)

    def forward(self, x: torch.Tensor):
        return self.w3(self.w2(x) * F.silu(self.w1(x)))


class MPNN(nn.Module):
    """Message passing (Gilmer et al., 2017), author implementation."""

    def __init__(self, dim: int, mp_steps: int, dimensionality: int = 3):
        super().__init__()
        self.message_fns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * dim + dimensionality, dim),
                nn.GELU(),
                nn.LayerNorm(dim),
            ) for _ in range(mp_steps)
        ])
        self.update_fns = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * dim, dim), nn.LayerNorm(dim))
            for _ in range(mp_steps)
        ])

    def layer(self, message_fn, update_fn, h, edge_attr, edge_index):
        row, col = edge_index
        messages = message_fn(torch.cat([h[row], h[col], edge_attr], dim=-1))
        # scatter-free grouped mean (see grouped_mean): identical math under
        # the kNN edge-layout contract, and CUDA-Graph capturable
        message = grouped_mean(messages, h.size(0))
        update = update_fn(torch.cat([h, message], dim=-1))
        return h + update

    @torch.no_grad()
    def compute_edge_attr(self, pos, edge_index):
        return pos[edge_index[0]] - pos[edge_index[1]]

    def forward(self, x, pos, edge_index):
        edge_attr = self.compute_edge_attr(pos, edge_index)
        for message_fn, update_fn in zip(self.message_fns, self.update_fns):
            x = self.layer(message_fn, update_fn, x, edge_attr, edge_index)
        return x


class ErwinEmbedding(nn.Module):
    """Linear projection -> MPNN."""

    def __init__(self, in_dim: int, dim: int, mp_steps: int,
                 dimensionality: int = 3):
        super().__init__()
        self.mp_steps = mp_steps
        self.embed_fn = nn.Linear(in_dim, dim)
        self.mpnn = MPNN(dim, mp_steps, dimensionality)

    def forward(self, x, pos, edge_index):
        x = self.embed_fn(x)
        return self.mpnn(x, pos, edge_index) if self.mp_steps > 0 else x


@dataclass
class Node:
    x: torch.Tensor
    pos: torch.Tensor
    batch_idx: torch.Tensor
    tree_idx_rot: torch.Tensor | None = None
    children: "Node | None" = None


class BallPooling(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, stride: int,
                 dimensionality: int = 3):
        super().__init__()
        self.stride = stride
        self.proj = nn.Linear(stride * in_dim + stride * dimensionality,
                              out_dim)
        self.norm = nn.BatchNorm1d(out_dim)

    def forward(self, node: Node) -> Node:
        if self.stride == 1:
            return Node(x=node.x, pos=node.pos, batch_idx=node.batch_idx,
                        children=node)
        with torch.no_grad():
            s_ = self.stride
            batch_idx = node.batch_idx[::s_]
            pos = node.pos.view(-1, s_, node.pos.shape[-1])
            centers = pos.mean(dim=1)
            rel_pos = (pos - centers[:, None]).reshape(pos.shape[0], -1)
        x = torch.cat([node.x.reshape(-1, s_ * node.x.shape[-1]), rel_pos],
                      dim=1)
        x = self.norm(self.proj(x))
        return Node(x=x, pos=centers, batch_idx=batch_idx, children=node)


class BallUnpooling(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, stride: int,
                 dimensionality: int = 3):
        super().__init__()
        self.stride = stride
        self.proj = nn.Linear(in_dim + stride * dimensionality,
                              stride * out_dim)
        self.norm = nn.BatchNorm1d(out_dim)

    def forward(self, node: Node) -> Node:
        with torch.no_grad():
            s_ = self.stride
            rel_pos = node.children.pos.view(-1, s_,
                                             node.children.pos.shape[-1])                 - node.pos[:, None]
            rel_pos = rel_pos.reshape(rel_pos.shape[0], -1)
        x = torch.cat([node.x, rel_pos], dim=-1)
        up = self.proj(x)
        node.children.x = self.norm(
            node.children.x + up.reshape(-1, up.shape[-1] // s_))
        return node.children


class BallMSA(nn.Module):
    """Ball multi-head self-attention with the distance bias."""

    def __init__(self, dim: int, num_heads: int, ball_size: int,
                 dimensionality: int = 3):
        super().__init__()
        self.num_heads = num_heads
        self.ball_size = ball_size
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.pe_proj = nn.Linear(dimensionality, dim)
        self.sigma_att = nn.Parameter(-1 + 0.01 * torch.randn(
            (1, num_heads, 1, 1)))

    @torch.no_grad()
    def create_attention_mask(self, pos: torch.Tensor):
        pos = pos.view(-1, self.ball_size, pos.shape[-1])
        return torch.cdist(pos, pos, p=2).unsqueeze(1)

    @torch.no_grad()
    def compute_rel_pos(self, pos: torch.Tensor):
        num_balls, dim = pos.shape[0] // self.ball_size, pos.shape[1]
        pos = pos.view(num_balls, self.ball_size, dim)
        return (pos - pos.mean(dim=1, keepdim=True)).view(-1, dim)

    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        x = x + self.pe_proj(self.compute_rel_pos(pos))
        H, m = self.num_heads, self.ball_size
        qkv = self.qkv(x)                       # [(n m), 3*dim]
        E = qkv.shape[-1] // (3 * H)
        # author layout "(n m) (H E K) -> K n H m E": the LAST axis of the
        # projection is (H, E, K), so unpack in that order then move K out
        qkv = qkv.view(-1, m, H, E, 3).permute(4, 0, 2, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]        # [n, H, m, E]
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=self.sigma_att * self.create_attention_mask(pos))
        x = x.permute(0, 2, 1, 3).reshape(-1, H * E)
        return self.proj(x)


class ErwinTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ball_size: int,
                 mlp_ratio: int, dimensionality: int = 3):
        super().__init__()
        self.ball_size = ball_size
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        self.BMSA = BallMSA(dim, num_heads, ball_size, dimensionality)
        self.swiglu = SwiGLU(dim, dim * mlp_ratio)

    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        x = x + self.BMSA(self.norm1(x), pos)
        return x + self.swiglu(self.norm2(x))


class BasicLayer(nn.Module):
    def __init__(self, direction: Literal["down", "up", None], depth: int,
                 stride: int | None, in_dim: int, out_dim: int,
                 num_heads: int, ball_size: int, mlp_ratio: int,
                 rotate: bool, dimensionality: int = 3):
        super().__init__()
        hidden_dim = in_dim if direction == "down" else out_dim
        self.blocks = nn.ModuleList([
            ErwinTransformerBlock(hidden_dim, num_heads, ball_size,
                                  mlp_ratio, dimensionality)
            for _ in range(depth)])
        self.rotate = [i % 2 for i in range(depth)] if rotate \
            else [False] * depth
        self.pool = lambda node: node
        self.unpool = lambda node: node
        if direction == "down" and stride is not None:
            self.pool = BallPooling(hidden_dim, out_dim, stride,
                                    dimensionality)
        elif direction == "up" and stride is not None:
            self.unpool = BallUnpooling(in_dim, hidden_dim, stride,
                                        dimensionality)

    def forward(self, node: Node) -> Node:
        node = self.unpool(node)
        if len(self.rotate) > 1 and self.rotate[1]:
            assert node.tree_idx_rot is not None, \
                "tree_idx_rot must be provided for rotation"
            tree_idx_rot_inv = torch.argsort(node.tree_idx_rot)
        for rotate, blk in zip(self.rotate, self.blocks):
            if rotate:
                node.x = blk(node.x[node.tree_idx_rot],
                             node.pos[node.tree_idx_rot])[tree_idx_rot_inv]
            else:
                node.x = blk(node.x, node.pos)
        return self.pool(node)


class ErwinTransformer(nn.Module):
    """Author U-Net assembly. Tree/kNN inputs are REQUIRED (built outside)."""

    def __init__(self, c_in: int, c_hidden: List, ball_sizes: List,
                 enc_num_heads: List, enc_depths: List, dec_num_heads: List,
                 dec_depths: List, strides: List, rotate: int,
                 decode: bool = True, mlp_ratio: int = 4,
                 dimensionality: int = 3, mp_steps: int = 3):
        super().__init__()
        assert len(enc_num_heads) == len(enc_depths) == len(ball_sizes) \
            == len(c_hidden)
        assert len(dec_num_heads) == len(dec_depths) == len(strides) \
            == len(c_hidden) - 1
        self.rotate = rotate
        self.decode = decode
        self.ball_sizes = list(ball_sizes)
        self.strides = list(strides)
        self.dimensionality = dimensionality
        self.embed = ErwinEmbedding(c_in, c_hidden[0], mp_steps,
                                    dimensionality)
        num_layers = len(enc_depths) - 1
        self.encoder = nn.ModuleList()
        for i in range(num_layers):
            self.encoder.append(BasicLayer(
                direction="down", depth=enc_depths[i], stride=strides[i],
                in_dim=c_hidden[i], out_dim=c_hidden[i + 1],
                num_heads=enc_num_heads[i], ball_size=ball_sizes[i],
                rotate=rotate > 0, mlp_ratio=mlp_ratio,
                dimensionality=dimensionality))
        self.bottleneck = BasicLayer(
            direction=None, depth=enc_depths[-1], stride=None,
            in_dim=c_hidden[-1], out_dim=c_hidden[-1],
            num_heads=enc_num_heads[-1], ball_size=ball_sizes[-1],
            rotate=rotate > 0, mlp_ratio=mlp_ratio,
            dimensionality=dimensionality)
        if decode:
            self.decoder = nn.ModuleList()
            for i in range(num_layers - 1, -1, -1):
                self.decoder.append(BasicLayer(
                    direction="up", depth=dec_depths[i], stride=strides[i],
                    in_dim=c_hidden[i + 1], out_dim=c_hidden[i],
                    num_heads=dec_num_heads[i], ball_size=ball_sizes[i],
                    rotate=rotate > 0, mlp_ratio=mlp_ratio,
                    dimensionality=dimensionality))
        self.in_dim = c_in
        self.out_dim = c_hidden[0]
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, mean=0., std=0.02, a=-2., b=2.)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, node_features: torch.Tensor,
                node_positions: torch.Tensor, batch_idx: torch.Tensor,
                edge_index: torch.Tensor | None, tree_idx: torch.Tensor,
                tree_mask: torch.Tensor, tree_idx_rot: list,
                restore_idx: torch.Tensor | None = None, **kwargs):
        """restore_idx: the STATIC unpad-and-reorder permutation from
        balltree.restore_index (int64 [N], precomputed on CPU next to
        the tree). Every compiled call path must pass it: the boolean
        branch at the tail contains aten.nonzero (dynamic output shape)
        and serves uncompiled eager callers only.
        """
        assert tree_idx is not None and tree_mask is not None, \
            "ball tree must be built OUTSIDE the model (balltree.build_tree)"
        if self.embed.mp_steps:
            assert edge_index is not None, \
                "edge_index required when mp_steps > 0 (balltree.knn_edges)"
        x = self.embed(node_features, node_positions, edge_index)
        rot = list(tree_idx_rot)
        node = Node(x=x[tree_idx], pos=node_positions[tree_idx],
                    batch_idx=batch_idx[tree_idx], tree_idx_rot=None)
        for layer in self.encoder:
            node.tree_idx_rot = rot.pop(0)
            node = layer(node)
        node.tree_idx_rot = rot.pop(0)
        node = self.bottleneck(node)
        if self.decode:
            for layer in self.decoder:
                node = layer(node)
            if restore_idx is not None:
                # static shape -> compiles under fullgraph + dynamic=False
                return node.x.index_select(0, restore_idx)
            # eager path (boolean mask -> aten.nonzero), never reached
            # from a compiled producer
            return node.x[tree_mask][torch.argsort(tree_idx[tree_mask])]
        return node.x


class GaussianRFF(nn.Module):
    """The author repo's ShapeNet positional encoder: a fixed random linear
    projection with N(0, 1/sigma) weights, features = [cos || sin]. (The
    airfoil-experiment encoder in the same repo only ever encodes the first
    coordinate and pushes frequencies to 2^63*pi — numerically meaningless —
    so this repo standardizes on the ShapeNet encoder for both datasets;
    disclosed in the paper's deviations appendix.)"""

    def __init__(self, num_features: int, dimensionality: int,
                 sigma: float = 1.0):
        super().__init__()
        assert num_features % 2 == 0
        self.linear = nn.Linear(dimensionality, num_features // 2, bias=False)
        nn.init.normal_(self.linear.weight, 0.0, 1.0 / sigma)
        self.linear.weight.requires_grad_(False)   # fixed random features

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        proj = self.linear(pos)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


class ErwinPointModel(nn.Module):
    """RFF(pos) [+ extra channels] -> ErwinTransformer -> two heads.
    Rows keep the sample's [volume; surface] layout; heads split statically
    via n_vol. The pred-head shape (Linear -> GELU -> Linear) is the author
    wrapper's."""

    def __init__(self, ecfg, out_dim_vol: int, extra_in: int = 0):
        super().__init__()
        self.__name__ = "Erwin"
        self.extra_in = int(extra_in)
        self.pos_enc = GaussianRFF(int(ecfg.c_in), int(ecfg.dimensionality),
                                   float(ecfg.rff_sigma))
        self.main = ErwinTransformer(
            c_in=int(ecfg.c_in) + self.extra_in,
            c_hidden=list(ecfg.c_hidden), ball_sizes=list(ecfg.ball_sizes),
            enc_num_heads=list(ecfg.enc_num_heads),
            enc_depths=list(ecfg.enc_depths),
            dec_num_heads=list(ecfg.dec_num_heads),
            dec_depths=list(ecfg.dec_depths), strides=list(ecfg.strides),
            rotate=int(ecfg.rotate), mlp_ratio=int(ecfg.mlp_ratio),
            dimensionality=int(ecfg.dimensionality),
            mp_steps=int(ecfg.mp_steps))
        d = self.main.out_dim
        self.vol_head = nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                      nn.Linear(d, out_dim_vol))
        self.surf_head = nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                       nn.Linear(d, 1))

    def forward(self, pos: torch.Tensor, extra: torch.Tensor | None,
                edge_index, tree_idx, tree_mask, tree_idx_rot,
                restore_idx: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Both heads on EVERY row ([1, N, C_vol], [1, N, 1]); the loss
        masks rows by is_volume — the uniform-sampling protocol contract
        shared with the Transolver recipe (rows arrive in pool order, mixed
        volume/surface; no stratified layout needed)."""
        feats = self.pos_enc(pos)
        if self.extra_in:
            feats = torch.cat([feats, extra], dim=-1)
        batch_idx = torch.zeros(pos.shape[0], dtype=torch.long,
                                device=pos.device)
        h = self.main(feats, pos, batch_idx, edge_index, tree_idx,
                      tree_mask, tree_idx_rot, restore_idx)
        return (self.vol_head(h).unsqueeze(0),
                self.surf_head(h).unsqueeze(0))


def build_erwin_model(ecfg, out_dim_vol: int,
                      extra_in: int = 0) -> ErwinPointModel:
    return ErwinPointModel(ecfg, out_dim_vol=out_dim_vol, extra_in=extra_in)
