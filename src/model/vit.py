"""Global processor: a pre-norm Transformer over the latent tokens.

Blocks use QK-RMSNorm, 3D rotary position embeddings (derived on the GPU
from the latent positions), full scaled-dot-product attention and SwiGLU
feed-forward layers, followed by a final RMSNorm. Optional components:

  adaLN       the five morphology conditioners modulate every block through
              a shared trunk Linear(5, 64) -> SiLU -> Linear(64, 64) and one
              zero-initialized head per block producing (gamma1, beta1,
              alpha1, gamma2, beta2, alpha2); at initialization every block
              is the identity modulation.
  registers   learned tokens appended to the sequence (zero rotary angle).
  long skips  gated skips between mirrored early and late blocks,
              x = x + gate * proj(RMSNorm(cached input)), gate initialized
              to 0.01.

forward returns the full [B, L + R, D] sequence, registers included.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import RMSNorm
from .rope import apply_rope, precompute_rope_3d


def _init_linear(module: nn.Linear):
    nn.init.trunc_normal_(module.weight, std=0.02)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.gate_up = nn.Linear(dim, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.drop(self.down(F.silu(gate) * up))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int,
                 ffn_hidden: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.norm1 = RMSNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, ffn_hidden, dropout)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor,
                rope_sin: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        """mod [B, 6*dim]: per-layer adaLN modulation (zero at init)."""
        B, T, D = x.shape
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = mod.chunk(6, dim=-1)
        h = self.norm1(x) * (1 + gamma1[:, None, :]) + beta1[:, None, :]
        qkv = self.qkv(h).view(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each [B, H, T, hd]
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rope(q, k, rope_cos, rope_sin)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        x = x + alpha1[:, None, :] * self.o_proj(attn.transpose(1, 2).reshape(B, T, D))
        h2 = self.norm2(x) * (1 + gamma2[:, None, :]) + beta2[:, None, :]
        x = x + alpha2[:, None, :] * self.ffn(h2)
        return x


class ViTProcessor(nn.Module):
    def __init__(self, dim: int, n_layers: int, num_heads: int, head_dim: int,
                 ffn_hidden: int, n_register: int, dropout: float = 0.1,
                 rope_base: float = 100.0, rope_scale_xy: float = 64.0,
                 rope_scale_z: float = 20.0,
                 rope_dims: tuple[int, int, int] = (36, 36, 24),
                 global_cond_dim: int = 5, adaln_hidden: int = 64,
                 use_adaln: bool = True, use_unet_skips: bool = True):
        super().__init__()
        self.n_register = n_register
        self.head_dim = head_dim
        self.rope_base = rope_base
        self.rope_scale_xy = rope_scale_xy
        self.rope_scale_z = rope_scale_z
        self.rope_dims = rope_dims
        self.register = nn.Parameter(torch.randn(n_register, dim) * 0.02) \
            if n_register > 0 else None
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, head_dim, ffn_hidden, dropout)
            for _ in range(n_layers)
        ])
        self.n_layers = n_layers
        # use_unet_skips=False -> n_skip=0 -> skip_pairs/save_set/consume_map
        # all become empty, so forward()'s `if i in ...` guards fall through
        # for every layer with no branch of their own.
        n_skip = (n_layers // 2) if use_unet_skips else 0
        # pair (i, n_layers - 1 - i): cache the INPUT of encoder-side layer i,
        # add its gated projection into the INPUT of the mirror layer.
        self.skip_pairs = [(i, n_layers - 1 - i) for i in range(n_skip)]
        self.save_set = {i for i, _ in self.skip_pairs}
        self.consume_map = {j: si for si, (_, j) in enumerate(self.skip_pairs)}
        self.skip_proj = nn.ModuleList([nn.Linear(dim, dim) for _ in self.skip_pairs])
        self.skip_norm = nn.ModuleList([RMSNorm(dim) for _ in self.skip_pairs])
        for w in self.skip_proj:
            _init_linear(w)
        # small-scale, LayerScale-style gate — an untested additive
        # correction on top of an already-complete residual stream must
        # start small. NOT zero, NOT trunc_normal(0.02).
        self.skip_gate = nn.Parameter(
            torch.full((len(self.skip_pairs),), 1e-2)) \
            if self.skip_pairs else None
        self.final_norm = RMSNorm(dim)

        # adaLN: shared trunk (one SiLU inside, one more before the heads),
        # zero-initialized per-block heads (gamma = beta = alpha = 0 at init)
        self.use_adaln = use_adaln
        if use_adaln:
            self.adaln_trunk = nn.Sequential(
                nn.Linear(global_cond_dim, adaln_hidden), nn.SiLU(),
                nn.Linear(adaln_hidden, adaln_hidden))
            self.adaln_heads = nn.ModuleList([
                nn.Linear(adaln_hidden, 6 * dim) for _ in range(n_layers)])
            for head in self.adaln_heads:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        else:
            # IDENTITY modulation, NOT zeros. mod order is (gamma1, beta1,
            # alpha1, gamma2, beta2, alpha2) and TransformerBlock computes
            # norm*(1+gamma)+beta then x + alpha*branch, so identity needs
            # gamma=beta=0 and alpha=1. A zero mod would set alpha=0 and
            # freeze the ViT as a permanent no-op — with adaLN ON that is
            # only the INIT state, which training then opens up.
            self.adaln_trunk = None
            self.adaln_heads = None
            ident = torch.cat([torch.zeros(dim), torch.zeros(dim),
                               torch.ones(dim), torch.zeros(dim),
                               torch.zeros(dim), torch.ones(dim)])
            self.register_buffer("_identity_mod", ident[None, :])

    def forward(self, leaf_tokens: torch.Tensor,
                leaf_pos_norm: torch.Tensor,
                global_cond: torch.Tensor) -> torch.Tensor:
        """leaf_tokens [B, L, D]; leaf_pos_norm [B, L, 3] (the rotary
        source); global_cond [B, 5] (global_morph_norm)
        -> [B, L + R, D], registers included."""
        B, L, D = leaf_tokens.shape
        if self.n_register > 0:
            reg = self.register.unsqueeze(0).expand(B, -1, -1).to(leaf_tokens.dtype)
            x = torch.cat([leaf_tokens, reg], dim=1)  # [B, L + R, D]
        else:
            x = leaf_tokens                          # [B, L, D]

        # sin/cos in fp32 (the scaled coordinate loses precision in bf16),
        # cast to the compute dtype once for all blocks
        rope_cos, rope_sin = precompute_rope_3d(
            leaf_pos_norm, self.head_dim, self.rope_dims, self.rope_scale_xy,
            self.rope_scale_z, self.rope_base, self.n_register)
        rope_cos = rope_cos.to(x.dtype)
        rope_sin = rope_sin.to(x.dtype)

        if self.use_adaln:
            h_act = F.silu(self.adaln_trunk(global_cond))  # [B, adaln_hidden]
        else:
            mod_ident = self._identity_mod.to(x.dtype).expand(B, -1)

        cache: list[torch.Tensor] = []
        for i, block in enumerate(self.blocks):
            if i in self.consume_map:
                si = self.consume_map[i]
                x = x + self.skip_gate[si] * self.skip_proj[si](self.skip_norm[si](cache[si]))
            if i in self.save_set:
                cache.append(x)
            x = block(x, rope_cos, rope_sin,
                      self.adaln_heads[i](h_act) if self.use_adaln
                      else mod_ident)
        x = self.final_norm(x)
        return x  # [B, L+R, D] — decoder stage 2 reads the register slice
