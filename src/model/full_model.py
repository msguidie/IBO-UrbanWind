"""IBO (Importance-Balanced Operator): the full model.

encode() runs the local encoder (PointNet over each leaf's members + the
leaf token projection) and the global processor once per case; decode()
predicts any set of query points from their k nearest latent points. The
two are separate so that full-field inference encodes once and decodes in
fixed-size chunks through the same code path as training. Every forward
takes explicit tensors, which keeps torch.compile(fullgraph=True) graphs
clean.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig
from .decoder import TwoStageCrossAttentionDecoder
from .encoder import Encoder
from .pointnet import PointNetEncoder
from .vit import ViTProcessor


class IBO(nn.Module):
    def __init__(self, mcfg: ModelConfig):
        super().__init__()
        mcfg.validate()
        self.pointnet = PointNetEncoder(mcfg.pn_in_dim, 64, mcfg.pn_out_dim)
        self.encoder = Encoder(mcfg.encoder_in_dim, mcfg.latent_dim)
        self.vit = ViTProcessor(
            mcfg.latent_dim, mcfg.n_vit_layers, mcfg.num_heads, mcfg.head_dim,
            mcfg.ffn_hidden, mcfg.n_register, mcfg.dropout,
            rope_base=mcfg.rope_base, rope_scale_xy=mcfg.rope_scale_xy,
            rope_scale_z=mcfg.rope_scale_z, rope_dims=mcfg.rope_dims,
            global_cond_dim=mcfg.global_cond_dim, adaln_hidden=mcfg.adaln_hidden,
            use_adaln=not mcfg.disable_adaln,
            use_unet_skips=not mcfg.disable_unet_skips)
        self.decoder = TwoStageCrossAttentionDecoder(
            mcfg.latent_dim, mcfg.decoder_dh, mcfg.idw_k,
            mcfg.fourier_freqs_xy, mcfg.fourier_freqs_z,
            n_heads=mcfg.decoder_n_heads, n_register=mcfg.n_register,
            use_register_attn=mcfg.decoder_register_attn,
            use_stage1=not mcfg.disable_decoder_stage1,
            xy_fourier_abs=mcfg.decoder_xy_fourier_abs)

    def encode(
        self,
        pn_points_feat: torch.Tensor,     # [B, L, k, 13]
        pn_mask: torch.Tensor,            # [B, L, k] bool
        leaf_stats: torch.Tensor,         # [B, L, 19]
        leaf_pos_norm: torch.Tensor,      # [B, L, 3] (also the RoPE source)
        global_morph_norm: torch.Tensor,  # [B, 5] z-scored morphology
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (encoder_out [B, L, D], vit_features [B, L+R, D],
        register tokens included)."""
        pn = self.pointnet(pn_points_feat, pn_mask)
        encoder_out = self.encoder(pn, leaf_stats, leaf_pos_norm)
        vit_features = self.vit(encoder_out, leaf_pos_norm, global_morph_norm)
        return encoder_out, vit_features

    def decode(
        self,
        encoder_out: torch.Tensor,
        vit_features: torch.Tensor,
        query_pos_norm: torch.Tensor,
        query_sdf_norm: torch.Tensor,
        query_sdf_dir: torch.Tensor,
        query_support_norm: torch.Tensor,
        region_dim1: torch.Tensor,
        region_dim2: torch.Tensor,
        region_dim3: torch.Tensor,
        idw_indices: torch.Tensor,
        leaf_pos_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decoder(
            encoder_out, vit_features,
            query_pos_norm, query_sdf_norm, query_sdf_dir, query_support_norm,
            region_dim1, region_dim2, region_dim3, idw_indices, leaf_pos_norm)

    def forward(
        self,
        pn_points_feat: torch.Tensor,
        pn_mask: torch.Tensor,
        leaf_stats: torch.Tensor,
        leaf_pos_norm: torch.Tensor,
        global_morph_norm: torch.Tensor,
        query_pos_norm: torch.Tensor,
        query_sdf_norm: torch.Tensor,
        query_sdf_dir: torch.Tensor,
        query_support_norm: torch.Tensor,
        region_dim1: torch.Tensor,
        region_dim2: torch.Tensor,
        region_dim3: torch.Tensor,
        idw_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_out, vit_features = self.encode(
            pn_points_feat, pn_mask, leaf_stats, leaf_pos_norm,
            global_morph_norm)
        return self.decode(
            encoder_out, vit_features,
            query_pos_norm, query_sdf_norm, query_sdf_dir, query_support_norm,
            region_dim1, region_dim2, region_dim3, idw_indices, leaf_pos_norm)
