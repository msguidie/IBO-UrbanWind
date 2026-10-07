"""IBO full-field inference and the metric tail shared by every recipe.

Batch 1: encode once, then decode the query pool in fixed-length chunks
through the same model code as training, under torch.inference_mode(). The
decoder computes all query geometry (distances, IDW prior) itself from the
normalized positions. The final short chunk is padded to the chunk length
and the pad rows are sliced off before any metric, so torch.compile
(dynamic=False) sees one shape. encode() has case-independent shapes
(n_leaves, n_pn_neighbors, the morphology vector).
"""
from __future__ import annotations

import numpy as np
import torch

from ..model.full_model import IBO
from .metrics import (case_metrics, inverse_transform_surf,
                      inverse_transform_vol)

DECODE_CHUNK_KEYS = ("query_pos_norm", "query_sdf_norm", "query_sdf_dir",
                     "query_support_norm", "region_dim1", "region_dim2",
                     "region_dim3", "idw_indices")


def _unsq(t: torch.Tensor, device, cast_fp32: bool) -> torch.Tensor:
    t = t.unsqueeze(0).to(device)
    if cast_fp32 and t.is_floating_point() and t.dtype == torch.bfloat16:
        t = t.float()
    return t


def _autocast(use_cuda: bool):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda)


@torch.inference_mode()
def encode_case(model: IBO, case: dict, device: torch.device,
                use_cuda: bool) -> tuple[torch.Tensor, torch.Tensor]:
    cast = not use_cuda  # CPU: bf16 -> fp32
    with _autocast(use_cuda):
        return model.encode(
            _unsq(case["pn_points_feat"], device, cast),
            _unsq(case["pn_mask"], device, cast),
            _unsq(case["leaf_stats"], device, cast),
            _unsq(case["leaf_pos_norm"], device, cast),
            _unsq(case["global_morph_norm"], device, cast))


def _pad_rows(t: torch.Tensor, pad: int) -> torch.Tensor:
    """Repeat the last row ``pad`` times (the padded predictions are
    discarded; repeating keeps indices valid and the math finite)."""
    return torch.cat([t, t[-1:].expand(pad, *t.shape[1:])], dim=0)


@torch.inference_mode()
def decode_pool(model: IBO, encoder_out: torch.Tensor,
                vit_features: torch.Tensor, case: dict,
                device: torch.device, chunk: int,
                use_cuda: bool) -> tuple[np.ndarray, np.ndarray]:
    """Chunked decode of one query pool -> (pred_vol_z, pred_surf_z) [N, C].

    Slice, pad to the fixed chunk length, decode, copy back. Every chunk
    has the same shape, so the compiled decode is built once (a ragged final
    chunk would recompile)."""
    cast = not use_cuda
    n_total = case["N_vol"] + case["N_surf"]
    leaf_pos_dev = _unsq(case["leaf_pos_norm"], device, cast)

    pv_parts, ps_parts = [], []
    for lo in range(0, n_total, chunk):
        hi = min(lo + chunk, n_total)
        n_valid = hi - lo
        pad = chunk - n_valid
        chunk_dev = {}
        for k in DECODE_CHUNK_KEYS:
            t = case[k][lo:hi]
            if pad:
                t = _pad_rows(t, pad)
            chunk_dev[k] = _unsq(t, device, cast)
        with _autocast(use_cuda):
            pv, ps = model.decode(
                encoder_out, vit_features,
                chunk_dev["query_pos_norm"], chunk_dev["query_sdf_norm"],
                chunk_dev["query_sdf_dir"], chunk_dev["query_support_norm"],
                chunk_dev["region_dim1"], chunk_dev["region_dim2"],
                chunk_dev["region_dim3"], chunk_dev["idw_indices"],
                leaf_pos_dev)
        # mask out the padding: slice to the valid rows before anything
        # downstream (metrics, saved predictions) can see them
        pv_parts.append(pv.squeeze(0)[:n_valid].float().cpu().numpy())
        ps_parts.append(ps.squeeze(0)[:n_valid].float().cpu().numpy())
    return np.concatenate(pv_parts, axis=0), np.concatenate(ps_parts, axis=0)


@torch.inference_mode()
def infer_case(model: IBO, case: dict, device: torch.device,
               chunk: int, use_cuda: bool) -> tuple[np.ndarray, np.ndarray]:
    """Full-field prediction: (pred_vol_z [N_vol, 5], pred_surf_z [N_surf, 1]) fp32."""
    encoder_out, vit_features = encode_case(model, case, device, use_cuda)
    if use_cuda:
        # encode and decode are compiled separately with CUDA graphs, and
        # decode reads encoder_out / vit_features once per chunk. Those
        # tensors live in encode's graph-private pool, which a later replay
        # may overwrite; cloning moves them into ordinary CUDA memory.
        encoder_out = encoder_out.clone()
        vit_features = vit_features.clone()
    pred_all_vol, pred_all_surf = decode_pool(
        model, encoder_out, vit_features, case, device, chunk, use_cuda)
    n_vol = case["N_vol"]
    return pred_all_vol[:n_vol], pred_all_surf[n_vol:]


def metrics_from_predictions(name: str, case: dict, stats: dict,
                             pred_vol_z: np.ndarray, pred_surf_z: np.ndarray
                             ) -> tuple[dict, np.ndarray, np.ndarray]:
    """Inverse transforms and per-case metrics from full-field z-score
    predictions. Shared by every recipe: only the prediction producer
    differs between models, never the metric definitions."""
    n_vol = case["N_vol"]
    true_vol_z = case["target_vol"].float().numpy()
    true_surf_z = case["target_surf"].float().numpy()
    z_phys = case["query_z_phys"]
    if z_phys is None:
        z_phys = case["query_pos_norm"][:, 2].float() * stats["coord_divisor"]
    z_phys = z_phys.numpy()
    z_vol, z_surf = z_phys[:n_vol], z_phys[n_vol:]

    pred_vol_phys = inverse_transform_vol(pred_vol_z, z_vol, stats)
    true_vol_phys = inverse_transform_vol(true_vol_z, z_vol, stats)
    pred_surf_phys = inverse_transform_surf(pred_surf_z, z_surf, stats)
    true_surf_phys = inverse_transform_surf(true_surf_z, z_surf, stats)

    metrics = case_metrics(pred_vol_z, true_vol_z, pred_surf_z, true_surf_z,
                           pred_vol_phys, true_vol_phys,
                           pred_surf_phys, true_surf_phys)
    metrics["case_name"] = name
    metrics["N_vol"] = n_vol
    metrics["N_surf"] = case["N_surf"]
    return metrics, pred_vol_phys, pred_surf_phys

