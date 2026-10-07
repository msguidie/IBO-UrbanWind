"""AB-UPT full-field inference.

Anchors are deterministic per-branch draws (per_case_epoch_seed namespaces
with a fixed seed value); the queries are each branch's remaining rows in
their original order.

Anchors are encoded once per case, capturing every block's post-RoPE anchor
K/V (model/abupt.py encode_anchor_cache); each fixed-length query chunk
then runs only its own rows against that cache (decode_queries_cached).
Because queries only ever read anchor K/V and anchors never see queries,
this equals the reference forward_with_queries, which the model keeps as
the equivalence reference.

Chunks have a fixed length (cfg.abupt.fullfield_query_chunk_per_branch rows
per branch; final or exhausted chunks padded by row repetition, pad rows
sliced off), so torch.compile(dynamic=False) sees one shape. Predictions
are scattered back to the original row order, giving the (pred_vol_z
[N_vol, 5], pred_surf_z [N_surf, 1]) contract every recipe shares.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..data.abupt_estate import check_abupt_coord_bounds
from ..data.dataset import CURVE_SAMPLE_MARKER
from ..utils.seed import per_case_epoch_seed
from .test_worker import _autocast, _pad_rows, _unsq

_GEO_SAMPLE_KEYS = ("sdf", "sdf_dir", "support", "region_dim2", "region_dim3")


# ---------------------------------------------------------------------------
# Full-field inference phase
# ---------------------------------------------------------------------------

def _fixed_anchor_indices(case_name: str, n: int, n_anchors: int,
                          tag: str, draw: int = 0) -> torch.Tensor:
    """A deterministic anchor subset per case and draw (per_case_epoch_seed
    namespace with the fixed CURVE_SAMPLE_MARKER seed value). Evaluation
    reports the mean +- std over cfg.abupt.infer_draws draws."""
    g = torch.Generator()
    suffix = f"_d{int(draw)}" if draw else ""
    g.manual_seed(per_case_epoch_seed(f"{case_name}__abupt_{tag}{suffix}",
                                      CURVE_SAMPLE_MARKER))
    return torch.randperm(n, generator=g)[:n_anchors].sort().values


def _complement(n: int, idx: torch.Tensor) -> torch.Tensor:
    """Rows of [0, n) not in idx, ascending (original row order)."""
    mask = torch.zeros(n, dtype=torch.bool)
    mask[idx] = True
    return torch.nonzero(~mask, as_tuple=False).squeeze(1)


def _branch_rows(case: dict, local_idx: torch.Tensor, is_surface: bool,
                 geometry_features: bool, n_vol: int) -> dict:
    """Model-input dict (side-prefixed keys, CPU) for the given local rows
    of one branch, gathered from a full case's combined per-point arrays
    (wind frame, estate.materialize layout)."""
    side = "surface" if is_surface else "volume"
    global_idx = local_idx + n_vol if is_surface else local_idx
    out = {f"{side}_position": case["query_pos_norm"][global_idx]}
    if geometry_features:
        out[f"{side}_sdf"] = case["query_sdf_norm"][global_idx].float()
        out[f"{side}_sdf_dir"] = case["query_sdf_dir"][global_idx].float()
        out[f"{side}_support"] = case["query_support_norm"][global_idx].float()
        out[f"{side}_region_dim2"] = case["region_dim2"][global_idx]
        out[f"{side}_region_dim3"] = case["region_dim3"][global_idx]
    return out


def _query_chunk(case: dict, query_idx: torch.Tensor, pad_row: dict, lo: int,
                 chunk: int, is_surface: bool, geometry_features: bool,
                 n_vol: int) -> tuple[dict, int]:
    """One fixed-length query chunk for one branch: rows [lo, lo+chunk) of
    the branch's query list, padded to exactly `chunk` rows. Returns
    (cpu tensor dict, n_valid); n_valid == 0 means a pure-padding chunk (the
    branch is exhausted — the lockstep partner still has work)."""
    n_q = int(query_idx.shape[0])
    hi = min(lo + chunk, n_q)
    valid = max(hi - lo, 0)
    if valid > 0:
        rows = _branch_rows(case, query_idx[lo:hi], is_surface,
                            geometry_features, n_vol)
        pad = chunk - valid
        if pad:
            rows = {k: _pad_rows(v, pad) for k, v in rows.items()}
    else:
        # exhausted branch: repeat the pad-source row (real data, finite
        # math; every pad prediction is sliced off before use)
        rows = {k: v.expand(chunk, *v.shape[1:]) for k, v in pad_row.items()}
    return rows, valid


def clone_cache(cache):
    """Deep-clone the anchor-K/V cache pytree. MANDATORY after a compiled
    (reduce-overhead) encode: its outputs live in graph-owned memory that
    the next replay overwrites — the cache is read by EVERY decode chunk,
    so it must be moved to owned storage exactly once, here."""
    if isinstance(cache, torch.Tensor):
        return cache.clone()
    if isinstance(cache, dict):
        return {k: clone_cache(v) for k, v in cache.items()}
    if isinstance(cache, (list, tuple)):
        out = [clone_cache(v) for v in cache]
        return type(cache)(out) if isinstance(cache, tuple) else out
    return cache


@torch.inference_mode()
def infer_case_fullfield(model, case: dict, cfg, device: torch.device,
                         use_cuda: bool, encode_fn=None, decode_fn=None,
                         draw: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Full-field prediction for one (estate, direction) case:
    (pred_vol_z [N_vol, 5], pred_surf_z [N_surf, 1]) fp32, original row
    order, no padding rows. Anchor-K/V-cached — see the module docstring.
    encode_fn/decode_fn: the recipe's COMPILED entry points (None = eager);
    the cache is cloned off graph-owned memory right after encode."""
    n_vol = int(case["N_vol"])
    n_surf = int(case["N_surf"])
    n_anchors = int(cfg.abupt.n_anchors)
    chunk = int(cfg.abupt.fullfield_query_chunk_per_branch)
    geometry = bool(cfg.abupt.geometry_features)
    name = case["case_name"]
    cast = not use_cuda

    check_abupt_coord_bounds(case["query_pos_norm"], name)
    if n_vol < n_anchors or n_surf < n_anchors:
        raise RuntimeError(f"{name}: N_vol={n_vol} N_surf={n_surf} < "
                           f"n_anchors={n_anchors}")

    vol_anchor = _fixed_anchor_indices(name, n_vol, n_anchors, "vol", draw)
    surf_anchor = _fixed_anchor_indices(name, n_surf, n_anchors, "surf",
                                        draw)
    vol_query = _complement(n_vol, vol_anchor)
    surf_query = _complement(n_surf, surf_anchor)
    n_vq = int(vol_query.shape[0])
    n_sq = int(surf_query.shape[0])

    surf_anchor_dev = {k: _unsq(v, device, cast) for k, v in
                       _branch_rows(case, surf_anchor, True, geometry,
                                    n_vol).items()}
    vol_anchor_dev = {k: _unsq(v, device, cast) for k, v in
                      _branch_rows(case, vol_anchor, False, geometry,
                                   n_vol).items()}

    out_vol = np.empty((n_vol, 5), dtype=np.float32)
    out_surf = np.empty((n_surf, 1), dtype=np.float32)

    # ---- anchors: encoded exactly ONCE, K/V cached per block ------------
    encode = encode_fn if encode_fn is not None else model.encode_anchor_cache
    decode = decode_fn if decode_fn is not None else model.decode_queries_cached
    with _autocast(use_cuda):
        ps_anchor, pv_anchor, cache = encode(**surf_anchor_dev,
                                             **vol_anchor_dev)
    if encode_fn is not None:
        cache = clone_cache(cache)     # off graph-owned memory, exactly once
    out_surf[surf_anchor.numpy()] = ps_anchor[0].float().cpu().numpy()
    out_vol[vol_anchor.numpy()] = pv_anchor[0].float().cpu().numpy()

    if n_sq == 0 and n_vq == 0:
        return out_vol, out_surf

    # pad source per branch: the last query row when the branch has any
    # queries, else its last anchor row (a branch can be query-empty while
    # its lockstep partner still has chunks to run)
    surf_pad_row = _branch_rows(
        case, (surf_query[-1:] if n_sq else surf_anchor[-1:]), True,
        geometry, n_vol)
    vol_pad_row = _branch_rows(
        case, (vol_query[-1:] if n_vq else vol_anchor[-1:]), False,
        geometry, n_vol)

    n_steps = max(math.ceil(n_sq / chunk), math.ceil(n_vq / chunk))
    for step in range(n_steps):
        lo = step * chunk
        s_rows, s_valid = _query_chunk(case, surf_query, surf_pad_row, lo,
                                       chunk, True, geometry, n_vol)
        v_rows, v_valid = _query_chunk(case, vol_query, vol_pad_row, lo,
                                       chunk, False, geometry, n_vol)
        kwargs = {}
        for key, t in s_rows.items():
            kwargs[key] = _unsq(t, device, cast)
        for key, t in v_rows.items():
            kwargs[key] = _unsq(t, device, cast)
        with _autocast(use_cuda):
            ps_q, pv_q = decode(cache, **kwargs)
        if s_valid:
            out_surf[surf_query[lo:lo + s_valid].numpy()] = \
                ps_q[0, :s_valid].float().cpu().numpy()
        if v_valid:
            out_vol[vol_query[lo:lo + v_valid].numpy()] = \
                pv_q[0, :v_valid].float().cpu().numpy()
    return out_vol, out_surf

