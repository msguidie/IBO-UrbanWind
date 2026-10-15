"""Sample-level helpers shared by the training datasets: the loader worker
initializer, the x-mirror of a drawn sample and the Ux normalization
constants that mirror needs.
"""
from __future__ import annotations

from ..config import Config
from ..utils.env import limit_worker_threads

# Fixed "epoch" of evaluation-time draws (e.g. AB-UPT's inference anchors):
# training epochs are >= 1, so -1 never collides with a training draw.
CURVE_SAMPLE_MARKER = -1


def _worker_init(_worker_id: int) -> None:
    """Every loader worker runs single-threaded math libraries."""
    limit_worker_threads()


def mirror_slim_sample(sample: dict, vol_mean_ux: float,
                       vol_std_ux: float) -> dict:
    """The x-mirrored twin of one drawn sample. In the wind frame the mean
    flow is along -y and the domain is symmetric about x = 0, so x -> -x is
    an exact symmetry of the flow problem. Query-side tensors are flipped
    row-wise, the latent side whole; the input dict is not modified.

    Invariant under the mirror: sdf magnitudes, the decoder k-NN (queries
    and latent points reflect together), region codes (wind labels use
    n . wind with wind = (0, -1); left / right sides share one code), the
    surface target and the morphology conditioners.
    """
    mirrored = dict(sample)

    qpn = sample["query_pos_norm"].clone()
    qpn[:, 0] = -qpn[:, 0]
    mirrored["query_pos_norm"] = qpn

    qsd = sample["query_sdf_dir"].clone()
    qsd[:, 0] = -qsd[:, 0]
    mirrored["query_sdf_dir"] = qsd

    # support columns are [+x, -x, +y, -y, +z, -z]: the x rays swap
    qsup = sample["query_support_norm"].clone()
    qsup[:, 0] = sample["query_support_norm"][:, 1]
    qsup[:, 1] = sample["query_support_norm"][:, 0]
    mirrored["query_support_norm"] = qsup

    # Ux is stored z-scored, so negating the physical value maps
    # z -> -z - 2 mean / std; volume rows only (surface rows of target_vol
    # are loss-masked zeros)
    tv = sample["target_vol"].clone()
    iv = sample["is_volume"]
    tv[iv, 0] = -tv[iv, 0] - 2.0 * (vol_mean_ux / vol_std_ux)
    mirrored["target_vol"] = tv

    lpn = sample["leaf_pos_norm"].clone()
    lpn[:, 0] = -lpn[:, 0]
    mirrored["leaf_pos_norm"] = lpn

    ls = sample["leaf_stats"].clone()
    ls[:, 6] = -ls[:, 6]   # cov_xy / trace
    ls[:, 7] = -ls[:, 7]   # cov_xz / trace
    mirrored["leaf_stats"] = ls

    ppf = sample["pn_points_feat"].clone()
    ppf[..., 0] = -ppf[..., 0]                      # relative x
    ppf[..., 4] = -ppf[..., 4]                      # sdf_dir x
    ppf[..., 7] = sample["pn_points_feat"][..., 8]  # support +x / -x swap
    ppf[..., 8] = sample["pn_points_feat"][..., 7]
    mirrored["pn_points_feat"] = ppf
    return mirrored


def load_ux_norm_stats(cfg: Config) -> tuple[float, float]:
    """(mean, std) of Ux, the pair the mirror's z-score flip needs."""
    from ..evaluation.metrics import load_norm_stats
    stats = load_norm_stats(cfg.data.norm_stats_path)
    return float(stats["vol_mean"][0]), float(stats["vol_std"][0])
