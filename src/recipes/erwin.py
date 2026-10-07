"""Recipe "erwin" — 3D Erwin on UrbanWind (settings in
config.ErwinConfig). Native input = coordinates only through the Gaussian
RFF encoder; uniform 128K mixed sampling with masked heads (the same
protocol contract as Transolver); AdamW 5e-4 / wd 0.01 / no warmup /
per-STEP cosine to 1e-6 (the native cosmology granularity). Ball tree +
kNN graph are built on the CPU inside the loader workers by the
self-contained builder, outside any compiled graph."""
from __future__ import annotations

import math

import numpy as np
import torch

from ..data.estate import VOL_CHANNEL_NAMES, stream_full_estate_cases
from ..data.pointcloud import PointcloudDataset, collate_single
from ..evaluation.flops import sdpa_flops
from ..model.erwin import build_erwin_model
from ..model.erwin.balltree import build_tree, knn_edges, restore_index
from ..training.losses import compute_loss
from ..training.schedules import build_warmup_cosine
from .base import Recipe


class ErwinRecipe(Recipe):
    name = "erwin"
    family = "erwin"
    tex_row_name = "Erwin"
    adapter_version = "coords_rff_v1"
    supported_profiles = ("a100", "h100")

    def apply_optim_defaults(self, cfg) -> None:
        cfg.training.lr_base = cfg.erwin.lr
        cfg.training.weight_decay_base = cfg.erwin.weight_decay

    def build_train_dataset(self, cfg, estate_shard, dirs_by_estate):
        return PointcloudDataset(cfg, estate_shard, "erwin",
                                 dirs_by_estate=dirs_by_estate)

    def collate(self):
        return collate_single

    def build_model(self, cfg) -> torch.nn.Module:
        return build_erwin_model(cfg.erwin,
                                 out_dim_vol=len(VOL_CHANNEL_NAMES),
                                 extra_in=0)

    def compile_plan(self, cfg):
        # BatchNorm running-stat buffer mutation inside cudagraph replay is
        # the documented fragile combination — the plan is plain
        # torch.compile without cudagraph capture, stated here rather than
        # discovered as a silent fallback. fullgraph stays on: the tree/kNN
        # are inputs, so the graph is clean.
        return ("default", True,
                "BatchNorm buffers x cudagraphs: reduce-overhead "
                "deliberately not used for Erwin")

    def forward_loss(self, model, batch, cfg):
        tc = cfg.training
        pv, ps = model(batch["pos"], None, batch.get("edge_index"),
                       batch["tree_idx"], batch["tree_mask"],
                       batch["tree_idx_rot"], batch["restore_idx"])
        return compute_loss(pv, ps, batch["target_vol"].unsqueeze(0),
                            batch["target_surf"].unsqueeze(0),
                            batch["is_volume"].unsqueeze(0),
                            tc.loss_vol_weights, tc.loss_shape,
                            tc.huber_delta)

    def build_optimizer(self, cfg, model, opt_steps_per_epoch: int):
        e = cfg.erwin
        optimizer = torch.optim.AdamW(model.parameters(), lr=e.lr,
                                      weight_decay=e.weight_decay)
        total = cfg.training.num_epochs * opt_steps_per_epoch
        sched = build_warmup_cosine(optimizer, total, e.lr, e.min_lr,
                                    warmup_fraction=0.0)
        return optimizer, sched, "step"

    # ---- inference --------------------------------------------------------
    def stream_cases(self, cfg, names):
        return stream_full_estate_cases(cfg, names, with_idw=False)

    def prepare_infer_model(self, model, cfg, use_cuda: bool) -> dict:
        """GPU: eval-mode inference core compiled with fullgraph +
        dynamic=False + reduce-overhead — in eval BatchNorm reads frozen
        buffers, so the training-side BN x cudagraph hazard does not apply.
        Every chunk has the SAME row count (infer_chunk; final chunk pads by
        cycling real rows — erwin-native padding semantics, pad predictions
        discarded), so the tree's padded length is one static shape. Tree +
        kNN stay on CPU OUTSIDE the graph. Verified after warmup."""
        self._infer = {"compiled": None}
        if not use_cuda:
            return {"mode": "eager", "note": "CPU (tests)",
                    "expect_compiled": False}
        from ..utils.compile_health import compile_health, \
            reset_compile_state
        reset_compile_state()
        e = cfg.erwin
        R = int(e.infer_chunk)
        fwd_c = torch.compile(model, mode="reduce-overhead", fullgraph=True,
                              dynamic=False)
        device = next(model.parameters()).device
        pos = torch.rand(R, int(e.dimensionality))
        ti, tm, rot = build_tree(pos, e.strides, e.ball_sizes, e.rotate)
        ei = knn_edges(pos, int(e.knn)) if e.mp_steps > 0 else None
        ri = restore_index(ti, tm)
        with torch.inference_mode(), torch.autocast("cuda",
                                                    dtype=torch.bfloat16):
            fwd_c(pos.to(device), None,
                  None if ei is None else ei.to(device),
                  ti.to(device), tm.to(device),
                  [r.to(device) for r in rot], ri.to(device))
        torch.cuda.synchronize(device)
        status = compile_health("erwin-infer", expect_cudagraphs=True)
        self._infer["compiled"] = fwd_c
        return {"mode": "reduce-overhead", "fullgraph": True,
                "dynamic": False, "expect_compiled": True,
                "health": status,
                "note": "eval-mode core in-graph; CPU tree/kNN per chunk "
                        "outside the graph and INSIDE the latency timer "
                        "(model-side online compute)"}

    @torch.inference_mode()
    def infer_case(self, model, case, cfg, device, use_cuda):
        e = cfg.erwin
        n_vol = int(case["N_vol"])
        n = n_vol + int(case["N_surf"])
        pos_all = case["query_pos_norm"].float()
        out_vol = np.empty((n_vol, len(VOL_CHANNEL_NAMES)), dtype=np.float32)
        out_surf = np.empty((n - n_vol, 1), dtype=np.float32)
        compiled = getattr(self, "_infer", {}).get("compiled")
        R = int(e.infer_chunk)
        fixed = use_cuda and compiled is not None
        n_chunks = max(1, math.ceil(n / R))
        fwd = compiled if fixed else model
        for c in range(n_chunks):
            rows = torch.arange(c, n, n_chunks)   # density-preserving stride
            n_real = int(rows.shape[0])
            if fixed and n_real < R:
                reps = (R + n_real - 1) // n_real
                rows_run = rows.repeat(reps)[:R]  # cycle real rows (static)
            else:
                rows_run = rows
            pos = pos_all[rows_run]
            ti, tm, rot = build_tree(pos, e.strides, e.ball_sizes, e.rotate)
            ei = knn_edges(pos, int(e.knn)) if e.mp_steps > 0 else None
            ri = restore_index(ti, tm)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=use_cuda):
                pv, ps = fwd(pos.to(device), None,
                             None if ei is None else ei.to(device),
                             ti.to(device), tm.to(device),
                             [r.to(device) for r in rot], ri.to(device))
            # copy off graph-owned memory before the next replay
            pv = pv.squeeze(0).float().cpu().numpy()[:n_real]
            ps = ps.squeeze(0).float().cpu().numpy()[:n_real]
            vol_mask = (rows < n_vol).numpy()
            vol_rows = rows[rows < n_vol]
            surf_rows = rows[rows >= n_vol]
            out_vol[vol_rows.numpy()] = pv[vol_mask]
            out_surf[(surf_rows - n_vol).numpy()] = ps[~vol_mask]
        return out_vol, out_surf

    def infer_flops_planner(self, cfg, sample_case, device):
        """Eager probe of ONE fixed-size chunk (real tree over random
        coordinates — identical shapes) on a FRESH eager instance; per case
        = n_chunks x per-chunk."""
        from ..evaluation.flops import FlopMeter
        e = cfg.erwin
        R = int(e.infer_chunk)
        probe = self.build_model(cfg).to(device)
        probe.eval()
        pos = torch.rand(R, int(e.dimensionality))
        ti, tm, rot = build_tree(pos, e.strides, e.ball_sizes, e.rotate)
        ei = knn_edges(pos, int(e.knn)) if e.mp_steps > 0 else None
        ri = restore_index(ti, tm)
        with FlopMeter(probe) as m, torch.inference_mode():
            probe(pos.to(device), None,
                  None if ei is None else ei.to(device),
                  ti.to(device), tm.to(device),
                  [r.to(device) for r in rot], ri.to(device))
        hooked = m.total
        manual = sum(self._bmsa_terms(cfg,
                                      self._padded_rows(cfg, R)).values())
        del probe
        if device.type == "cuda":
            torch.cuda.empty_cache()

        def plan(n_vol: int, n_surf: int) -> float:
            n_chunks = max(1, math.ceil((n_vol + n_surf) / R))
            return n_chunks * (hooked + manual)

        return plan

    # ---- FLOPs manual terms -----------------------------------------------
    def _bmsa_terms(self, cfg, n_rows_padded):
        """Per-level SDPA + distance-mask cdist + rel-pos work; level sizes
        follow the strides."""
        e = cfg.erwin
        terms = {}
        rows = float(n_rows_padded)
        levels = list(zip(e.c_hidden, e.enc_num_heads, e.enc_depths,
                          e.ball_sizes))
        dec = list(zip(e.c_hidden[:-1], e.dec_num_heads, e.dec_depths,
                       e.ball_sizes[:-1]))
        sizes = [rows]
        for s in e.strides:
            sizes.append(sizes[-1] / s)
        total_sdpa = 0.0
        total_cdist = 0.0
        for i, (dim, h, depth, ball) in enumerate(levels):
            r = sizes[i]
            balls = r / ball
            total_sdpa += depth * balls * sdpa_flops(ball, ball, h, dim // h)
            total_cdist += depth * balls * ball * ball \
                * (2 * e.dimensionality + 1)
        for i, (dim, h, depth, ball) in enumerate(dec):
            r = sizes[i]
            balls = r / ball
            total_sdpa += depth * balls * sdpa_flops(ball, ball, h, dim // h)
            total_cdist += depth * balls * ball * ball \
                * (2 * e.dimensionality + 1)
        terms["bmsa_sdpa"] = total_sdpa
        terms["bmsa_distance_mask"] = total_cdist
        if e.mp_steps > 0:
            # scatter-mean adds over E = N*k edges, per mp step
            terms["mpnn_scatter"] = e.mp_steps * rows * e.knn * e.c_hidden[0]
        terms["rff_trig"] = rows * e.c_in
        # SwiGLU's F.silu gate is functional (not module-hooked): 4/elem on
        # the ffn width, per block, encoder + decoder levels
        act = 0.0
        for i, (dim, _h, depth, _b) in enumerate(levels):
            act += depth * sizes[i] * dim * e.mlp_ratio * 4.0
        for i, (dim, _h, depth, _b) in enumerate(dec):
            act += depth * sizes[i] * dim * e.mlp_ratio * 4.0
        terms["swiglu_silu_act"] = act
        return terms

    def _padded_rows(self, cfg, n):
        ball = cfg.erwin.ball_sizes[0]
        return ball * max(1, 2 ** math.ceil(math.log2(max(n / ball, 1))))

    def manual_train_flops(self, cfg):
        return self._bmsa_terms(cfg,
                                self._padded_rows(cfg, cfg.training.n_query))

    def manual_infer_flops(self, cfg, n_vol, n_surf):
        n = n_vol + n_surf
        n_chunks = max(1, math.ceil(n / int(cfg.erwin.infer_chunk)))
        per_chunk = self._bmsa_terms(
            cfg, self._padded_rows(cfg, math.ceil(n / n_chunks)))
        return {k: v * n_chunks for k, v in per_chunk.items()}
