"""Recipe "spider" — SpiderSolver on UrbanWind (settings in
config.SpiderConfig). Six wind-frame surface regions,
onion-10 shells, fixed 500 innermost participants, 128K stratified sample
(125952 vol + 2048 surf), Adam 1e-3 OneCycle(0.3, 25, 1000) per optimizer
step; effective batch 4 = microbatch 1 x 4 GPUs, grad_accum 1; no
activation checkpointing; training compiles with CUDA Graphs.

Full-field inference uses disjoint, density-preserving stride chunks, every
point exactly once (an adaptation, not a whole-estate global pass): sector
tokens, onion bins and the innermost set are recomputed per chunk as they
are per training sample (a stride chunk is a uniform subsample at training
density); the surface-row count per chunk follows the case's own proportion
rather than the training quota."""
from __future__ import annotations

import numpy as np
import torch

from ..data.estate import VOL_CHANNEL_NAMES, load_estate, materialize
from ..data.pointcloud import (PointcloudDataset, attach_spider_derived,
                               build_spider_derived, collate_single,
                               onion_shell_ids, spider_derived_dir)
from ..evaluation.flops import sdpa_flops, segment_mean_flops
from ..model.spider import build_spider_model
from ..training.losses import compute_loss_branch
from ..training.schedules import build_onecycle
from .base import Recipe


def _region_classes_from_dirs(sdf_dir_surf: torch.Tensor) -> torch.Tensor:
    axis = sdf_dir_surf.abs().argmax(dim=-1)
    sign_neg = torch.gather(sdf_dir_surf, 1,
                            axis.unsqueeze(1)).squeeze(1) < 0
    return axis * 2 + sign_neg.long()


class SpiderRecipe(Recipe):
    name = "spider"
    family = "spider"
    tex_row_name = "SpiderSolver"
    adapter_version = "native7_v1+spider_v1"
    supported_profiles = ("a100", "h100")
    effective_batch = 4
    supports_activation_checkpointing = False
    # DDPOptimizer cannot split this graph (int split point + index tensors)
    disable_ddp_optimizer = True


    def apply_optim_defaults(self, cfg) -> None:
        cfg.training.lr_base = cfg.spider.lr
        cfg.training.weight_decay_base = 0.0     # native Adam, no decay

    def compile_plan(self, cfg):
        """Training: reduce-overhead (inductor + CUDA Graphs), fullgraph,
        with torch._dynamo's DDPOptimizer off (it cannot split this graph:
        int split point + index tensors)."""
        return ("reduce-overhead", True,
                "CUDA Graphs on; DDPOptimizer off for this graph")

    def check_pool(self, cfg, pool: dict) -> None:
        # the stratified draw needs surf_quota surface rows per case; the
        # 400 pool covers 16384 by construction, but check the real widths
        # rather than assuming
        import json
        from pathlib import Path
        idx_path = Path(cfg.data.cache_estate_dir) / "_index.json"
        if idx_path.exists():
            with open(idx_path, encoding="utf-8") as f:
                index = json.load(f)
            bad = [e for e in pool["splits"]["train_estates"]
                   if e in index
                   and int(index[e]["N_surf"]) < cfg.spider.surf_quota]
            if bad:
                raise SystemExit(
                    f"spider surf_quota={cfg.spider.surf_quota} exceeds "
                    f"N_surf of {len(bad)} pool estates (e.g. {bad[:3]})")

    def ensure_derived(self, cfg, estates: list[str]) -> None:
        out = spider_derived_dir(cfg)
        missing = [e for e in estates if not (out / f"{e}.pt").exists()]
        if missing:
            raise SystemExit(
                f"spider derived cache incomplete: {len(missing)} estates "
                f"missing under {out} (e.g. {missing[:3]}). Build once on a "
                f"CPU node:\n  python main.py derive --model spider "
                f"--manifest <pool manifest>")

    def run_derive(self, cfg, estates: list[str]) -> None:
        build_spider_derived(cfg, estates,
                             workers=int(cfg.preprocess.n_workers))

    def build_train_dataset(self, cfg, estate_shard, dirs_by_estate):
        self.ensure_derived(cfg, estate_shard)
        return PointcloudDataset(cfg, estate_shard, "spider",
                                 dirs_by_estate=dirs_by_estate)

    def collate(self):
        return collate_single

    def build_model(self, cfg) -> torch.nn.Module:
        return build_spider_model(
            cfg.spider, out_dim_vol=len(VOL_CHANNEL_NAMES),
            surf_token_mode="per_region",
            use_checkpoint=cfg.training.activation_checkpointing)

    def forward_loss(self, model, batch, cfg):
        tc = cfg.training
        n_vol = batch["target_vol"].shape[0]
        pv, ps = model(batch["x"].unsqueeze(0), n_vol, batch["vol_gid"],
                       batch["inner_idx"], batch["surf_region_id"])
        return compute_loss_branch(pv, ps, batch["target_vol"].unsqueeze(0),
                                   batch["target_surf"].unsqueeze(0),
                                   tc.loss_vol_weights, tc.loss_shape,
                                   tc.huber_delta)

    def build_optimizer(self, cfg, model, opt_steps_per_epoch: int):
        s = cfg.spider
        optimizer = torch.optim.Adam(model.parameters(), lr=s.lr)
        total = cfg.training.num_epochs * opt_steps_per_epoch
        sched = build_onecycle(optimizer, total, s.lr, s.onecycle_pct_start,
                               s.onecycle_div_factor,
                               s.onecycle_final_div_factor)
        return optimizer, sched, "step"

    # ---- inference --------------------------------------------------------
    def stream_cases(self, cfg, names):
        """Full cases + the per-estate derived nearest-surface index
        attached (one pack read per estate — the estate-granular shard)."""
        from ..data.estate import _group_by_estate
        for est, dirs in _group_by_estate(names).items():
            pack = load_estate(cfg, est, resident_mode=False,
                               build_idw_tree=False)
            attach_spider_derived(cfg, pack)
            nearest = pack["_spider_nearest_surf"]
            for d in dirs:
                case = materialize(pack, d, keep_z_phys=True, with_idw=False)
                case["_spider_nearest"] = nearest
                yield f"{est}_{d}", case
            del pack

    def checkpointed_modules(self, model) -> list:
        from ..training.swa import unwrap_model
        return list(unwrap_model(model).blocks)

    @staticmethod
    def _quota_shapes(cfg) -> tuple[int, int]:
        """The ONE static inference shape: the TRAINING sample geometry
        (n_query - surf_quota volume rows + surf_quota surface rows).
        Every chunk of every case runs at exactly this shape."""
        nv = int(cfg.training.n_query) - int(cfg.spider.surf_quota)
        return nv, int(cfg.spider.surf_quota)

    def prepare_infer_model(self, model, cfg, use_cuda: bool) -> dict:
        """GPU: compile the WHOLE forward (static [1, NV+NS, 7] chunk at the
        training quota shape) with fullgraph + dynamic=False +
        reduce-overhead, warm on synthetic tensors of the real shapes, then
        VERIFY graph capture. CPU: the eager natural-proportion path."""
        self._infer = {"compiled": None}
        if not use_cuda:
            return {"mode": "eager", "note": "CPU (tests) — natural-"
                    "proportion stride chunks", "expect_compiled": False}
        from ..utils.compile_health import compile_health, \
            reset_compile_state
        reset_compile_state()
        s = cfg.spider
        NV, NS = self._quota_shapes(cfg)
        fwd_c = torch.compile(model, mode="reduce-overhead", fullgraph=True,
                              dynamic=False)
        device = next(model.parameters()).device
        x = torch.randn(1, NV + NS, int(s.in_dim), device=device)
        gid = torch.randint(0, int(s.onion_num) * int(s.n_regions), (NV,),
                            device=device)
        inner = torch.arange(int(s.n_inner), device=device)
        reg = torch.randint(0, int(s.n_regions), (NS,), device=device)
        with torch.inference_mode(), torch.autocast("cuda",
                                                    dtype=torch.bfloat16):
            fwd_c(x, NV, gid, inner, reg)
        torch.cuda.synchronize(device)
        status = compile_health("spider-infer", expect_cudagraphs=True)
        self._infer["compiled"] = fwd_c
        return {"mode": "reduce-overhead", "fullgraph": True,
                "dynamic": False, "expect_compiled": True,
                "health": status,
                "note": f"fixed training-quota chunks ({NV}+{NS}); final "
                        f"chunks pad by cycling REAL rows (predictions for "
                        f"pad duplicates discarded; disclosed)"}

    @staticmethod
    def _pad_cycle(idx: torch.Tensor, target: int) -> torch.Tensor:
        """Pad an index list to `target` by cycling its own (real) entries —
        static shapes without synthetic geometry; duplicated rows perturb
        that chunk's count-normalised aggregation by <= pad/target
        (disclosed); their predictions are discarded."""
        n = int(idx.shape[0])
        if n == target:
            return idx
        assert 0 < n < target
        reps = (target + n - 1) // n
        return idx.repeat(reps)[:target]

    @torch.inference_mode()
    def infer_case(self, model, case, cfg, device, use_cuda):
        s = cfg.spider
        n_vol = int(case["N_vol"])
        n = n_vol + int(case["N_surf"])
        x_all = torch.cat([case["query_pos_norm"].float(),
                           case["query_sdf_norm"].float().unsqueeze(-1),
                           case["query_sdf_dir"].float()], dim=-1)
        surf_cls_full = _region_classes_from_dirs(
            case["query_sdf_dir"][n_vol:].float())
        nearest = case["_spider_nearest"].long()
        out_vol = np.empty((n_vol, len(VOL_CHANNEL_NAMES)), dtype=np.float32)
        out_surf = np.empty((n - n_vol, 1), dtype=np.float32)

        import math
        compiled = getattr(self, "_infer", {}).get("compiled")
        if use_cuda and compiled is not None:
            # fixed training-quota chunks: every forward is the ONE compiled
            # static shape; every real row appears in exactly one chunk
            NV, NS = self._quota_shapes(cfg)
            n_surf = n - n_vol
            n_chunks = max(1, math.ceil(n_vol / NV),
                           math.ceil(max(n_surf, 1) / NS))
            for c in range(n_chunks):
                vol_rows = torch.arange(c, n_vol, n_chunks)
                surf_rows = torch.arange(c, n_surf, n_chunks)
                nv_real = int(vol_rows.shape[0])
                ns_real = int(surf_rows.shape[0])
                vol_pad = self._pad_cycle(vol_rows, NV)
                surf_pad = self._pad_cycle(surf_rows, NS) if ns_real \
                    else torch.zeros(NS, dtype=torch.long)
                x = torch.cat([x_all[vol_pad],
                               x_all[surf_pad + n_vol]])
                sdf_vol = x[:NV, 3]
                shell = onion_shell_ids(sdf_vol, int(s.onion_num))
                vol_cls = surf_cls_full[nearest[vol_pad]]
                gid = shell * int(s.n_regions) + vol_cls
                inner = torch.topk(sdf_vol.abs(), int(s.n_inner),
                                   largest=False).indices.sort().values
                reg = surf_cls_full[surf_pad]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pv, ps = compiled(x.unsqueeze(0).to(device), NV,
                                      gid.to(device), inner.to(device),
                                      reg.to(device))
                # graph-owned outputs: copy off-device before the next replay
                pv_c = pv.squeeze(0).float().cpu().numpy()
                ps_c = ps.squeeze(0).float().cpu().numpy()
                out_vol[vol_rows.numpy()] = pv_c[:nv_real]
                if ns_real:
                    out_surf[surf_rows.numpy()] = ps_c[:ns_real]
            return out_vol, out_surf

        # eager path (CPU tests): natural-proportion stride chunks
        n_chunks = max(1, math.ceil(n / int(s.infer_chunk)))
        for c in range(n_chunks):
            rows = torch.arange(c, n, n_chunks)          # stride partition
            vol_rows = rows[rows < n_vol]
            surf_rows = rows[rows >= n_vol]
            order = torch.cat([vol_rows, surf_rows])
            x = x_all[order]
            nv = int(vol_rows.shape[0])
            sdf_vol = x[:nv, 3]
            shell = onion_shell_ids(sdf_vol, int(s.onion_num))
            vol_cls = surf_cls_full[nearest[vol_rows]]
            gid = shell * int(s.n_regions) + vol_cls
            n_inner = min(int(s.n_inner), nv)
            inner = torch.topk(sdf_vol.abs(), n_inner,
                               largest=False).indices.sort().values
            reg = surf_cls_full[surf_rows - n_vol]
            xd = x.unsqueeze(0).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=use_cuda):
                pv, ps = model(xd, nv, gid.to(device), inner.to(device),
                               reg.to(device))
            out_vol[vol_rows.numpy()] = pv.squeeze(0).float().cpu().numpy()
            out_surf[(surf_rows - n_vol).numpy()] = \
                ps.squeeze(0).float().cpu().numpy()
        return out_vol, out_surf

    def infer_flops_planner(self, cfg, sample_case, device):
        """Eager probe of ONE training-quota chunk on a FRESH eager
        instance; per case = n_chunks x per-chunk (all chunks identical)."""
        from ..evaluation.flops import FlopMeter
        s = cfg.spider
        NV, NS = self._quota_shapes(cfg)
        probe = self.build_model(cfg).to(device)
        probe.eval()
        x = torch.randn(1, NV + NS, int(s.in_dim), device=device)
        gid = torch.randint(0, int(s.onion_num) * int(s.n_regions), (NV,),
                            device=device)
        inner = torch.arange(int(s.n_inner), device=device)
        reg = torch.randint(0, int(s.n_regions), (NS,), device=device)
        with FlopMeter(probe) as m, torch.inference_mode():
            probe(x, NV, gid, inner, reg)
        hooked = m.total
        manual = sum(self._attn_terms(cfg, NV, NS,
                                      int(s.n_inner)).values()) \
            * int(s.n_layers)
        del probe
        if device.type == "cuda":
            torch.cuda.empty_cache()
        import math

        def plan(n_vol: int, n_surf: int) -> float:
            n_chunks = max(1, math.ceil(n_vol / NV),
                           math.ceil(max(n_surf, 1) / NS))
            return n_chunks * (hooked + manual)

        return plan

    # ---- FLOPs manual terms -----------------------------------------------
    def _attn_terms(self, cfg, nv, ns, n_inner):
        s = cfg.spider
        C = s.n_hidden
        H = s.n_heads
        D = C // H
        G = s.onion_num * s.n_regions
        T = G + s.n_regions
        coarse_rows = n_inner + ns
        return {
            "coarse_sdpa": sdpa_flops(coarse_rows, coarse_rows, H, D),
            "fine_sdpa": sdpa_flops(T, T, H, D),
            "sector_segment": segment_mean_flops(nv, C, G)
            + segment_mean_flops(ns, C, s.n_regions),
            "t2p_matmul": 2.0 * (nv * G * C + ns * G * C
                                 + ns * s.n_regions * C),
        }

    def manual_train_flops(self, cfg):
        s = cfg.spider
        nv = cfg.training.n_query - s.surf_quota
        terms = self._attn_terms(cfg, nv, s.surf_quota, s.n_inner)
        return {k: v * s.n_layers for k, v in terms.items()}

    def manual_infer_flops(self, cfg, n_vol, n_surf):
        import math
        s = cfg.spider
        n = n_vol + n_surf
        n_chunks = max(1, math.ceil(n / int(s.infer_chunk)))
        nv_c = n_vol / n_chunks
        ns_c = n_surf / n_chunks
        terms = self._attn_terms(cfg, nv_c, ns_c,
                                 min(s.n_inner, int(nv_c)))
        return {k: v * s.n_layers * n_chunks for k, v in terms.items()}
