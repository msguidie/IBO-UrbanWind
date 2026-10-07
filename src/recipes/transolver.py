"""Recipe "transolver" — Transolver-3 on UrbanWind (see
config.TransolverConfig). Native 7-channel input, joint vol(5)+surf(1)
heads, uniform 128K mixed sampling, AdamW 1e-3 / wd 0.01 / no warmup /
per-EPOCH cosine to 1e-5 (the native scheduler granularity). Effective
batch 4 = microbatch 1 x 4 GPUs, grad_accum 1 (one update per
forward/backward, identical to every other model); no activation
checkpointing; training compiles with CUDA Graphs.
Full-field inference = the EXACT global two-pass slice aggregation
(model/transolver.py forward_chunks) over the whole case — never a
chunk-local approximation."""
from __future__ import annotations

import numpy as np
import torch

from ..data.estate import stream_full_estate_cases, VOL_CHANNEL_NAMES
from ..data.pointcloud import PointcloudDataset, collate_single
from ..evaluation.flops import einsum_flops, sdpa_flops
from ..model.transolver import build_transolver_model
from ..training.losses import compute_loss
from ..training.schedules import build_warmup_cosine
from .base import Recipe


class TransolverRecipe(Recipe):
    name = "transolver"
    family = "transolver"
    tex_row_name = "Transolver-3"
    adapter_version = "native7_v1"
    supported_profiles = ("a100", "h100")
    effective_batch = 4
    supports_activation_checkpointing = False

    def apply_optim_defaults(self, cfg) -> None:
        cfg.training.lr_base = cfg.transolver.lr
        cfg.training.weight_decay_base = cfg.transolver.weight_decay

    def compile_plan(self, cfg):
        """TRAINING: reduce-overhead (inductor + CUDA Graphs), fullgraph.

        History: with the reference's activation checkpointing ON this
        aborted backward ("accessing tensor output of CUDAGraphs that has
        been overwritten") — the checkpointed region recomputes while the
        graph buffers have been reused. Checkpointing is now OFF (measured
        peak 2.4-9 GiB of 40 GiB), so the hazard is gone and the graph is
        captured. A capture refusal shows up as a logged inductor skip, not
        as wrong numbers."""
        return ("reduce-overhead", True,
                "CUDA Graphs on: activation checkpointing removed "
                "(memory headroom was never the constraint)")

    def build_train_dataset(self, cfg, estate_shard, dirs_by_estate):
        return PointcloudDataset(cfg, estate_shard, "transolver",
                                 dirs_by_estate=dirs_by_estate)

    def collate(self):
        return collate_single

    def build_model(self, cfg) -> torch.nn.Module:
        return build_transolver_model(
            cfg.transolver, out_dim_vol=len(VOL_CHANNEL_NAMES),
            use_checkpoint=cfg.training.activation_checkpointing)

    def forward_loss(self, model, batch, cfg):
        tc = cfg.training
        pv, ps = model(batch["x"].unsqueeze(0))
        return compute_loss(pv, ps, batch["target_vol"].unsqueeze(0),
                            batch["target_surf"].unsqueeze(0),
                            batch["is_volume"].unsqueeze(0),
                            tc.loss_vol_weights, tc.loss_shape,
                            tc.huber_delta)

    def build_optimizer(self, cfg, model, opt_steps_per_epoch: int):
        t = cfg.transolver
        optimizer = torch.optim.AdamW(model.parameters(), lr=t.lr,
                                      weight_decay=t.weight_decay)
        # native granularity: ONE scheduler step per EPOCH (Transolver-3's
        # CosineAnnealingLR is stepped outside the batch loop), no warmup
        sched = build_warmup_cosine(optimizer, cfg.training.num_epochs,
                                    t.lr, t.min_lr, warmup_fraction=0.0)
        return optimizer, sched, "epoch"

    def checkpointed_modules(self, model) -> list:
        from ..training.swa import unwrap_model
        return list(unwrap_model(model).blocks)

    # ---- inference --------------------------------------------------------
    def stream_cases(self, cfg, names):
        return stream_full_estate_cases(cfg, names, with_idw=False)

    @staticmethod
    def _case_x(case: dict) -> torch.Tensor:
        return torch.cat([case["query_pos_norm"].float(),
                          case["query_sdf_norm"].float().unsqueeze(-1),
                          case["query_sdf_dir"].float()], dim=-1)

    def prepare_infer_model(self, model, cfg, use_cuda: bool) -> dict:
        """GPU: compile the per-block static-chunk entry points (fixed
        infer_chunk rows, final chunk PADDED and exactly masked out of the
        global slice statistics) with fullgraph + dynamic=False +
        reduce-overhead, warm them on synthetic tensors of the REAL shapes,
        then VERIFY graph capture — a downgrade hard-fails. The tiny GxG
        slice attention between the two passes stays eager (recorded).
        CPU: the exact eager forward_chunks path (tests)."""
        self._infer = {"chunk": int(cfg.transolver.infer_chunk),
                       "compiled": None}
        if not use_cuda:
            return {"mode": "eager", "note": "CPU (tests) — exact "
                    "forward_chunks path", "expect_compiled": False}
        from ..utils.compile_health import compile_health, \
            reset_compile_state
        reset_compile_state()
        kw = dict(mode="reduce-overhead", fullgraph=True, dynamic=False)
        embed_c = torch.compile(model.embed, **kw)
        stats_c = [torch.compile(b.stats_masked, **kw) for b in model.blocks]
        apply_c = [torch.compile(b.apply_chunk, **kw) for b in model.blocks]

        def heads(fx):
            h = model.ln_out(fx)
            return model.vol_head(h), model.surf_head(h)

        heads_c = torch.compile(heads, **kw)
        # warmup at the REAL static shapes (values are irrelevant to shape/
        # graph capture); the mask carries a zero tail like a padded chunk
        device = next(model.parameters()).device
        chunk = self._infer["chunk"]
        x = torch.randn(1, chunk, int(cfg.transolver.in_dim), device=device)
        mask = torch.ones(1, chunk, device=device)
        mask[:, -7:] = 0.0
        with torch.inference_mode(), torch.autocast("cuda",
                                                    dtype=torch.bfloat16):
            fx = embed_c(x).clone()
            for li, block in enumerate(model.blocks):
                num, den = stats_c[li](fx, mask)
                token = num / (den[..., None] + 1e-5)
                out_slice = block.Attn.slice_attend(token)
                fx = apply_c[li](fx, out_slice).clone()
            heads_c(fx)
        torch.cuda.synchronize(device)
        status = compile_health("transolver-infer", expect_cudagraphs=True)
        self._infer["compiled"] = (embed_c, stats_c, apply_c, heads_c)
        return {"mode": "reduce-overhead", "fullgraph": True,
                "dynamic": False, "expect_compiled": True,
                "health": status,
                "note": "masked fixed chunks (padding exactly removed from "
                        "global slice stats); slice_attend (GxG) eager"}

    @torch.inference_mode()
    def infer_case(self, model, case, cfg, device, use_cuda):
        n_vol = int(case["N_vol"])
        x = self._case_x(case)
        n = x.shape[0]
        chunk = int(cfg.transolver.infer_chunk)
        compiled = getattr(self, "_infer", {}).get("compiled")
        if not use_cuda or compiled is None:
            # exact eager path (CPU tests); the last chunk is simply shorter
            chunks = [x[lo:lo + chunk].unsqueeze(0).to(device)
                      for lo in range(0, n, chunk)]
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=use_cuda):
                outs = model.forward_chunks(chunks)
            pv = torch.cat([o[0].squeeze(0).float().cpu() for o in outs])
            ps = torch.cat([o[1].squeeze(0).float().cpu() for o in outs])
            return (pv[:n_vol].numpy().astype(np.float32),
                    ps[n_vol:].numpy().astype(np.float32))

        embed_c, stats_c, apply_c, heads_c = compiled
        import math
        n_chunks = max(1, math.ceil(n / chunk))
        pad = n_chunks * chunk - n
        x_pad = torch.cat([x, x.new_zeros(pad, x.shape[1])]) if pad else x
        xs = [x_pad[c * chunk:(c + 1) * chunk].unsqueeze(0).to(device)
              for c in range(n_chunks)]
        masks = []
        for c in range(n_chunks):
            m = torch.ones(1, chunk, device=device)
            if c == n_chunks - 1 and pad:
                m[:, chunk - pad:] = 0.0
            masks.append(m)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # .clone() everywhere a compiled output must survive the next
            # CUDA-Graph replay — graph-owned memory is reused per replay
            fx = [embed_c(xc).clone() for xc in xs]
            for li in range(len(stats_c)):
                num, den = None, None
                for c in range(n_chunks):
                    nk, dk = stats_c[li](fx[c], masks[c])
                    num = nk.clone() if num is None else num + nk
                    den = dk.clone() if den is None else den + dk
                token = num / (den[..., None] + 1e-5)
                from ..training.swa import unwrap_model
                block = unwrap_model(model).blocks[li]
                out_slice = block.Attn.slice_attend(token)
                fx = [apply_c[li](fx[c], out_slice).clone()
                      for c in range(n_chunks)]
            pv_parts, ps_parts = [], []
            for c in range(n_chunks):
                pv_c, ps_c = heads_c(fx[c])
                pv_parts.append(pv_c.squeeze(0).float().cpu())
                ps_parts.append(ps_c.squeeze(0).float().cpu())
        pv = torch.cat(pv_parts)[:n]
        ps = torch.cat(ps_parts)[:n]
        return (pv[:n_vol].numpy().astype(np.float32),
                ps[n_vol:].numpy().astype(np.float32))

    def infer_flops_planner(self, cfg, sample_case, device):
        """Eager probe of ONE padded chunk (the real static shape) on a
        FRESH eager instance — Linears/norms/activations metered by hooks,
        the einsum/softmax/SDPA terms added in closed form; per case =
        n_chunks x per-chunk (every chunk is shape-identical)."""
        from ..evaluation.flops import FlopMeter
        probe = self.build_model(cfg).to(device)
        probe.eval()
        chunk = int(cfg.transolver.infer_chunk)
        x = torch.randn(1, chunk, int(cfg.transolver.in_dim), device=device)
        mask = torch.ones(1, chunk, device=device)
        with FlopMeter(probe) as m, torch.inference_mode():
            fx = probe.embed(x)
            for block in probe.blocks:
                num, den = block.stats_masked(fx, mask)
                token = num / (den[..., None] + 1e-5)
                out_slice = block.Attn.slice_attend(token)
                fx = block.apply_chunk(fx, out_slice)
            h = probe.ln_out(fx)
            probe.vol_head(h), probe.surf_head(h)
        hooked_per_chunk = m.total
        L = int(cfg.transolver.n_layers)
        manual_per_chunk = sum(self._block_einsum_terms(
            cfg, chunk, n_passes_weights=2.0).values()) * L
        del probe
        if device.type == "cuda":
            torch.cuda.empty_cache()
        import math

        def plan(n_vol: int, n_surf: int) -> float:
            n_chunks = max(1, math.ceil((n_vol + n_surf) / chunk))
            return n_chunks * (hooked_per_chunk + manual_per_chunk)

        return plan

    # ---- FLOPs manual terms -----------------------------------------------
    def _block_einsum_terms(self, cfg, n_rows, n_passes_weights=1.0):
        """Per BLOCK non-Linear terms for n_rows points: slice-weight
        logits einsum, x_agg einsum, num projection einsum, out-projection
        einsum, deslice einsum, softmax; the slice attention SDPA."""
        t = cfg.transolver
        H, G = t.n_heads, t.slice_num
        C = t.n_hidden
        D = C // H
        weights = einsum_flops(n_rows * H * G, C) + 3.0 * n_rows * H * G
        agg = einsum_flops(H * G * C, n_rows)
        num = einsum_flops(H * G * D, C)
        proj = einsum_flops(H * G * C, D)
        deslice = einsum_flops(n_rows * C, H * G)
        attn = sdpa_flops(G, G, H, D)
        return {
            "slice_weights": n_passes_weights * weights,
            "slice_agg": agg, "slice_num_proj": num,
            "slice_attention": attn, "deslice_proj": proj,
            "deslice": deslice,
        }

    def manual_train_flops(self, cfg):
        n = cfg.training.n_query
        L = cfg.transolver.n_layers
        # single-chunk training path computes the slice weights once per
        # block (forward()) — reuse inside the block
        terms = self._block_einsum_terms(cfg, n, n_passes_weights=1.0)
        return {k: v * L for k, v in terms.items()}

    def manual_infer_flops(self, cfg, n_vol, n_surf):
        n = n_vol + n_surf
        L = cfg.transolver.n_layers
        # forward_chunks recomputes the slice weights in pass 2 (deslice) —
        # count them twice, faithfully to the executed algorithm
        terms = self._block_einsum_terms(cfg, n, n_passes_weights=2.0)
        return {k: v * L for k, v in terms.items()}
