"""Recipe "abupt": AB-UPT (Alkin et al., 2025) without the geometry branch.
16384 + 16384 anchors, supervised on its anchor points; Lion 5e-5, weight
decay 0.05, 5% warmup, cosine to 1e-6 (the AB-UPT paper's recipe).
Full-field inference caches the anchors' keys and values once per case and
decodes the remaining points in fixed-size chunks."""
from __future__ import annotations

import torch

from ..data.abupt_estate import RamEstateDatasetABUPT
from ..data.estate import stream_full_estate_cases
from ..evaluation.flops import sdpa_flops
from ..model.abupt import build_abupt_model
from ..training.losses import compute_loss_branch
from ..training.lion import Lion
from ..training.schedules import build_warmup_cosine
from ..utils.seed import train_seed_salt
from .base import Recipe, cpu_defloat_bf16

_GEO_BATCH_KEYS = (
    "surface_sdf", "surface_sdf_dir", "surface_support",
    "surface_region_dim2", "surface_region_dim3",
    "volume_sdf", "volume_sdf_dir", "volume_support",
    "volume_region_dim2", "volume_region_dim3")


def collate_abupt(samples: list[dict]) -> dict:
    keys = samples[0].keys()
    return {k: torch.stack([s[k] for s in samples]).contiguous()
            for k in keys}


class AbuptRecipe(Recipe):
    name = "abupt"
    family = "abupt"
    tex_row_name = "AB-UPT"
    supports_train_seed = True
    adapter_version = "abupt_reference_port"
    supported_profiles = ("a100", "h100")

    def apply_optim_defaults(self, cfg) -> None:
        ac = cfg.abupt
        cfg.training.lr_base = ac.lr_base
        cfg.training.weight_decay_base = ac.weight_decay_base

    def check_pool(self, cfg, pool: dict) -> None:
        if not pool.get("abupt_eligible", False):
            raise SystemExit(
                "this pool is not AB-UPT-eligible (its estates do not all "
                "cover 16384 surface anchors). Train AB-UPT on "
                "manifest_400.json.")

    def build_train_dataset(self, cfg, estate_shard, dirs_by_estate):
        assert not cfg.training.mirror_augment, \
            "baseline recipes run mirror-free by protocol"
        return RamEstateDatasetABUPT(cfg, estate_shard, mirror_augment=False,
                                     dirs_by_estate=dirs_by_estate,
                                     seed_salt=train_seed_salt(cfg))

    def collate(self):
        return collate_abupt

    def build_model(self, cfg) -> torch.nn.Module:
        return build_abupt_model(cfg)

    def forward_loss(self, model, batch, cfg):
        tc = cfg.training
        kwargs = {"surface_position": batch["surface_position"],
                  "volume_position": batch["volume_position"]}
        if cfg.abupt.geometry_features:
            for key in _GEO_BATCH_KEYS:
                kwargs[key] = batch[key]
        pred_surf, pred_vol = model(**kwargs)
        return compute_loss_branch(pred_vol, pred_surf, batch["target_vol"],
                                   batch["target_surf"],
                                   tc.loss_vol_weights, tc.loss_shape,
                                   tc.huber_delta)

    def build_optimizer(self, cfg, model, opt_steps_per_epoch: int):
        ac, tc = cfg.abupt, cfg.training
        optimizer = Lion(model.parameters(), lr=ac.lr_base,
                         weight_decay=ac.weight_decay_base)
        total = tc.num_epochs * opt_steps_per_epoch
        sched = build_warmup_cosine(optimizer, total, ac.lr_base, ac.end_lr,
                                    ac.warmup_fraction)
        return optimizer, sched, "step"

    # ---- inference --------------------------------------------------------
    def stream_cases(self, cfg, names):
        return stream_full_estate_cases(cfg, names, with_idw=False)

    def prepare_infer_model(self, model, cfg, use_cuda: bool) -> dict:
        """GPU: BOTH cached-path entry points compiled (fullgraph +
        dynamic=False + reduce-overhead) — encode_anchor_cache has ONE
        static shape (n_anchors per branch) and decode_queries_cached ONE
        static chunk shape (fullfield_query_chunk_per_branch, padded). The
        cache pytree that crosses the boundary is CLONED right after encode
        (abupt_eval), so no graph-owned tensor is ever read after a later
        replay overwrote it. Warmed on synthetic real-shape tensors, then
        graph capture is VERIFIED. CPU: eager (tests)."""
        self._infer = {"encode": None, "decode": None}
        if not use_cuda:
            return {"mode": "eager", "note": "CPU (tests)",
                    "expect_compiled": False}
        from ..utils.compile_health import compile_health, \
            reset_compile_state
        reset_compile_state()
        ac = cfg.abupt
        kw = dict(mode="reduce-overhead", fullgraph=True, dynamic=False)
        enc_c = torch.compile(model.encode_anchor_cache, **kw)
        dec_c = torch.compile(model.decode_queries_cached, **kw)
        device = next(model.parameters()).device
        na, chunk = int(ac.n_anchors), int(ac.fullfield_query_chunk_per_branch)
        sp = torch.rand(1, na, 3, device=device)
        vp = torch.rand(1, na, 3, device=device)
        sq = torch.rand(1, chunk, 3, device=device)
        vq = torch.rand(1, chunk, 3, device=device)
        from ..evaluation.abupt_eval import clone_cache
        with torch.inference_mode(), torch.autocast("cuda",
                                                    dtype=torch.bfloat16):
            _, _, cache = enc_c(surface_position=sp, volume_position=vp)
            cache = clone_cache(cache)
            dec_c(cache, surface_position=sq, volume_position=vq)
        torch.cuda.synchronize(device)
        status = compile_health("abupt-infer", expect_cudagraphs=True)
        self._infer["encode"] = enc_c
        self._infer["decode"] = dec_c
        return {"mode": "reduce-overhead", "fullgraph": True,
                "dynamic": False, "expect_compiled": True,
                "health": status,
                "note": "encode-once (anchor shape) + fixed padded decode "
                        "chunks; anchor-KV cache cloned off graph-owned "
                        "memory at the boundary"}

    def infer_draws(self, cfg) -> int:
        return int(cfg.abupt.infer_draws)

    def infer_case(self, model, case, cfg, device, use_cuda):
        from ..evaluation.abupt_eval import infer_case_fullfield
        if not use_cuda:
            case = cpu_defloat_bf16(case)
        infer = getattr(self, "_infer", {})
        return infer_case_fullfield(model, case, cfg, device, use_cuda,
                                    encode_fn=infer.get("encode"),
                                    decode_fn=infer.get("decode"),
                                    draw=int(self.infer_draw))

    def infer_flops_planner(self, cfg, sample_case, device):
        """Eager decomposition on a FRESH instance: encode is a constant
        (anchor shape), decode a per-padded-chunk constant; the SDPA/RoPE
        terms come from the closed forms."""
        from ..evaluation.flops import FlopMeter
        ac = cfg.abupt
        probe = self.build_model(cfg).to(device)
        probe.eval()
        na, chunk = int(ac.n_anchors), int(ac.fullfield_query_chunk_per_branch)
        sp = torch.rand(1, na, 3, device=device)
        vp = torch.rand(1, na, 3, device=device)
        sq = torch.rand(1, chunk, 3, device=device)
        vq = torch.rand(1, chunk, 3, device=device)
        with FlopMeter(probe) as m1, torch.inference_mode():
            _, _, cache = probe.encode_anchor_cache(surface_position=sp,
                                                    volume_position=vp)
        encode_hooked = m1.total
        with FlopMeter(probe) as m2, torch.inference_mode():
            probe.decode_queries_cached(cache, surface_position=sq,
                                        volume_position=vq)
        decode_hooked = m2.total
        del probe, cache
        if device.type == "cuda":
            torch.cuda.empty_cache()
        import math

        def plan(n_vol: int, n_surf: int) -> float:
            n_sq = max(n_surf - na, 0)
            n_vq = max(n_vol - na, 0)
            n_steps = max(math.ceil(n_sq / chunk), math.ceil(n_vq / chunk))
            manual = sum(self.manual_infer_flops(cfg, n_vol,
                                                 n_surf).values())
            return encode_hooked + n_steps * decode_hooked + manual

        return plan

    # ---- FLOPs manual terms -----------------------------------------------
    def _trunk_terms(self, cfg, ns_a, nv_a):
        ac = cfg.abupt
        H, hd = ac.num_heads, ac.hidden_dim // ac.num_heads
        t = 0.0
        for ch in ac.shared_blocks:
            if ch == "s":
                t += sdpa_flops(ns_a, ns_a, H, hd) + sdpa_flops(nv_a, nv_a, H, hd)
            else:
                t += sdpa_flops(ns_a, nv_a, H, hd) + sdpa_flops(nv_a, ns_a, H, hd)
        return t

    def manual_train_flops(self, cfg):
        ac = cfg.abupt
        H, hd = ac.num_heads, ac.hidden_dim // ac.num_heads
        n = ac.n_anchors
        branch = (ac.num_surface_blocks + ac.num_volume_blocks) \
            * sdpa_flops(n, n, H, hd)
        rope = 2.0 * 2 * n * hd * (len(ac.shared_blocks)
                                   + ac.num_surface_blocks)
        return {"trunk_sdpa": self._trunk_terms(cfg, n, n),
                "branch_sdpa": branch, "rope_trig": rope}

    def manual_infer_flops(self, cfg, n_vol, n_surf):
        import math
        ac = cfg.abupt
        H, hd = ac.num_heads, ac.hidden_dim // ac.num_heads
        na = ac.n_anchors
        chunk = ac.fullfield_query_chunk_per_branch
        n_sq = max(n_surf - na, 0)
        n_vq = max(n_vol - na, 0)
        n_steps = max(math.ceil(n_sq / chunk), math.ceil(n_vq / chunk), 0)
        q_rows = n_steps * chunk * 2                     # both branches, padded
        # anchor pass (once) + query rows reading cached anchor K/V in every
        # trunk + branch block
        n_blocks = len(ac.shared_blocks) + ac.num_surface_blocks
        anchors = self._trunk_terms(cfg, na, na) \
            + (ac.num_surface_blocks + ac.num_volume_blocks) \
            * sdpa_flops(na, na, H, hd)
        queries = n_blocks * sdpa_flops(q_rows, na, H, hd)
        rope = 2.0 * (2 * na + q_rows) * hd * n_blocks
        return {"anchor_pass_sdpa": anchors, "query_sdpa": queries,
                "rope_trig": rope}
