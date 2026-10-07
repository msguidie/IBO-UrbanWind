"""Recipe "ibo": the IBO model on the importance-weighted partition, and
the two layout ablations of the paper with the same architecture:
"ibo_equal" (adaptive partition with unit weights, i.e. equal point count
per leaf) and "ibo_grid" (a regular 32 x 32 x 16 grid).
"""
from __future__ import annotations

import torch

from ..data.dataloader import batch_to_device, collate
from ..data.estate import RamEstateDataset, stream_full_estate_cases
from ..evaluation.flops import sdpa_flops
from ..model.full_model import IBO
from ..training.losses import compute_loss
from ..training.schedules import build_ibo_scheduler
from ..training.lion import Lion
from ..utils.seed import train_seed_salt
from .base import Recipe, cpu_defloat_bf16


class IBORecipe(Recipe):
    name = "ibo"
    family = "ibo"
    tex_row_name = "IBO"
    adapter_version = "ibo_native_v1"
    supported_profiles = ("a100", "h100")
    supports_train_seed = True

    def apply_optim_defaults(self, cfg) -> None:
        # IBO's optimizer settings live in cfg.training already
        pass

    def build_train_dataset(self, cfg, estate_shard, dirs_by_estate):
        return RamEstateDataset(cfg, estate_shard, cfg.training.n_query,
                                mirror_augment=cfg.training.mirror_augment,
                                dirs_by_estate=dirs_by_estate,
                                seed_salt=train_seed_salt(cfg))

    def collate(self):
        return collate

    def batch_to_device(self, batch, device):
        return batch_to_device(batch, device)

    def build_model(self, cfg) -> torch.nn.Module:
        return IBO(cfg.model)

    def forward_loss(self, model, batch, cfg):
        tc = cfg.training
        pred_vol, pred_surf = model(
            pn_points_feat=batch["pn_points_feat"], pn_mask=batch["pn_mask"],
            leaf_stats=batch["leaf_stats"],
            leaf_pos_norm=batch["leaf_pos_norm"],
            global_morph_norm=batch["global_morph_norm"],
            query_pos_norm=batch["query_pos_norm"],
            query_sdf_norm=batch["query_sdf_norm"],
            query_sdf_dir=batch["query_sdf_dir"],
            query_support_norm=batch["query_support_norm"],
            region_dim1=batch["region_dim1"], region_dim2=batch["region_dim2"],
            region_dim3=batch["region_dim3"],
            idw_indices=batch["idw_indices"])
        return compute_loss(pred_vol, pred_surf, batch["target_vol"],
                            batch["target_surf"], batch["is_volume"],
                            tc.loss_vol_weights, tc.loss_shape,
                            tc.huber_delta)

    def build_optimizer(self, cfg, model, opt_steps_per_epoch: int):
        tc = cfg.training
        optimizer = Lion(model.parameters(), lr=tc.lr,
                         weight_decay=tc.weight_decay)
        total = tc.num_epochs * opt_steps_per_epoch
        ref = tc.lr_schedule_ref_epochs * opt_steps_per_epoch \
            if tc.lr_schedule_ref_epochs is not None else None
        sched = build_ibo_scheduler(
            optimizer, total, tc.warmup_fraction, tc.min_lr_fraction,
            tc.lr_schedule, tc.wsd_stable_fraction, ref_total_steps=ref,
            warmup_steps=tc.warmup_steps)
        return optimizer, sched, "step"

    # ---- inference --------------------------------------------------------
    def stream_cases(self, cfg, names):
        return stream_full_estate_cases(cfg, names, with_idw=True)

    def prepare_infer_model(self, model, cfg, use_cuda: bool) -> dict:
        if not use_cuda:
            return {"mode": "eager", "note": "CPU",
                    "expect_compiled": False}
        # encode and decode compiled separately, one static shape each
        # (decode pads its final chunk); graph health is verified by the
        # post driver after the warm-up case
        from ..utils.compile_health import reset_compile_state
        reset_compile_state()
        model.encode = torch.compile(model.encode, mode="reduce-overhead",
                                     fullgraph=True, dynamic=False)
        model.decode = torch.compile(model.decode, mode="reduce-overhead",
                                     fullgraph=True, dynamic=False)
        return {"mode": "reduce-overhead(encode)/reduce-overhead(decode)",
                "fullgraph": True, "dynamic": False, "expect_compiled": True,
                "note": "encode-once + fixed-chunk decode, pads final chunk"}

    def infer_case(self, model, case, cfg, device, use_cuda):
        from ..evaluation.test_worker import infer_case as ibo_infer
        if not use_cuda:
            case = cpu_defloat_bf16(case)
        return ibo_infer(model, case, device,
                         cfg.evaluation.test_chunk_size, use_cuda)

    def infer_flops_planner(self, cfg, sample_case, device):
        from ..evaluation.post import _IBOFlopPlanner
        return _IBOFlopPlanner(cfg, self, sample_case, device=device)

    # ---- FLOPs manual terms -----------------------------------------------
    @staticmethod
    def _functional_act_terms(m, L, n_rows) -> dict:
        """F.silu sites the module hooks cannot see (4 FLOPs/elem): the ViT
        SwiGLU gate, the adaLN trunk, and the decoder's silu-activated
        projections (vol+surf self, pair, head-256)."""
        return {
            "vit_swiglu_silu": 4.0 * m.n_vit_layers * L * m.ffn_hidden,
            "adaln_silu": 4.0 * m.adaln_hidden,
            "decoder_silu": 4.0 * n_rows * (2.0 * m.decoder_dh
                                            + m.idw_k * m.decoder_dh + 256.0),
        }

    def manual_train_flops(self, cfg):
        m = cfg.model
        L = m.n_leaves + m.n_register
        n_q = cfg.training.n_query * cfg.training.batch_size
        terms = {
            "vit_sdpa": m.n_vit_layers * sdpa_flops(L, L, m.num_heads,
                                                    m.head_dim),
            "vit_rope_trig": 2.0 * m.n_vit_layers * L * m.head_dim,
            # decoder per-query neighbour attention: scores + value agg per
            # stage-2 head over idw_k neighbours (+ softmax)
            "decoder_neighbour_attn": n_q * (
                4.0 * m.idw_k * m.decoder_dh + 3.0 * m.decoder_n_heads
                * m.idw_k),
            "decoder_fourier_trig": 2.0 * n_q * m.fourier_dim,
            "pointnet_pooling": 3.0 * m.n_leaves * m.n_pn_neighbors
            * m.pn_out_dim,
        }
        terms.update(self._functional_act_terms(m, L, n_q))
        return terms

    def manual_infer_flops(self, cfg, n_vol, n_surf):
        m = cfg.model
        L = m.n_leaves + m.n_register
        n = n_vol + n_surf
        chunk = cfg.evaluation.test_chunk_size
        import math
        n_padded = int(math.ceil(n / chunk)) * chunk
        terms = {
            "vit_sdpa": m.n_vit_layers * sdpa_flops(L, L, m.num_heads,
                                                    m.head_dim),
            "vit_rope_trig": 2.0 * m.n_vit_layers * L * m.head_dim,
            "decoder_neighbour_attn": n_padded * (
                4.0 * m.idw_k * m.decoder_dh
                + 3.0 * m.decoder_n_heads * m.idw_k),
            "decoder_fourier_trig": 2.0 * n_padded * m.fourier_dim,
            "pointnet_pooling": 3.0 * m.n_leaves * m.n_pn_neighbors
            * m.pn_out_dim,
        }
        terms.update(self._functional_act_terms(m, L, n_padded))
        return terms


class IBOEqualRecipe(IBORecipe):
    """"ibo_equal": the IBO architecture on the adaptive partition with unit
    weights (equal point count per leaf); reads the `--grid equal` cache
    generation."""
    name = "ibo_equal"
    tex_row_name = "IBO (equal count)"
    adapter_version = "ibo_native_v1+equal"

    def apply(self, cfg, world: int) -> None:
        cfg.preprocess.kdtree_weighting = "equal"
        cfg.resolve_paths()
        super().apply(cfg, world)


class IBOGridRecipe(IBORecipe):
    """"ibo_grid": the IBO architecture on a regular 32 x 32 x 16 grid;
    reads the `--grid structured` cache generation."""
    name = "ibo_grid"
    tex_row_name = "IBO (uniform grid)"
    adapter_version = "ibo_native_v1+grid"

    def apply(self, cfg, world: int) -> None:
        cfg.preprocess.kdtree_weighting = "structured"
        cfg.resolve_paths()
        super().apply(cfg, world)
