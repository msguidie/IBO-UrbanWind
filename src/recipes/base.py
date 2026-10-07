"""Recipe protocol: the one interface through which the trainer and the
evaluation pass drive every model.

A recipe (one `--model` entry) binds together:
  - the model builder,
  - its view of the shared estate packs (what it loads, how it samples),
  - the training forward + loss (always reporting the same breakdown keys),
  - the optimizer and scheduler at the model's own step granularity,
  - the full-case inference producer (every point exactly once),
  - the non-Linear FLOPs terms (evaluation/flops.py meters the Linear part
    exactly; recipes add attention / einsum / segment / trigonometric
    terms),
  - model-specific derived preprocessing (only SpiderSolver has one).

The trainer itself contains no per-model branches.
"""
from __future__ import annotations

import torch


def move_to_device(batch, device: torch.device):
    """Generic device mover for recipe batches: tensors move, lists recurse,
    everything else passes through."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(v, device) for v in batch)
    return batch


def cpu_defloat_bf16(batch):
    """CPU fallback (tests/diagnostics): bf16 tensors -> fp32."""
    if isinstance(batch, torch.Tensor):
        return batch.float() if batch.is_floating_point() \
            and batch.dtype == torch.bfloat16 else batch
    if isinstance(batch, dict):
        return {k: cpu_defloat_bf16(v) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(cpu_defloat_bf16(v) for v in batch)
    return batch


class Recipe:
    name: str = ""
    family: str = ""
    tex_row_name: str = ""
    adapter_version: str = ""
    # hardware profiles (labels) this recipe may be launched with
    supported_profiles: tuple[str, ...] = ("a100", "h100")
    # fixed effective global batch (None = world_size * batch_size, no
    # accumulation); every recipe on the menu uses 4 = 1 x 4 GPUs
    effective_batch: int | None = None
    # the model supports activation checkpointing (off by default)
    supports_activation_checkpointing: bool = False
    # build_train_dataset salts its per-step draws with the training seed
    # (utils.seed.train_seed_salt); only such recipes accept `train --seed`
    supports_train_seed: bool = False
    # torch._dynamo's DDPOptimizer (graph splitting by all-reduce bucket)
    # cannot trace every graph (SpiderSolver's forward takes an int split
    # point plus index tensors); such a recipe turns it off. The math is
    # unchanged, only the backward / all-reduce overlap.
    disable_ddp_optimizer: bool = False

    # ---- config plumbing -------------------------------------------------
    def apply(self, cfg, world: int) -> None:
        """Install this recipe's training settings and run identity into cfg.
        Called once by main.py before config_used.yaml is written; explicit
        CLI flags are applied afterwards and win."""
        cfg.run.recipe = self.name
        cfg.run.family = self.family
        cfg.run.native_input_adapter = self.adapter_version
        if cfg.hardware.profile not in self.supported_profiles:
            raise SystemExit(
                f"recipe {self.name!r} does not define an "
                f"{cfg.hardware.profile!r} hardware profile "
                f"(supported: {self.supported_profiles})")
        eff = self.effective_batch
        if eff is None:
            eff = world * cfg.training.batch_size
            cfg.training.grad_accum = 1
        else:
            denom = world * cfg.training.batch_size
            if eff % denom != 0:
                raise SystemExit(
                    f"recipe {self.name!r}: effective batch {eff} is not "
                    f"divisible by world*microbatch = {denom} — launch at a "
                    f"world size that divides it")
            cfg.training.grad_accum = eff // denom
        cfg.run.effective_batch = eff
        cfg.run.world_size = world
        if self.supports_activation_checkpointing:
            cfg.training.activation_checkpointing = \
                (cfg.hardware.profile == "a100")
        else:
            cfg.training.activation_checkpointing = False
        self.apply_optim_defaults(cfg)

    def apply_optim_defaults(self, cfg) -> None:
        """Install the recipe's optimizer numbers into cfg.training so the
        run's config_used.yaml records the values actually used."""

    def check_pool(self, cfg, pool: dict) -> None:
        """Eligibility check against the pool manifest, before any pack
        loads."""

    # ---- data ------------------------------------------------------------
    def build_train_dataset(self, cfg, estate_shard, dirs_by_estate):
        raise NotImplementedError

    def collate(self):
        raise NotImplementedError

    def batch_to_device(self, batch, device):
        return move_to_device(batch, device)

    # ---- model -----------------------------------------------------------
    def build_model(self, cfg) -> torch.nn.Module:
        raise NotImplementedError

    def compile_plan(self, cfg) -> tuple[str, bool, str]:
        """(mode, fullgraph, note); mode "" disables compile. The trainer
        prints and records the choice."""
        return (cfg.training.compile_mode, True, "default whole-model plan")

    # ---- training step ----------------------------------------------------
    def forward_loss(self, model, batch, cfg):
        raise NotImplementedError

    def checkpointed_modules(self, model) -> list:
        """The modules wrapped by activation checkpointing when
        cfg.training.activation_checkpointing is on — the train-FLOPs meter
        adds ONE extra forward of exactly these (the recompute that actually
        executes in backward). Recipes that support checkpointing must keep
        their manual FLOPs terms inside these modules."""
        return []

    def build_optimizer(self, cfg, model, opt_steps_per_epoch: int):
        """-> (optimizer, scheduler, granularity in {"step", "epoch"})."""
        raise NotImplementedError

    # ---- inference ---------------------------------------------------------
    def stream_cases(self, cfg, names: list[str]):
        """Yield (case_name, full case dict) for the post pass."""
        raise NotImplementedError

    def prepare_infer_model(self, model, cfg, use_cuda: bool) -> dict:
        """Optionally compile inference entry points. Returns the explicit
        status dict recorded into telemetry (mode/why). A recipe that
        declares compiled inference sets "expect_compiled": True and either
        verifies its own graph health here ("health": {...}) or leaves it to
        the post driver's post-warmup check — a downgrade is a hard failure
        either way, never a silent eager fallback."""
        return {"mode": "eager", "note": "recipe default: eager inference",
                "expect_compiled": False}

    def infer_flops_planner(self, cfg, sample_case: dict, device):
        """FLOPs plan for a COMPILED producer (hooks are blind inside
        compiled graphs): a callable (n_vol, n_surf) -> flops built from an
        EAGER probe at the exact static chunk/padding shapes the producer
        executes. None (default) = the producer is eager and Phase P meters
        it directly."""
        return None

    # Stochastic producers (AB-UPT's anchor draw): post sets `infer_draw`
    # to 0..infer_draws-1 around each infer_case call; draw 0 is the fixed
    # per-case seed whose predictions are persisted. Deterministic
    # producers ignore it and report infer_draws() == 1.
    infer_draw: int = 0

    def infer_draws(self, cfg) -> int:
        return 1

    def infer_case(self, model, case: dict, cfg, device, use_cuda: bool):
        """-> (pred_vol_z [N_vol, C_vol], pred_surf_z [N_surf, 1]) fp32
        numpy, original row order, every point exactly once."""
        raise NotImplementedError

    # ---- FLOPs (non-Linear manual terms; Linears are metered exactly) ----
    def manual_train_flops(self, cfg) -> dict[str, float]:
        return {}

    def manual_infer_flops(self, cfg, n_vol: int, n_surf: int
                           ) -> dict[str, float]:
        return {}

    # ---- derived preprocessing --------------------------------------------
    def ensure_derived(self, cfg, estates: list[str]) -> None:
        """Raise (with build instructions) if required derived caches are
        missing. Default: nothing required."""

    def run_derive(self, cfg, estates: list[str]) -> None:
        raise SystemExit(f"recipe {self.name!r} has no derived "
                         f"preprocessing to build")
