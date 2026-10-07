"""Configuration dataclasses: the single source of every tunable.

Derived quantities (number of heads, FFN width, encoder input width, cache
key, SWA start epoch, ...) are computed here and nowhere else. Defaults are
the UrbanWind setting of the paper's main table.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


def _expand(p: str | None) -> str | None:
    if p is None:
        return None
    return str(Path(os.path.expanduser(str(p))))


@dataclass
class DataConfig:
    """Two independent choices:

    WHICH ESTATES: pool_manifest_path, one of the pool manifests
    (src/manifest.py), set with `--manifest`.

    WHICH PACKS: cache_estate_dir and norm_stats_path, derived by
    Config.resolve_paths() from the settings that change what is baked into
    a pack (n_leaves, cache_idw_k, n_pn_neighbors, kdtree_weighting,
    pn_surf_first). norm_stats.json always lives inside the cache directory:
    the targets were normalized with it.
    """
    raw_pt_dir: str = "data/raw"
    work_dir: str = "work"
    coord_divisor: float = 550.0
    cache_estate_dir: str | None = None
    norm_stats_path: str | None = None
    pool_manifest_path: str | None = None

    def __post_init__(self):
        self.raw_pt_dir = _expand(self.raw_pt_dir)
        self.work_dir = _expand(self.work_dir)
        w = Path(self.work_dir)
        for k, v in (("cache_estate_dir", w / "cache_estate"),
                     ("norm_stats_path", w / "norm_stats.json")):
            cur = getattr(self, k)
            setattr(self, k, _expand(cur) if cur is not None else str(v))
        self.pool_manifest_path = _expand(self.pool_manifest_path)


@dataclass
class ModelConfig:
    # processor token width; a multiple of head_dim
    latent_dim: int = 384
    # processor head width: latent_dim // head_dim heads
    head_dim: int = 32
    # PointNet output width; the encoder consumes its max/mean/std concat
    pn_out_dim: int = 256
    # decoder width and heads (decoder head width = decoder_dh // heads)
    decoder_dh: int = 192
    decoder_n_heads: int = 6
    # with registers present, let the decoder attend them through a separate
    # softmax (no parameter-shape effect)
    decoder_register_attn: bool = False
    # Fourier ladder on absolute x and y in the query self-embedding (changes
    # the embedding width)
    decoder_xy_fourier_abs: bool = True
    n_register: int = 0
    n_vit_layers: int = 12
    n_leaves: int = 16384
    # PointNet member slots per leaf
    n_pn_neighbors: int = 32
    # decoder neighbours per query, sliced at load time from the k-NN; no
    # weight shape depends on it
    idw_k: int = 8
    # k-NN width of the cache key (>= idw_k)
    cache_idw_k: int = 16
    global_cond_dim: int = 5
    adaln_hidden: int = 64
    fourier_freqs_xy: int = 8
    fourier_freqs_z: int = 6
    dropout: float = 0.1
    # RoPE: per axis, `scale` sets the finest wavelength (2 pi 550 / scale
    # metres) and `base` how far the ladder stretches from there; separate
    # bases because the domain is 1100 m wide and 160 m tall
    rope_base: float | tuple[float, float, float] = (450.0, 450.0, 64.0)
    # RoPE dimensions per axis (x, y, z); must sum to head_dim
    rope_dims: tuple[int, int, int] = (12, 12, 8)
    rope_scale_xy: float = 500.0
    rope_scale_z: float = 500.0
    # optional components (the UrbanWind setting uses none of them)
    disable_unet_skips: bool = True        # gated long skips in the processor
    disable_decoder_stage1: bool = True    # decoder stage reading the encoder
    disable_adaln: bool = False            # morphology conditioning (adaLN)

    def __post_init__(self):
        self.rope_dims = tuple(int(d) for d in self.rope_dims)
        if isinstance(self.rope_base, (list, tuple)):
            self.rope_base = tuple(float(b) for b in self.rope_base)

    @property
    def num_heads(self) -> int:
        return self.latent_dim // self.head_dim

    @property
    def ffn_hidden(self) -> int:
        return 4 * self.latent_dim

    @property
    def encoder_in_dim(self) -> int:
        # PointNet max|mean|std (3 * pn_out_dim) + leaf_stats (19)
        # + latent position (3)
        return 3 * self.pn_out_dim + 19 + 3

    @property
    def pn_in_dim(self) -> int:
        # relative position (3) + sdf (1) + sdf direction (3) + support (6)
        return 13

    @property
    def n_kd_levels(self) -> int:
        return int(round(math.log2(self.n_leaves)))

    @property
    def cache_key(self) -> str:
        return f"L{self.n_leaves}_idw{self.cache_idw_k}_pn{self.n_pn_neighbors}"

    @property
    def fourier_dim(self) -> int:
        # xy branch: 4 inputs x (1 + 2K); z branch: 3 inputs x (1 + 2K)
        return 4 * (1 + 2 * self.fourier_freqs_xy) \
            + 3 * (1 + 2 * self.fourier_freqs_z)

    def validate(self):
        assert self.latent_dim % self.head_dim == 0, \
            "latent_dim must be divisible by head_dim"
        bases = self.rope_base if isinstance(self.rope_base, tuple) \
            else (self.rope_base,) * 3
        assert len(bases) == 3 and all(b > 1.0 for b in bases), \
            "rope_base must be a scalar > 1 or a 3-tuple (x, y, z) of them"
        assert self.fourier_freqs_xy >= 0 and self.fourier_freqs_z >= 0
        assert sum(self.rope_dims) == self.head_dim, (
            f"rope_dims {self.rope_dims} must sum to head_dim {self.head_dim}")
        assert all(d % 2 == 0 for d in self.rope_dims), "rope_dims must be even"
        assert self.decoder_dh % self.decoder_n_heads == 0, \
            "decoder_dh must be divisible by decoder_n_heads"
        assert 0 < self.idw_k <= self.cache_idw_k, (
            f"idw_k={self.idw_k} must be in (0, cache_idw_k="
            f"{self.cache_idw_k}]")
        assert 2 ** self.n_kd_levels == self.n_leaves, \
            "n_leaves must be a power of 2"
        assert self.n_leaves <= 2 ** 18, \
            "n_leaves above 2^18 leaves only a few points per leaf"
        assert self.n_pn_neighbors > 0
        assert self.global_cond_dim > 0
        assert self.adaln_hidden > 0


@dataclass
class PreprocessConfig:
    n_workers: int = 32
    # footprint extraction from CFD surface points
    dbscan_eps: float = 5.0                # metres, xy clustering
    dbscan_min_samples: int = 10
    wall_nz_threshold: float = 0.3         # |n_z| below this -> wall point
    concave_hull_ratio: float = 0.3
    dp_tol: float = 0.5                    # Douglas-Peucker tolerance, metres
    min_edge_collapse: float = 1.0         # shorter edges collapse, metres
    # mean flow direction of the wind-aligned frame
    wind_direction: tuple[float, float] = (0.0, -1.0)
    wind_face_dot_threshold: float = 0.3   # |n . wind| windward / leeward
    threshold_edge: float = 0.5            # metres, surface edge proximity
    near_building_sdf_threshold: float = 10.0  # metres, volume region code
    near_ground_z_threshold: float = 4.0       # metres, volume region code
    # Partition of the latent layout: "invsqrt" (importance weight
    # 1 / sqrt(max(d, invsqrt_floor)), d = distance to the nearest building),
    # "equal" (unit weights, equal point count per leaf) or "structured" (a
    # regular grid_dims grid). Baked into the packs and part of the cache
    # directory name.
    kdtree_weighting: str = "invsqrt"
    invsqrt_floor: float = 2.0             # metres
    grid_dims: tuple[int, int, int] = (32, 32, 16)
    # surface points take PointNet slots before volume points
    pn_surf_first: bool = True
    support_max: float = 550.0             # metres, ray support cap
    geom_chunk: int = 16384                # points per distance-kernel chunk
    anomaly_concordance_threshold: float = 5.0  # MAD z, >= 2 fields -> drop
    anomaly_extreme_threshold: float = 8.0      # MAD z, any indicator -> drop
    sdf_qc: bool = True                    # log 2.5D vs raw distance agreement

    def __post_init__(self):
        self.wind_direction = tuple(float(v) for v in self.wind_direction)
        self.grid_dims = tuple(int(v) for v in self.grid_dims)


@dataclass
class TrainingConfig:
    num_epochs: int = 400
    batch_size: int = 1
    world_size: int = 1  # detected from DDP at runtime
    n_query: int = 128000
    # training seed of a seed study (`train --seed`). None uses the top-level
    # seed, which also drives the split and the cache; a different value
    # moves only weight init, dropout, case order and the per-step draws
    seed: int | None = None
    lr_base: float = 2e-4                  # Lion peak learning rate
    weight_decay_base: float = 0.01
    lr_schedule: str = "cosine"            # constant | cosine | wsd
    # warmup length: warmup_steps when set, else warmup_fraction of the run
    warmup_fraction: float = 0.05
    warmup_steps: int | None = None
    min_lr_fraction: float = 0.10          # cosine floor, fraction of peak
    wsd_stable_fraction: float = 0.70
    # anchor warmup + decay to this many epochs; later epochs follow a short
    # warm restart (None: the schedule spans num_epochs)
    lr_schedule_ref_epochs: int | None = None
    max_grad_norm: float = 1.0
    # x-mirror augmentation: every case also yields its mirrored twin
    mirror_augment: bool = False
    loss_vol_weights: tuple[float, float, float, float, float] = \
        (2.0, 1.0, 2.0, 1.0, 1.0)          # Ux, Uy, Uz, p, nut
    loss_shape: str = "mse"                # mse | huber
    huber_delta: float = 1.75              # z-score units (huber only)
    swa_enabled: bool = True
    # the final weights average the epoch-end weights of the last
    # swa_fraction of training
    swa_fraction: float = 0.25
    compile_mode: str = "reduce-overhead"  # "" disables torch.compile
    # loader workers per rank (clamped, see data/dataloader.py)
    num_workers: int = 8
    prefetch_factor: int = 4
    # gradient-accumulation microsteps per optimizer update, resolved from
    # the recipe's effective batch at launch
    grad_accum: int = 1
    activation_checkpointing: bool = False

    @property
    def swa_start(self) -> int:
        # first SWA epoch (1-based)
        return math.ceil((1.0 - self.swa_fraction) * self.num_epochs) + 1

    @property
    def lr(self) -> float:
        return self.lr_base

    @property
    def weight_decay(self) -> float:
        return self.weight_decay_base


@dataclass
class CheckpointConfig:
    save_dir: str = "work/runs/default"
    rolling_keep: int = 3                  # resumable epochs kept on disk

    def __post_init__(self):
        self.save_dir = _expand(self.save_dir)


@dataclass
class EvaluationConfig:
    test_chunk_size: int = 262144          # decoder query chunk


@dataclass
class ProfilingConfig:
    first_window: tuple[int, int] = (10, 30)
    periodic_every: int = 5000
    periodic_len: int = 10

    def __post_init__(self):
        self.first_window = tuple(int(v) for v in self.first_window)


@dataclass
class AbuptConfig:
    """AB-UPT (recipe "abupt"): dual-branch surface/volume model, width 192,
    3 heads, 16384 anchors per branch, without the geometry branch.
    geometry_features=True additionally feeds sdf, sdf_dir, support and
    region codes."""
    hidden_dim: int = 192
    num_heads: int = 3
    shared_blocks: str = "sscscs"          # weight-shared trunk (s=split, c=cross)
    num_surface_blocks: int = 6
    num_volume_blocks: int = 6
    n_anchors: int = 16384                 # per branch, without replacement
    geometry_features: bool = False
    # fixed affine from the /550 frame to AB-UPT's non-negative coordinates:
    # xy in [-550, 550] m, z in [0, 160] m -> (pos_norm + [1, 1, 0]) * 500
    pos_offset: tuple[float, float, float] = (1.0, 1.0, 0.0)
    pos_scale: float = 500.0
    # full-field inference: query chunk length per branch
    fullfield_query_chunk_per_branch: int = 65536
    # independent anchor draws per test case (inference only); draw 0 is
    # the persisted one
    infer_draws: int = 5
    # optimizer (AB-UPT paper): Lion, 5% warmup, cosine to an absolute floor
    lr_base: float = 5e-5
    weight_decay_base: float = 0.05
    warmup_fraction: float = 0.05
    end_lr: float = 1e-6

    def __post_init__(self):
        self.pos_offset = tuple(float(v) for v in self.pos_offset)

    def validate(self):
        assert self.hidden_dim % self.num_heads == 0
        assert set(self.shared_blocks) <= {"s", "c"}, \
            "shared_blocks may contain only 's' and 'c'"
        assert self.n_anchors > 0


@dataclass
class TransolverConfig:
    """Transolver-3 (recipe "transolver"): 16 layers, width 256, 8 heads,
    64 slices, MLP ratio 2; input [xyz, sdf, sdf_dir]; one backbone with
    volume and surface heads. AdamW 1e-3, weight decay 0.01, no warmup,
    per-epoch cosine to 1e-5."""
    n_layers: int = 16
    n_hidden: int = 256
    n_heads: int = 8
    slice_num: int = 64
    mlp_ratio: int = 2
    dropout: float = 0.0
    unified_pos: bool = False
    in_dim: int = 7
    lr: float = 1e-3
    weight_decay: float = 0.01
    min_lr: float = 1e-5
    effective_batch: int = 4
    # full-field inference: rows per chunk of the exact two-pass slice
    # aggregation
    infer_chunk: int = 128000

    def validate(self):
        assert self.n_hidden % self.n_heads == 0
        assert self.effective_batch >= 1


@dataclass
class SpiderConfig:
    """SpiderSolver (recipe "spider"): 8 layers, width 256, 8 heads, MLP
    ratio 2, 10 onion shells, 6 surface regions (dominant signed axis of the
    surface normal), 500 innermost volume participants, a stratified sample
    of (n_query - surf_quota) volume + surf_quota surface rows. Adam with
    OneCycle (peak 1e-3, pct_start 0.3, div 25, final div 1000)."""
    n_layers: int = 8
    n_hidden: int = 256
    n_heads: int = 8
    mlp_ratio: int = 2
    dropout: float = 0.0
    onion_num: int = 10
    n_regions: int = 6
    n_inner: int = 500
    surf_quota: int = 2048
    in_dim: int = 7
    lr: float = 1e-3
    onecycle_pct_start: float = 0.3
    onecycle_div_factor: float = 25.0
    onecycle_final_div_factor: float = 1000.0
    effective_batch: int = 4
    # full-field inference: disjoint stride chunks at training density
    infer_chunk: int = 128000

    def validate(self):
        assert self.n_hidden % self.n_heads == 0
        assert 0 < self.n_inner
        assert 0 < self.surf_quota


@dataclass
class ErwinConfig:
    """Erwin (recipe "erwin"): four-level ball-tree hierarchy, widths
    64-512, encoder depths 2/2/6/2, ball size 256, stride 2, 45° rotated
    trees, 3 message-passing steps over an 8-NN graph; coordinates through a
    Gaussian random-Fourier encoder; volume and surface heads. AdamW 5e-4,
    weight decay 0.01, no warmup, per-step cosine to 1e-6. Trees and graphs
    are built on the CPU in the loader workers (model/erwin/balltree.py)."""
    c_in: int = 64
    c_hidden: tuple[int, ...] = (64, 128, 256, 512)
    ball_sizes: tuple[int, ...] = (256, 256, 256, 256)
    enc_num_heads: tuple[int, ...] = (2, 4, 8, 16)
    enc_depths: tuple[int, ...] = (2, 2, 6, 2)
    dec_num_heads: tuple[int, ...] = (2, 4, 8)
    dec_depths: tuple[int, ...] = (2, 2, 2)
    strides: tuple[int, ...] = (2, 2, 2)
    rotate: int = 45
    mp_steps: int = 3
    knn: int = 8
    mlp_ratio: int = 4
    dimensionality: int = 3
    rff_sigma: float = 1.0
    lr: float = 5e-4
    weight_decay: float = 0.01
    min_lr: float = 1e-6
    effective_batch: int = 4
    infer_chunk: int = 128000

    def __post_init__(self):
        for f in ("c_hidden", "ball_sizes", "enc_num_heads", "enc_depths",
                  "dec_num_heads", "dec_depths", "strides"):
            setattr(self, f, tuple(int(v) for v in getattr(self, f)))

    def validate(self):
        assert len(self.enc_num_heads) == len(self.enc_depths) == \
            len(self.ball_sizes) == len(self.c_hidden)
        assert len(self.dec_num_heads) == len(self.dec_depths) == \
            len(self.strides) == len(self.c_hidden) - 1
        for d, h in zip(self.c_hidden, self.enc_num_heads):
            assert d % h == 0, f"c_hidden {d} not divisible by heads {h}"


@dataclass
class HardwareConfig:
    """A label recorded with every run (a100 | h100). The paper's runs used
    4 GPUs with micro-batch 1 for every model."""
    profile: str = "a100"

    def validate(self):
        assert self.profile in {"a100", "h100"}, \
            f"hardware.profile must be a100|h100, got {self.profile!r}"


@dataclass
class RunMeta:
    """Run identity and completion metadata, written into config_used.yaml
    by main.py and the trainer."""
    schema_version: int = 2
    recipe: str = "ibo"
    family: str = "ibo"
    dataset: str = "urbanwind"
    native_input_adapter: str = ""
    mode: str = "formal"
    config_hash: str = ""
    code_digest: str = ""
    manifest_hash: str = ""
    # sha256 over the ordered train/test roster of the pool manifest
    roster_hash: str = ""
    train_completed: bool = False
    completed_epochs: int = 0
    world_size: int = 0
    effective_batch: int = 0
    peak_train_mem_bytes: int = 0
    peak_train_mem_profile: str = ""
    train_flops_per_opt_step: float = 0.0
    compile_mode: str = ""
    compile_fullgraph: bool = False
    compile_notes: str = ""


@dataclass
class Config:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    abupt: AbuptConfig = field(default_factory=AbuptConfig)
    transolver: TransolverConfig = field(default_factory=TransolverConfig)
    spider: SpiderConfig = field(default_factory=SpiderConfig)
    erwin: ErwinConfig = field(default_factory=ErwinConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    run: RunMeta = field(default_factory=RunMeta)

    @property
    def recipe_name(self) -> str:
        return self.run.recipe

    def resolve_paths(self):
        """Derive the cache paths from the settings baked into the packs.
        Call after all CLI overrides; the pool manifest is not touched."""
        w = Path(self.data.work_dir)
        key = self.model.cache_key
        weight_tag = f"_{self.preprocess.kdtree_weighting}"
        sample_tag = "_sf" if self.preprocess.pn_surf_first else ""
        estate_dir = w / "cache_estate" / f"{key}{weight_tag}{sample_tag}_v1"
        self.data.cache_estate_dir = str(estate_dir)
        self.data.norm_stats_path = str(estate_dir / "norm_stats.json")

    def validate(self):
        self.model.validate()
        self.abupt.validate()
        self.transolver.validate()
        self.spider.validate()
        self.erwin.validate()
        self.hardware.validate()
        assert self.training.n_query > 0
        assert self.training.lr_base > 0.0
        assert self.training.weight_decay_base >= 0.0
        assert self.training.grad_accum >= 1
        assert len(self.training.loss_vol_weights) == 5
        assert all(float(w) >= 0.0 for w in self.training.loss_vol_weights)
        assert self.training.loss_shape in {"mse", "huber"}, \
            "training.loss_shape must be one of: mse, huber"
        assert self.training.huber_delta > 0.0
        assert self.training.lr_schedule in {"constant", "cosine", "wsd"}, \
            "training.lr_schedule must be one of: constant, cosine, wsd"
        assert self.training.lr_schedule_ref_epochs is None or \
            0 < self.training.lr_schedule_ref_epochs <= self.training.num_epochs, \
            "training.lr_schedule_ref_epochs must be in (0, num_epochs]"
        assert self.preprocess.kdtree_weighting in {"invsqrt", "equal",
                                                    "structured"}, \
            "preprocess.kdtree_weighting must be one of: invsqrt, equal, " \
            "structured"
        assert self.preprocess.invsqrt_floor > 0.0
        gd = self.preprocess.grid_dims
        assert len(gd) == 3 and all(d >= 1 and (d & (d - 1)) == 0 for d in gd), \
            f"preprocess.grid_dims {gd} must be three powers of two"
        assert int(gd[0]) * int(gd[1]) * int(gd[2]) == self.model.n_leaves, \
            (f"preprocess.grid_dims {gd} multiplies to "
             f"{gd[0] * gd[1] * gd[2]}, but model.n_leaves is "
             f"{self.model.n_leaves}")
        assert 0.0 <= self.training.warmup_fraction < 1.0
        assert self.training.warmup_steps is None or \
            self.training.warmup_steps > 0, \
            "training.warmup_steps must be a positive step count or null"
        assert 0.0 <= self.training.min_lr_fraction <= 1.0
        assert 0.0 <= self.training.wsd_stable_fraction < 1.0
        assert 0.0 < self.training.swa_fraction <= 1.0
        assert self.run.mode in {"formal", "debug"}, \
            "run.mode must be one of: formal, debug"


_SECTIONS = {
    "data": DataConfig, "model": ModelConfig, "preprocess": PreprocessConfig,
    "training": TrainingConfig,
    "checkpoint": CheckpointConfig, "evaluation": EvaluationConfig,
    "profiling": ProfilingConfig,
    "abupt": AbuptConfig, "transolver": TransolverConfig,
    "spider": SpiderConfig, "erwin": ErwinConfig,
    "hardware": HardwareConfig, "run": RunMeta,
}


def save_config(cfg: Config, path: str | Path):
    """Write the effective config (CLI overrides applied) atomically."""
    import dataclasses

    payload = {"seed": cfg.seed}
    for name in _SECTIONS:
        d = dataclasses.asdict(getattr(cfg, name))
        payload[name] = {k: (list(v) if isinstance(v, tuple) else v)
                         for k, v in d.items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    Path(tmp).replace(path)


def load_config(path: str | Path, quiet: bool = False,
                strict: bool = True) -> Config:
    """Read a config file. strict=False ignores keys this code does not know
    (e.g. reading only the model / preprocess settings of a config written
    by another version); strict=True rejects them."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    kwargs = {"seed": int(raw.get("seed", 42))}
    for name, cls in _SECTIONS.items():
        section = raw.get(name) or {}
        known = {k: v for k, v in section.items()
                 if k in cls.__dataclass_fields__}
        unknown = set(section) - set(known)
        if unknown:
            if strict:
                raise KeyError(f"Unknown keys in config section '{name}': "
                               f"{sorted(unknown)}")
            if not quiet:
                print(f"[config] {Path(path).name}: ignoring '{name}.' keys "
                      f"{sorted(unknown)}", flush=True)
        # None means "use the dataclass default / derived path"
        known = {k: v for k, v in known.items() if v is not None}
        kwargs[name] = cls(**known)
    cfg = Config(**kwargs)
    cfg.validate()
    return cfg


def config_hash(cfg: Config) -> str:
    """sha256 over every result-changing field of the resolved config (run
    outputs and machine-local paths excluded), so the same experiment hashes
    identically on any machine and a resume can refuse a different one."""
    import hashlib
    import json as _json

    blob = _json.dumps(config_hash_payload(cfg), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def config_hash_payload(cfg: Config) -> dict:
    """The dict config_hash digests: every result-changing field, nothing
    machine-local."""
    import dataclasses

    payload = {"seed": cfg.seed}
    for name in _SECTIONS:
        if name == "run":
            continue
        d = dataclasses.asdict(getattr(cfg, name))
        payload[name] = {k: (list(v) if isinstance(v, tuple) else v)
                         for k, v in d.items()}
    for k in ("raw_pt_dir", "work_dir", "cache_estate_dir", "norm_stats_path",
              "pool_manifest_path"):
        payload["data"].pop(k, None)
    payload["checkpoint"].pop("save_dir", None)
    # training.seed is hashed only when it changes the training
    ts = payload["training"].get("seed")
    if ts is None or int(ts) == int(cfg.seed):
        payload["training"].pop("seed", None)
    # world_size is detected at runtime; bind the value the run used
    payload["training"]["world_size"] = int(cfg.run.world_size) \
        if cfg.run.world_size else int(cfg.training.world_size)
    payload["run"] = {"recipe": cfg.run.recipe, "family": cfg.run.family,
                      "dataset": cfg.run.dataset,
                      "native_input_adapter": cfg.run.native_input_adapter,
                      "mode": cfg.run.mode,
                      "world_size": int(cfg.run.world_size),
                      "effective_batch": int(cfg.run.effective_batch),
                      "code_digest": cfg.run.code_digest,
                      "manifest_hash": cfg.run.manifest_hash,
                      "roster_hash": cfg.run.roster_hash}
    return payload


def code_digest() -> str:
    """sha256 over this repository's Python sources, recorded with every
    run."""
    import hashlib

    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    files = sorted((root / "src").rglob("*.py")) + [root / "main.py"]
    for f in files:
        if not f.exists():
            continue
        h.update(str(f.relative_to(root)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]
