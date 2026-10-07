"""FLOPs accounting.

The Linear-dominant part of every forward is counted exactly by a
hook meter over the real modules during a real pass (each nn.Linear
contributes 2*rows*in*out + rows*out; norms 5*elements; embeddings are
gathers = 0 arithmetic), so no per-architecture transcription errors and no
row-scaling assumptions. The parts hooks cannot see — SDPA, einsum
contractions, segment/index_add reductions, cdist, RoPE/Fourier
trigonometry — are added by per-recipe closed forms in the recipe modules,
computed from the SAME shapes the pass actually ran.

Conventions (recorded into every FLOPs JSON):
  - 1 MAC = 2 FLOPs;
  - SDPA(q, kv, H, d) = 4*H*q*kv*d (QK^T and AV matmuls) + 3*H*q*kv
    (softmax exp/sum/div);
  - einsum contraction over c: 2 * (output elements) * c;
  - segment/index_add mean: adds = rows*C, divides = groups*C;
  - activations (GELU/SiLU): 4 FLOPs/element; norms 5 FLOPs/element;
  - sin/cos: 1 FLOP each (trig unit-cost convention);
  - gathers / index selection / sort / tree search: 0 FLOPs — recorded
    separately as index-op counts where meaningful, never dressed up as
    floating-point work;
  - backward = 2x forward (the standard dense-layer convention), applied
    uniformly to every model;
  - optimizer element-wise update is reported in the breakdown but EXCLUDED
    from the headline train number (uniform choice across models — Lion and
    AdamW have different state math and including it would compare
    optimizers, not models).

torch.profiler(with_flops=True) covers only the matmul subset (it
undercounts SDPA, scatter and segment reductions), so it is not used here.
"""
from __future__ import annotations

from contextlib import contextmanager

import torch.nn as nn

FLOPS_CONVENTIONS = {
    "mac": 2, "sdpa": "4*H*q*kv*d + 3*H*q*kv", "norm_per_elem": 5,
    "act_per_elem": 4, "trig_per_elem": 1, "backward_multiplier": 2.0,
    "gather_index_ops": 0, "optimizer_update": "breakdown-only",
}


class FlopMeter:
    """Exact Linear/Conv/norm/ACTIVATION counting via forward hooks over one
    real EAGER pass. Module activations (GELU/SiLU/ReLU/...) are hooked at 4
    FLOPs/element — the documented convention is now actually counted, not
    just declared. Functional activations (F.silu inside fused expressions)
    cannot be hooked; the recipes add those as manual closed forms. Softmax
    modules are deliberately NOT hooked: every softmax in these models is
    already inside a manual attention/slice term (the 3*H*q*kv convention)
    and hooking it would double-count. Hooks see nothing inside a compiled
    graph — compiled producers use their recipe's eager probe/planner."""

    _NORM_TYPES = (nn.LayerNorm, nn.BatchNorm1d)
    _ACT_TYPES = (nn.GELU, nn.SiLU, nn.ReLU, nn.LeakyReLU, nn.Sigmoid,
                  nn.Tanh, nn.ELU)

    def __init__(self, model: nn.Module):
        self.model = model
        self.linear = 0.0
        self.norm = 0.0
        self.activation = 0.0
        self.extra = 0.0
        self.extra_terms: dict[str, float] = {}
        self._handles = []
        norm_types = list(self._NORM_TYPES)
        if hasattr(nn, "RMSNorm"):
            norm_types.append(nn.RMSNorm)
        self._norm_types = tuple(norm_types)

    def _hook_linear(self, mod, inputs, output):
        rows = output.numel() / output.shape[-1]
        self.linear += 2.0 * rows * mod.in_features * mod.out_features
        if mod.bias is not None:
            self.linear += rows * mod.out_features

    def _hook_conv(self, mod, inputs, output):
        k = 1.0
        for s in mod.kernel_size:
            k *= s
        self.linear += output.numel() * 2.0 * (mod.in_channels / mod.groups) \
            * k
        if mod.bias is not None:
            self.linear += output.numel()

    def _hook_norm(self, mod, inputs, output):
        self.norm += 5.0 * output.numel()

    def _hook_act(self, mod, inputs, output):
        self.activation += 4.0 * output.numel()

    def __enter__(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                self._handles.append(m.register_forward_hook(self._hook_linear))
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                self._handles.append(m.register_forward_hook(self._hook_conv))
            elif isinstance(m, self._norm_types):
                self._handles.append(m.register_forward_hook(self._hook_norm))
            elif isinstance(m, self._ACT_TYPES):
                self._handles.append(m.register_forward_hook(self._hook_act))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False

    def add(self, name: str, flops: float):
        self.extra += float(flops)
        self.extra_terms[name] = self.extra_terms.get(name, 0.0) + float(flops)

    @property
    def total(self) -> float:
        return self.linear + self.norm + self.activation + self.extra

    def breakdown(self) -> dict:
        return {"linear": self.linear, "norm": self.norm,
                "activation": self.activation,
                "manual": dict(self.extra_terms), "total": self.total,
                "conventions": FLOPS_CONVENTIONS}


def sdpa_flops(q_rows: float, kv_rows: float, heads: float,
               head_dim: float) -> float:
    return 4.0 * heads * q_rows * kv_rows * head_dim \
        + 3.0 * heads * q_rows * kv_rows


def einsum_flops(out_elems: float, contracted: float) -> float:
    return 2.0 * out_elems * contracted


def segment_mean_flops(rows: float, channels: float, groups: float) -> float:
    return rows * channels + groups * channels


@contextmanager
def metered(model: nn.Module):
    with FlopMeter(model) as m:
        yield m
