"""Preprocessing.

  geometry   exact 2.5D building geometry (signed distance, ray support,
             footprint extraction and canonicalization)
  kdtree     weighted KD-tree / regular-grid partitions
  features   per-point and per-leaf model inputs, shared by the dataset
             builder and the new-case pipeline
  build      raw CFD cases -> estate packs, releaf, pool-manifest commands

The pack-building entry points below resolve lazily, so importing
`src.preprocess.geometry` does not pull in the whole builder.
"""
from __future__ import annotations

_LAZY = {
    "run_estate_preprocess": "build",
    "run_releaf": "build",
    "pool_manifest_paths": "build",
    "write_pool_manifests": "build",
    "rebuild_pool_manifests": "build",
    "verify_pool_manifests": "build",
    "retest_pool_manifest": "build",
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    mod_name = _LAZY.get(name)
    if mod_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(f"{__name__}.{mod_name}"), name)


def __dir__():
    return sorted(__all__)
