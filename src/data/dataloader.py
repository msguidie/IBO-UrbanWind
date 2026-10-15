"""Batched DataLoader over the RAM-resident datasets and the CUDA
prefetcher that overlaps the next batch's host-to-device copy with compute.

Each dataset item is one already-drawn sample dict (the mirror twin is a
separate index), so collation is a plain stack of BATCH_KEYS.
"""
from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader, Sampler

from .dataset import _worker_init

BATCH_KEYS = ("query_pos_norm", "query_sdf_norm", "query_sdf_dir",
              "query_support_norm", "region_dim1", "region_dim2", "region_dim3",
              "idw_indices", "target_vol", "target_surf", "is_volume",
              "pn_points_feat", "pn_mask", "leaf_stats", "leaf_pos_norm",
              "global_morph_norm")
# RoPE cos/sin are not part of the batch: the processor derives them on the
# GPU from leaf_pos_norm.


def collate(samples: list[dict]) -> dict:
    return {k: torch.stack([s[k] for s in samples]).contiguous()
            for k in BATCH_KEYS}


def batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


# Upper bound on loader workers per rank (training.num_workers is clamped to
# it). Each worker draws samples from resident packs, including the decoder
# k-NN query; raise LOADER_WORKERS if the step profiler reports
# data_wait > forward.
RESIDENT_MAX_WORKERS = max(1, int(os.environ.get("LOADER_WORKERS", "8")))


class EpochSampler(Sampler):
    """Yields (item_idx, epoch) for one epoch, reshuffled by set_epoch()."""

    def __init__(self, n_items: int, seed: int, shuffle: bool = True):
        self.n_items = int(n_items)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self) -> int:
        return self.n_items

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            order = torch.randperm(self.n_items, generator=g).tolist()
        else:
            order = range(self.n_items)
        return iter([(i, self.epoch) for i in order])


def build_dataloader(dataset, batch_size: int, num_workers: int,
                     prefetch_factor: int, seed: int, shuffle: bool = True,
                     drop_last: bool = True,
                     collate_fn=None) -> tuple[DataLoader, EpochSampler]:
    """Batched DataLoader over a resident dataset. Pinning runs on
    PyTorch's background thread; the worker pool persists across epochs."""
    num_workers = min(int(num_workers), RESIDENT_MAX_WORKERS)
    sampler = EpochSampler(len(dataset), seed, shuffle)
    kwargs = dict(batch_size=batch_size, sampler=sampler,
                  collate_fn=collate_fn if collate_fn is not None else collate,
                  num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                  drop_last=drop_last)
    if num_workers > 0:
        kwargs.update(prefetch_factor=prefetch_factor, persistent_workers=True,
                      worker_init_fn=_worker_init)
    return DataLoader(dataset, **kwargs), sampler


class CudaPrefetcher:
    """Copies the next batch to the GPU on a side stream while the current
    batch computes. The DataLoader already returns pinned batches."""

    def __init__(self, loader: DataLoader, device: torch.device):
        self._loader_iter = iter(loader)
        self._device = device
        self._stream = torch.cuda.Stream(device)
        self._next_batch: dict | None = None
        self._preload()

    def _preload(self):
        try:
            batch = next(self._loader_iter)
        except StopIteration:
            self._next_batch = None
            return
        def _mv(v):
            if isinstance(v, torch.Tensor):
                return v.to(self._device, non_blocking=True)
            if isinstance(v, (list, tuple)):
                return type(v)(_mv(x) for x in v)
            return v
        with torch.cuda.stream(self._stream):
            self._next_batch = {k: _mv(v) for k, v in batch.items()}

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        if self._next_batch is None:
            raise StopIteration
        batch = self._next_batch
        torch.cuda.current_stream().wait_stream(self._stream)
        # the tensors were allocated on the side stream: tell the allocator
        # the compute stream uses them, so their memory is not reused early
        cur_stream = torch.cuda.current_stream()

        def _rec(v):
            if isinstance(v, torch.Tensor):
                v.record_stream(cur_stream)
            elif isinstance(v, (list, tuple)):
                for x in v:
                    _rec(x)
        for v in batch.values():
            _rec(v)
        self._preload()
        return batch
