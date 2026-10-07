"""DDP setup and teardown under torchrun; without torchrun everything runs
as a single process."""
from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist


def init_ddp() -> tuple[int, int, int]:
    """Returns (rank, world_size, local_rank). 1800 s NCCL timeout: the first
    compiled forward takes minutes per rank at different speeds."""
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend, timeout=timedelta(seconds=1800))
        rank = dist.get_rank()
        world = dist.get_world_size()
        local = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank, world, local = 0, 1, 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    return rank, world, local


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def barrier():
    if is_distributed():
        dist.barrier()


def all_reduce_sum(t: torch.Tensor) -> torch.Tensor:
    if is_distributed():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def cleanup_ddp():
    if is_distributed():
        dist.destroy_process_group()
