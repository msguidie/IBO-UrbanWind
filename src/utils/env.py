"""Process environment hygiene. Import this module BEFORE importing torch.

The allocator / NCCL variables only take effect if set before CUDA init;
entrypoints (main.py, the post/test workers) import this first.
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "1")
os.environ.setdefault("NCCL_DEBUG", "WARN")


def limit_worker_threads():
    """Force single-threaded math libs inside multiprocessing pool workers."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "TBB_NUM_THREADS"):
        os.environ[var] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass


def quiet_inductor_perf_hints() -> None:
    """Silence inductor's per-graph "skipping cudagraphs due to ..." spam.

    These are performance hints, not errors: the IBO decoder's
    per-neighbour gather has an index_put(accumulate=True) backward, so
    inductor declines CUDA-Graph capture for the training graph. That does
    not affect correctness or the inference graphs that latency is measured
    on, and the message repeats per compiled subgraph per rank.

    Inference capture is still checked: utils/compile_health.py reads the
    dynamo / inductor counters after warmup and fails a producer that
    declared CUDA Graphs and did not get them. Set
    TORCH_LOGS="+perf_hints" to see the hints again.
    """
    import logging
    try:
        import torch._logging
        torch._logging.set_logs(perf_hints=logging.ERROR)
        return
    except Exception:
        pass
    for name in ("torch._inductor.utils", "torch._inductor.cudagraph_trees",
                 "torch._inductor.compile_fx"):
        try:
            logging.getLogger(name).setLevel(logging.ERROR)
        except Exception:
            pass
