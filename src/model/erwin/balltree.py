"""Self-contained ball-tree builder for the Erwin port.

Implements the ball tree the Erwin paper defines — "recursively partition
the points into two equal halves along the axis of largest spread, down to
balls of a fixed size" — plus the rotated-tree variants used for cross-ball
mixing, with the exact tensor contract models/erwin/erwin.py consumes:

  tree_idx  [P] long   P = ball_size * 2^k >= N; x[tree_idx] lays the cloud
                       out ball-contiguously; padded slots repeat a real
                       row's index (finite math, masked out downstream).
  tree_mask [P] bool   True exactly once per original row —
                       tree_idx[tree_mask] is a PERMUTATION of range(N), so
                       out[tree_mask][argsort(tree_idx[tree_mask])] restores
                       the original row order (the decode contract).
  tree_idx_rot         one [P_i] permutation per attention level (encoder
                       levels + bottleneck): the ROTATED tree's ball order
                       over that level's rows, where level i's positions are
                       the stride-group means of level i-1's (exactly what
                       BallPooling will hold at runtime).

This replaces the external `balltree-erwin` C++ package: the algorithm is
the published one; only tie-breaking details could differ from that binary,
and they are documented here (stable sort, ties by row index). Being pure
torch/numpy it runs identically in loader workers on any node — no compiled
dependency, no train/test drift.

Splitting under padding: a node with capacity `cap` holding m <= cap real
rows sends left_m = min(cap/2, max(m/2 rounded up? no — median) rows to the
left child: left_m = min(cap // 2, max(m // 2, m - cap // 2)). At cap == m
this is the exact median split; when m < cap it stays the median until the
capacity bound bites. Leaves pad to ball_size by repeating their last real
row (mask False).

Rotation: `rotate` degrees about the z axis for 2D; for 3D, the rotation
about z composed with the same angle about y (a fixed, documented mixing
choice). Rotated trees are built with NO padding (level sizes are exact
multiples of the ball size by construction — asserted).
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

#: CPU threads per loader worker for tree / kNN construction (override with
#: ERWIN_TREE_WORKERS). The total is ranks x loader workers x TREE_WORKERS.
#: The tree builds of one step (main tree + one rotated tree per level) are
#: independent and run concurrently; every builder writes its own array, so
#: the result equals the serial path.
TREE_WORKERS = int(os.environ.get("ERWIN_TREE_WORKERS", "4"))

_POOL = None


def _pool() -> ThreadPoolExecutor:
    """One lazily created, process-wide pool (never one per call)."""
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=TREE_WORKERS,
                                   thread_name_prefix="balltree")
    return _POOL


def _reset_pool_after_fork() -> None:
    """Threads do NOT survive fork: a child that inherited the parent's
    pool object would submit work to threads that no longer exist and hang.
    DataLoader workers are forked, so the child drops the handle and builds
    its own pool on first use."""
    global _POOL
    _POOL = None


if hasattr(os, "register_at_fork"):          # POSIX only; Windows is spawn
    os.register_at_fork(after_in_child=_reset_pool_after_fork)


def _split_perm(pos: np.ndarray, idx: np.ndarray, cap: int,
                ball: int, out: list) -> None:
    m = idx.shape[0]
    if cap == ball:
        out.append(idx)
        return
    half = cap // 2
    if m == 0:
        # a fully empty subtree still owes cap/ball leaves to the layout
        for _ in range(cap // ball):
            out.append(idx)
        return
    span = pos[idx].max(axis=0) - pos[idx].min(axis=0)
    axis = int(np.argmax(span))
    order = np.lexsort((idx, pos[idx, axis]))   # stable, ties by row index
    # median split until the capacity bound bites (left side rounds up, so a
    # lone point always lands left and leaves fill front-to-back)
    left_m = min(half, max((m + 1) // 2, m - half))
    left = idx[order[:left_m]]
    right = idx[order[left_m:]]
    _split_perm(pos, left, half, ball, out)
    _split_perm(pos, right, half, ball, out)


def _build_single(pos: np.ndarray, ball: int) -> tuple[np.ndarray, np.ndarray]:
    """(tree_idx [P], tree_mask [P]) over one point cloud."""
    n = pos.shape[0]
    n_balls = max(1, 2 ** math.ceil(math.log2(max(n / ball, 1))))
    cap = n_balls * ball
    chunks: list[np.ndarray] = []
    _split_perm(pos, np.arange(n, dtype=np.int64), cap, ball, chunks)
    tree_idx = np.empty(cap, dtype=np.int64)
    tree_mask = np.zeros(cap, dtype=bool)
    off = 0
    last_seen = 0
    for c in chunks:
        m = c.shape[0]
        tree_idx[off:off + m] = c
        tree_mask[off:off + m] = True
        if m > 0:
            last_seen = c[-1]
        if m < ball:
            tree_idx[off + m:off + ball] = last_seen   # pad: repeat a real row
        off += ball
    assert off == cap
    return tree_idx, tree_mask


def _rot_matrix(dim: int, degrees: float) -> np.ndarray:
    t = math.radians(degrees)
    c, s = math.cos(t), math.sin(t)
    if dim == 2:
        return np.array([[c, -s], [s, c]], dtype=np.float64)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    return ry @ rz


def build_tree(pos: torch.Tensor, strides, ball_sizes, rotate: float,
               workers: int = TREE_WORKERS
               ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """One point cloud [N, d] (micro-batch 1) ->
    (tree_idx, tree_mask, tree_idx_rot). tree_idx_rot has
    len(ball_sizes) entries (= encoder levels + bottleneck); entries are
    identity-free permutations of that level's padded row count.

    workers: threads used for the INDEPENDENT builds (main tree + one
    rotated tree per level). The algorithm is untouched — each build owns
    its own arrays — so the result is bit-identical to workers=1."""
    p = pos.detach().cpu().numpy().astype(np.float64, copy=False)
    strides = [int(s) for s in strides]
    ball_sizes = [int(b) for b in ball_sizes]
    main = _pool().submit(_build_single, p, ball_sizes[0])         if workers > 1 else None
    tree_idx, tree_mask = main.result() if main is not None         else _build_single(p, ball_sizes[0])

    # level positions: level 0 = padded layout; level i = stride-group means
    level_pos = [p[tree_idx]]
    for s in strides:
        prev = level_pos[-1]
        assert prev.shape[0] % s == 0
        level_pos.append(prev.reshape(-1, s, prev.shape[1]).mean(axis=1))

    rot_perms: list[torch.Tensor] = []
    if rotate:
        rot = _rot_matrix(p.shape[1], float(rotate))
        for lvl, ball in enumerate(ball_sizes):
            assert level_pos[lvl].shape[0] % ball == 0, (
                f"level {lvl}: {level_pos[lvl].shape[0]} rows not divisible "
                f"by ball {ball} — raise n_query or lower the stride depth")
        rotated = [level_pos[lvl] @ rot.T for lvl in range(len(ball_sizes))]
        if workers > 1:
            futs = [_pool().submit(_build_single, rotated[lvl], ball)
                    for lvl, ball in enumerate(ball_sizes)]
            built = [f.result() for f in futs]
        else:
            built = [_build_single(rotated[lvl], ball)
                     for lvl, ball in enumerate(ball_sizes)]
        for r_idx, r_mask in built:
            assert bool(r_mask.all()), "rotated level build must be pad-free"
            rot_perms.append(torch.from_numpy(r_idx))
    else:
        rot_perms = [None] * len(ball_sizes)   # type: ignore[list-item]

    return (torch.from_numpy(tree_idx), torch.from_numpy(tree_mask),
            rot_perms)


def knn_edges(pos: torch.Tensor, k: int,
              workers: int = TREE_WORKERS) -> torch.Tensor:
    """kNN graph with self-loops, edge_index [2, N*k] = (source=neighbour,
    target=node) — the flow the MPNN's scatter-at-target aggregation
    expects (torch_cluster.knn_graph(loop=True) semantics, built with
    scipy's cKDTree instead of the CUDA extension)."""
    from scipy.spatial import cKDTree

    p = pos.detach().cpu().numpy()
    tree = cKDTree(p)
    _, nbr = tree.query(p, k=k, workers=int(workers))
    if nbr.ndim == 1:
        nbr = nbr[:, None]
    n = p.shape[0]
    tgt = np.repeat(np.arange(n, dtype=np.int64), nbr.shape[1])
    src = nbr.astype(np.int64).reshape(-1)
    # LAYOUT CONTRACT (relied on by MPNN.layer's grouped mean): every node
    # receives EXACTLY k edges, laid out contiguously in node order, so
    # edge_index[1] == repeat(arange(N), k) by construction.
    assert tgt.shape[0] == n * nbr.shape[1]
    return torch.from_numpy(np.stack([src, tgt], axis=0))


def restore_index(tree_idx: torch.Tensor,
                  tree_mask: torch.Tensor) -> torch.Tensor:
    """STATIC integer permutation replacing the decoder's dynamic-shape
    unpad-and-reorder step.

    The author's tail is
        out = node.x[tree_mask][argsort(tree_idx[tree_mask])]
    which needs a boolean mask -> aten.nonzero -> an output length unknown
    at trace time, so it cannot live inside a fullgraph/dynamic=False
    compiled region. The same rows, in the same order, are addressable by a
    fixed-length index computed once on CPU next to the tree:

        keep    = nonzero(tree_mask)      padded positions of the real rows
        orig    = tree_idx[keep]          their original row ids (a
                                          permutation of 0..N-1)
        restore = keep[argsort(orig)]     length N, int64

    Then node.x.index_select(0, restore) is ELEMENTWISE IDENTICAL to the
    boolean expression (gate: tests/test_gates.py). Contract asserted here:
    int64, length N == tree_mask.sum(), values inside [0, P), and the
    induced mapping is a bijection onto 0..N-1."""
    assert tree_mask.dtype == torch.bool, "tree_mask must be boolean"
    keep = torch.nonzero(tree_mask, as_tuple=False).squeeze(1)
    orig = tree_idx[keep].long()
    restore = keep[torch.argsort(orig)].to(torch.int64)
    n = int(keep.numel())
    assert restore.dtype == torch.int64 and restore.shape == (n,)
    assert int(tree_mask.sum()) == n
    assert torch.equal(tree_idx[restore].long(),
                       torch.arange(n, dtype=torch.int64)), \
        "restore_index: tree_idx/tree_mask do not form a 0..N-1 bijection"
    return restore
