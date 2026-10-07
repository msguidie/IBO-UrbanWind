"""Weighted KD-tree partition with stored split planes.

Implicit complete binary tree in heap layout: internal node i has children
2i+1 / 2i+2; a depth-d tree has 2^d - 1 internal nodes and 2^d leaves. Each
node splits along its axis of largest extent at the weighted median of its
points (cumulative weight reaches half). Membership is then re-derived by the
same strict `<` descent used for arbitrary query points, so the stored split
values reproduce the partition exactly.

build_structured_grid expresses a regular nx x ny x nz cuboid grid as the
same KDTree structure (split planes at cell midpoints), so every consumer of
the weighted tree works on it unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class KDTree:
    depth: int
    split_axes: np.ndarray    # [2^depth - 1] int8
    split_values: np.ndarray  # [2^depth - 1] float64
    perm: np.ndarray          # [N] int64, points permuted into leaf order
    leaf_offsets: np.ndarray  # [2^depth + 1] int64, leaf l owns perm[off[l]:off[l+1]]
    build_stats: dict | None = None

    @property
    def n_leaves(self) -> int:
        return 2 ** self.depth

    def leaf_members(self, leaf: int) -> np.ndarray:
        return self.perm[self.leaf_offsets[leaf]:self.leaf_offsets[leaf + 1]]

    def leaf_ids(self, n_points: int) -> np.ndarray:
        """Leaf id of every original point index."""
        counts = np.diff(self.leaf_offsets)
        ids = np.repeat(np.arange(self.n_leaves, dtype=np.int64), counts)
        out = np.empty(n_points, dtype=np.int64)
        out[self.perm] = ids
        return out


def build_weighted_kdtree(points: np.ndarray, weights: np.ndarray,
                          depth: int) -> KDTree:
    """points [N, D], weights [N] (positive), depth >= 1.

    No empty leaves: every node holding at least two distinct points is split
    so that both children keep at least one point, and the recorded split
    value reproduces that split under coordinate descent. When the
    weighted-median cut falls inside a run of equal coordinates (common for
    cell centres of a structured mesh), the cut moves to whichever end of
    the run leaves the weight closest to half. Nodes with a single point or
    with exactly coincident points cannot be split and are counted in
    build_stats.
    """
    points = np.asarray(points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    N, _ = points.shape
    n_internal = 2 ** depth - 1
    split_axes = np.zeros(n_internal, dtype=np.int8)
    split_values = np.zeros(n_internal, dtype=np.float64)
    perm = np.arange(N, dtype=np.int64)
    n_tie_fixed = n_singleton = n_duplicate = 0
    max_skew = 0.0                      # |left weight fraction - 0.5|

    ranges = [(0, N)]  # node ranges of the current level, in heap order
    node_base = 0
    for level in range(depth):
        next_ranges = []
        for local, (start, end) in enumerate(ranges):
            node = node_base + local
            count = end - start
            if count == 0:
                next_ranges.append((start, start))
                next_ranges.append((start, start))
                continue
            sub_idx = perm[start:end]
            sub = points[sub_idx]
            spreads = sub.max(axis=0) - sub.min(axis=0)
            axis = int(np.argmax(spreads))
            vals = sub[:, axis]

            if count == 1:
                n_singleton += 1
                split_axes[node] = axis
                split_values[node] = vals[0]
                next_ranges.append((start, start))       # empty left
                next_ranges.append((start, end))         # single point right
                continue

            order = np.argsort(vals, kind="stable")
            v_sorted = vals[order]
            w_sorted = weights[sub_idx[order]]
            cum = np.cumsum(w_sorted)
            half = cum[-1] * 0.5
            mid = int(np.searchsorted(cum, half))
            mid = max(1, min(mid, count - 1))

            if v_sorted[mid - 1] == v_sorted[mid]:
                v = v_sorted[mid]
                lo = int(np.searchsorted(v_sorted, v, "left"))
                hi = int(np.searchsorted(v_sorted, v, "right"))
                cand = [m for m in (lo, hi) if 1 <= m <= count - 1]
                if cand:
                    mid = min(cand, key=lambda m: abs(cum[m - 1] - half))
                    n_tie_fixed += 1
                else:
                    # the run spans the whole node on its widest axis, so
                    # the points coincide and no plane separates them
                    n_duplicate += 1
                    split_axes[node] = axis
                    split_values[node] = v
                    next_ranges.append((start, start))
                    next_ranges.append((start, end))
                    continue

            left_max = v_sorted[mid - 1]
            right_min = v_sorted[mid]
            # midpoint of the gap, so an external point (a surface point or
            # a query) falls on the geometrically nearer side; guarded
            # against the midpoint rounding onto left_max
            sv = 0.5 * (left_max + right_min)
            if not sv > left_max:
                sv = right_min

            split_axes[node] = axis
            split_values[node] = sv
            perm[start:end] = sub_idx[order]
            next_ranges.append((start, start + mid))
            next_ranges.append((start + mid, end))
            max_skew = max(max_skew, abs(cum[mid - 1] / cum[-1] - 0.5))
        ranges = next_ranges
        node_base += 2 ** level

    n_leaves = 2 ** depth
    struct = KDTree(depth=depth, split_axes=split_axes,
                    split_values=split_values,
                    perm=np.empty(N, dtype=np.int64),
                    leaf_offsets=np.empty(n_leaves + 1, dtype=np.int64))
    leaf_ids = assign_points_to_leaves(points, struct)
    perm = np.argsort(leaf_ids, kind="stable").astype(np.int64)
    counts = np.bincount(leaf_ids, minlength=n_leaves).astype(np.int64)
    offsets = np.zeros(n_leaves + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    built = np.fromiter((e - s for s, e in ranges), dtype=np.int64,
                        count=n_leaves)
    stats = {"tie_fixed": n_tie_fixed, "singleton": n_singleton,
             "duplicate": n_duplicate, "max_skew": float(max_skew),
             "empty": int((counts == 0).sum()),
             "descent_mismatch": int((counts != built).sum())}
    return KDTree(depth=depth, split_axes=split_axes,
                  split_values=split_values, perm=perm, leaf_offsets=offsets,
                  build_stats=stats)


def build_structured_grid(points: np.ndarray, dims: tuple[int, int, int]
                          ) -> KDTree:
    """A regular nx x ny x nz grid of equal cuboids over the points' bounding
    box, expressed as a KDTree of depth log2(nx) + log2(ny) + log2(nz):
    every node splits at the midpoint of its own cell, x levels first, then
    y, then z, so the leaf index is (ix * ny + iy) * nz + iz.

    The grid ignores the data, so cells inside buildings or above the
    highest roof are empty; downstream they get a cell-centre latent point
    and an all-masked PointNet row.
    """
    points = np.asarray(points, dtype=np.float64)
    dims = tuple(int(d) for d in dims)
    assert len(dims) == 3 and all(d >= 1 for d in dims)
    bits = [int(round(math.log2(d))) for d in dims]
    assert all(2 ** b == d for b, d in zip(bits, dims)), \
        f"grid_dims {dims} must all be powers of two"
    depth = sum(bits)
    n_leaves = 2 ** depth
    axis_of_level = np.repeat([0, 1, 2], bits).astype(np.int8)

    lo0 = points.min(axis=0)
    hi0 = points.max(axis=0)
    span = np.maximum(hi0 - lo0, 1e-12)
    hi0 = lo0 + span

    n_internal = n_leaves - 1
    split_axes = np.empty(n_internal, dtype=np.int8)
    split_values = np.empty(n_internal, dtype=np.float64)
    node_lo = np.empty((n_internal, 3), dtype=np.float64)
    node_hi = np.empty((n_internal, 3), dtype=np.float64)
    node_lo[0], node_hi[0] = lo0, hi0
    for node in range(n_internal):
        d = int(node + 1).bit_length() - 1
        a = int(axis_of_level[d])
        mid = 0.5 * (node_lo[node, a] + node_hi[node, a])
        split_axes[node] = a
        split_values[node] = mid
        left, right = 2 * node + 1, 2 * node + 2
        if left < n_internal:
            node_lo[left] = node_lo[node]
            node_hi[left] = node_hi[node]
            node_hi[left, a] = mid
            node_lo[right] = node_lo[node]
            node_lo[right, a] = mid
            node_hi[right] = node_hi[node]

    tree = KDTree(depth=depth, split_axes=split_axes,
                  split_values=split_values,
                  perm=np.empty(0, dtype=np.int64),
                  leaf_offsets=np.zeros(n_leaves + 1, dtype=np.int64))
    leaf_ids = assign_points_to_leaves(points, tree)
    perm = np.argsort(leaf_ids, kind="stable").astype(np.int64)
    counts = np.bincount(leaf_ids, minlength=n_leaves).astype(np.int64)
    offsets = np.zeros(n_leaves + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    tree.perm = perm
    tree.leaf_offsets = offsets
    tree.build_stats = {"structured": True, "grid_dims": list(dims),
                        "empty": int((counts == 0).sum()),
                        "descent_mismatch": 0}
    return tree


def assign_points_to_leaves(points: np.ndarray, tree: KDTree) -> np.ndarray:
    """Vectorized descent of arbitrary points through the stored splits.

    points [M, D] -> leaf ids [M] int64. Strictly `<` goes left.
    """
    points = np.asarray(points, dtype=np.float64)
    M = len(points)
    node = np.zeros(M, dtype=np.int64)
    rows = np.arange(M)
    for _ in range(tree.depth):
        axes = tree.split_axes[node].astype(np.intp)
        vals = tree.split_values[node]
        go_left = points[rows, axes] < vals
        node = np.where(go_left, 2 * node + 1, 2 * node + 2)
    return node - (2 ** tree.depth - 1)


def compute_leaf_bounds(tree: KDTree, domain_lo: np.ndarray,
                        domain_hi: np.ndarray) -> np.ndarray:
    """Cuboid bounds of every leaf: walk each leaf up to the root, clipping
    the domain envelope by the split planes. Returns [n_leaves, 2, D]."""
    D = len(domain_lo)
    L = tree.n_leaves
    n_internal = L - 1
    bounds = np.empty((L, 2, D), dtype=np.float64)
    for leaf in range(L):
        lo = np.array(domain_lo, dtype=np.float64)
        hi = np.array(domain_hi, dtype=np.float64)
        idx = leaf + n_internal
        while idx > 0:
            parent = (idx - 1) // 2
            axis = int(tree.split_axes[parent])
            sv = tree.split_values[parent]
            if idx == 2 * parent + 1:
                hi[axis] = min(hi[axis], sv)
            else:
                lo[axis] = max(lo[axis], sv)
            idx = parent
        bounds[leaf, 0] = lo
        bounds[leaf, 1] = np.maximum(hi, lo)
    return bounds
