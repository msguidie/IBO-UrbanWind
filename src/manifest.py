"""Pool manifests: the split files.

A pool manifest is one self-contained json that names the estates training
and evaluation touch:

  * the estates and the wind directions each one has;
  * the train / test / train_eval name lists;
  * the per-rank assignment (rank r trains on ranks[r]["estates"]).

Two pools share the same test and train_eval lists:

  manifest_400.json   400 training estates (1600 cases)
  manifest_full.json  every clean estate except the test estates

`build_pool_manifests` copies given name lists; the only thing it computes is
each pool's rank assignment.

A manifest names estates and nothing else. Which cache the estates are read
from (importance-weighted, equal-weighted or grid leaves) is a separate
choice, made by `--grid` and resolved into cache_estate_dir / norm_stats_path
by config.resolve_paths(), so one manifest serves every cache.

AB-UPT eligibility: AB-UPT draws n_anchors (16384) surface and 16384 volume
anchors per sample without replacement, so an estate is usable only if both
point sets reach that count and it has all four directions. The pool records
this as `abupt_eligible` (true for manifest_400, false for manifest_full),
and `load_pool_manifest(..., require_abupt=True)` refuses an ineligible pool
at startup.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

SCHEMA = "pool_v1"

DIRECTION_SUFFIXES = ("N", "E", "S", "W")

# AB-UPT's per-branch anchor count; a split rule (it decides which estates
# may appear in an AB-UPT-eligible pool), so it lives here.
N_ANCHORS = 16384

# Resident bytes per point (not disk bytes): pos fp32 12 + sdf bf16 2 +
# direction fp16 angles 4 + support bf16 12 + region_dim2/dim3 int8 2.
_BYTES_PER_POINT = 12 + 2 + 4 + 12 + 2


def estate_of(case_name: str) -> str:
    """``<estate>_E`` -> ``<estate>``. Fails on a name that does not end in a
    direction segment: splits are stated per estate, so a wrong estate key
    would mix train and test."""
    stem, sep, direction = case_name.rpartition("_")
    if not sep or direction not in DIRECTION_SUFFIXES:
        raise ValueError(
            f"case name {case_name!r} does not end in _<dir> with dir in "
            f"{DIRECTION_SUFFIXES} — cannot derive its estate")
    return stem


def resident_bytes(n_vol: int, n_surf: int, n_dirs: int,
                   n_leaves: int, n_pn_neighbors: int) -> int:
    """Approximate resident bytes of one estate pack, used only to balance
    the rank assignment (only the relative ordering matters)."""
    n = n_vol + n_surf
    L, K = n_leaves, n_pn_neighbors
    geom = (n * _BYTES_PER_POINT
            + L * (12 + 19 * 2) + L * K * (13 * 2 + 1))
    per_dir = n_vol * 5 * 2 + n_surf * 1 * 2 + n_surf * 2  # fp16 tgts + int8
    return geom + n_dirs * per_dir


def assignment_hash(ranks: list[dict]) -> str:
    """Hash of the per-rank estate-name assignment. `--auto-resume` refuses a
    changed assignment (it would change steps_per_epoch and which rank sees
    which data); the predicted byte numbers are advisory."""
    payload = json.dumps([sorted(r["estates"]) for r in ranks],
                         sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def atomic_json(obj, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    Path(tmp).replace(path)


def read_json(path: str | Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def load_pool_manifest(path: str | Path, world: int | None = None,
                       require_abupt: bool = False) -> dict:
    """Load and validate; every consistency rule the run depends on is
    checked here, once, at startup."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"pool manifest {path} not found")
    m = read_json(path)
    if m.get("schema") != SCHEMA:
        raise RuntimeError(
            f"{path}: schema is {m.get('schema')!r}, expected {SCHEMA!r}")

    est, splits, ranks = m["estates"], m["splits"], m["ranks"]
    if world is not None and int(m["world"]) != int(world):
        raise RuntimeError(
            f"{path} was partitioned for world={m['world']} but this launch "
            f"has world={world}. Re-partition with `python main.py manifest "
            f"repartition --world {world}` (name lists are kept; only the "
            f"per-rank assignment changes).")
    if assignment_hash(ranks) != m["assignment_sha256"]:
        raise RuntimeError(f"{path}: assignment hash mismatch — hand-edited?")
    counts = {r["n_cases"] for r in ranks}
    if len(counts) != 1:
        raise RuntimeError(f"{path}: per-rank case counts differ: {counts}")

    overlap = set(splits["train_estates"]) & set(splits["test_estates"])
    if overlap:
        raise RuntimeError(f"{path}: {len(overlap)} estates are in BOTH "
                           f"train and test: {sorted(overlap)[:5]}")
    missing = [e for e in splits["train_estates"] + splits["test_estates"]
               if e not in est]
    if missing:
        raise RuntimeError(f"{path}: {len(missing)} split members absent "
                           f"from `estates`: {missing[:5]}")
    stray = [e for r in ranks for e in r["estates"]
             if e not in set(splits["train_estates"])]
    if stray:
        raise RuntimeError(f"{path}: rank assignment names non-train "
                           f"estates: {stray[:5]}")

    if require_abupt and not m.get("abupt_eligible", False):
        raise RuntimeError(
            f"{path} is not an AB-UPT-eligible pool. AB-UPT draws "
            f"{m.get('n_anchors_min', N_ANCHORS)} surface + volume anchors per "
            f"sample without replacement, and this pool contains estates "
            f"whose N_surf is below that (or fewer than four wind "
            f"directions). Train AB-UPT on manifest_400.json; both pools "
            f"share the same test split.")
    return m


def dirs_by_estate(m: dict) -> dict[str, list[str]]:
    """Direction lists come from the manifest, not from pack["dirs"]: the
    per-rank case counts are computed from these lists."""
    return {e: info["dirs"] for e, info in m["estates"].items()}


def roster_sha256(m: dict) -> str:
    """Content hash of the ordered roster: the train and test estate lists
    as stored plus each estate's direction list. Recorded with a run and
    with its evaluation, so results are never reused across rosters."""
    payload = {
        "train_estates": list(m["splits"]["train_estates"]),
        "test_estates": list(m["splits"]["test_estates"]),
        "dirs": {e: list(info["dirs"])
                 for e, info in sorted(m["estates"].items())},
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def train_estates_for_rank(m: dict, rank: int) -> list[str]:
    return list(m["ranks"][rank]["estates"])


def test_case_names(m: dict, rank: int, world: int) -> list[str]:
    """Evaluation shard at estate granularity: rank r takes
    sorted(test_estates)[r::world] whole, so each test pack is opened once
    by one rank."""
    est = m["estates"]
    mine = sorted(m["splits"]["test_estates"])[rank::world]
    return [f"{e}_{d}" for e in mine for d in est[e]["dirs"]]


# ---------------------------------------------------------------------------
# Rank assignment
# ---------------------------------------------------------------------------

def _solve_quotas(n_by_bucket: dict[int, int], world: int
                  ) -> tuple[dict, dict[int, int]]:
    """Per-(rank, direction-count bucket) estate quotas whose per-rank case
    sums are strictly equal, dropping the fewest estates to get there.
    Unequal per-rank step counts would deadlock DDP at the end of an
    epoch."""
    buckets = sorted(n_by_bucket)
    for total_drop in range(0, sum(n_by_bucket.values()) + 1):
        for drop_combo in itertools.product(
                *[range(min(total_drop, n_by_bucket[b]) + 1) for b in buckets]):
            if sum(drop_combo) != total_drop:
                continue
            adj = {b: n_by_bucket[b] - dc
                   for b, dc in zip(buckets, drop_combo)}
            base = {b: adj[b] // world for b in buckets}
            left = {b: adj[b] % world for b in buckets}
            base_cases = sum(b * base[b] for b in buckets)
            per_bucket_opts = []
            for b in buckets:
                opts = [c for c in itertools.product(range(left[b] + 1),
                                                     repeat=world)
                        if sum(c) == left[b]]
                per_bucket_opts.append(opts)
            for combo in itertools.product(*per_bucket_opts):
                rank_cases = [base_cases + sum(b * combo[bi][r]
                                               for bi, b in enumerate(buckets))
                              for r in range(world)]
                if len(set(rank_cases)) == 1:
                    quotas = {r: {b: base[b] + combo[bi][r]
                                  for bi, b in enumerate(buckets)}
                              for r in range(world)}
                    return quotas, dict(zip(buckets, drop_combo))
    raise RuntimeError("quota solver found no equal-count partition "
                       "(should be unreachable)")


def build_rank_assignment(entries: list[dict], world: int,
                          train_eval: list[str], seed: int) -> dict:
    """entries: [{estate, dirs, bytes}] over the training pool.

    Hard constraints: per-rank case counts strictly equal (fewest estate
    drops), and the train_eval estates spread evenly over the ranks.
    Soft objective: balance the predicted resident bytes (largest-first
    seeding, then same-class pairwise swaps). Deterministic given the
    inputs.
    """
    by_name = {e["estate"]: e for e in entries}
    for te in train_eval:
        if te not in by_name:
            raise RuntimeError(
                f"train_eval estate {te} is not in the train pool — the "
                f"train_eval list must be a subset of train_estates")
        if len(by_name[te]["dirs"]) != 4:
            raise RuntimeError(f"train_eval estate {te} is not 4-direction")
    if train_eval and len(train_eval) % world:
        raise RuntimeError(f"len(train_eval)={len(train_eval)} not divisible "
                           f"by world={world}")

    dropped: list[str] = []
    pool = list(entries)
    n_by_bucket: dict[int, int] = {}
    for e in pool:
        n_by_bucket[len(e["dirs"])] = n_by_bucket.get(len(e["dirs"]), 0) + 1
    quotas, drops_per_bucket = _solve_quotas(n_by_bucket, world)
    for b, dc in drops_per_bucket.items():
        if not dc:
            continue
        cand = sorted((e for e in pool if len(e["dirs"]) == b
                       and e["estate"] not in train_eval),
                      key=lambda e: (e["bytes"], e["estate"]))
        if len(cand) < dc:
            raise RuntimeError(f"cannot drop {dc} estates from bucket {b} "
                               f"without touching train_eval")
        dropped += [e["estate"] for e in cand[:dc]]
        pool = [e for e in pool if e["estate"] not in dropped]

    per_rank_te = len(train_eval) // world if train_eval else 0
    for r in range(world):
        if quotas[r].get(4, 0) < per_rank_te:
            raise RuntimeError(f"rank {r} bucket-4 quota {quotas[r].get(4, 0)} "
                               f"< train_eval share {per_rank_te}")

    assign: dict[int, list[dict]] = {r: [] for r in range(world)}
    used = {r: {b: 0 for b in n_by_bucket} for r in range(world)}
    rbytes = [0] * world
    te_count = [0] * world
    te_set = set(train_eval)

    def place(e: dict, r: int):
        assign[r].append(e)
        used[r][len(e["dirs"])] += 1
        rbytes[r] += e["bytes"]

    for e in sorted((by_name[t] for t in train_eval),
                    key=lambda e: (-e["bytes"], e["estate"])):
        r = min((r for r in range(world) if te_count[r] < per_rank_te),
                key=lambda r: rbytes[r])
        place(e, r)
        te_count[r] += 1
    for e in sorted((e for e in pool if e["estate"] not in te_set),
                    key=lambda e: (-e["bytes"], e["estate"])):
        b = len(e["dirs"])
        r = min((r for r in range(world) if used[r][b] < quotas[r][b]),
                key=lambda r: rbytes[r])
        place(e, r)

    # Swapping two estates of the same (bucket, train_eval) class keeps both
    # hard constraints.
    def cls(e):
        return (len(e["dirs"]), e["estate"] in te_set)

    for _ in range(500):
        hi = max(range(world), key=lambda r: rbytes[r])
        lo = min(range(world), key=lambda r: rbytes[r])
        spread = rbytes[hi] - rbytes[lo]
        best = None
        for eh in assign[hi]:
            for el in assign[lo]:
                if cls(eh) != cls(el):
                    continue
                delta = eh["bytes"] - el["bytes"]
                if 0 < delta < spread and (best is None or delta > best[0]):
                    if max(rbytes[hi] - delta, rbytes[lo] + delta) < rbytes[hi]:
                        best = (delta, eh, el)
        if best is None:
            break
        _, eh, el = best
        assign[hi].remove(eh)
        assign[lo].remove(el)
        assign[hi].append(el)
        assign[lo].append(eh)
        rbytes[hi] += el["bytes"] - eh["bytes"]
        rbytes[lo] += eh["bytes"] - el["bytes"]

    counts = [sum(len(e["dirs"]) for e in assign[r]) for r in range(world)]
    assert len(set(counts)) == 1, f"per-rank case counts differ: {counts}"
    ranks = [{
        "rank": r,
        "estates": sorted(e["estate"] for e in assign[r]),
        "n_cases": counts[r],
        "bytes_pred": rbytes[r],
        "train_eval": sorted(e["estate"] for e in assign[r]
                             if e["estate"] in te_set),
    } for r in range(world)]
    return {"ranks": ranks, "dropped": sorted(dropped), "seed": seed,
            "case_count_per_rank": counts[0], "bytes_pred_per_rank": rbytes,
            "max_minus_min_bytes": max(rbytes) - min(rbytes),
            "assignment_sha256": assignment_hash(ranks)}


# ---------------------------------------------------------------------------
# Building the pool manifests
# ---------------------------------------------------------------------------

def _assemble(pool: str, estates: dict, test_estates: list[str],
              train_estates: list[str], train_eval: list[str],
              part: dict, world: int, seed: int,
              abupt_eligible: bool, n_anchors: int, source: dict) -> dict:
    keep = sorted(set(train_estates) | set(test_estates))
    return {
        "schema": SCHEMA,
        "pool": pool,
        "seed": seed,
        "world": world,
        "abupt_eligible": bool(abupt_eligible),
        "n_anchors_min": int(n_anchors),
        "source": source,
        "estates": {e: {"dirs": estates[e]["dirs"],
                        "canonical": estates[e]["canonical"]} for e in keep},
        "splits": {"test_estates": sorted(test_estates),
                   "train_estates": sorted(train_estates),
                   "train_eval_estates": sorted(train_eval)},
        "ranks": part["ranks"],
        "dropped": part["dropped"],
        "case_count_per_rank": part["case_count_per_rank"],
        "bytes_pred_per_rank": part["bytes_pred_per_rank"],
        "max_minus_min_bytes": part["max_minus_min_bytes"],
        "assignment_sha256": part["assignment_sha256"],
        "totals": {
            "estates": len(keep),
            "train_estates": len(train_estates),
            "test_estates": len(test_estates),
            "train_cases": sum(len(estates[e]["dirs"]) for e in train_estates),
            "test_cases": sum(len(estates[e]["dirs"]) for e in test_estates),
        },
    }


def build_pool_manifests(estates: dict, index: dict,
                         test_estates: list[str], train_400: list[str],
                         train_eval: list[str], *, world: int, seed: int,
                         n_leaves: int, n_pn_neighbors: int, source: dict,
                         n_anchors: int = N_ANCHORS,
                         available: set[str] | None = None
                         ) -> dict[str, dict]:
    """The two pool manifests from given name lists.

    estates:   {estate: {dirs, canonical}}, the clean estates.
    index:     {estate: {N_vol, N_surf}}, for the byte balance and the
               AB-UPT eligibility check.
    available: estates present, with identical direction lists, in every
               cache on disk (a manifest must be readable through any of
               them). The full pool is restricted to this set; the given
               lists are checked against it, never silently shrunk.

    test_estates / train_400 / train_eval are copied as given. The full
    pool's train list is everything else that is available, and the only
    computed quantity is each pool's rank assignment.
    """
    for name, lst in (("test", test_estates), ("train400", train_400),
                      ("train_eval", train_eval)):
        missing = [e for e in lst if e not in estates]
        if missing:
            raise RuntimeError(f"{len(missing)} {name} estates are absent "
                               f"from the clean pool: {missing[:5]}")
        missing_idx = [e for e in lst if e not in index]
        if missing_idx:
            raise RuntimeError(f"{len(missing_idx)} {name} estates are "
                               f"absent from _index.json: {missing_idx[:5]}")
    overlap = set(test_estates) & set(train_400)
    if overlap:
        raise RuntimeError(f"the 400 training estates and the test set "
                           f"overlap on {len(overlap)} estates: "
                           f"{sorted(overlap)[:5]}")
    stray = set(train_eval) - set(train_400)
    if stray:
        raise RuntimeError(f"train_eval must be a subset of the 400: "
                           f"{sorted(stray)[:5]}")

    excluded: list[str] = []
    if available is not None:
        gone = sorted((set(test_estates) | set(train_400)) - available)
        if gone:
            raise RuntimeError(
                f"{len(gone)} estates in the test/400 lists are missing from "
                f"at least one cache: {gone[:5]}. Rebuild the incomplete "
                f"cache (`main.py preprocess --grid <which>`).")
        excluded = sorted(set(estates) - available)

    # the 400 and the test estates must all be AB-UPT-eligible
    bad = [(e, index[e]["N_surf"], index[e]["N_vol"])
           for e in sorted(set(train_400) | set(test_estates))
           if min(int(index[e]["N_vol"]),
                  int(index[e]["N_surf"])) < n_anchors
           or len(estates[e]["dirs"]) != 4]
    if bad:
        raise RuntimeError(
            f"{len(bad)} estates in the 400/test lists are not "
            f"AB-UPT-eligible (need 4 directions and "
            f"min(N_vol, N_surf) >= {n_anchors}): {bad[:5]}")

    def entries(names: list[str]) -> list[dict]:
        return [{"estate": e, "dirs": estates[e]["dirs"],
                 "bytes": resident_bytes(int(index[e]["N_vol"]),
                                         int(index[e]["N_surf"]),
                                         len(estates[e]["dirs"]),
                                         n_leaves, n_pn_neighbors)}
                for e in names]

    out = {}
    test_set = set(test_estates)
    excl_set = set(excluded)
    if excluded:
        print(f"[manifest] full pool: excluded {len(excluded)} estate(s) "
              f"absent from at least one cache: "
              f"{excluded[:8]}{' ...' if len(excluded) > 8 else ''}",
              flush=True)
    source = {**source, "excluded_not_in_every_cache": excluded}
    for pool, train_names, eligible in (
            ("400", sorted(train_400), True),
            ("full", sorted(e for e in estates
                            if e not in test_set and e not in excl_set),
             False)):
        part = build_rank_assignment(entries(train_names), world,
                                     sorted(train_eval), seed)
        placed = {e for r in part["ranks"] for e in r["estates"]}
        out[pool] = _assemble(pool, estates, test_estates,
                              sorted(placed), train_eval, part,
                              world, seed, eligible, n_anchors, source)
    return out


POOL_FILES = {"400": "manifest_400.json", "full": "manifest_full.json"}


def cache_generations(work_dir: Path) -> list[Path]:
    """Every built cache under work_dir (each directory with an
    _index.json)."""
    root = work_dir / "cache_estate"
    return sorted(p.parent for p in root.glob("*/_index.json")) \
        if root.exists() else []


def available_estates(cache_dirs: list[Path]) -> tuple[set[str], dict]:
    """(estates present with identical direction lists in every cache, the
    first cache's index). A manifest must be readable through every
    `--grid`, and matching direction lists keep the per-rank case counts
    identical across caches."""
    if not cache_dirs:
        raise FileNotFoundError("no cache carries an _index.json")
    indexes = [read_json(d / "_index.json") for d in cache_dirs]
    common = set(indexes[0])
    for idx in indexes[1:]:
        common &= set(idx)
    common = {e for e in common
              if len({tuple(sorted(idx[e]["dirs"])) for idx in indexes}) == 1}
    for d, idx in zip(cache_dirs, indexes):
        missing = len(set().union(*(set(i) for i in indexes)) - set(idx))
        print(f"[manifest] cache {d.name}: {len(idx)} estates"
              f"{f' ({missing} absent vs the union)' if missing else ''}",
              flush=True)
    print(f"[manifest] usable in every cache: {len(common)}", flush=True)
    return common, indexes[0]


def repartition(manifests: dict[str, dict], index: dict, *, world: int,
                n_leaves: int, n_pn_neighbors: int) -> dict[str, dict]:
    """Re-run only the rank assignment at a new world size; every name list
    is kept, so a world change never moves an estate between train and
    test."""
    out = {}
    for pool, m in manifests.items():
        estates = m["estates"]
        train = list(m["splits"]["train_estates"])
        entries = [{"estate": e, "dirs": estates[e]["dirs"],
                    "bytes": resident_bytes(int(index[e]["N_vol"]),
                                            int(index[e]["N_surf"]),
                                            len(estates[e]["dirs"]),
                                            n_leaves, n_pn_neighbors)}
                   for e in train]
        part = build_rank_assignment(entries, world,
                                     list(m["splits"]["train_eval_estates"]),
                                     m["seed"])
        placed = sorted({e for r in part["ranks"] for e in r["estates"]})
        out[pool] = _assemble(
            pool, estates, m["splits"]["test_estates"], placed,
            m["splits"]["train_eval_estates"], part, world, m["seed"],
            m["abupt_eligible"], m["n_anchors_min"],
            {**m.get("source", {}), "repartitioned_to_world": world})
    return out


def describe(m: dict) -> str:
    t = m["totals"]
    gb = [b / 1024 ** 3 for b in m["bytes_pred_per_rank"]]
    return (f"pool={m['pool']} world={m['world']} "
            f"abupt_eligible={m['abupt_eligible']}\n"
            f"  train {t['train_estates']} estates / {t['train_cases']} cases"
            f"  ({m['case_count_per_rank']} cases per rank, strict)\n"
            f"  test  {t['test_estates']} estates / {t['test_cases']} cases"
            f"  train_eval {len(m['splits']['train_eval_estates'])} estates\n"
            f"  resident/rank {['%.1fGiB' % g for g in gb]} "
            f"(max-min {m['max_minus_min_bytes'] / 1024 ** 3:.2f}GiB)\n"
            f"  dropped for equal case counts: {m['dropped'] or 'none'}")


# ---------------------------------------------------------------------------
# Narrowing the test list inside an existing manifest
# ---------------------------------------------------------------------------

def stratified_test_subset(names: list[str], n_buildings: dict[str, int],
                           n_test: int, n_strata: int, seed: int) -> list[str]:
    """Pick n_test names out of `names`, stratified by building count.

    Sort by (n_buildings, name), cut into n_strata contiguous strata of
    near-equal size, and draw the same number from each with one seeded RNG,
    so sparse and dense estates are both represented. Deterministic given
    the inputs; the result is a subset of `names`.
    """
    import numpy as np

    if n_test > len(names):
        raise ValueError(f"asked for {n_test} test estates out of "
                         f"{len(names)} candidates")
    ordered = sorted(names, key=lambda e: (int(n_buildings[e]), e))
    rng = np.random.default_rng(seed)
    bounds = np.linspace(0, len(ordered), n_strata + 1).round().astype(int)
    quota = [n_test // n_strata] * n_strata
    for i in range(n_test - sum(quota)):      # remainder to the densest strata
        quota[-1 - i] += 1

    picked: list[str] = []
    for s in range(n_strata):
        block = ordered[bounds[s]:bounds[s + 1]]
        k = min(quota[s], len(block))
        picked += [block[i] for i in
                   sorted(rng.choice(len(block), size=k, replace=False))]
    # a stratum smaller than its quota leaves the draw short: top up from
    # what is left, densest first
    if len(picked) < n_test:
        rest = [e for e in reversed(ordered) if e not in set(picked)]
        picked += rest[:n_test - len(picked)]
    return sorted(picked)


def apply_test_subset(m: dict, keep_test: list[str], selection: dict) -> dict:
    """A copy of pool manifest `m` whose test split is `keep_test`.

    Changes splits.test_estates, the estates roster, totals and a new
    test_selection record. The rank assignment partitions the training list
    only, so it (and assignment_sha256) stays unchanged; asserted below.
    """
    keep_test = sorted(keep_test)
    missing = [e for e in keep_test if e not in m["estates"]]
    if missing:
        raise ValueError(f"{len(missing)} of the requested test estates are "
                         f"not in this manifest: {missing[:5]}")
    stray = set(keep_test) & set(m["splits"]["train_estates"])
    if stray:
        raise ValueError(f"test/train overlap: {sorted(stray)[:5]}")

    out = json.loads(json.dumps(m))     # deep copy, plain json types only
    before = out["assignment_sha256"]
    train = out["splits"]["train_estates"]
    out["splits"]["test_estates"] = keep_test
    keep = sorted(set(train) | set(keep_test))
    out["estates"] = {e: m["estates"][e] for e in keep}
    out["totals"] = {
        "estates": len(keep),
        "train_estates": len(train),
        "test_estates": len(keep_test),
        "train_cases": sum(len(m["estates"][e]["dirs"]) for e in train),
        "test_cases": sum(len(m["estates"][e]["dirs"]) for e in keep_test),
    }
    out["test_selection"] = selection
    assert out["assignment_sha256"] == before, \
        "the rank assignment must not change when only the test list does"
    return out
