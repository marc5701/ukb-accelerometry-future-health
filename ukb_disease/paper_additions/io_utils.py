"""Shared IO + a fast Harrell C-index for the paper-additions bootstraps.

The canonical per-disease point estimates are produced by
`ukb_disease.baseline.per_disease_eval` with lifelines. lifelines is exact but slow
(too slow for a joint subject-level bootstrap over 390 diseases x ~1000
replicates). Here we provide a Fenwick-tree O(n log n) Harrell C that matches
lifelines to < 1e-3 on the full data (verified in `__main__`), used ONLY for
bootstrap variability. Published point estimates stay the lifelines values.

Run `python -m ukb_disease.paper_additions.io_utils` to validate that
(a) the fast C matches lifelines, and (b) the seed-averaged mean-C reproduces
the published Embedding 0.6880 / Sequence_model 0.6865.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ukb_disease.baseline.config import resolve_oak_path
from ukb_disease.paths import UKB_ROOT

RUNS_ROOT = f"{UKB_ROOT}/runs_cox"
SEEDS = (42, 123, 456, 789)
MIN_EVAL = 10  # per_disease_eval floor: >= 10 evaluable subjects + >= 1 event


# Fast Harrell C-index (Fenwick tree)

def cindex(hazard: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """Harrell C-index. Higher hazard => higher risk => shorter survival.

    Comparable pair (i, j): the earlier-time subject is an event. Concordant if
    that subject has the higher hazard. Tied hazards count 0.5. Tied times are
    not comparable (standard). O(n log n) via a Fenwick tree over hazard ranks.
    Returns NaN if no comparable pair exists.
    """
    hazard = np.asarray(hazard, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=np.float64) > 0.5
    n = hazard.shape[0]
    if n < 2 or not event.any():
        return float("nan")

    # Compress hazards to 1..R ranks (ties share a rank).
    uniq = np.unique(hazard)
    rank = np.searchsorted(uniq, hazard) + 1  # 1-based
    R = uniq.shape[0]

    # Fenwick tree (count of inserted subjects per hazard-rank).
    tree = np.zeros(R + 1, dtype=np.int64)

    def _update(i: int) -> None:
        while i <= R:
            tree[i] += 1
            i += i & (-i)

    def _prefix(i: int) -> int:  # count with rank <= i
        s = 0
        while i > 0:
            s += tree[i]
            i -= i & (-i)
        return s

    # Process subjects by DESCENDING time, in groups of equal time. For each
    # group: first SCORE its events against everything already inserted (all of
    # which have strictly greater time), then INSERT the whole group.
    order = np.argsort(-time, kind="stable")
    h_o = hazard[order]
    t_o = time[order]
    e_o = event[order]
    r_o = rank[order]

    concordant = 0.0
    ties = 0.0
    comparable = 0.0
    inserted = 0

    i = 0
    while i < n:
        j = i
        while j < n and t_o[j] == t_o[i]:
            j += 1
        # Score events in [i, j) against the `inserted` greater-time subjects.
        for k in range(i, j):
            if e_o[k]:
                ri = int(r_o[k])
                lt = _prefix(ri - 1)          # inserted hazards strictly less
                le = _prefix(ri)              # inserted hazards <= ri
                eq = le - lt                  # inserted hazards equal
                concordant += lt
                ties += eq
                comparable += inserted
        # Insert the whole group.
        for k in range(i, j):
            _update(int(r_o[k]))
            inserted += 1
        i = j

    if comparable == 0:
        return float("nan")
    return float((concordant + 0.5 * ties) / comparable)


# Run loaders

def load_run(run_name: str, root: str = RUNS_ROOT) -> dict:
    d = Path(resolve_oak_path(f"{root}/{run_name}"))
    cfg = json.load(open(d / "config.json"))
    return {
        "eids": np.load(d / "test_eids.npy"),
        "hazards": np.load(d / "test_hazards.npy"),
        "times": np.load(d / "test_event_times.npy"),
        "events": np.load(d / "test_is_event.npy"),
        "mask": np.load(d / "test_eval_mask.npy").astype(bool),
        "phecodes": list(cfg["phecodes"]),
    }


def load_seed_avg(prefix: str, seeds=SEEDS, root: str = RUNS_ROOT) -> dict:
    """Seed-average the per-subject hazards for `<prefix>_seed<S>` runs.

    All seeds share the same subject-disjoint test set; `_aggregate_to_subject`
    returns eids sorted ascending, so the rows align across seeds (asserted).
    Labels/masks are identical across seeds (same subjects) -> taken from seed[0].
    """
    runs = [load_run(f"{prefix}_seed{s}", root) for s in seeds]
    base = runs[0]
    for r in runs[1:]:
        if not np.array_equal(r["eids"], base["eids"]):
            raise ValueError(f"eid mismatch between {prefix} seeds; cannot seed-average")
        if r["phecodes"] != base["phecodes"]:
            raise ValueError(f"phecode mismatch between {prefix} seeds")
    haz = np.mean([r["hazards"] for r in runs], axis=0).astype(np.float32)
    return {
        "eids": base["eids"],
        "hazards": haz,
        "times": base["times"],
        "events": base["events"],
        "mask": base["mask"],
        "phecodes": base["phecodes"],
        "per_seed_hazards": [r["hazards"] for r in runs],
    }


def evaluable_diseases(run: dict, min_eval: int = MIN_EVAL) -> np.ndarray:
    """Indices of phecodes passing the eval floor (>= min_eval evaluable, >= 1 event)."""
    m = run["mask"]
    e = run["events"]
    keep = []
    for p in range(len(run["phecodes"])):
        mm = m[:, p]
        if int(mm.sum()) >= min_eval and float(e[mm, p].sum()) > 0:
            keep.append(p)
    return np.asarray(keep, dtype=np.int64)


def per_disease_c(run: dict, disease_idx: np.ndarray, cfn=cindex) -> np.ndarray:
    """C-index per disease over its evaluable subjects (NaN if degenerate)."""
    haz, t, e, m = run["hazards"], run["times"], run["events"], run["mask"]
    out = np.full(len(disease_idx), np.nan)
    for i, p in enumerate(disease_idx):
        mm = m[:, p]
        if mm.sum() >= 2 and e[mm, p].sum() > 0:
            out[i] = cfn(haz[mm, p], t[mm, p], e[mm, p])
    return out


def mean_c(run: dict, min_eval: int = MIN_EVAL, cfn=cindex) -> tuple[float, int]:
    idx = evaluable_diseases(run, min_eval)
    cs = per_disease_c(run, idx, cfn)
    cs = cs[~np.isnan(cs)]
    return float(cs.mean()), int(len(cs))


# Validation

def _validate() -> None:
    from lifelines.utils import concordance_index as li_ci

    def li_c(h, t, e):
        return float(li_ci(t, -h, e))

    print("== fast-C vs lifelines (Embedding seed42, 8 diseases) ==")
    r = load_run("embedding_disjoint_split_seed42")
    idx = evaluable_diseases(r)[:8]
    maxdiff = 0.0
    for p in idx:
        mm = r["mask"][:, p]
        h, t, e = r["hazards"][mm, p], r["times"][mm, p], r["events"][mm, p]
        a, b = cindex(h, t, e), li_c(h, t, e)
        maxdiff = max(maxdiff, abs(a - b))
        print(f"  {r['phecodes'][p]:14s} fast={a:.5f} lifelines={b:.5f} d={a-b:+.5f}")
    print(f"  max|diff| = {maxdiff:.6f}")

    print("\n== seed-averaged mean-C (fast C) ==")
    for prefix, pub in [("embedding_disjoint_split", 0.6880), ("sequence_model_disjoint_split", 0.6865)]:
        sa = load_seed_avg(prefix)
        mc, n = mean_c(sa)
        # also avg-of-per-seed for comparison
        per_seed = []
        for s in SEEDS:
            rr = load_run(f"{prefix}_seed{s}")
            per_seed.append(mean_c(rr)[0])
        print(f"  {prefix:18s} seed-avg-hazard mean-C={mc:.4f} (n={n})  "
              f"avg-of-seed mean-C={np.mean(per_seed):.4f}  published={pub}")


if __name__ == "__main__":
    _validate()
