"""Shared utilities for per-phecode sequence-vs-pooled bootstrap analysis.

Companion to `per_phecode_seq_vs_pool.py`. Holds I/O, alignment, and
cluster-resample-bootstrap primitives that are testable in isolation.

Conventions:
- An "arm" is one trained disease-prediction model run.
- Arm artifacts on disk: test_hazards.npy (N_rows, N_phecodes) float32,
  test_event_times.npy (N_rows, N_phecodes) float32, test_is_event.npy
  (N_rows, N_phecodes) uint8/bool, test_eval_mask.npy (N_rows, N_phecodes)
  bool, test_eids.npy (N_rows,) int64. test_eids is sorted; an EID may
  appear up to ~4 times (one per evaluation window).
- C-index convention (lifelines): higher hazard means shorter survival, so
  pass `-hazards` to concordance_index.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index as li_concordance_index


# ---------------------------------------------------------------------------
# Arm loader
# ---------------------------------------------------------------------------


@dataclass
class Arm:
    name: str
    run_dir: Path
    eids: np.ndarray            # (N_rows,) int64, sorted
    hazards: np.ndarray         # (N_rows, N_phecodes) float32
    event_times: np.ndarray     # (N_rows, N_phecodes) float32
    is_event: np.ndarray        # (N_rows, N_phecodes) bool
    eval_mask: np.ndarray       # (N_rows, N_phecodes) bool
    phecodes: list[str]


def load_arm(name: str, run_dir: str | Path) -> Arm:
    rd = Path(run_dir)
    with open(rd / "config.json") as f:
        cfg = json.load(f)
    phecodes = list(cfg["phecodes"])
    eids = np.load(rd / "test_eids.npy")
    hz = np.load(rd / "test_hazards.npy")
    et = np.load(rd / "test_event_times.npy")
    ev = np.load(rd / "test_is_event.npy").astype(bool)
    mk = np.load(rd / "test_eval_mask.npy").astype(bool)
    assert hz.shape == et.shape == ev.shape == mk.shape, \
        f"{name}: shape mismatch {hz.shape} {et.shape} {ev.shape} {mk.shape}"
    assert hz.shape[0] == eids.shape[0], \
        f"{name}: eids length {eids.shape[0]} != hazards rows {hz.shape[0]}"
    assert hz.shape[1] == len(phecodes), \
        f"{name}: hazards cols {hz.shape[1]} != n_phecodes {len(phecodes)}"
    assert (eids[:-1] <= eids[1:]).all(), f"{name}: eids not sorted"
    return Arm(name=name, run_dir=rd, eids=eids, hazards=hz,
               event_times=et, is_event=ev, eval_mask=mk, phecodes=phecodes)


# ---------------------------------------------------------------------------
# EID intersection + positional alignment
# ---------------------------------------------------------------------------


def _eid_to_row_indices(eids: np.ndarray) -> dict[int, np.ndarray]:
    """Return mapping eid -> array of row indices (sorted ascending)."""
    out: dict[int, list[int]] = {}
    for i, e in enumerate(eids):
        out.setdefault(int(e), []).append(i)
    return {k: np.asarray(v, dtype=np.int64) for k, v in out.items()}


def align_arms_by_eid(arms: Sequence[Arm]) -> tuple[list[np.ndarray], list[int]]:
    """For each arm, return an array of row indices such that aligning the
    arrays produces one row per (EID, position-within-EID) tuple that exists
    in EVERY arm. Returns (index_arrays, common_eid_list_unique).

    Within each EID, we take the first min(count_per_arm) rows from each arm,
    in order. This gives row-by-row pairing that uses as many rows as the
    most-restrictive arm allows.
    """
    arm_maps = [_eid_to_row_indices(a.eids) for a in arms]
    common_eids = sorted(set.intersection(*[set(m.keys()) for m in arm_maps]))
    idx_per_arm: list[list[int]] = [[] for _ in arms]
    for e in common_eids:
        per_arm_rows = [m[e] for m in arm_maps]
        k = min(len(r) for r in per_arm_rows)
        for ai, rows in enumerate(per_arm_rows):
            idx_per_arm[ai].extend(rows[:k].tolist())
    return [np.asarray(x, dtype=np.int64) for x in idx_per_arm], common_eids


def build_aligned_panel(arms: Sequence[Arm]) -> tuple[list[np.ndarray], np.ndarray,
                                                       np.ndarray, np.ndarray,
                                                       np.ndarray]:
    """Build aligned per-row arrays for paired analysis.

    Returns
    -------
    hazards_per_arm : list of (N_aligned, N_phecodes) float32, one per arm
    event_times     : (N_aligned, N_phecodes) float32, shared (asserted)
    is_event        : (N_aligned, N_phecodes) bool, shared (asserted)
    eval_mask       : (N_aligned, N_phecodes) bool, AND across arms
    row_to_eid      : (N_aligned,) int64, for cluster bootstrap on EIDs

    Asserts that event_times and is_event are byte-identical across arms
    (they come from labels, not model output, so divergence = bug).
    """
    assert arms, "need at least one arm"
    assert all(a.phecodes == arms[0].phecodes for a in arms), \
        "phecode lists differ across arms"
    idx_per_arm, _ = align_arms_by_eid(arms)
    n_aligned = len(idx_per_arm[0])
    assert all(len(idx) == n_aligned for idx in idx_per_arm), \
        "alignment row counts disagree"

    hazards_per_arm = [a.hazards[idx] for a, idx in zip(arms, idx_per_arm)]
    event_times = arms[0].event_times[idx_per_arm[0]]
    is_event = arms[0].is_event[idx_per_arm[0]]
    eval_mask = arms[0].eval_mask[idx_per_arm[0]].copy()
    row_to_eid = arms[0].eids[idx_per_arm[0]]

    for a, idx in zip(arms[1:], idx_per_arm[1:]):
        et_b = a.event_times[idx]
        ev_b = a.is_event[idx]
        mk_b = a.eval_mask[idx]
        if not np.array_equal(et_b, event_times):
            diffs = int((et_b != event_times).sum())
            raise AssertionError(
                f"event_times differ between arm {arms[0].name} and {a.name} "
                f"on {diffs} cells after alignment (label divergence)")
        if not np.array_equal(ev_b, is_event):
            diffs = int((ev_b != is_event).sum())
            raise AssertionError(
                f"is_event differs between arm {arms[0].name} and {a.name} "
                f"on {diffs} cells after alignment")
        eval_mask &= mk_b

    return hazards_per_arm, event_times, is_event, eval_mask, row_to_eid


# ---------------------------------------------------------------------------
# C-index + paired bootstrap
# ---------------------------------------------------------------------------


def c_index(h: np.ndarray, t: np.ndarray, e: np.ndarray) -> float:
    if e.sum() == 0 or len(e) < 2:
        return float("nan")
    try:
        return float(li_concordance_index(t, -h, e))
    except ZeroDivisionError:
        return float("nan")


def point_estimates(hazards_per_arm: list[np.ndarray], event_times: np.ndarray,
                    is_event: np.ndarray, eval_mask: np.ndarray
                    ) -> dict[str, np.ndarray]:
    """Per-phecode point estimates: c per arm, n_eval, n_pos."""
    n_arms = len(hazards_per_arm)
    n_phecodes = event_times.shape[1]
    c_per_arm = np.full((n_arms, n_phecodes), np.nan, dtype=np.float64)
    n_eval = np.zeros(n_phecodes, dtype=np.int64)
    n_pos = np.zeros(n_phecodes, dtype=np.int64)
    for p in range(n_phecodes):
        m = eval_mask[:, p]
        n_eval[p] = int(m.sum())
        n_pos[p] = int(is_event[m, p].sum())
        if n_pos[p] == 0 or n_eval[p] < 2:
            continue
        t = event_times[m, p]
        e = is_event[m, p]
        for a, h_a in enumerate(hazards_per_arm):
            c_per_arm[a, p] = c_index(h_a[m, p], t, e)
    return {"c_per_arm": c_per_arm, "n_eval": n_eval, "n_pos": n_pos}


def _bootstrap_single_iter(
    seed_offset: int,
    hazards_per_arm: list[np.ndarray],
    event_times: np.ndarray,
    is_event: np.ndarray,
    eval_mask: np.ndarray,
    eid_to_rows: dict[int, np.ndarray],
    unique_eids: np.ndarray,
    base_seed: int,
) -> np.ndarray:
    """One bootstrap iteration. Returns (n_arms, n_phecodes) C-index array."""
    n_arms = len(hazards_per_arm)
    n_phecodes = event_times.shape[1]
    rng = np.random.default_rng(base_seed + seed_offset)
    n_eids = len(unique_eids)
    sampled_eids = rng.choice(unique_eids, size=n_eids, replace=True)
    chunks = [eid_to_rows[int(e)] for e in sampled_eids]
    rows = np.concatenate(chunks)
    out = np.full((n_arms, n_phecodes), np.nan, dtype=np.float64)
    for p in range(n_phecodes):
        m = eval_mask[rows, p]
        if not m.any():
            continue
        rows_m = rows[m]
        if is_event[rows_m, p].sum() == 0 or rows_m.size < 2:
            continue
        t = event_times[rows_m, p]
        e_flag = is_event[rows_m, p]
        for a, h_a in enumerate(hazards_per_arm):
            out[a, p] = c_index(h_a[rows_m, p], t, e_flag)
    return out


def cluster_paired_bootstrap(
    hazards_per_arm: list[np.ndarray],
    event_times: np.ndarray,
    is_event: np.ndarray,
    eval_mask: np.ndarray,
    row_to_eid: np.ndarray,
    n_boot: int,
    seed: int,
    n_jobs: int = 1,
    log_every: int | None = None,
) -> np.ndarray:
    """Cluster-resampled paired bootstrap.

    Resamples unique EIDs with replacement; the resampled rows = all rows
    belonging to the drawn EIDs. Per iteration, recompute per-arm per-phecode
    C-index on the masked subset. Returns boot_c with shape
    (n_boot, n_arms, n_phecodes).

    Parallelizes across iterations when n_jobs > 1 (joblib loky backend).
    Each worker draws from `seed + iter_index` so iterations are independent
    and the result is reproducible regardless of n_jobs.
    """
    n_arms = len(hazards_per_arm)
    n_phecodes = event_times.shape[1]

    # Pre-compute EID -> row-indices map once (shared across iterations).
    eid_to_rows: dict[int, list[int]] = {}
    for i, e in enumerate(row_to_eid):
        eid_to_rows.setdefault(int(e), []).append(i)
    eid_to_rows = {k: np.asarray(v, dtype=np.int64) for k, v in eid_to_rows.items()}
    unique_eids = np.fromiter(eid_to_rows.keys(), dtype=np.int64)

    out = np.full((n_boot, n_arms, n_phecodes), np.nan, dtype=np.float64)

    if n_jobs == 1:
        for b in range(n_boot):
            if log_every and b % log_every == 0:
                print(f"[bootstrap] iter {b}/{n_boot}", flush=True)
            out[b] = _bootstrap_single_iter(
                b, hazards_per_arm, event_times, is_event, eval_mask,
                eid_to_rows, unique_eids, seed)
        return out

    from joblib import Parallel, delayed
    print(f"[bootstrap] running {n_boot} iters with n_jobs={n_jobs}", flush=True)
    results = Parallel(n_jobs=n_jobs, backend="loky",
                       verbose=10 if log_every else 0)(
        delayed(_bootstrap_single_iter)(
            b, hazards_per_arm, event_times, is_event, eval_mask,
            eid_to_rows, unique_eids, seed)
        for b in range(n_boot)
    )
    for b, r in enumerate(results):
        out[b] = r
    return out


# ---------------------------------------------------------------------------
# Phecode metadata
# ---------------------------------------------------------------------------


TIME_TO_DEATH_META = {"name": "All-cause mortality", "category": "Mortality"}


def load_phecode_meta(map_csv: str | Path) -> dict[str, dict[str, str]]:
    """Phecode → {name, category}. Adds time_to_death as Mortality."""
    df = pd.read_csv(map_csv)[["Phecode", "PhecodeString", "PhecodeCategory"]]
    df = df.drop_duplicates("Phecode")
    out: dict[str, dict[str, str]] = {}
    for _, r in df.iterrows():
        p = str(r["Phecode"])
        out[p] = {
            "name": str(r["PhecodeString"]) if pd.notna(r["PhecodeString"]) else p,
            "category": str(r["PhecodeCategory"]) if pd.notna(r["PhecodeCategory"]) else "Other",
        }
    out["time_to_death"] = TIME_TO_DEATH_META
    return out


def attach_meta(phecodes: list[str], meta: dict[str, dict[str, str]]) -> tuple[list[str], list[str]]:
    names, cats = [], []
    for p in phecodes:
        m = meta.get(p) or meta.get(str(p)) or {"name": str(p), "category": "Other"}
        names.append(m["name"])
        cats.append(m["category"])
    return names, cats


# ---------------------------------------------------------------------------
# FDR + summary stats
# ---------------------------------------------------------------------------


def bh_fdr(p: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg step-up. Returns (q_values, rejected_at_alpha).
    NaN p-values pass through as NaN q-values and False rejections.
    """
    p = np.asarray(p, dtype=np.float64)
    valid = ~np.isnan(p)
    q = np.full_like(p, np.nan)
    rej = np.zeros_like(p, dtype=bool)
    pv = p[valid]
    m = pv.size
    if m == 0:
        return q, rej
    order = np.argsort(pv)
    ranked = pv[order]
    # q-values in rank order, then scatter back to original positions.
    qv_sorted = ranked * m / (np.arange(m) + 1)
    qv_sorted = np.minimum.accumulate(qv_sorted[::-1])[::-1]
    qv_sorted = np.clip(qv_sorted, 0.0, 1.0)
    q_full = np.empty(m)
    q_full[order] = qv_sorted
    q[valid] = q_full
    # BH step-up rejection in rank order, then scatter back.
    rej_sorted = ranked <= (np.arange(m) + 1) * alpha / m
    rej_sorted = np.maximum.accumulate(rej_sorted[::-1])[::-1]
    rej_full = np.zeros(m, dtype=bool)
    rej_full[order] = rej_sorted
    rej[valid] = rej_full
    return q, rej


def percentile_ci(boot: np.ndarray, lo: float = 2.5, hi: float = 97.5,
                  axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    lo_v = np.nanpercentile(boot, lo, axis=axis)
    hi_v = np.nanpercentile(boot, hi, axis=axis)
    return lo_v, hi_v
