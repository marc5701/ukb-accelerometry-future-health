"""Label-side logic for the baseline: phecode filtering, prevalent handling, survival-array build.

The phecode CSV at `MORTALITY_PHECODES_CSV` is indexed by `eid` and has one
column per phecode plus a `time_to_death` column. Cell semantics (years since
recording):

    NaN            no event during follow-up (censored)
    > 7/365.25     incident, event occurred post-baseline
    [0, 7/365.25]  within the 7-day exclusion zone, treated as
                   "effective prevalent" under the survival-model labeling policy
    < 0            prevalent, event occurred BEFORE baseline (already had the disease)

Train-side rule for effective_prevalent (prevalent or 7-day-zone):
    include with `is_event=1, time=0.0001`.
Test-side rule for effective_prevalent: exclude this (subject, label) pair
from eval.

Phecode filter: keep columns with `>= min_prevalence` incident-only train positives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ukb_disease.baseline.config import (
    ACC_DEM_CSV,
    MIN_PREVALENCE,
    MORTALITY_PHECODES_CSV,
    SEVEN_DAY_YEARS,
    resolve_oak_path,
)


def load_phecodes(path: str = MORTALITY_PHECODES_CSV) -> pd.DataFrame:
    """Load the phecode + time_to_death CSV (indexed by `eid`).

    Float columns are downcast to float32 to halve memory. Phecode times
    are recorded in years to 4-decimal precision, which fits in float32
    headroom (~7 significant digits) with no loss.
    """
    df = pd.read_csv(resolve_oak_path(path), index_col="eid")
    float_cols = df.select_dtypes(include=["float64"]).columns
    if len(float_cols):
        df[float_cols] = df[float_cols].astype(np.float32)
    return df


def _join_descriptor_csv(
    df: pd.DataFrame, csv: str, cols: list[str] | None, tag: str
) -> pd.DataFrame:
    """Left-join z-scored numeric descriptors from `csv` onto `df` (NaN set to 0).

    Shared by the informative-missingness and circadian rest-activity-rhythm
    covariate families. z-score stats are global over the descriptor CSV
    (train-only would be marginally cleaner); residual join-NaN set to 0 so the
    WindowDataset no-NaN covariate assert holds.
    """
    md = pd.read_csv(resolve_oak_path(csv)).set_index("eid")
    md = md[list(cols)] if cols else md.select_dtypes(include=[np.number])
    mu = md.mean(axis=0)
    sd = md.std(axis=0).replace(0.0, 1.0)
    md = (md - mu) / sd
    new_cols = list(md.columns)
    df = df.join(md, how="left")
    df[new_cols] = df[new_cols].fillna(0.0)
    print(f"[load_covariates] joined {len(new_cols)} {tag} descriptors: {new_cols}")
    return df


def load_covariates(
    path: str = ACC_DEM_CSV,
    missingness_csv: str | None = None,
    missingness_cols: list[str] | None = None,
    circadian_csv: str | None = None,
    circadian_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Load demographics [age, sex] indexed by `eid`.

    If `missingness_csv` (informative-missingness descriptors) and/or
    `circadian_csv` (rest-activity rhythm descriptors) are given, left-join their
    named descriptor columns, z-scored on their own non-NaN values and imputing
    NaN to 0 (WindowDataset asserts no-NaN covariates). z-score stats are global
    over the descriptor CSV, a standard covariate-scaling convenience.
    """
    df = pd.read_csv(resolve_oak_path(path))
    df = df.set_index("eid")[["age_at_first_run", "sex"]].rename(
        columns={"age_at_first_run": "age"}
    )
    if missingness_csv is not None:
        df = _join_descriptor_csv(df, missingness_csv, missingness_cols, "missingness")
    if circadian_csv is not None:
        df = _join_descriptor_csv(df, circadian_csv, circadian_cols, "circadian")
    return df


def filter_phecodes_by_prevalence(
    df: pd.DataFrame,
    train_eids: list[int] | np.ndarray,
    min_prevalence: float = MIN_PREVALENCE,
    seven_day_years: float = SEVEN_DAY_YEARS,
    include_prevalent_in_count: bool = True,
) -> tuple[list[str], dict[str, int]]:
    """Keep phecode columns whose prevalence in train exceeds `min_prevalence`.

    Default: any-positive count (incident + prevalent), matching the reference
    `dropna(thresh=N)` prevalence definition. Set
    `include_prevalent_in_count=False` for the stricter incident-only count
    (t > 7d) when tighter statistical power per phecode is wanted.

    Returns:
        (phecode_columns, stats_dict)
        stats_dict has total counts under both definitions for transparency.
    """
    eids = np.asarray(train_eids)
    train_df = df.reindex(index=eids)
    n_train = len(train_df)
    candidate_cols = [c for c in train_df.columns if c != "time_to_death"]
    # `time_to_death` is included as a phecode-style column too (the project
    # treats mortality as one of the survival outcomes).
    candidate_cols = list(train_df.columns)

    incident_only: dict[str, float] = {}
    any_positive: dict[str, float] = {}
    for col in candidate_cols:
        t = train_df[col]
        n_incident = int(((t > seven_day_years) & t.notna()).sum())
        n_prev = int(((t < 0) & t.notna()).sum())
        incident_only[col] = n_incident / n_train
        any_positive[col] = (n_incident + n_prev) / n_train

    kept = []
    for col in candidate_cols:
        prev_value = any_positive[col] if include_prevalent_in_count else incident_only[col]
        if prev_value >= min_prevalence:
            kept.append(col)

    stats = {
        "n_train": n_train,
        "n_cols_total": len(candidate_cols),
        "n_incident_only_>=thr": sum(1 for c in candidate_cols if incident_only[c] >= min_prevalence),
        "n_any_positive_>=thr": sum(1 for c in candidate_cols if any_positive[c] >= min_prevalence),
        "min_prevalence": min_prevalence,
        "definition": "any_positive" if include_prevalent_in_count else "incident_only",
        "n_kept": len(kept),
    }
    return kept, stats


def build_survival_arrays(
    df: pd.DataFrame,
    phecodes: list[str],
    train_eids: list[int] | np.ndarray,
    test_eids: list[int] | np.ndarray,
    seven_day_years: float = SEVEN_DAY_YEARS,
    val_eids: list[int] | np.ndarray | None = None,
) -> dict:
    """Build (event_times, is_event, eval_mask) for train + test under the split rules.

    If `val_eids` is given (an explicit, subject-disjoint validation partition),
    an extra "val" entry is built using the SAME rules as "train"
    (prevalent/zone becomes immediate event), because val is consumed as an
    early-stopping val-loss and must be comparable to the train objective.

    Returns:
        {
            "train": {"event_times", "is_event", "eval_mask", "eids"},
            "test":  {"event_times", "is_event", "eval_mask", "eids"},
            ["val":  {...}  if val_eids given]
            "max_follow_up": float,
        }
    Where arrays have shape (N_split, n_phecodes) and `eval_mask` is bool;
    False means this (subject, phecode) pair is excluded from eval/loss
    contribution.
    """
    # Project-wide max positive time + 1 yr as global censoring time.
    # Compute column-by-column to avoid the ~2x memory hit of a full block copy.
    max_pos = 0.0
    for col in phecodes:
        col_vals = df[col].to_numpy()
        pos = col_vals[col_vals >= 0]
        if pos.size:
            cmax = float(pos.max())
            if cmax > max_pos:
                max_pos = cmax
    max_follow_up = max_pos + 1.0

    out: dict = {"max_follow_up": max_follow_up}

    splits = [("train", train_eids), ("test", test_eids)]
    if val_eids is not None:
        splits.append(("val", val_eids))
    for split_name, eids in splits:
        eids = np.asarray(eids, dtype=np.int64)
        sub = df.reindex(index=eids)[phecodes]
        t = sub.to_numpy(dtype=np.float32)  # (N, P), may contain NaN

        is_event = np.zeros_like(t, dtype=np.float32)
        event_times = np.full_like(t, max_follow_up, dtype=np.float32)
        eval_mask = np.ones_like(t, dtype=bool)

        nan_mask = np.isnan(t)
        # 1. NaN: censored at max_follow_up (already set)

        # 2. t > 7-day: incident
        incident = (~nan_mask) & (t > seven_day_years)
        is_event[incident] = 1.0
        event_times[incident] = t[incident].astype(np.float32)

        # 3. 0 <= t <= 7-day: 7-day-zone, treat like effective prevalent
        #    (reverse-causality risk; disease likely present at recording but
        #    coded a few days later). Train includes as immediate event; test
        #    excludes from eval, mirroring the prevalent rule below.
        zone = (~nan_mask) & (t >= 0) & (t <= seven_day_years)
        # 4. t < 0: prevalent
        prevalent = (~nan_mask) & (t < 0)
        effective_prevalent = zone | prevalent

        if split_name in ("train", "val"):
            is_event[effective_prevalent] = 1.0
            event_times[effective_prevalent] = 0.0001
            # eval_mask stays True (we want the loss to use these as positives)
        else:  # test
            eval_mask[effective_prevalent] = False
            # is_event stays 0, event_times stays at max (effectively censored)

        out[split_name] = {
            "event_times": event_times,
            "is_event": is_event,
            "eval_mask": eval_mask,
            "eids": eids,
        }
    return out


def join_split_to_phecodes(
    df: pd.DataFrame,
    eids: list[int] | np.ndarray,
) -> np.ndarray:
    """Return the boolean mask `eid_in_phecode_csv` for `eids`."""
    eids = np.asarray(eids, dtype=np.int64)
    return np.isin(eids, df.index.to_numpy())


__all__ = [
    "build_survival_arrays",
    "filter_phecodes_by_prevalence",
    "join_split_to_phecodes",
    "load_covariates",
    "load_phecodes",
]
