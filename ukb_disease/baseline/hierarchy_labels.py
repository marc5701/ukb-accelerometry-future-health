"""Phecode-hierarchy auxiliary survival columns.

Builds higher-prevalence aggregate phecode columns (category roll-ups and
integer parent codes) from the full phecode table. These are added as extra
survival heads trained jointly with the 289 leaf phecodes and dropped at
inference. They inject dense, low-variance gradient into the shared transformer
trunk every batch. The 13 category aggregates fire roughly 15 to 21 events per
batch at batch=64, versus about 0.3 for a median leaf, which keeps the in-batch
Cox / discrete loss from starving while retaining the multilabel regularizer
over all leaves.

Phecode column names are `<CAT>_<num>` (e.g. `GI_522.12`, `CV_404.2`). An
aggregate is positive for a subject if any of its member columns is, with event
time equal to the per-subject nanmin over members' raw time values. A prevalent
member (t<0) makes the aggregate prevalent, otherwise the earliest incident time
is used, and an all-NaN row is censored. Taking the raw-time nanmin lets the
downstream `build_survival_arrays` apply the same incident / 7-day-zone /
prevalent / censored rules uniformly to leaves and aggregates.
"""

from __future__ import annotations

import re
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd


def _safe_nanmin(block: np.ndarray, axis: int) -> np.ndarray:
    """np.nanmin over `axis`, returning NaN (not warning) for all-NaN slices.

    All-NaN rows are expected and correct here (a subject with no event in any
    member column → the aggregate is censored). Suppress the bounded, cosmetic
    `All-NaN slice encountered` RuntimeWarning so it does not clutter logs.
    """
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmin(block, axis=axis)

_NAME_RE = re.compile(r"^(?P<cat>[A-Za-z]+)_(?P<num>\d+(?:\.\d+)?)$")

AUX_CAT_PREFIX = "AUXCAT_"   # category roll-up,   e.g. AUXCAT_GI
AUX_PAR_PREFIX = "AUXPAR_"   # integer-parent code, e.g. AUXPAR_GI_522


def parse_phecode(name: str) -> tuple[str, int] | None:
    """Return (category, integer_part) for a `<CAT>_<num>` column, else None."""
    m = _NAME_RE.match(name)
    if not m:
        return None
    cat = m.group("cat")
    try:
        integer_part = int(float(m.group("num")))
    except ValueError:
        return None
    return cat, integer_part


def _any_positive_prevalence(
    df: pd.DataFrame,
    cols: list[str],
    train_eids: np.ndarray,
    seven_day_years: float,
) -> float:
    """any-positive (incident + prevalent) fraction of an aggregate over train.

    Mirrors filter_phecodes_by_prevalence's any_positive definition, applied to
    the per-subject nanmin across `cols`.
    """
    sub = df.reindex(index=train_eids)[cols].to_numpy(dtype=np.float64)  # (N, k)
    agg = _safe_nanmin(sub, axis=1)  # (N,) NaN where all-NaN
    finite = np.isfinite(agg)
    n_incident = int((finite & (agg > seven_day_years)).sum())
    n_prev = int((finite & (agg < 0)).sum())
    return (n_incident + n_prev) / max(len(train_eids), 1)


def _aggregate_time(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Per-subject nanmin over member columns (NaN where all members NaN)."""
    block = df[cols].to_numpy(dtype=np.float64)
    agg = _safe_nanmin(block, axis=1)
    return pd.Series(agg.astype(np.float32), index=df.index)


def build_aux_columns(
    df_pheco_full: pd.DataFrame,
    train_eids: np.ndarray,
    min_prevalence: float = 0.01,
    categories_only: bool = False,
    seven_day_years: float = 7.0 / 365.25,
) -> tuple[list[str], pd.DataFrame]:
    """Construct hierarchy aggregate columns.

    Args:
        df_pheco_full: full phecode table (eid-indexed; includes `time_to_death`).
        train_eids:    train eids for the prevalence screen.
        min_prevalence: keep an integer-parent aggregate only if its any-positive
            train prevalence >= this. Category roll-ups are always kept.
        categories_only: if True, build only the 13 category roll-ups (the
            densest, highest-event subset).

    Returns:
        (aux_names, df_aux) where df_aux is eid-indexed with one column per
        aggregate, named with AUXCAT_/AUXPAR_ prefixes so they never collide with
        leaf phecode names.
    """
    train_eids = np.asarray(train_eids, dtype=np.int64)
    cols = [c for c in df_pheco_full.columns if c != "time_to_death"]

    by_cat: dict[str, list[str]] = defaultdict(list)
    by_parent: dict[tuple[str, int], list[str]] = defaultdict(list)
    for c in cols:
        parsed = parse_phecode(c)
        if parsed is None:
            continue
        cat, integer_part = parsed
        by_cat[cat].append(c)
        by_parent[(cat, integer_part)].append(c)

    aux_data: dict[str, pd.Series] = {}

    # Category roll-ups (always kept; densest gradient).
    for cat, members in sorted(by_cat.items()):
        if len(members) < 2:
            continue
        aux_data[f"{AUX_CAT_PREFIX}{cat}"] = _aggregate_time(df_pheco_full, members)

    # Integer-parent aggregates (kept only if >=2 children AND prevalent enough).
    if not categories_only:
        for (cat, integer_part), members in sorted(by_parent.items()):
            if len(members) < 2:
                continue  # a singleton parent == its one leaf; no aggregation value
            prev = _any_positive_prevalence(df_pheco_full, members, train_eids, seven_day_years)
            if prev < min_prevalence:
                continue
            aux_data[f"{AUX_PAR_PREFIX}{cat}_{integer_part}"] = _aggregate_time(
                df_pheco_full, members
            )

    if not aux_data:
        return [], pd.DataFrame(index=df_pheco_full.index)
    df_aux = pd.DataFrame(aux_data, index=df_pheco_full.index)
    aux_names = list(df_aux.columns)
    return aux_names, df_aux


__all__ = ["parse_phecode", "build_aux_columns", "AUX_CAT_PREFIX", "AUX_PAR_PREFIX"]
