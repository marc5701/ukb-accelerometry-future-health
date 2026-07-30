"""`screening_prevalent_v1` split construction (per-disease, independent classifiers).

Experimental split for the cross-entropy *current-disease* screener. The Cox
pipeline excludes prevalent cases at test; here they become the screening
positives. Because prevalent cases are rare (cohort totals PD 132 / AD 16 /
dementia 21) they cannot populate a held-out test AND leave enough to train on,
so we concentrate ALL prevalent cases into test and TRAIN on incident cases.

Per disease `d` (status from the phecode cell `t` = years-since-wear):
    prevalent (current): notna & t <= SEVEN_DAY_YEARS   (t<0 or the 0-7d zone)
    incident  (future):  notna & t >  SEVEN_DAY_YEARS
    control:             isna

    test_pos  = prevalent                          (pulled from ANY hash bucket)
    test_ctrl = control  & bucket==test            (shared test-hash control pool)
    test_excl = incident & bucket==test            (masked; ambiguous, not scored)
    train_pos = incident & bucket==train           (near-term variant: & t<=NEARTERM)
    train_ctrl= control  & bucket==train
    val_pos   = incident & bucket==val             (hyper-parameter selection only)
    val_ctrl  = control  & bucket==val

Disjointness: prevalent cases never enter `d`'s train (they're test positives).
A subject may be `d`-test and `d'`-train, which is harmless because classifiers
are independent per disease. PPV/NPV use the *population* prevalence
|test_pos| / N_total, never the enriched test prevalence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ukb_disease.baseline._split_hash import hash_split
from ukb_disease.baseline.config import SEVEN_DAY_YEARS


def assign_buckets(eids: np.ndarray) -> np.ndarray:
    """Deterministic md5 hash bucket ('train'/'val'/'test') per subject eid."""
    return np.array([hash_split(int(e)) for e in eids], dtype=object)


def disease_masks(
    t: np.ndarray,
    bucket: np.ndarray,
    nearterm_years: float | None = None,
    seven: float = SEVEN_DAY_YEARS,
) -> dict[str, np.ndarray]:
    """Boolean masks (over the subject axis) for one disease under the split rules.

    `t` is the per-subject phecode time (NaN/neg/pos), `bucket` the hash bucket.
    `nearterm_years` restricts train positives to incident cases within that
    window (the sensitivity variant); None = all future onset (primary).
    """
    notna = ~np.isnan(t)
    prevalent = notna & (t <= seven)
    incident = notna & (t > seven)
    control = ~notna

    is_train = bucket == "train"
    is_val = bucket == "val"
    is_test = bucket == "test"

    train_pos = incident & is_train
    if nearterm_years is not None:
        train_pos = train_pos & (t <= nearterm_years)

    return {
        "test_pos": prevalent,
        "test_ctrl": control & is_test,
        "test_excl": incident & is_test,
        "train_pos": train_pos,
        "train_ctrl": control & is_train,
        "val_pos": incident & is_val,
        "val_ctrl": control & is_val,
    }


def disease_masks_same_dist_cv(
    t: np.ndarray,
    cv_assignment: np.ndarray,
    cv_fold: int,
    seven: float = SEVEN_DAY_YEARS,
) -> dict[str, np.ndarray]:
    """Per-fold geometry for the SAME-DISTRIBUTION CV screener (train+test on prevalent).

    Unlike the transfer split, incident cases are dropped entirely; the universe is
    prevalent positives + controls, partitioned into folds by ``cv_assignment`` (an
    md5(eid)%K vector, see ``baseline.make_cv_folds.fold_of_eid``). For the held-out
    fold ``cv_fold`` we return the test masks; everything else is the training pool.
    The caller carves a validation slice from the train pool (for C selection) and
    subsamples train controls for fitting. A subject's out-of-fold prediction therefore
    always comes from a model that never saw it (no leakage).
    """
    notna = ~np.isnan(t)
    prevalent = notna & (t <= seven)
    control = ~notna
    in_fold = cv_assignment == cv_fold
    return {
        "test_pos": prevalent & in_fold,
        "test_ctrl": control & in_fold,
        "trainpool_pos": prevalent & ~in_fold,
        "trainpool_ctrl": control & ~in_fold,
    }


def audit_panel(
    phe: pd.DataFrame,
    bucket: np.ndarray,
    min_test_pos: int = 10,
    min_train_pos: int = 50,
    nearterm_years: float | None = None,
) -> pd.DataFrame:
    """Per-disease case counts + viability flag for every phecode column.

    `phe` is the phecode table already reindexed to the embedding subject
    universe (one row per subject, columns = phecodes incl. time_to_death).
    `bucket` is aligned to `phe.index`.
    """
    n_total = len(phe)
    rows = []
    for col in phe.columns:
        t = phe[col].to_numpy(dtype=np.float64)
        m = disease_masks(t, bucket, nearterm_years=nearterm_years)
        n_test_pos = int(m["test_pos"].sum())
        n_train_pos = int(m["train_pos"].sum())
        rows.append(
            {
                "phecode": col,
                "n_test_pos": n_test_pos,
                "n_test_ctrl": int(m["test_ctrl"].sum()),
                "n_test_excl": int(m["test_excl"].sum()),
                "n_train_pos": n_train_pos,
                "n_train_ctrl": int(m["train_ctrl"].sum()),
                "n_val_pos": int(m["val_pos"].sum()),
                "pop_prevalence": n_test_pos / n_total,
                "viable": (n_test_pos >= min_test_pos) and (n_train_pos >= min_train_pos),
            }
        )
    return pd.DataFrame(rows).sort_values("n_test_pos", ascending=False).reset_index(drop=True)


def viable_phecodes(audit: pd.DataFrame) -> list[str]:
    return audit.loc[audit["viable"], "phecode"].tolist()
