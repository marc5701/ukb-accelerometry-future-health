"""Frozen-embedding logistic readout (the "probe").

A lightweight L2-penalized LogisticRegression trained on the 512-d wrist
embedding ONLY. The probe asserts the feature matrix is exactly 512-d so age
or sex can never leak in (WRIST-ONLY constraint).

Two estimators:
  (a) Repeated stratified k-fold CV (default 5-fold x 10 repeats), out-of-fold
      predictions -> AUROC + AUPRC; bootstrap 95% CI on the OOF scores.
  (b) Single stratified 80/20 split -> AUROC + AUPRC.

Controls are subject-disjoint by construction (each subject is one row, in one
cohort), so a stratified split over rows never leaks a subject across the
train/test of a fold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedShuffleSplit,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EMB_DIM = 512


def _make_probe(C: float = 1.0, seed: int = 42):
    """Standardize then L2-logistic (class_weight balanced for imbalance)."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="l2", C=C, max_iter=2000,
            class_weight="balanced", random_state=seed,
        ),
    )


def _assert_wrist_only(X: np.ndarray) -> None:
    if X.ndim != 2 or X.shape[1] != EMB_DIM:
        raise AssertionError(
            f"probe feature matrix must be {EMB_DIM}-d wrist embedding only, "
            f"got shape {X.shape}. age/sex must NEVER enter the probe.")


def _bootstrap_ci(y: np.ndarray, score: np.ndarray, n: int = 1000,
                  seed: int = 42):
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    aps, aucs = [], []
    for _ in range(n):
        bi = np.concatenate([rng.choice(pos, len(pos), True),
                             rng.choice(neg, len(neg), True)])
        yi, si = y[bi], score[bi]
        aps.append(average_precision_score(yi, si))
        aucs.append(roc_auc_score(yi, si))
    pc = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return {
        "auprc_mean": float(np.mean(aps)), "auprc_ci": pc(aps),
        "auroc_mean": float(np.mean(aucs)), "auroc_ci": pc(aucs),
    }


@dataclass
class ProbeResult:
    estimator: str
    n: int
    n_pos: int
    auroc: float
    auprc: float
    auroc_ci: tuple[float, float]
    auprc_ci: tuple[float, float]
    extra: dict


def cv_probe(X: np.ndarray, y: np.ndarray, *, n_splits: int = 5,
             n_repeats: int = 10, C: float = 1.0, seed: int = 42,
             n_boot: int = 1000) -> ProbeResult:
    """Repeated stratified k-fold CV with out-of-fold predictions.

    Each repeat produces a full OOF score vector; we average OOF scores across
    repeats per subject (each subject is test exactly once per repeat) then
    score once + bootstrap a CI. Also reports the across-repeat AUROC/AUPRC
    spread as a sanity check.
    """
    _assert_wrist_only(X)
    y = np.asarray(y)
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    oof_sum = np.zeros(len(y))
    oof_cnt = np.zeros(len(y))
    per_repeat_auroc, per_repeat_auprc = [], []
    cur = np.full(len(y), np.nan)
    fold_in_repeat = 0
    for tr, te in rskf.split(X, y):
        clf = _make_probe(C=C, seed=seed)
        clf.fit(X[tr], y[tr])
        s = clf.predict_proba(X[te])[:, 1]
        oof_sum[te] += s
        oof_cnt[te] += 1
        cur[te] = s
        fold_in_repeat += 1
        if fold_in_repeat == n_splits:
            per_repeat_auroc.append(roc_auc_score(y, cur))
            per_repeat_auprc.append(average_precision_score(y, cur))
            cur = np.full(len(y), np.nan)
            fold_in_repeat = 0

    oof = oof_sum / np.maximum(oof_cnt, 1)
    auroc = float(roc_auc_score(y, oof))
    auprc = float(average_precision_score(y, oof))
    ci = _bootstrap_ci(y, oof, n=n_boot, seed=seed)
    return ProbeResult(
        estimator=f"cv_{n_splits}x{n_repeats}",
        n=int(len(y)), n_pos=int(y.sum()),
        auroc=auroc, auprc=auprc,
        auroc_ci=ci["auroc_ci"], auprc_ci=ci["auprc_ci"],
        extra={
            "oof_scores": oof,
            "per_repeat_auroc_mean": float(np.mean(per_repeat_auroc)),
            "per_repeat_auroc_std": float(np.std(per_repeat_auroc)),
            "per_repeat_auprc_mean": float(np.mean(per_repeat_auprc)),
            "per_repeat_auprc_std": float(np.std(per_repeat_auprc)),
        },
    )


def single_split_probe(X: np.ndarray, y: np.ndarray, *, test_size: float = 0.2,
                       C: float = 1.0, seed: int = 42, n_boot: int = 1000
                       ) -> ProbeResult:
    """Single stratified 80/20 split -> AUROC + AUPRC with bootstrap CI."""
    _assert_wrist_only(X)
    y = np.asarray(y)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                 random_state=seed)
    (tr, te), = sss.split(X, y)
    clf = _make_probe(C=C, seed=seed)
    clf.fit(X[tr], y[tr])
    s = clf.predict_proba(X[te])[:, 1]
    yte = y[te]
    auroc = float(roc_auc_score(yte, s))
    auprc = float(average_precision_score(yte, s))
    ci = _bootstrap_ci(yte, s, n=n_boot, seed=seed)
    return ProbeResult(
        estimator="single_80_20",
        n=int(len(yte)), n_pos=int(yte.sum()),
        auroc=auroc, auprc=auprc,
        auroc_ci=ci["auroc_ci"], auprc_ci=ci["auprc_ci"],
        extra={"test_scores": s, "test_y": yte},
    )
