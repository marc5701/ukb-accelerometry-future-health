"""Cell_correlation diagnostics: what is cleanly separable vs entangled.

Two axes:
  * cell correlation: across-subject correlation of the pooled cell blocks
    ({day,night}x{AR,HA}, or AR vs HA). High off-diagonal means the cells share
    variance, so a marginal attribution double-counts and must be read against
    the conditional one.
  * leakage = marginal dC minus conditional dC, per (cell, disease). Large positive
    leakage means the cell's apparent standalone value is mostly its correlation with
    the others (entangled); small leakage means a genuine unique contribution.

Per-axis verdict: clean if |r|<0.3 and leakage<25% of marginal; inseparable if
|r|>=0.6 or leakage>=50%; else partially-entangled.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ukb_disease.baseline.config import resolve_oak_path

AR_DIM = 768


def per_subject_cell_means(cache: str, split: str, cell_specs: dict):
    """Cross-subject pooled cell vectors to per-cell (n_subj, dim) means.

    cell_specs: {name: (tensor 'ar'/'ha', start, end)}.
    Returns (eids_unique, {name: (n_subj, dim)}).
    """
    d = resolve_oak_path(os.path.join(cache, split))
    ar = np.load(os.path.join(d, "ar.npy"), mmap_mode="r")
    ha = np.load(os.path.join(d, "ha.npy"), mmap_mode="r")
    vm = np.load(os.path.join(d, "valid_day_mask.npy"))
    idx = pd.read_parquet(os.path.join(d, "index.parquet"))
    eids = idx["eid"].to_numpy()
    uniq, inv = np.unique(eids, return_inverse=True)
    out = {}
    for name, (t, s, e) in cell_specs.items():
        arr = ar if t == "ar" else ha
        per_rec = np.zeros((arr.shape[0], e - s), dtype=np.float64)
        for i in range(arr.shape[0]):
            m = vm[i]
            if m.any():
                per_rec[i] = np.asarray(arr[i, m, s:e]).mean(axis=0)
        agg = np.zeros((len(uniq), e - s)); cnt = np.zeros(len(uniq))
        np.add.at(agg, inv, per_rec); np.add.at(cnt, inv, 1.0)
        out[name] = agg / cnt[:, None]
    return uniq, out


def cell_correlation_matrix(cell_means: dict) -> tuple[list, np.ndarray]:
    """Mean absolute cross-correlation between cells (averaged over the shared dims).

    For two cells X (n,D), Y (n,D): r = mean_d corr(X[:,d], Y[:,d]), a summary of
    how aligned the per-dimension subject variation is.
    """
    names = list(cell_means)
    K = len(names)
    R = np.eye(K)
    for a in range(K):
        for b in range(a + 1, K):
            X, Y = cell_means[names[a]], cell_means[names[b]]
            D = min(X.shape[1], Y.shape[1])
            rs = []
            for d in range(D):
                xa, yb = X[:, d], Y[:, d]
                if xa.std() > 1e-9 and yb.std() > 1e-9:
                    rs.append(np.corrcoef(xa, yb)[0, 1])
            R[a, b] = R[b, a] = float(np.nanmean(rs)) if rs else np.nan
    return names, R


def verdict(abs_r: float, leakage_frac: float) -> str:
    if abs_r >= 0.6 or leakage_frac >= 0.5:
        return "inseparable"
    if abs_r < 0.3 and leakage_frac < 0.25:
        return "clean"
    return "partially-entangled"


def leakage(marg_mean: np.ndarray, cond_mean: np.ndarray) -> np.ndarray:
    """marginal minus conditional dC (per cell, disease). Positive means a shared-variance leak."""
    return marg_mean - cond_mean
