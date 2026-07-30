"""Uncertainty and confounding statistics for the BMI analysis.

- Pearson r with subject-bootstrap CI (Fisher-z percentile).
- Paired Δr bootstrap (shared resample cancels shared subject variance), used to
  compare two predictors.
- Four confounding controls: r_demo (true-demographic proxy floor),
  r_proxy_ceiling (decoded-demographic proxy), r_partial (age/sex-residualized
  adjusted r), r_within (within sex x age-decile cells).
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# correlation + bootstrap
# ---------------------------------------------------------------------------
def pearson(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def _fisher_ci(boot, lo=2.5, hi=97.5):
    z = np.arctanh(np.clip(boot, -0.999999, 0.999999))
    return float(np.tanh(np.percentile(z, lo))), float(np.tanh(np.percentile(z, hi)))


def boot_r_ci(yhat, y, n_boot=2000, seed=0) -> dict:
    yhat = np.asarray(yhat, float)
    y = np.asarray(y, float)
    ok = np.isfinite(yhat) & np.isfinite(y)
    yhat, y = yhat[ok], y[ok]
    n = len(y)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b] = pearson(yhat[idx], y[idx])
    lo, hi = _fisher_ci(boot)
    return {"point": pearson(yhat, y), "lo": lo, "hi": hi,
            "se": float(np.std(boot)), "n": int(n)}


def paired_dr(yhat_a, yhat_b, y, n_boot=2000, seed=0) -> dict:
    """Δr = r(B,y) - r(A,y) with a shared bootstrap draw (paired)."""
    a = np.asarray(yhat_a, float)
    b = np.asarray(yhat_b, float)
    y = np.asarray(y, float)
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(y)
    a, b, y = a[ok], b[ok], y[ok]
    n = len(y)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[k] = pearson(b[idx], y[idx]) - pearson(a[idx], y[idx])
    point = pearson(b, y) - pearson(a, y)
    p = 2.0 * min(float((boot <= 0).mean()), float((boot >= 0).mean()))
    return {"point": float(point), "lo": float(np.percentile(boot, 2.5)),
            "hi": float(np.percentile(boot, 97.5)), "p": p}


# ---------------------------------------------------------------------------
# OLS helpers + demographic design
# ---------------------------------------------------------------------------
def demo_design(age, sex, age_mean, age_std):
    a = (np.asarray(age, float) - age_mean) / (age_std + 1e-9)
    s = np.asarray(sex, float)
    return np.column_stack([np.ones_like(a), a, s, a * a, a * s])


def _ols_fit(D, y):
    beta, *_ = np.linalg.lstsq(D, y, rcond=None)
    return beta


def _ols_resid(y, D):
    return y - D @ _ols_fit(D, y)


# ---------------------------------------------------------------------------
# confounding controls
# ---------------------------------------------------------------------------
def r_demo(bmi_tr, age_tr, sex_tr, bmi_ho, age_ho, sex_ho) -> float:
    """True-demographic proxy: OLS BMI~poly(age,sex) fit on TRAIN, scored held-out."""
    am, asd = float(np.mean(age_tr)), float(np.std(age_tr))
    beta = _ols_fit(demo_design(age_tr, sex_tr, am, asd), bmi_tr)
    pred = demo_design(age_ho, sex_ho, am, asd) @ beta
    return pearson(pred, bmi_ho)


def r_proxy_ceiling(emb_tr, bmi_tr, age_tr, sex_tr, emb_ho, bmi_ho,
                    ridge_alpha=10.0) -> float:
    """Decoded-demographic proxy ceiling: decode age,sex from embedding (TRAIN),
    map decoded demographics -> BMI (TRAIN OLS), score held-out."""
    from sklearn.linear_model import Ridge, LogisticRegression

    age_dec = Ridge(alpha=ridge_alpha).fit(emb_tr, age_tr)
    sex_dec = LogisticRegression(max_iter=2000, C=1.0).fit(emb_tr, sex_tr.astype(int))
    a_tr = age_dec.predict(emb_tr)
    s_tr = sex_dec.predict_proba(emb_tr)[:, 1]
    a_ho = age_dec.predict(emb_ho)
    s_ho = sex_dec.predict_proba(emb_ho)[:, 1]
    am, asd = float(np.mean(a_tr)), float(np.std(a_tr))
    beta = _ols_fit(demo_design(a_tr, s_tr, am, asd), bmi_tr)
    pred = demo_design(a_ho, s_ho, am, asd) @ beta
    return pearson(pred, bmi_ho)


def r_partial(yhat, bmi, age, sex, n_boot=2000, seed=0) -> dict:
    """Partial Pearson r of (yhat, bmi) controlling age+sex (adjusted r).

    Nuisance regressed out of BOTH series on the held-out set (textbook partial
    correlation). Bootstrap CI by resampling subjects and re-residualizing.
    """
    am, asd = float(np.mean(age)), float(np.std(age))
    D = demo_design(age, sex, am, asd)
    point = pearson(_ols_resid(yhat, D), _ols_resid(bmi, D))
    n = len(bmi)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        Db = demo_design(age[idx], sex[idx], am, asd)
        boot[b] = pearson(_ols_resid(yhat[idx], Db), _ols_resid(bmi[idx], Db))
    lo, hi = _fisher_ci(boot)
    return {"point": point, "lo": lo, "hi": hi}


def r_within(yhat, bmi, age, sex, n_deciles=10, min_cell=20,
             n_boot=1000, seed=0) -> dict:
    """Sample-weighted Fisher-z mean of per-cell r over sex x age-decile cells."""
    yhat = np.asarray(yhat, float)
    bmi = np.asarray(bmi, float)
    age = np.asarray(age, float)
    sex = np.asarray(sex, float)
    edges = np.quantile(age, np.linspace(0, 1, n_deciles + 1))
    edges[-1] += 1e-6
    abin = np.clip(np.digitize(age, edges[1:-1]), 0, n_deciles - 1)

    def _pooled(yh, bm, ab, sx):
        zsum, wsum, cells = 0.0, 0.0, 0
        for sval in (0, 1):
            for d in range(n_deciles):
                m = (sx == sval) & (ab == d)
                if m.sum() >= min_cell:
                    rc = pearson(yh[m], bm[m])
                    if np.isfinite(rc):
                        w = m.sum() - 1
                        zsum += w * np.arctanh(np.clip(rc, -0.999999, 0.999999))
                        wsum += w
                        cells += 1
        return (np.tanh(zsum / wsum) if wsum > 0 else np.nan), cells

    point, n_cells = _pooled(yhat, bmi, abin, sex)
    n = len(bmi)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b] = _pooled(yhat[idx], bmi[idx], abin[idx], sex[idx])[0]
    lo, hi = _fisher_ci(boot[np.isfinite(boot)]) if np.isfinite(boot).any() else (np.nan, np.nan)
    return {"point": float(point), "lo": lo, "hi": hi, "n_cells": int(n_cells)}
