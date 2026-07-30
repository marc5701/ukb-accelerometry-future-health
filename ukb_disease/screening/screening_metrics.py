"""Screening metric panel: AUROC, AUPRC, sensitivity@95%-specificity, PPV/NPV
at the *true population prevalence*, with bootstrap CIs.

Why population-prevalence PPV/NPV: the screening test set is ENRICHED (all
prevalent cases are pulled in), so the empirical test prevalence is far above
the real cohort prevalence. AUROC/AUPRC are computed directly on the (enriched)
scores, but PPV/NPV are Bayes-adjusted from sensitivity + specificity to the
population prevalence the screener would actually face.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

SPEC_TARGET = 0.95


def _sens_at_spec(y: np.ndarray, s: np.ndarray, spec: float) -> tuple[float, float]:
    """Sensitivity at a target specificity; returns (sensitivity, threshold).

    Threshold = the `spec` quantile of the NEGATIVE scores (so specificity ~= spec);
    sensitivity = fraction of positives scoring >= that threshold.
    """
    neg = s[y == 0]
    pos = s[y == 1]
    if len(neg) == 0 or len(pos) == 0:
        return float("nan"), float("nan")
    thr = float(np.quantile(neg, spec))
    sens = float((pos >= thr).mean())
    return sens, thr


def _ppv_npv(sens: float, spec: float, prevalence: float) -> tuple[float, float]:
    """Bayes PPV/NPV from sens/spec at a given population prevalence."""
    if not np.isfinite(sens) or not np.isfinite(spec):
        return float("nan"), float("nan")
    tp = sens * prevalence
    fp = (1.0 - spec) * (1.0 - prevalence)
    fn = (1.0 - sens) * prevalence
    tn = spec * (1.0 - prevalence)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    return ppv, npv


def point_metrics(y: np.ndarray, s: np.ndarray, prevalence: float,
                  spec_target: float = SPEC_TARGET) -> dict[str, float]:
    """All point estimates for one (y, score) pair. NaN if degenerate."""
    out = {"auroc": np.nan, "auprc": np.nan, "sens_at_spec": np.nan,
           "ppv": np.nan, "npv": np.nan}
    if len(y) < 2 or len(np.unique(y)) < 2:
        return out
    out["auroc"] = float(roc_auc_score(y, s))
    out["auprc"] = float(average_precision_score(y, s))
    sens, _ = _sens_at_spec(y, s, spec_target)
    out["sens_at_spec"] = sens
    ppv, npv = _ppv_npv(sens, spec_target, prevalence)
    out["ppv"], out["npv"] = ppv, npv
    return out


def metrics_with_ci(y: np.ndarray, s: np.ndarray, prevalence: float,
                    n_boot: int = 1000, seed: int = 42,
                    spec_target: float = SPEC_TARGET) -> dict[str, float]:
    """Point metrics + percentile bootstrap CIs (resample subjects with replacement)."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    pt = point_metrics(y, s, prevalence, spec_target)
    keys = ["auroc", "auprc", "sens_at_spec", "ppv", "npv"]
    boot = {k: np.full(n_boot, np.nan) for k in keys}
    n = len(y)
    if n >= 2 and len(np.unique(y)) >= 2:
        rng = np.random.default_rng(seed)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            yb, sb = y[idx], s[idx]
            if len(np.unique(yb)) < 2:
                continue
            mb = point_metrics(yb, sb, prevalence, spec_target)
            for k in keys:
                boot[k][b] = mb[k]
    res = dict(pt)
    for k in keys:
        vals = boot[k][~np.isnan(boot[k])]
        if len(vals):
            res[f"{k}_lo"] = float(np.percentile(vals, 2.5))
            res[f"{k}_hi"] = float(np.percentile(vals, 97.5))
        else:
            res[f"{k}_lo"] = np.nan
            res[f"{k}_hi"] = np.nan
    return res
