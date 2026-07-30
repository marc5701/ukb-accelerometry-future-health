"""Per-disease baseline evaluation.

Loads a run directory produced by `run_baseline.py` and computes per-phecode
C-Index (lifelines, fast), 6-yr AUROC, 95% bootstrap CIs (1000 resamples),
asymptotic-bootstrap one-sided p-value (H0: C-Index <= 0.5), and Bonferroni
+ FDR-BH corrected p-values across all evaluated phecodes.

Outputs into the same run directory:
    per_disease.csv   one row per phecode
    headline_plot.png bootstrap CI bars for the first 5 (or user-specified) phecodes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines.utils import concordance_index as li_concordance_index
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

from ukb_disease.baseline.config import HORIZON_YEARS, N_BOOTSTRAP


def c_index_lifelines(hazards: np.ndarray, times: np.ndarray, events: np.ndarray) -> float:
    """lifelines.utils.concordance_index returns the C-Index for survival data.

    Note: lifelines expects HIGHER predicted scores to mean LONGER survival.
    Our hazards encode HIGHER = SHORTER survival (more risk). So pass `-hazards`.
    """
    if len(events) < 2 or np.sum(events) == 0:
        return float("nan")
    try:
        return float(li_concordance_index(times, -hazards, events))
    except ZeroDivisionError:
        return float("nan")


def auroc_at_horizon(hazards: np.ndarray, times: np.ndarray, events: np.ndarray,
                     horizon: float = HORIZON_YEARS) -> float:
    """Fixed-horizon AUROC.

    Positives: event before horizon (times < horizon & event).
    Negatives: everyone else who reached the horizon or had no event at all.
    Censored-before-horizon subjects are ambiguous, so excluded (standard).
    """
    events_b = events > 0
    pos = (times < horizon) & events_b
    censored_before = (times < horizon) & ~events_b
    keep = ~censored_before
    y = pos[keep].astype(int)
    s = hazards[keep]
    if len(y) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def bootstrap_scores(
    metric_fn,
    hazards: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
    n_boot: int,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(hazards)
    if n < 2:
        return np.array([])
    scores = np.full(n_boot, np.nan, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s = metric_fn(hazards[idx], times[idx], events[idx])
        scores[b] = s
    return scores[~np.isnan(scores)]


def ci_from_scores(scores: np.ndarray) -> tuple[float, float, float]:
    if len(scores) == 0:
        return (float("nan"), float("nan"), float("nan"))
    return (float(scores.mean()), float(np.percentile(scores, 2.5)),
            float(np.percentile(scores, 97.5)))


def evaluate_run(run_dir: Path, n_boot: int = N_BOOTSTRAP, min_eval: int = 10,
                 progress_every: int = 25) -> pd.DataFrame:
    import sys
    import time
    hazards = np.load(run_dir / "test_hazards.npy")
    times = np.load(run_dir / "test_event_times.npy")
    events = np.load(run_dir / "test_is_event.npy")
    mask = np.load(run_dir / "test_eval_mask.npy")
    # Granularity guard: if a producer saved per-recording rows (duplicate eids),
    # collapse to one row per subject before scoring so multi-recording subjects
    # are not over-counted in the C-index. Hazards become the subject mean across
    # recordings; labels/times/mask are identical per subject, so take the first row.
    eids_path = run_dir / "test_eids.npy"
    if eids_path.exists():
        _eids = np.load(eids_path)
        _uniq, _first = np.unique(_eids, return_index=True)
        if len(_uniq) != len(_eids):
            import warnings
            warnings.warn(f"[per_disease_eval] {run_dir.name}: collapsing {len(_eids)} "
                          f"recordings to {len(_uniq)} subjects (per-recording rows "
                          "detected); scoring per subject.")
            _inv = np.searchsorted(_uniq, _eids)
            _hs = np.zeros((len(_uniq), hazards.shape[1]), dtype=np.float64)
            _cnt = np.zeros(len(_uniq), dtype=np.float64)
            np.add.at(_hs, _inv, hazards.astype(np.float64))
            np.add.at(_cnt, _inv, 1.0)
            hazards = (_hs / _cnt[:, None]).astype(np.float32)
            times, events, mask = times[_first], events[_first], mask[_first]
    with open(run_dir / "config.json") as f:
        config = json.load(f)
    phecodes = config["phecodes"]
    print(f"[per_disease_eval] {run_dir.name}: starting eval on {len(phecodes)} phecodes "
          f"with n_boot={n_boot}", flush=True)
    t0 = time.time()

    rows: list[dict] = []
    for p_idx, p in enumerate(phecodes):
        if p_idx > 0 and p_idx % progress_every == 0:
            elapsed = time.time() - t0
            rate = p_idx / elapsed
            eta_sec = (len(phecodes) - p_idx) / rate if rate > 0 else 0
            print(f"[per_disease_eval] {run_dir.name}: {p_idx}/{len(phecodes)} phecodes  "
                  f"({rate*60:.1f}/min  eta {eta_sec/60:.1f}min)", flush=True)
        m = mask[:, p_idx]
        h = hazards[m, p_idx]
        t = times[m, p_idx]
        e = events[m, p_idx]
        n_eval = int(m.sum())
        n_pos = int(e.sum())
        row = {
            "phecode": p,
            "n_eval": n_eval,
            "n_pos": n_pos,
            "c_index": np.nan,
            "c_lo": np.nan, "c_hi": np.nan,
            "auroc_6yr": np.nan,
            "auroc_lo": np.nan, "auroc_hi": np.nan,
            "pvalue_raw": np.nan,
        }
        if n_pos > 0 and n_eval >= min_eval:
            row["c_index"] = c_index_lifelines(h, t, e)
            row["auroc_6yr"] = auroc_at_horizon(h, t, e)
            c_scores = bootstrap_scores(c_index_lifelines, h, t, e, n_boot)
            _, c_lo, c_hi = ci_from_scores(c_scores)
            row["c_lo"], row["c_hi"] = c_lo, c_hi
            # One-sided p-value: frac of bootstraps with C <= 0.5
            row["pvalue_raw"] = float((c_scores <= 0.5).mean()) if len(c_scores) else np.nan
            a_scores = bootstrap_scores(auroc_at_horizon, h, t, e, n_boot)
            _, a_lo, a_hi = ci_from_scores(a_scores)
            row["auroc_lo"], row["auroc_hi"] = a_lo, a_hi
        rows.append(row)

    df = pd.DataFrame(rows)
    valid = df["pvalue_raw"].notna()
    df["pvalue_bonferroni"] = np.nan
    df["pvalue_fdr_bh"] = np.nan
    if valid.sum() > 0:
        _, p_bonf, _, _ = multipletests(df.loc[valid, "pvalue_raw"], method="bonferroni")
        _, p_fdr, _, _ = multipletests(df.loc[valid, "pvalue_raw"], method="fdr_bh")
        df.loc[valid, "pvalue_bonferroni"] = p_bonf
        df.loc[valid, "pvalue_fdr_bh"] = p_fdr
    return df


def make_headline_plot(df: pd.DataFrame, headline_phecodes: list[str], out_path: Path,
                       title: str = "") -> None:
    pd_keep = df.set_index("phecode").reindex(headline_phecodes).reset_index()
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(pd_keep)), 4))
    x = np.arange(len(pd_keep))
    err_lo = pd_keep["c_index"] - pd_keep["c_lo"]
    err_hi = pd_keep["c_hi"] - pd_keep["c_index"]
    ax.errorbar(x, pd_keep["c_index"], yerr=[err_lo, err_hi], fmt="o", capsize=4)
    ax.axhline(0.5, ls="--", color="grey", alpha=0.6, label="random")
    ax.set_xticks(x)
    ax.set_xticklabels(pd_keep["phecode"], rotation=30, ha="right")
    ax.set_ylabel("Test C-Index (95% bootstrap CI)")
    ax.set_title(title or "Baseline, headline phecodes")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--headline-phecodes", nargs="+", default=None,
                    help="Specific phecodes for the headline plot (default: first 5 evaluated)")
    ap.add_argument("--min-eval", type=int, default=10,
                    help="Min eval-mask-True subjects to attempt a phecode")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    df = evaluate_run(run_dir, n_boot=args.n_boot, min_eval=args.min_eval)
    out_csv = run_dir / "per_disease.csv"
    df.to_csv(out_csv, index=False)
    print(f"[per_disease_eval] wrote {out_csv}: {len(df)} phecodes")

    summary = df[df["c_index"].notna()].copy()
    print(f"[per_disease_eval] {len(summary)} phecodes with valid metrics")
    if len(summary):
        print(summary[["phecode", "n_eval", "n_pos", "c_index", "c_lo", "c_hi",
                       "auroc_6yr", "pvalue_raw", "pvalue_fdr_bh"]]
              .sort_values("c_index", ascending=False).head(15).to_string(index=False))

    hp = args.headline_phecodes if args.headline_phecodes else summary["phecode"].head(5).tolist()
    if hp:
        out_png = run_dir / "headline_plot.png"
        make_headline_plot(df, hp, out_png,
                           title=f"Baseline ({run_dir.name})")
        print(f"[per_disease_eval] wrote {out_png}")


if __name__ == "__main__":
    main()
