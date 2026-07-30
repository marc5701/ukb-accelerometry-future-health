"""PRS non-linearity check (supplementary sensitivity analysis).

The headline model enters each matched PRS as a single standardised linear term.
PRS often has a flat middle and diverges only in the tails, so a linear term can
miss real signal. This tests whether a non-linear encoding captures anything the
linear term does not, without changing the frozen headline model.

Three tests per matched disease (EUR), in both contexts (standalone PRS to disease
and the actigraphy + PRS fusion):
  (1) LR test   : nested Cox, PRS-linear vs PRS-RCS (4 knots), in-sample
                  partial-likelihood ratio; chi2 with df = (rcs_df - 1).
  (2) OOF dC    : out-of-fold C-index, RCS vs linear, standalone [prs] and
                  fusion [hazard, prs]; paired subject bootstrap on the frozen OOF
                  linear predictors.
  (3) Decile    : per-decile univariate hazard ratio (ref = middle decile),
                  which reveals a flat-middle / steep-tail shape if present.
Plus extreme-tail indicators (top/bottom 5% and 1%) as single-df HRs.

Data loading mirrors the per-disease CI analysis exactly (same matched crosswalk,
same EUR subset, same per-disease z-scoring), so results are directly comparable
to the linear fusion result.

Run:
  python3 -m ukb_disease.genetics.prs_nonlinear --out prs_nonlinear
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
from scipy.stats import chi2 as _chi2

from ukb_disease.benchmark.concordance_benchmark import DEFAULT_RUNS, load_model
from ukb_disease.baseline.cox_crossfit_features import load_age_covars
from ukb_disease.baseline.config import resolve_oak_path
from ukb_disease.genetics.cox_crossfit_prs import build_phecode_to_trait, TRACK_PREFIX, FEATURES
from ukb_disease.genetics.perdisease_ci import oof_lp, c_from, boot_ci

# Harrell knot quantiles for k=4 (fixed placement; robust, no data-snooping).
KNOT_Q4 = np.array([0.05, 0.35, 0.65, 0.95])
LR_PEN = 1e-3  # tiny ridge, applied identically to both nested models (approx-valid LR)


def rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Restricted (natural) cubic spline basis, Harrell parameterisation.

    Returns (n, k-1): column 0 is linear x; columns 1..k-2 are the non-linear
    terms. Knots are given in the units of x.
    """
    x = np.asarray(x, float)
    t = np.asarray(knots, float)
    k = len(t)

    def cube(z):
        z = np.where(z > 0, z, 0.0)
        return z ** 3

    denom = (t[-1] - t[0]) ** 2
    cols = [x]
    for j in range(k - 2):
        term = (cube(x - t[j])
                - cube(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
                + cube(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2]))
        cols.append(term / denom)
    return np.column_stack(cols)


def _fit_ll(X, T, E, pen=LR_PEN):
    """Fit a lifelines Cox on design X; return (partial log-likelihood, ok)."""
    cols = [f"x{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df["T"] = T
    df["E"] = E
    try:
        cph = CoxPHFitter(penalizer=pen, l1_ratio=0.0).fit(df, "T", "E")
        return float(cph.log_likelihood_), True
    except (ConvergenceError, ValueError, np.linalg.LinAlgError):
        return np.nan, False


def lr_test(base, prs, T, E, knots):
    """LR test of PRS non-linearity: [base, prs] vs [base, rcs(prs)].

    base: (n, b) array of fixed covariates held identical across the two nested
    models (empty for the standalone test, [hazard] for the fusion-adjusted test).
    Returns dict(chi2, df, p, ok).
    """
    R = rcs_basis(prs, knots)
    df_nl = R.shape[1] - 1  # non-linear degrees of freedom beyond the linear term
    Xlin = np.column_stack([base, prs]) if base.size else prs.reshape(-1, 1)
    Xrcs = np.column_stack([base, R]) if base.size else R
    ll0, ok0 = _fit_ll(Xlin, T, E)
    ll1, ok1 = _fit_ll(Xrcs, T, E)
    if not (ok0 and ok1) or not np.isfinite(ll0) or not np.isfinite(ll1):
        return {"chi2": np.nan, "df": df_nl, "p": np.nan, "ok": False}
    stat = max(0.0, 2.0 * (ll1 - ll0))
    return {"chi2": stat, "df": df_nl, "p": float(_chi2.sf(stat, df_nl)), "ok": True}


def _single_hr(ind, T, E, adjust=None):
    """Univariate (optionally hazard-adjusted) log-HR + se for a 0/1 indicator."""
    if ind.sum() < 3 or (~ind.astype(bool)).sum() < 3:
        return np.nan, np.nan
    X = ind.reshape(-1, 1).astype(float)
    if adjust is not None:
        X = np.column_stack([X, adjust])
    cols = [f"x{i}" for i in range(X.shape[1])]
    d = pd.DataFrame(X, columns=cols); d["T"] = T; d["E"] = E
    try:
        cph = CoxPHFitter(penalizer=LR_PEN).fit(d, "T", "E")
        return float(cph.params_["x0"]), float(cph.standard_errors_["x0"])
    except (ConvergenceError, ValueError, np.linalg.LinAlgError):
        return np.nan, np.nan


def decile_curve(prs, T, E, ref=4):
    """Per-decile univariate log-HR vs the reference decile (default middle-ish).

    Returns list of dict(decile, n, n_ev, logHR, se) with the ref decile HR=0.
    """
    edges = np.quantile(prs, np.linspace(0, 1, 11))
    edges[0] -= 1e-9
    dec = np.clip(np.digitize(prs, edges[1:-1]), 0, 9)
    rows = []
    ref_mask = dec == ref
    for d in range(10):
        m = dec == d
        n, n_ev = int(m.sum()), int(E[m].sum())
        if d == ref:
            rows.append({"decile": d, "n": n, "n_ev": n_ev, "logHR": 0.0, "se": 0.0})
            continue
        sub = m | ref_mask
        ind = m[sub].astype(float)
        lo, se = _single_hr(ind, T[sub], E[sub])
        rows.append({"decile": d, "n": n, "n_ev": n_ev, "logHR": lo, "se": se})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="std")
    ap.add_argument("--subset", default="eur_genetic")
    ap.add_argument("--deep-prefix", default="embedding_disjoint_split")
    ap.add_argument("--runs-dir", default=DEFAULT_RUNS)
    ap.add_argument("--age-dir", default="age_sex_disjoint_split/sequence_model_age")
    ap.add_argument("--features", default=FEATURES)
    ap.add_argument("--min-events", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out_dir = resolve_oak_path(args.out); os.makedirs(out_dir, exist_ok=True)

    runs_dir = resolve_oak_path(args.runs_dir)
    mean_h, _bs, base, seeds = load_model(runs_dir, args.deep_prefix)
    T, E, M, eids = base["T"], base["E"], base["M"], base["eids"]
    N, P = T.shape
    names = json.load(open(f"{runs_dir}/{args.deep_prefix}_seed{seeds[0]}/config.json"))["phecodes"]
    age_dir = args.age_dir
    if not os.path.isabs(age_dir):
        age_dir = os.path.join(os.path.dirname(runs_dir), age_dir)
    _ac = load_age_covars(age_dir, eids)  # loaded for parity/QC; age/sex not needed for the PRS shape

    feat = pd.read_parquet(args.features).set_index("eid").reindex(index=eids)
    prefix = TRACK_PREFIX[args.track]
    keep_subj = np.ones(N, bool) if args.subset == "all" else \
        (feat[args.subset].fillna(0).to_numpy() == 1)
    p2t = build_phecode_to_trait(names)

    rows, dec_rows = [], []
    for j in range(P):
        ph = names[j]; trait = p2t.get(ph)
        col = f"{prefix}{trait}" if trait else None
        if col not in feat.columns:
            continue
        m = M[:, j] & keep_subj
        if int(m.sum()) < args.min_events or int(E[m, j].sum()) < args.min_events:
            continue
        Tj = T[m, j].astype(float); Ej = E[m, j].astype(float)
        prs = pd.to_numeric(feat[col], errors="coerce").to_numpy()[m]
        prs = np.nan_to_num((prs - np.nanmean(prs)) / (np.nanstd(prs) or 1.0))
        h = mean_h[m, j]
        knots = np.quantile(prs, KNOT_Q4)
        if len(np.unique(knots)) < len(knots):
            continue  # degenerate (ties at knots), skip RCS for this disease

        # (1) LR tests: standalone (base empty) and fusion-adjusted (base=[hazard])
        lr_s = lr_test(np.empty((len(prs), 0)), prs, Tj, Ej, knots)
        lr_f = lr_test(h.reshape(-1, 1), prs, Tj, Ej, knots)

        # (2) OOF dC: RCS vs linear, standalone and fusion (paired subject bootstrap)
        R = rcs_basis(prs, knots)
        lp_lin_s = oof_lp(prs.reshape(-1, 1), Tj, Ej)
        lp_rcs_s = oof_lp(R, Tj, Ej)
        cS_lin, cS_rcs, dS_lo, dS_hi, _ = boot_ci(lp_lin_s, lp_rcs_s, Tj, Ej, n_boot=args.n_boot)
        lp_lin_f = oof_lp(np.column_stack([h, prs]), Tj, Ej)
        lp_rcs_f = oof_lp(np.column_stack([h, R]), Tj, Ej)
        cF_lin, cF_rcs, dF_lo, dF_hi, _ = boot_ci(lp_lin_f, lp_rcs_f, Tj, Ej, n_boot=args.n_boot)

        # (3) extreme-tail indicators (hazard-adjusted single-df HR)
        top5 = (prs >= np.quantile(prs, 0.95)).astype(float)
        bot5 = (prs <= np.quantile(prs, 0.05)).astype(float)
        top1 = (prs >= np.quantile(prs, 0.99)).astype(float)
        bot1 = (prs <= np.quantile(prs, 0.01)).astype(float)
        t5, _ = _single_hr(top5, Tj, Ej, adjust=h.reshape(-1, 1))
        b5, _ = _single_hr(bot5, Tj, Ej, adjust=h.reshape(-1, 1))
        t1, _ = _single_hr(top1, Tj, Ej, adjust=h.reshape(-1, 1))
        b1, _ = _single_hr(bot1, Tj, Ej, adjust=h.reshape(-1, 1))

        rows.append({
            "phecode": ph, "trait": trait, "n_ev": int(Ej.sum()), "n_eval": int(m.sum()),
            "lr_chi2_standalone": lr_s["chi2"], "lr_p_standalone": lr_s["p"],
            "lr_chi2_fusion": lr_f["chi2"], "lr_df": lr_f["df"], "lr_p_fusion": lr_f["p"],
            "c_lin_standalone": cS_lin, "c_rcs_standalone": cS_rcs,
            "dC_standalone": cS_rcs - cS_lin, "dC_s_lo": dS_lo, "dC_s_hi": dS_hi,
            "c_lin_fusion": cF_lin, "c_rcs_fusion": cF_rcs,
            "dC_fusion": cF_rcs - cF_lin, "dC_f_lo": dF_lo, "dC_f_hi": dF_hi,
            "top5_logHR": t5, "bot5_logHR": b5, "top1_logHR": t1, "bot1_logHR": b1,
        })
        for dr in decile_curve(prs, Tj, Ej):
            dr.update({"phecode": ph, "trait": trait})
            dec_rows.append(dr)

    df = pd.DataFrame(rows).sort_values("lr_p_fusion")
    dec = pd.DataFrame(dec_rows)
    df.to_csv(f"{out_dir}/prs_nonlinear_perdisease.csv", index=False)
    dec.to_csv(f"{out_dir}/prs_decile_curve.csv", index=False)

    # pooled summary + BH-FDR on the fusion LR test
    ok = df[df.lr_p_fusion.notna()].copy()
    if len(ok):
        p = ok.lr_p_fusion.to_numpy()
        order = np.argsort(p)
        m_ = len(p)
        bh = np.empty(m_); bh[order] = np.minimum.accumulate((p[order] * m_ / np.arange(1, m_ + 1))[::-1])[::-1]
        ok["lr_fdr_fusion"] = np.clip(bh, 0, 1)
        ok.to_csv(f"{out_dir}/prs_nonlinear_perdisease.csv", index=False)
    print(df[["phecode", "trait", "n_ev", "lr_p_standalone", "lr_p_fusion",
              "dC_standalone", "dC_fusion"]].round(4).to_string(index=False))
    n_sig = int((ok.get("lr_fdr_fusion", pd.Series(dtype=float)) < 0.05).sum()) if len(ok) else 0
    print(f"\nmatched n={len(df)}  LR(fusion) tested={len(ok)}  "
          f"non-linear at FDR<0.05: {n_sig}")
    print(f"mean dC standalone (RCS-lin) = {df.dC_standalone.mean():+.4f}  "
          f"fusion = {df.dC_fusion.mean():+.4f}")
    print(f"wrote {out_dir}/prs_nonlinear_perdisease.csv + prs_decile_curve.csv")


if __name__ == "__main__":
    main()
