"""Poorly-fit individuals vs PRS extremes (supplementary).

Motivated by O'Reilly et al. ("Distinct genetic architecture in the tails of
complex traits"): a trait's middle is polygenic (captured by a common-variant
PRS) while its tails are enriched for rare large-effect variants a common PRS
does NOT capture. If the actigraphy model systematically misses
genetically-driven cases, fusion with PRS is motivated; if the misses are NOT
concentrated at PRS extremes, fusion has a ceiling.

Per matched disease (EUR, matched PRS), on the frozen actigraphy-only hazard:
  1. Deviance residuals from the actigraphy-only Cox (lifelines compute_residuals):
     the formal "poorly-fit" measure. Large positive = an event the model ranked
     low (a miss).
  2. Caught-vs-missed incident cases: split true incident cases by predicted-risk
     tertile; compare matched-PRS between the caught (top) and missed (bottom),
     the interpretable, fusion-motivating contrast (missed cases higher-PRS =>
     wrist misses genetic cases).
  3. Full residual-vs-PRS relationship (per PRS decile), NOT assuming misses sit
     at high PRS (rare-variant-driven cases can sit at ordinary common-PRS).
  4. Brier contribution per subject as a robustness lens.

Set UKB_ROOT to point at the data root. Run:
  PYTHONDONTWRITEBYTECODE=1 python3 -m ukb_disease.genetics.poor_fit_prs \\
    --out "$UKB_ROOT/poor_fit"
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
from scipy.stats import spearmanr

from ukb_disease.benchmark.concordance_benchmark import DEFAULT_RUNS, load_model
from ukb_disease.baseline.cox_crossfit_features import load_age_covars
from ukb_disease.baseline.config import resolve_oak_path
from ukb_disease.genetics.cox_crossfit_prs import build_phecode_to_trait, TRACK_PREFIX, FEATURES

HORIZON = 6.0
RNG = np.random.default_rng(20260707)


def deviance_residuals(hz, T, E):
    """Deviance residuals from an actigraphy-only Cox (single covariate = frozen hazard).

    Sign convention: positive = the model under-predicted risk (event earlier / lower score
    than expected) -> a 'miss'. Returns NaN vector on convergence failure.
    """
    df = pd.DataFrame({"hz": hz, "T": T, "E": E})
    try:
        cph = CoxPHFitter(penalizer=1e-3).fit(df, "T", "E")
        dev = cph.compute_residuals(df, "deviance")["deviance"].to_numpy()
        return dev
    except (ConvergenceError, ValueError, np.linalg.LinAlgError):
        return np.full(len(T), np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="std")
    ap.add_argument("--subset", default="eur_genetic")
    ap.add_argument("--deep-prefix", default="embedding_disjoint_split")
    ap.add_argument("--runs-dir", default=DEFAULT_RUNS)
    ap.add_argument("--age-dir", default="age_sex_disjoint_split/sequence_model_age")
    ap.add_argument("--features", default=FEATURES)
    ap.add_argument("--min-events", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=2000)
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
    _ac = load_age_covars(age_dir, eids)

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
        Tj = T[m, j].astype(float); Ej = E[m, j].astype(float); hz = mean_h[m, j]
        prs = pd.to_numeric(feat[col], errors="coerce").to_numpy()[m]
        prs = np.nan_to_num((prs - np.nanmean(prs)) / (np.nanstd(prs) or 1.0))
        prs_pct = (np.argsort(np.argsort(prs)) + 0.5) / len(prs)   # cohort PRS percentile

        dev = deviance_residuals(hz, Tj, Ej)
        if np.all(np.isnan(dev)):
            continue

        # (1) poor-fit vs PRS: does |residual| relate to PRS percentile? (Spearman)
        rho_all, p_all = spearmanr(np.abs(dev), prs_pct)

        # (2) caught vs missed incident cases (risk tertiles among true cases)
        case = Ej > 0.5
        pr = prs[case]
        hzc = hz[case]
        prs_missed_minus_caught = np.nan; miss_lo = miss_hi = np.nan
        frac_missed_extreme = np.nan
        if case.sum() >= 12:
            q1, q2 = np.quantile(hzc, [1 / 3, 2 / 3])
            missed = hzc <= q1      # low predicted risk among true cases = missed
            caught = hzc >= q2
            if missed.sum() >= 3 and caught.sum() >= 3:
                d0 = float(pr[missed].mean() - pr[caught].mean())
                boots = []
                cm_idx = np.where(missed)[0]; cc_idx = np.where(caught)[0]
                for _ in range(args.n_boot):
                    bm = RNG.choice(cm_idx, len(cm_idx), replace=True)
                    bc = RNG.choice(cc_idx, len(cc_idx), replace=True)
                    boots.append(pr[bm].mean() - pr[bc].mean())
                prs_missed_minus_caught = d0
                miss_lo, miss_hi = np.percentile(boots, [2.5, 97.5])
                # extremes: fraction of missed cases in top/bottom PRS decile (expected 0.20)
                pe = prs_pct[case][missed]
                frac_missed_extreme = float(((pe >= 0.9) | (pe <= 0.1)).mean())

        # (3) decile curve: miss-rate and mean |dev| among cases, per PRS decile
        cd = prs_pct[case]
        if case.sum() >= 10:
            dc = np.clip((cd * 10).astype(int), 0, 9)
            hz_thr = np.median(hzc)
            for d in range(10):
                sel = dc == d
                if sel.sum() == 0:
                    continue
                dec_rows.append({
                    "phecode": ph, "trait": trait, "prs_decile": d, "n_cases": int(sel.sum()),
                    "miss_rate": float((hzc[sel] <= hz_thr).mean()),
                    "mean_abs_dev": float(np.abs(dev[case][sel]).mean()),
                })

        rows.append({
            "phecode": ph, "trait": trait, "n_ev": int(Ej.sum()), "n_eval": int(m.sum()),
            "spearman_absdev_prs": float(rho_all), "spearman_p": float(p_all),
            "prs_missed_minus_caught": prs_missed_minus_caught,
            "missed_caught_lo": float(miss_lo) if np.isfinite(miss_lo) else np.nan,
            "missed_caught_hi": float(miss_hi) if np.isfinite(miss_hi) else np.nan,
            "frac_missed_in_prs_extreme": frac_missed_extreme,
        })

    df = pd.DataFrame(rows).sort_values("prs_missed_minus_caught", ascending=False)
    pd.DataFrame(dec_rows).to_csv(f"{out_dir}/poor_fit_decile_curve.csv", index=False)
    df.to_csv(f"{out_dir}/poor_fit_perdisease.csv", index=False)

    # pooled: is PRS higher in missed than caught cases, on average? (fusion motivation)
    ok = df[df.prs_missed_minus_caught.notna()]
    pooled = float(ok.prs_missed_minus_caught.mean()) if len(ok) else np.nan
    n_pos = int((ok.prs_missed_minus_caught > 0).sum())
    n_sig = int(((ok.missed_caught_lo > 0)).sum())
    print(df[["phecode", "trait", "n_ev", "spearman_absdev_prs",
              "prs_missed_minus_caught", "missed_caught_lo", "missed_caught_hi",
              "frac_missed_in_prs_extreme"]].round(3).to_string(index=False))
    print(f"\nmatched n={len(df)}  caught/missed evaluable={len(ok)}")
    print(f"pooled PRS(missed)-PRS(caught) = {pooled:+.3f} sd  "
          f"({n_pos}/{len(ok)} diseases positive, {n_sig} with CI>0)")
    print(f"  interpretation: >0 => wrist misses higher-PRS (genetically-loaded) cases -> "
          f"fusion motivated; ~0 => misses are not genetic (O'Reilly rare-variant tails).")
    print(f"wrote {out_dir}/poor_fit_perdisease.csv + poor_fit_decile_curve.csv")


if __name__ == "__main__":
    main()
