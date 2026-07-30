"""High-specificity screening x genetics (supplementary).

Tests whether adding a polygenic risk score helps screening at stringent operating
points, especially for diseases the wrist already screens well (e.g. Parkinson's).
C-index is a global rank metric; high-specificity sensitivity is a tail metric, so
PRS could help in the tail even where it does not move C-index.

Five arms per matched disease (EUR), incident-primary framing (predict onset within 6y):
  actig        : frozen wrist hazard
  actig+PRS    : learned OOF fusion [hazard, PRS]  (canonical fusion, perdisease_ci.oof_lp)
  PRS          : standalone [PRS]
  clinical     : [age, age^2, sex, bmi, bmi^2]
  clinical+PRS : [age, age^2, sex, bmi, bmi^2, PRS]
Metrics per arm: full AUROC, partial AUROC (FPR<=0.10, McClish-standardised), and sensitivity
at 90/95/99/99.9% specificity with the incident-rate enrichment (PPV/prevalence). The added
value of PRS (Delta-sensitivity, actig to actig+PRS and clinical to clinical+PRS) gets a paired
subject-bootstrap CI. PD is the pre-specified headline; thin cells are power-flagged.

Run:
  python3 -m ukb_disease.paper_experiments.screening_genetics --out screening_genetics
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from ukb_disease.benchmark.concordance_benchmark import DEFAULT_RUNS, load_model
from ukb_disease.baseline.cox_crossfit_features import load_age_covars
from ukb_disease.baseline.config import resolve_oak_path
from ukb_disease.genetics.cox_crossfit_prs import build_phecode_to_trait, TRACK_PREFIX, FEATURES
from ukb_disease.genetics.perdisease_ci import oof_lp
from ukb_disease.paper_experiments import io

HORIZON = 6.0
SEVEN = 7.0 / 365.25
SPECS = [0.90, 0.95, 0.99, 0.999]
PD = "NS_324.11"
RNG = np.random.default_rng(20260707)


def sens_at_spec(score, y, spec):
    """Sensitivity at fixed specificity; threshold = spec-quantile of control scores."""
    ctrl = score[y == 0]; case = score[y == 1]
    if len(ctrl) < 5 or len(case) < 1:
        return np.nan, np.nan
    thr = np.quantile(ctrl, spec)
    return float((case >= thr).mean()), float(thr)


def pauc(score, y, max_fpr=0.10):
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y, score, max_fpr=max_fpr))
    except ValueError:
        return np.nan


def paired_dsens_ci(sA, sB, y, spec, n_boot=500):
    """Paired subject-bootstrap CI of Delta-sensitivity (B - A) at fixed spec."""
    n = len(y)
    d0 = (sens_at_spec(sB, y, spec)[0] or np.nan) - (sens_at_spec(sA, y, spec)[0] or np.nan)
    ds = []
    for _ in range(n_boot):
        ix = RNG.integers(0, n, n)
        a = sens_at_spec(sA[ix], y[ix], spec)[0]
        b = sens_at_spec(sB[ix], y[ix], spec)[0]
        if np.isfinite(a) and np.isfinite(b):
            ds.append(b - a)
    if len(ds) < 2:
        return d0, np.nan, np.nan
    return d0, float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def oof_logistic(X, y, seed=0, n_splits=5):
    """Out-of-fold logistic probability for the prevalent (cross-sectional) detection task.
    Falls back to the first (frozen) feature (monotone, so ROC/sens are unchanged) if a
    fold cannot be formed."""
    y = np.asarray(y, int)
    oof = np.full(len(y), np.nan)
    if y.sum() < n_splits or (len(y) - y.sum()) < n_splits:
        return X[:, 0].astype(float)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        s = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000, C=1.0).fit(s.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(s.transform(X[te]))[:, 1]
    return oof


def eval_arms(sc, y, ph, trait, framing, n_boot):
    """Per-arm operating-point metrics + paired Delta-sensitivity for one disease.
    `sc` values are scores already restricted to the eval subset and aligned with `y`."""
    arm_rows, dsens_rows = [], []
    pi = float(y.mean()); n_case, n_ctrl = int(y.sum()), int((y == 0).sum())
    two = len(np.unique(y)) > 1
    # per-arm subject-bootstrap of sens@spec (marginal 95% CI per arm). Uses a SEPARATE RNG so the
    # shared module RNG that drives the paired dsens CIs below is untouched (panel b unchanged).
    n = len(y); arng = np.random.default_rng(20260708)
    aboot = {arm: {spec: [] for spec in SPECS} for arm in sc}
    if two:
        for _ in range(n_boot):
            ix = arng.integers(0, n, n); yb = y[ix]
            for arm, score in sc.items():
                sb = score[ix]
                for spec in SPECS:
                    se = sens_at_spec(sb, yb, spec)[0]
                    if np.isfinite(se):
                        aboot[arm][spec].append(se)
    for arm, score in sc.items():
        row = {"phecode": ph, "trait": trait, "framing": framing, "arm": arm,
               "n_case": n_case, "n_ctrl": n_ctrl, "prevalence": pi,
               "auroc": float(roc_auc_score(y, score)) if two else np.nan,
               "pauc_fpr10": pauc(score, y),
               "average_precision": float(average_precision_score(y, score)) if two else np.nan}
        for spec in SPECS:
            se, _ = sens_at_spec(score, y, spec)
            ppv = (se * pi) / (se * pi + (1 - spec) * (1 - pi)) if np.isfinite(se) else np.nan
            row[f"sens@{spec}"] = se
            row[f"enrich@{spec}"] = (ppv / pi) if (np.isfinite(ppv) and pi > 0) else np.nan
            bs = aboot[arm][spec]
            row[f"sens_lo@{spec}"] = float(np.percentile(bs, 2.5)) if len(bs) >= 2 else np.nan
            row[f"sens_hi@{spec}"] = float(np.percentile(bs, 97.5)) if len(bs) >= 2 else np.nan
        arm_rows.append(row)
    for baselab, addlab, tag in [("actig", "actig_prs", "wrist"), ("clinical", "clinical_prs", "clinical")]:
        for spec in SPECS:
            d, lo, hi = paired_dsens_ci(sc[baselab], sc[addlab], y, spec, n_boot)
            dsens_rows.append({"phecode": ph, "trait": trait, "framing": framing, "base": tag,
                               "spec": spec, "dsens": d, "lo": lo, "hi": hi,
                               "n_case": n_case, "n_ctrl": n_ctrl})
    return arm_rows, dsens_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="std")
    ap.add_argument("--subset", default="eur_genetic")
    ap.add_argument("--deep-prefix", default="embedding_disjoint_split")
    ap.add_argument("--runs-dir", default=DEFAULT_RUNS)
    ap.add_argument("--age-dir", default="age_sex_disjoint_split/sequence_model_age")
    ap.add_argument("--features", default=FEATURES)
    ap.add_argument("--min-events", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--framing", choices=["incident", "prevalent", "both"], default="both")
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
    age, sex, bmi, _cov = io.load_demographics(eids)
    az = (age - np.nanmean(age)) / (np.nanstd(age) + 1e-9)
    bz = (bmi - np.nanmean(bmi)) / (np.nanstd(bmi) + 1e-9)

    feat = pd.read_parquet(args.features).set_index("eid").reindex(index=eids)
    prefix = TRACK_PREFIX[args.track]
    eur = np.ones(N, bool) if args.subset == "all" else (feat[args.subset].fillna(0).to_numpy() == 1)
    p2t = build_phecode_to_trait(names)

    arm_rows, dsens_rows = [], []
    matched = []
    for j in range(P):
        ph = names[j]; trait = p2t.get(ph)
        col = f"{prefix}{trait}" if trait else None
        if col not in feat.columns:
            continue
        m = M[:, j] & eur
        if int(m.sum()) < args.min_events or int(E[m, j].sum()) < args.min_events:
            continue
        matched.append((j, ph, trait, col))
        if args.framing in ("incident", "both"):
            Tj, Ej = T[m, j].astype(float), E[m, j].astype(float); hz = mean_h[m, j]
            prs = pd.to_numeric(feat[col], errors="coerce").to_numpy()[m]
            prs = np.nan_to_num((prs - np.nanmean(prs)) / (np.nanstd(prs) or 1.0))
            a, a2, s, b, b2 = az[m], az[m] ** 2, sex[m], bz[m], bz[m] ** 2
            Xclin = np.column_stack([a, a2, s, b, b2])
            sc = {"actig": hz,
                  "actig_prs": oof_lp(np.column_stack([hz, prs]), Tj, Ej),
                  "prs": oof_lp(prs.reshape(-1, 1), Tj, Ej),
                  "clinical": oof_lp(Xclin, Tj, Ej),
                  "clinical_prs": oof_lp(np.column_stack([Xclin, prs]), Tj, Ej)}
            y6 = ((Tj <= HORIZON) & (Ej > 0.5)).astype(int)
            keep = (y6 == 1) | (Tj > HORIZON)
            sck = {k: v[keep] for k, v in sc.items()}
            ar, dr = eval_arms(sck, y6[keep], ph, trait, "incident", args.n_boot)
            arm_rows += ar; dsens_rows += dr

    # --- prevalent (cross-sectional detection) companion: cases = disease present at/near
    #     baseline (years-to-dx <= 7 days), controls = never; frozen wrist hazard + OOF-logistic
    #     fusion. More cases -> stable stringent operating points + the incident-vs-prevalent
    #     PRS-value contrast (PRS is time-invariant, so it should add more before onset). ---
    if args.framing in ("prevalent", "both"):
        want = set(["eid"] + [ph for _, ph, _, _ in matched])
        tim = (pd.read_csv(resolve_oak_path(io.MORTALITY_PHECODES_CSV),
                           usecols=lambda c: c in want).set_index("eid").reindex(eids))
        for j, ph, trait, col in matched:
            if ph not in tim.columns:
                continue
            cell = tim[ph].to_numpy(float)
            prevalent = (~np.isnan(cell)) & (cell <= SEVEN)
            sub = (prevalent | np.isnan(cell)) & eur
            if int(prevalent[sub].sum()) < args.min_events:
                continue
            hz = mean_h[sub, j]
            prs = pd.to_numeric(feat[col], errors="coerce").to_numpy()[sub]
            prs = np.nan_to_num((prs - np.nanmean(prs)) / (np.nanstd(prs) or 1.0))
            a, a2, s, b, b2 = az[sub], az[sub] ** 2, sex[sub], bz[sub], bz[sub] ** 2
            Xclin = np.column_stack([a, a2, s, b, b2])
            y = prevalent[sub].astype(int)
            sc = {"actig": hz,
                  "actig_prs": oof_logistic(np.column_stack([hz, prs]), y),
                  "prs": prs,
                  "clinical": oof_logistic(Xclin, y),
                  "clinical_prs": oof_logistic(np.column_stack([Xclin, prs]), y)}
            ar, dr = eval_arms(sc, y, ph, trait, "prevalent", args.n_boot)
            arm_rows += ar; dsens_rows += dr

    arms = pd.DataFrame(arm_rows)
    ds = pd.DataFrame(dsens_rows)
    arms.to_csv(f"{out_dir}/screening_arms.csv", index=False)
    ds.to_csv(f"{out_dir}/screening_dsens.csv", index=False)

    for fr in [f for f in ("incident", "prevalent") if (arms.framing == f).any()]:
        af, dfr = arms[arms.framing == fr], ds[ds.framing == fr]
        print(f"\n=== {fr.upper()}: PD (headline) ===")
        print(af[af.phecode == PD][["arm", "n_case", "n_ctrl", "auroc", "pauc_fpr10",
              "sens@0.9", "sens@0.95", "sens@0.99"]].round(3).to_string(index=False))
        dw = dfr[dfr.base == "wrist"]
        print(f"pooled +PRS over wrist ({fr}):")
        for spec in SPECS:
            sub = dw[(dw.spec == spec) & dw.dsens.notna()]
            print(f"  spec {spec}: mean dsens={sub.dsens.mean():+.4f}  ({int((sub.lo>0).sum())}/{len(sub)} CI>0)")
    print(f"\nmatched diseases={arms.phecode.nunique()}  wrote {out_dir}/screening_arms.csv + screening_dsens.csv")


if __name__ == "__main__":
    main()
