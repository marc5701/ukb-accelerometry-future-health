#!/usr/bin/env python3
"""UKB summarized-activity-features baseline vs the wrist embedding.

Mirror of `clinical_baseline.py` (age/sex/BMI) that swaps the covariate block for
the UK Biobank distributed accelerometer summary features. Uses the same test/val
eids, phecode panel, per-disease ridge-Cox fit on validation scored on test
(`cox_valfit._fit_apply`), 6-year AUROC and C-index, and subject-level paired
bootstrap as `winrate_ci.py`. Cohort, splits, outcomes, and metrics therefore
match the clinical baseline by construction.

Arms (per phecode):
  summary       ridge-Cox on the UKB summary features       (mirror of clinical-only)
  emb           the frozen embedding test hazard                   (matches clinical arm B)
  emb_summary   ridge-Cox [emb hazard, summary]              (does the FM add value?)
  clin_summary  ridge-Cox [age, sex, BMI, summary]           (strongest combined baseline)
  clin          ridge-Cox [age, sex, BMI]                    (reference; reproduces clinical_baseline)

Feature sets (`--feature-set`):
  full     100 activity fields: overall accel avg and SD (90012/90013), day-of-week
           averages (90019-90025), hour-of-day averages (90027-90050), acceleration
           intensity distribution fraction<=X mg (90092-90158). Primary.
  compact  the two canonical scalar UKB summaries {90012, 90013}. Sensitivity.

The summary and combined arms are fit on val and scored on test; the emb arm reuses
the frozen deep-model test hazard unchanged. Only the covariate block varies across
arms.

Usage:
  python -m ukb_disease.baseline.summary_baseline --feature-set full    --boot 1000
  python -m ukb_disease.baseline.summary_baseline --feature-set compact --boot 1000
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from ukb_disease.baseline.config import resolve_oak_path
from ukb_disease.baseline.cox_valfit import _fit_apply, seed_avg
from ukb_disease.baseline.per_disease_eval import auroc_at_horizon
from ukb_disease.benchmark.concordance_benchmark import harrell_counts
from ukb_disease.paths import UKB_ROOT

BASE = str(UKB_ROOT)
DEM = f"{UKB_ROOT}/metadata/acc_dem.csv"
BMI = f"{UKB_ROOT}/metadata/BMI_yes.csv"
PARQ = f"{UKB_ROOT}/caches/ukb_summary_activity/ukb_summary_activity.parquet"
CLIN_REF = f"{UKB_ROOT}/clinical_baseline/per_disease_3arm.csv"
OUT_ROOT = f"{UKB_ROOT}/summary_baseline/output"
HORIZON = 6.0
PEN_C0 = 10.0
BOOT_SEED = 20260701

# Activity include-lists (UKB Data-Dictionary showcase field IDs).
_FULL_IDS = [90012, 90013] + list(range(90019, 90026)) + list(range(90027, 90051)) + list(range(90092, 90159))
FULL_COLS = [f"{i}-0.0" for i in _FULL_IDS]        # 100 fields
COMPACT_COLS = ["90012-0.0", "90013-0.0"]          # 2 canonical scalars


def load_clinical(eids):
    """[age_z, sex, bmi_z] aligned to eids (copy of clinical_baseline.load_clinical, no side effects)."""
    dem = pd.read_csv(resolve_oak_path(DEM)).set_index("eid")
    bmi = (pd.read_csv(resolve_oak_path(BMI), usecols=["eid", "21001-0.0"])
           .rename(columns={"21001-0.0": "bmi"}).groupby("eid").mean())
    age = dem.reindex(eids)["age_at_first_run"].to_numpy(float)
    sex = dem.reindex(eids)["sex"].to_numpy(float)
    bm = bmi.reindex(eids)["bmi"].to_numpy(float)

    def z(x):
        x = np.where(np.isfinite(x), x, np.nanmean(x))
        return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)
    sex = np.where(np.isfinite(sex), sex, np.nanmean(sex))
    return np.column_stack([z(age), sex, z(bm)]), {"bmi_cov": float(np.isfinite(bm).mean())}


def load_summary(eids, cols):
    """Raw UKB summary features aligned to eids; _fit_apply standardizes + NaN-imputes on val."""
    df = pd.read_parquet(resolve_oak_path(PARQ)).set_index("eid")
    missing = [c for c in cols if c not in df.columns]
    assert not missing, f"summary cols missing from parquet: {missing}"
    X = df.reindex(eids)[cols].to_numpy(float)
    return X, {"n_feat": len(cols), "cov": float(np.isfinite(X).mean())}


def build(cols):
    """Return per-subject-per-disease risk matrices for every arm on the disjoint_split test set."""
    th, te, tT, tE, tM, tdirs = seed_avg(f"{BASE}/runs_cox/embedding_disjoint_split_seed*", "test")
    vh, ve, vT, vE, vM, _ = seed_avg(f"{BASE}/runs_cox_val_pred/embedding_val_pred_seed*", "val")
    names = json.load(open(f"{tdirs[0]}/config.json"))["phecodes"]
    P = th.shape[1]

    clin_t, covc = load_clinical(te); clin_v, _ = load_clinical(ve)
    summ_t, covs = load_summary(te, cols); summ_v, _ = load_summary(ve, cols)
    print(f"[summary] N_test={len(te)} N_val={len(ve)} P={P} | n_feat={covs['n_feat']} "
          f"| summary cov(test)={covs['cov']*100:.1f}% bmi cov(test)={covc['bmi_cov']*100:.1f}%")

    evaln = tM.sum(0); evn = (tE * tM).sum(0)
    panel = [j for j in range(P) if evaln[j] >= 10 and evn[j] >= 1]
    print(f"[summary] panel = {len(panel)} phecodes")

    N = len(te)
    R = {k: np.full((N, P), np.nan) for k in ["clin", "summary", "emb_summary", "clin_summary"]}
    R["emb"] = th.astype(float)
    n_fail = {k: 0 for k in ["clin", "summary", "emb_summary", "clin_summary"]}
    for j in panel:
        mv, mt = vM[:, j], tM[:, j]
        Tv, Ev = vT[mv, j].astype(float), vE[mv, j].astype(float)
        pen = max(0.1, PEN_C0 / max(1, int(Ev.sum())))

        def fit(Xv, Xt, key):
            try:
                return _fit_apply(Xv, Tv, Ev, Xt, pen)
            except Exception:
                n_fail[key] += 1
                return np.full(mt.sum(), np.nan)

        R["clin"][mt, j] = fit(clin_v[mv], clin_t[mt], "clin")
        R["summary"][mt, j] = fit(summ_v[mv], summ_t[mt], "summary")
        R["emb_summary"][mt, j] = fit(np.column_stack([vh[mv, j], summ_v[mv]]),
                                      np.column_stack([th[mt, j], summ_t[mt]]), "emb_summary")
        R["clin_summary"][mt, j] = fit(np.column_stack([clin_v[mv], summ_v[mv]]),
                                       np.column_stack([clin_t[mt], summ_t[mt]]), "clin_summary")
    print(f"[summary] per-disease fit failures (NaN, caught): {n_fail}")
    return {"R": R, "names": names, "panel": np.asarray(panel),
            "tT": tT.astype(float), "tE": tE.astype(float), "tM": tM.astype(bool),
            "evn": evn, "n_fail": n_fail, "cov_summary": covs["cov"], "n_feat": covs["n_feat"]}


def _auroc(risk, T, E):
    return auroc_at_horizon(risk, T, E, HORIZON)


def _auroc_fast(scores, times, events, horizon=HORIZON):
    """Vectorized fixed-horizon AUROC via the Mann-Whitney rank formula. Numerically
    identical to per_disease_eval.auroc_at_horizon (roc_auc_score) including average-rank
    tie handling, but about 15x faster (no sklearn per-call overhead), so used only
    inside the subject bootstrap. Same positive/negative/censored-before-horizon
    definition."""
    from scipy.stats import rankdata
    ev = events > 0
    pos = (times < horizon) & ev
    keep = ~((times < horizon) & ~ev)          # drop censored-before-horizon (ambiguous)
    y = pos[keep]
    s = scores[keep]
    fin = np.isfinite(s)
    y, s = y[fin], s[fin]
    npos = int(y.sum()); nneg = int(y.size - npos)
    if npos == 0 or nneg == 0 or y.size < 2:
        return np.nan
    r = rankdata(s)
    return float((r[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def _c(risk, T, E):
    c, comp = harrell_counts(risk, T, E)
    return c / comp if comp > 0 else np.nan


def per_disease_table(G):
    R, panel, names, tT, tE, tM, evn = (G["R"], G["panel"], G["names"], G["tT"], G["tE"],
                                        G["tM"], G["evn"])
    rows = []
    for j in panel:
        mt = tM[:, j]
        Tt, Et = tT[mt, j], tE[mt, j]
        rec = {"phecode": names[j], "category": names[j].split("_")[0], "n_event": int(evn[j])}
        for arm, M in R.items():
            rec[f"auroc_{arm}"] = _auroc(M[mt, j], Tt, Et)
            rec[f"c_{arm}"] = _c(M[mt, j], Tt, Et)
        rows.append(rec)
    return pd.DataFrame(rows)


def _arm_aurocs(R_arm, panel, tT, tE, tM, rows):
    """Per-disease fixed-horizon AUROC for ONE arm's risk matrix on subject index `rows`
    (fast rank AUROC). Precomputed once per bootstrap replicate, then reused across all
    comparisons that involve this arm."""
    out = np.full(len(panel), np.nan)
    for i, j in enumerate(panel):
        m = tM[rows, j]
        r = rows[m]
        if r.size < 10:
            continue
        out[i] = _auroc_fast(R_arm[r, j], tT[r, j], tE[r, j])
    return out


def _wr_da(Aa, Ab):
    """win-rate (Ab>Aa), mean AUROC delta (Ab-Aa), n over diseases where both are finite."""
    ok = np.isfinite(Aa) & np.isfinite(Ab)
    if not ok.any():
        return np.nan, np.nan, 0
    d = Ab[ok] - Aa[ok]
    return float((Ab[ok] > Aa[ok]).mean()), float(d.mean()), int(ok.sum())


# comparisons: (label, arm_A=baseline, arm_B=challenger), asking "does B beat A?"
COMPARISONS = [
    ("emb_vs_summary", "summary", "emb"),        # embedding vs UKB summaries
    ("emb_vs_clin", "clin", "emb"),              # sanity check: reproduce the clinical result
    ("summary_vs_clin", "clin", "summary"),      # do UKB summaries beat age/sex/BMI?
    ("embsummary_vs_emb", "emb", "emb_summary"), # do summaries add on top of the embedding?
    ("clinsummary_vs_emb", "emb", "clin_summary"),  # best combined simple baseline vs embedding
]
_ARMS = ["clin", "summary", "emb", "emb_summary", "clin_summary"]


def bootstrap(G, B):
    R, panel, tT, tE, tM = G["R"], G["panel"], G["tT"], G["tE"], G["tM"]
    N = R["emb"].shape[0]
    A0 = {a: _arm_aurocs(R[a], panel, tT, tE, tM, np.arange(N)) for a in _ARMS}
    out = {}
    for label, a, b in COMPARISONS:
        wr0, da0, n0 = _wr_da(A0[a], A0[b])
        out[label] = {"baseline": a, "challenger": b, "n_diseases": n0,
                      "winrate_point": wr0, "mean_dAUROC_point": da0, "_wr": [], "_da": []}
    rng = np.random.default_rng(BOOT_SEED)
    for bi in range(B):
        rows = rng.integers(0, N, N)
        Ab = {a: _arm_aurocs(R[a], panel, tT, tE, tM, rows) for a in _ARMS}
        for label, a, b in COMPARISONS:
            wr, da, _ = _wr_da(Ab[a], Ab[b])
            out[label]["_wr"].append(wr); out[label]["_da"].append(da)
        if (bi + 1) % 200 == 0:
            print(f"  boot {bi+1}/{B}", flush=True)
    for label in out:
        wr = np.array(out[label].pop("_wr")); da = np.array(out[label].pop("_da"))
        out[label]["winrate_ci"] = [float(np.nanpercentile(wr, 2.5)), float(np.nanpercentile(wr, 97.5))]
        out[label]["mean_dAUROC_ci"] = [float(np.nanpercentile(da, 2.5)), float(np.nanpercentile(da, 97.5))]
        out[label]["B"] = int(B)
    return out


def fairness_checks(df, G):
    """Assert the summary run shares the clinical baseline's panel + reproduces its arms."""
    checks = {}
    if os.path.exists(CLIN_REF):
        ref = pd.read_csv(CLIN_REF)
        m = df.merge(ref[["phecode", "auroc_embedding", "auroc_clinical"]], on="phecode", how="inner")
        checks["n_shared_phecodes"] = int(len(m))
        # emb arm uses the same frozen th, so it must match ref auroc_embedding almost exactly.
        de = (m["auroc_emb"] - m["auroc_embedding"]).abs()
        checks["max_abs_emb_vs_ref"] = float(de.max())
        # clin arm reproduces ref clinical (same 3 covariates, same pen), so it should match closely.
        dc = (m["auroc_clin"] - m["auroc_clinical"]).abs()
        checks["max_abs_clin_vs_ref"] = float(dc.max())
        checks["emb_reproduces_ref"] = bool(de.max() < 1e-6)
    return checks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--feature-set", choices=["full", "compact"], default="full")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    cols = FULL_COLS if args.feature_set == "full" else COMPACT_COLS
    out = f"{OUT_ROOT}/{args.feature_set}"
    os.makedirs(out, exist_ok=True)

    G = build(cols)
    df = per_disease_table(G)
    df.to_csv(f"{out}/per_disease_summary.csv", index=False)

    def m(col):
        return float(np.nanmean(df[col]))
    arms = ["clin", "summary", "emb", "emb_summary", "clin_summary"]
    summ = {
        "feature_set": args.feature_set, "n_feat": G["n_feat"], "n_panel": len(df),
        "summary_cov_test": G["cov_summary"], "horizon_years": HORIZON,
        "full_field_ids": _FULL_IDS if args.feature_set == "full" else [90012, 90013],
        "per_disease_fit_failures": G["n_fail"],
        "mean_auroc": {a: m(f"auroc_{a}") for a in arms},
        "mean_c": {a: m(f"c_{a}") for a in arms},
    }
    print("\n[summary] mean AUROC:", {k: round(v, 4) for k, v in summ["mean_auroc"].items()})
    print("[summary] mean C   :", {k: round(v, 4) for k, v in summ["mean_c"].items()})

    summ["fairness_checks"] = fairness_checks(df, G)
    print("[summary] fairness_checks:", summ["fairness_checks"])

    # validate the fast bootstrap AUROC reproduces the exact sklearn AUROC of the saved table
    A_emb_fast = _arm_aurocs(G["R"]["emb"], G["panel"], G["tT"], G["tE"], G["tM"],
                             np.arange(G["R"]["emb"].shape[0]))
    maxd = float(np.nanmax(np.abs(A_emb_fast - df["auroc_emb"].to_numpy())))
    print(f"[summary] fast-vs-sklearn AUROC max abs diff (emb arm): {maxd:.2e}")
    assert maxd < 1e-9, "fast AUROC diverges from sklearn auroc_at_horizon"
    summ["fast_auroc_maxdiff"] = maxd

    print(f"[summary] bootstrap B={args.boot} ...", flush=True)
    summ["comparisons"] = bootstrap(G, args.boot)
    for label, r in summ["comparisons"].items():
        print(f"  {label:22s} B>{r['baseline']:<12s} winrate {r['winrate_point']:.3f} "
              f"CI[{r['winrate_ci'][0]:.3f},{r['winrate_ci'][1]:.3f}]  "
              f"dAUROC {r['mean_dAUROC_point']:+.4f} CI[{r['mean_dAUROC_ci'][0]:+.4f},{r['mean_dAUROC_ci'][1]:+.4f}]")

    json.dump(summ, open(f"{out}/summary.json", "w"), indent=2, default=float)
    print(f"\n[summary] wrote {out}/per_disease_summary.csv + summary.json")


if __name__ == "__main__":
    main()
