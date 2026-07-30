#!/usr/bin/env python3
"""Signal beyond demographics: residualized partial-discrimination and in-model demographic ladder.

Two CPU-only experiments, reusing cached per-disease deep hazards and age/sex/BMI covariates.

EXP 1 (residual_partial), the headline. Per phecode, a within-test cross-fit increment of the deep
wrist hazard on top of a flexible age+sex+BMI model, in two metric spaces:
  * AUROC@6yr: regularized logistic on [age,sex,bmi] vs [age,sex,bmi,hazard] (shared OOF folds,
    paired delta-AUROC).
  * Harrell C: ridge-Cox on an expanded demographic basis [age,age^2,sex,bmi,bmi^2,age*sex,age*bmi,
    sex*bmi] vs basis+hazard, event-stratified within-fold cross-fit (sum conc/comp counts).
Two hazard sources: the primary `embedding_disjoint_split` (trained with age+sex) and `embedding_wrist_only_disjoint_split`
(no demographics). Plus a permuted-hazard null control (Cox) to calibrate the estimation cost of
adding any column. Aggregated per-disease and per-category with panel bootstrap CIs.

EXP 2 (ladder) closes the in-model demographic ablation, now with BMI. Scores the existing
covariate-ladder hazard caches per-disease (no fitting): wrist_only(0) -> ageonly(1) / sexonly(1) ->
agesex(2, primary) -> agesexbmi(3). Reports the mean-C / mean-AUROC ladder and the BMI / age / sex /
age+sex marginals with paired bootstrap CIs.

Run:
  python3 ukb_disease/investigations/residual_partial/residual_partial.py --exp both \
    --out output
Smoke: add --phecode-limit 8
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import time

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from ukb_disease.paths import UKB_ROOT
from ukb_disease.baseline.config import resolve_oak_path
from ukb_disease.baseline.cox_valfit import seed_avg, _fit_apply
from ukb_disease.baseline.per_disease_eval import auroc_at_horizon
from ukb_disease.baseline.cox_crossfit_features import adaptive_k, event_stratified_folds
from ukb_disease.benchmark.concordance_benchmark import harrell_counts

HORIZON = 6.0
CACHE = f"{UKB_ROOT}/runs_cox"
DEM = f"{UKB_ROOT}/metadata/acc_dem.csv"
BMI = f"{UKB_ROOT}/metadata/BMI_yes.csv"

# covariate ladder (n_covar in parens), glob suffix under CACHE
LADDER = [
    ("wrist_only", "embedding_wrist_only_disjoint_split_seed*"),   # 0
    ("bmionly",   "embedding_bmionly_disjoint_split_seed*"),     # 1 (BMI alone; --add-bmi --keep-covariates bmi)
    ("ageonly",   "embedding_ageonly_disjoint_split_seed*"),     # 1
    ("sexonly",   "embedding_sexonly_disjoint_split_seed*"),     # 1
    ("agesex",    "embedding_disjoint_split_seed*"),             # 2 (primary)
    ("agesexbmi", "embedding_clinical_embedding_disjoint_split_seed*"),     # 3 (--add-bmi)
]

# phecode prefix -> readable category (best-effort; falls back to the raw prefix)
CAT = {"NS": "neuro", "CV": "cardiovascular", "CA": "cardiovascular", "EM": "endocrine_metabolic",
       "RE": "respiratory", "GI": "digestive", "GU": "genitourinary", "MS": "musculoskeletal",
       "DE": "dermatologic", "HE": "hematologic", "NE": "neoplasm", "ID": "infectious",
       "PS": "psychiatric", "SO": "symptoms", "BI": "blood_immune", "SS": "sense_organ",
       "PG": "pregnancy", "IN": "injury", "MB": "mental_behavioral"}


def category_of(phecode: str) -> str:
    if phecode in ("time_to_death", "mortality"):
        return "mortality"
    pre = phecode.split("_", 1)[0]
    return CAT.get(pre, pre)


# ------------------------------------------------------------------------------------------------
def load_demo(eids: np.ndarray):
    """Return raw (age, sex, bmi) aligned to eids, NaN -> column mean. (Cox fit z-scores per fold;
    GBM uses raw.)"""
    dem = pd.read_csv(resolve_oak_path(DEM)).set_index("eid")
    bmi = (pd.read_csv(resolve_oak_path(BMI), usecols=["eid", "21001-0.0"])
           .rename(columns={"21001-0.0": "bmi"}).groupby("eid").mean())
    age = dem.reindex(eids)["age_at_first_run"].to_numpy(float)
    sex = dem.reindex(eids)["sex"].to_numpy(float)
    bm = bmi.reindex(eids)["bmi"].to_numpy(float)
    cov = {}
    for nm, a in (("age", age), ("sex", sex), ("bmi", bm)):
        cov[nm] = float(np.isfinite(a).mean())
        a[~np.isfinite(a)] = np.nanmean(a)
    return age, sex, bm, cov


def demo_basis(age, sex, bmi):
    """Flexible demographic basis: linear + quadratic + pairwise interactions (8 cols).
    _fit_apply z-scores each column per fold, so raw polynomial scale is fine."""
    return np.column_stack([age, age * age, sex, bmi, bmi * bmi, age * sex, age * bmi, sex * bmi])


def boot_mean_ci(x, n=1000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan, 0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n, x.size))
    means = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), int(x.size)


def crossfit_c(X, T_j, E_j, penalizer_c0=10.0, fallback_col=0):
    """Honest within-test cross-fit Harrell C: event-stratified adaptive-K folds, ridge-Cox fit on
    train folds, score held-out fold, SUM concordant/comparable counts across folds (not pooled LP)."""
    n_ev = int(E_j.sum())
    K = adaptive_k(n_ev)
    if K == 0:
        return np.nan
    fold = event_stratified_folds(E_j, K)
    conc = comp = 0.0
    for f in range(K):
        te = fold == f
        tr = ~te
        if tr.sum() < 3 or te.sum() < 1:
            continue
        try:
            pen = max(0.1, penalizer_c0 / max(1, int(E_j[tr].sum())))
            lp = _fit_apply(X[tr], T_j[tr], E_j[tr], X[te], pen)
        except Exception:
            lp = X[te, fallback_col]
        c, n = harrell_counts(lp, T_j[te], E_j[te])
        conc += c
        comp += n
    return conc / comp if comp > 0 else np.nan


def crossfit_aurocs(feat_list, T_j, E_j, K=5, seed=0):
    """Shared-fold OOF AUROC@horizon for several feature matrices (paired Δ). Uses a regularized
    logistic regression on the (standardized) flexible demographic basis, a fair strong comparator
    (GBM overfits on thin-event phecodes and would inflate the increment). Positives/keep-mask follow
    auroc_at_horizon: pos=(T<H & E==1); drop censored-before-horizon."""
    pos = (T_j < HORIZON) & (E_j == 1)
    keep = ~((T_j < HORIZON) & (E_j == 0))
    idx = np.where(keep)[0]
    y = pos[idx].astype(int)
    npos, nneg = int(y.sum()), int(len(y) - y.sum())
    if npos < 5 or nneg < 5:
        return [np.nan] * len(feat_list)
    k = int(min(K, npos, nneg))
    if k < 2:
        return [np.nan] * len(feat_list)
    splits = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(idx, y))
    outs = []
    for X in feat_list:
        Xk = X[idx]
        oof = np.full(len(idx), np.nan)
        for tr, te in splits:
            sc = StandardScaler().fit(Xk[tr])
            clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000)
            clf.fit(sc.transform(Xk[tr]), y[tr])
            oof[te] = clf.predict_proba(sc.transform(Xk[te]))[:, 1]
        try:
            outs.append(float(roc_auc_score(y, oof)))
        except Exception:
            outs.append(np.nan)
    return outs


# ------------------------------------------------------------------------------------------------
def run_exp2(out_dir, limit=None):
    """In-model demographic ablation ladder from the 5 existing caches (no fitting)."""
    print("\n========== EXP 2: in-model demographic ladder ==========", flush=True)
    base = None
    haz = {}
    present = []  # ladder rungs whose cache actually exists (graceful skip if a run hasn't landed)
    for nm, gl in LADDER:
        dirs = [d for d in sorted(_glob.glob(resolve_oak_path(f"{CACHE}/{gl}")))
                if os.path.exists(f"{d}/test_hazards.npy")]
        if not dirs:
            print(f"[exp2] SKIP {nm}: no cache yet ({gl})", flush=True)
            continue
        h, eids, T, E, M, used = seed_avg(f"{CACHE}/{gl}", "test")
        if base is None:
            base = dict(eids=eids, T=T, E=E, M=M,
                        names=json.load(open(f"{used[0]}/config.json"))["phecodes"])
        else:
            assert np.array_equal(eids, base["eids"]), f"{nm} eids differ from reference"
        haz[nm] = h
        present.append(nm)
    names, T, E, M = base["names"], base["T"], base["E"], base["M"]
    P = T.shape[1]
    ev = (E * M).sum(0)
    panel = [j for j in range(P) if ev[j] >= 10]
    if limit:
        panel = panel[:limit]
    print(f"[exp2] panel = {len(panel)} phecodes; models = {present}", flush=True)

    rows = []
    for j in panel:
        m = M[:, j]
        Tj, Ej = T[m, j].astype(float), E[m, j].astype(float)
        rec = {"phecode": names[j], "category": category_of(names[j]), "n_ev": int(Ej.sum())}
        for nm in present:
            hj = haz[nm][m, j]
            c, comp = harrell_counts(hj, Tj, Ej)
            rec[f"c_{nm}"] = c / comp if comp > 0 else np.nan
            rec[f"auroc_{nm}"] = auroc_at_horizon(hj, Tj, Ej, HORIZON)
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(f"{out_dir}/exp2_ladder_per_disease.csv", index=False)

    summ = {"n_panel": len(panel), "horizon": HORIZON, "models": present}
    for metric in ("c", "auroc"):
        summ[f"mean_{metric}"] = {nm: float(df[f"{metric}_{nm}"].mean()) for nm in present}
    # marginals (paired per-disease deltas, bootstrap CI); only computed if both rungs present
    def marg(metric, a, b, key):
        if a not in present or b not in present:
            return
        d = (df[f"{metric}_{a}"] - df[f"{metric}_{b}"]).to_numpy()
        mean, lo, hi, n = boot_mean_ci(d, seed=hash(key) % 2**31)
        summ.setdefault(f"marginals_{metric}", {})[key] = {
            "mean_delta": mean, "ci2.5": lo, "ci97.5": hi, "n_disease": n,
            "winrate": float(np.nanmean(d > 0))}
    for metric in ("c", "auroc"):
        marg(metric, "agesex", "wrist_only", "age+sex (vs wrist_only)")
        marg(metric, "ageonly", "wrist_only", "age (vs wrist_only)")
        marg(metric, "sexonly", "wrist_only", "sex (vs wrist_only)")
        marg(metric, "bmionly", "wrist_only", "BMI-only (vs wrist_only)")
        marg(metric, "agesexbmi", "agesex", "BMI (vs age+sex)")
        marg(metric, "agesexbmi", "wrist_only", "age+sex+BMI (vs wrist_only)")
    json.dump(summ, open(f"{out_dir}/exp2_ladder_summary.json", "w"), indent=2)

    print("\n[exp2] mean C-index ladder:", flush=True)
    for nm in present:
        print(f"   {nm:12s}  C={summ['mean_c'][nm]:.4f}  AUROC={summ['mean_auroc'][nm]:.4f}", flush=True)
    print("[exp2] marginals (mean ΔC [95% CI], winrate):", flush=True)
    for k, v in summ["marginals_c"].items():
        print(f"   {k:28s}  {v['mean_delta']:+.4f} [{v['ci2.5']:+.4f},{v['ci97.5']:+.4f}]  "
              f"win {v['winrate']*100:.0f}%", flush=True)
    return summ


def run_exp1(out_dir, limit=None):
    """Residualized partial-discrimination: deep hazard increment over a FLEXIBLE age+sex+BMI model."""
    print("\n========== EXP 1: flexible-demographic residualization ==========", flush=True)
    hc, eids, T, E, M, dirs = seed_avg(f"{CACHE}/embedding_disjoint_split_seed*", "test")          # age+sex
    hw, ewids, *_ = seed_avg(f"{CACHE}/embedding_wrist_only_disjoint_split_seed*", "test")          # wrist-only
    assert np.array_equal(eids, ewids), "age+sex vs wrist_only eids differ"
    names = json.load(open(f"{dirs[0]}/config.json"))["phecodes"]
    P = T.shape[1]
    age, sex, bmi, cov = load_demo(eids)
    print(f"[exp1] covariate coverage: {cov}", flush=True)

    ev = (E * M).sum(0)
    panel = [j for j in range(P) if ev[j] >= 10]
    if limit:
        panel = panel[:limit]
    print(f"[exp1] panel = {len(panel)} phecodes; N_test = {len(eids)}", flush=True)

    rows = []
    t0 = time.time()
    for i, j in enumerate(panel):
        m = M[:, j]
        Tj, Ej = T[m, j].astype(float), E[m, j].astype(float)
        a, s, b = age[m], sex[m], bmi[m]
        cj, wj = hc[m, j].astype(float), hw[m, j].astype(float)
        rng = np.random.default_rng(1234 + j)
        cj_perm = rng.permutation(cj)
        basis = demo_basis(a, s, b)
        rec = {"phecode": names[j], "category": category_of(names[j]), "n_ev": int(Ej.sum())}

        # ---- Harrell C (Cox, honest cross-fit) ----
        c_demo = crossfit_c(basis, Tj, Ej)
        c_dc = crossfit_c(np.column_stack([basis, cj]), Tj, Ej, fallback_col=basis.shape[1])
        c_dw = crossfit_c(np.column_stack([basis, wj]), Tj, Ej, fallback_col=basis.shape[1])
        c_dn = crossfit_c(np.column_stack([basis, cj_perm]), Tj, Ej, fallback_col=basis.shape[1])
        c_co, comp = harrell_counts(cj, Tj, Ej); c_co = c_co / comp if comp > 0 else np.nan
        c_wo, comp = harrell_counts(wj, Tj, Ej); c_wo = c_wo / comp if comp > 0 else np.nan
        rec.update(c_demo=c_demo, c_demo_primary=c_dc, c_demo_wrist=c_dw, c_demo_null=c_dn,
                   c_primary_only=c_co, c_wrist_only=c_wo,
                   dC_primary=c_dc - c_demo, dC_wrist=c_dw - c_demo, dC_null=c_dn - c_demo)

        # ---- AUROC@6yr (regularized logistic on flexible basis, shared-fold OOF) ----
        a_demo, a_dc, a_dw = crossfit_aurocs(
            [basis, np.column_stack([basis, cj]), np.column_stack([basis, wj])], Tj, Ej)
        a_co = auroc_at_horizon(cj, Tj, Ej, HORIZON)
        a_wo = auroc_at_horizon(wj, Tj, Ej, HORIZON)
        rec.update(auroc_demo=a_demo, auroc_demo_primary=a_dc, auroc_demo_wrist=a_dw,
                   auroc_primary_only=a_co, auroc_wrist_only=a_wo,
                   dA_primary=a_dc - a_demo, dA_wrist=a_dw - a_demo)
        rows.append(rec)

        if (i + 1) % 25 == 0 or (i + 1) == len(panel):
            dt = time.time() - t0
            print(f"[exp1] {i+1}/{len(panel)} phecodes  ({dt:.0f}s, {dt/(i+1):.2f}s/phe)", flush=True)
            pd.DataFrame(rows).to_csv(f"{out_dir}/exp1_per_disease.csv", index=False)  # partial flush

    df = pd.DataFrame(rows)
    df.to_csv(f"{out_dir}/exp1_per_disease.csv", index=False)

    summ = {"n_panel": len(panel), "horizon": HORIZON, "n_test": int(len(eids)),
            "covariate_coverage": cov}
    summ["mean_c"] = {k: float(df[f"c_{k}"].mean()) for k in
                      ["demo", "demo_primary", "demo_wrist", "demo_null", "primary_only", "wrist_only"]}
    summ["mean_auroc"] = {k: float(df[f"auroc_{k}"].mean()) for k in
                          ["demo", "demo_primary", "demo_wrist", "primary_only", "wrist_only"]}

    def delta(col, key, seed):
        d = df[col].to_numpy()
        mean, lo, hi, n = boot_mean_ci(d, seed=seed)
        return {"mean_delta": mean, "ci2.5": lo, "ci97.5": hi, "n_disease": n,
                "winrate": float(np.nanmean(d > 0)),
                "n_ge_0.02": int(np.nansum(d >= 0.02)), "n_ge_0.05": int(np.nansum(d >= 0.05))}
    summ["increment_over_flexible_demo"] = {
        "C_primary_hazard": delta("dC_primary", "c_primary", 1),
        "C_wrist_hazard": delta("dC_wrist", "c_wrist", 2),
        "C_null_permuted": delta("dC_null", "c_null", 3),
        "AUROC_primary_hazard": delta("dA_primary", "a_primary", 4),
        "AUROC_wrist_hazard": delta("dA_wrist", "a_wrist", 5),
    }
    # per-category mean delta (C, wrist hazard = cleanest "pure wrist beyond demographics")
    percat = {}
    for cat, g in df.groupby("category"):
        percat[cat] = {"n": int(len(g)),
                       "dC_wrist": float(g["dC_wrist"].mean()),
                       "dC_primary": float(g["dC_primary"].mean()),
                       "dC_null": float(g["dC_null"].mean()),
                       "dA_wrist": float(g["dA_wrist"].mean()),
                       "c_wrist_only": float(g["c_wrist_only"].mean())}
    summ["per_category"] = dict(sorted(percat.items(), key=lambda kv: -kv[1]["dC_wrist"]))
    json.dump(summ, open(f"{out_dir}/exp1_summary.json", "w"), indent=2)

    print("\n[exp1] mean over panel:", flush=True)
    print(f"   demo_flex(Cox)      C={summ['mean_c']['demo']:.4f}   AUROC={summ['mean_auroc']['demo']:.4f}", flush=True)
    print(f"   primary_hazard only   C={summ['mean_c']['primary_only']:.4f}   AUROC={summ['mean_auroc']['primary_only']:.4f}", flush=True)
    print(f"   wrist_hazard only   C={summ['mean_c']['wrist_only']:.4f}   AUROC={summ['mean_auroc']['wrist_only']:.4f}", flush=True)
    print("[exp1] increment of hazard OVER flexible age+sex+BMI (mean Δ [95% CI]):", flush=True)
    for k, v in summ["increment_over_flexible_demo"].items():
        print(f"   {k:22s}  {v['mean_delta']:+.4f} [{v['ci2.5']:+.4f},{v['ci97.5']:+.4f}]  "
              f"win {v['winrate']*100:.0f}%", flush=True)
    print("[exp1] per-category ΔC(wrist beyond demo), top & bottom:", flush=True)
    cats = list(summ["per_category"].items())
    for cat, v in cats[:6] + cats[-3:]:
        print(f"   {cat:20s} n={v['n']:3d}  ΔC_wrist={v['dC_wrist']:+.4f}  ΔC_null={v['dC_null']:+.4f}", flush=True)
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", choices=["1", "2", "both"], default="both")
    ap.add_argument("--out", required=True)
    ap.add_argument("--phecode-limit", type=int, default=None, help="smoke test: first N phecodes")
    args = ap.parse_args()
    out = resolve_oak_path(args.out)
    os.makedirs(out, exist_ok=True)
    print(f"[residual_partial] out={out}  exp={args.exp}  limit={args.phecode_limit}", flush=True)
    if args.exp in ("2", "both"):
        run_exp2(out, args.phecode_limit)
    if args.exp in ("1", "both"):
        run_exp1(out, args.phecode_limit)
    print("\n[residual_partial] DONE.", flush=True)


if __name__ == "__main__":
    main()
