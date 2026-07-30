#!/usr/bin/env python3
"""Clinical baseline comparison: per-disease 6-yr AUROC, three arms.

  A. clinical-only        [age, sex, BMI]              (clinical baseline)
  B. embedding-only       deep embedding per-disease hazard  (the FM model)
  C. clinical + embedding [embedding hazard, age, sex, BMI]  (combined arm)

Follows the standard per-disease AUROC comparison (clinical vs clinical plus
embedding). A and C are per-phecode ridge-Cox fit on the held-out VALIDATION set
and scored on TEST; B is the deep model's test hazard (train-fit deep, scored on
test). Reuses cox_valfit._fit_apply, per_disease_eval.auroc_at_horizon,
concordance_benchmark.harrell_counts.

Clinical covariates available: age (acc_dem), sex (acc_dem), BMI (BMI_yes.csv,
field 21001, ~100%% coverage). Smoking, alcohol, and ethnicity are not in the
reachable data yet; the loader is structured to add them via extra columns later.
"""
import os, json
import numpy as np, pandas as pd
from ukb_disease.paths import UKB_ROOT
from ukb_disease.baseline.cox_valfit import _fit_apply, seed_avg
from ukb_disease.baseline.per_disease_eval import auroc_at_horizon
from ukb_disease.benchmark.concordance_benchmark import harrell_counts
from ukb_disease.baseline.config import resolve_oak_path

BASE = f"{UKB_ROOT}"
DEM = f"{UKB_ROOT}/metadata/acc_dem.csv"
BMI = f"{UKB_ROOT}/metadata/BMI_yes.csv"
OUT = f"{UKB_ROOT}/figures_out/clinical_baseline"
os.makedirs(OUT, exist_ok=True)
HORIZON = 6.0
PEN_C0 = 10.0

def load_clinical(eids):
    """Return (n, 3) [age_z, sex(0/1), bmi_z] aligned to eids; NaN->mean impute, continuous z-scored."""
    dem = pd.read_csv(resolve_oak_path(DEM)).set_index("eid")
    bmi = pd.read_csv(resolve_oak_path(BMI), usecols=["eid", "21001-0.0"]).rename(
        columns={"21001-0.0": "bmi"}).groupby("eid").mean()
    age = dem.reindex(eids)["age_at_first_run"].to_numpy(float)
    sex = dem.reindex(eids)["sex"].to_numpy(float)
    bm = bmi.reindex(eids)["bmi"].to_numpy(float)
    def z(x):
        x = np.where(np.isfinite(x), x, np.nanmean(x))
        return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)
    sex = np.where(np.isfinite(sex), sex, np.nanmean(sex))
    return np.column_stack([z(age), sex, z(bm)]), {"bmi_cov": float(np.isfinite(bm).mean())}

def main():
    print("[clin] loading embedding test + val hazards/labels ...", flush=True)
    th, te, tT, tE, tM, tdirs = seed_avg(f"{BASE}/runs_cox/embedding_disjoint_split_seed*", "test")
    vh, ve, vT, vE, vM, _ = seed_avg(f"{BASE}/runs_cox_val_pred/embedding_val_pred_seed*", "val")
    names = json.load(open(f"{tdirs[0]}/config.json"))["phecodes"]
    P = th.shape[1]
    clin_t, covmeta = load_clinical(te)
    clin_v, _ = load_clinical(ve)
    print(f"[clin] N_test={len(te)} N_val={len(ve)} BMI coverage(test)={covmeta['bmi_cov']*100:.1f}%")

    evaln = tM.sum(0); evn = (tE * tM).sum(0)
    panel = [j for j in range(P) if evaln[j] >= 10 and evn[j] >= 1]
    print(f"[clin] panel = {len(panel)} phecodes")

    rows = []
    for j in panel:
        mt, mv = tM[:, j], vM[:, j]
        Tt, Et = tT[mt, j].astype(float), tE[mt, j].astype(float)
        Tv, Ev = vT[mv, j].astype(float), vE[mv, j].astype(float)
        pen = max(0.1, PEN_C0 / max(1, int(Ev.sum())))
        def score(risk):
            return (auroc_at_horizon(risk, Tt, Et, HORIZON),
                    (lambda c, n: c / n if n > 0 else np.nan)(*harrell_counts(risk, Tt, Et)))
        try:    rA = _fit_apply(clin_v[mv], Tv, Ev, clin_t[mt], pen)
        except Exception: rA = np.full(mt.sum(), np.nan)
        rB = th[mt, j]
        try:    rC = _fit_apply(np.column_stack([vh[mv, j], clin_v[mv]]), Tv, Ev,
                                np.column_stack([th[mt, j], clin_t[mt]]), pen)
        except Exception: rC = np.full(mt.sum(), np.nan)
        aA, cA = score(rA); aB, cB = score(rB); aC, cC = score(rC)
        rows.append({"phecode": names[j], "category": names[j].split("_")[0],
                     "n_event": int(evn[j]),
                     "auroc_clinical": aA, "auroc_embedding": aB, "auroc_clin_emb": aC,
                     "c_clinical": cA, "c_embedding": cB, "c_clin_emb": cC})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/per_disease_3arm.csv", index=False)

    def m(col): return float(np.nanmean(df[col]))
    ok = df[["auroc_clinical", "auroc_embedding", "auroc_clin_emb"]].notna().all(1)
    d = df[ok]
    summ = {
        "n_panel": len(df), "n_panel_auroc": int(ok.sum()), "horizon_years": HORIZON,
        "clinical_covariates": ["age", "sex", "BMI"], "bmi_coverage_test": covmeta["bmi_cov"],
        "mean_auroc": {"clinical": m("auroc_clinical"), "embedding": m("auroc_embedding"),
                       "clin_emb": m("auroc_clin_emb")},
        "mean_c": {"clinical": m("c_clinical"), "embedding": m("c_embedding"),
                   "clin_emb": m("c_clin_emb")},
        # Does the wrist embedding alone beat the clinical workup?
        "embedding_vs_clinical": {
            "mean_delta_auroc": float((d["auroc_embedding"] - d["auroc_clinical"]).mean()),
            "winrate_embedding": float((d["auroc_embedding"] > d["auroc_clinical"]).mean()),
            "n_diseases_emb_ge_clin": int((d["auroc_embedding"] >= d["auroc_clinical"]).sum()),
            "n_total": int(len(d))},
        # Incremental value of clinical covariates on top of the embedding.
        "clinical_increment_over_embedding": {
            "mean_delta_auroc": float((d["auroc_clin_emb"] - d["auroc_embedding"]).mean()),
            "winrate_clin_helps": float((d["auroc_clin_emb"] > d["auroc_embedding"]).mean()),
            "n_improved_ge_0.02": int((d["auroc_clin_emb"] - d["auroc_embedding"] >= 0.02).sum())},
        # Incremental value of the embedding on top of clinical covariates.
        "embedding_increment_over_clinical": {
            "mean_delta_auroc": float((d["auroc_clin_emb"] - d["auroc_clinical"]).mean()),
            "n_improved_ge_0.02": int((d["auroc_clin_emb"] - d["auroc_clinical"] >= 0.02).sum()),
            "n_improved_ge_0.05": int((d["auroc_clin_emb"] - d["auroc_clinical"] >= 0.05).sum())},
    }
    # per-category mean AUROC
    cat = d.groupby("category")[["auroc_clinical", "auroc_embedding", "auroc_clin_emb"]].mean()
    cat["n"] = d.groupby("category").size()
    summ["per_category_auroc"] = cat.round(4).reset_index().to_dict(orient="records")
    json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=2, default=float)

    print("\n[clin] ===== 3-ARM per-disease AUROC (6-yr incident, honest) =====")
    print(f"  mean AUROC: clinical {summ['mean_auroc']['clinical']:.4f} | "
          f"embedding {summ['mean_auroc']['embedding']:.4f} | clin+emb {summ['mean_auroc']['clin_emb']:.4f}")
    print(f"  mean C    : clinical {summ['mean_c']['clinical']:.4f} | "
          f"embedding {summ['mean_c']['embedding']:.4f} | clin+emb {summ['mean_c']['clin_emb']:.4f}")
    h1 = summ["embedding_vs_clinical"]
    print(f"  [H1 wrist-vs-clinic] embedding-only beats clinical on {h1['winrate_embedding']*100:.0f}% "
          f"of diseases ({h1['n_diseases_emb_ge_clin']}/{h1['n_total']} ≥), mean ΔAUROC {h1['mean_delta_auroc']:+.4f}")
    h2 = summ["clinical_increment_over_embedding"]
    print(f"  [H2 clinical adds?] clinical on top of embedding: mean ΔAUROC {h2['mean_delta_auroc']:+.4f}, "
          f"helps {h2['winrate_clin_helps']*100:.0f}%, ≥0.02 on {h2['n_improved_ge_0.02']} diseases")
    he = summ["embedding_increment_over_clinical"]
    print(f"  [Oxford framing] embedding on top of clinical: mean ΔAUROC {he['mean_delta_auroc']:+.4f}, "
          f"≥0.02 on {he['n_improved_ge_0.02']}, ≥0.05 on {he['n_improved_ge_0.05']}")
    print(f"[clin] wrote {OUT}/per_disease_3arm.csv + summary.json")

if __name__ == "__main__":
    main()
