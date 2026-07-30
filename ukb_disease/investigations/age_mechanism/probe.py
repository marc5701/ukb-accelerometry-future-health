#!/usr/bin/env python3
"""Age-mechanism probe: why age/sex adds only about +0.0045 to the deep disease
model, and why the wrist-derived age estimate (predicted age / ar_emb) does not help.

Every "does X help" is a val-fit (fit a per-phecode ridge Cox on the held-out
validation set, score on test) or a val->test decode. Reuses the canonical Harrell
C and the ridge-Cox fitter rather than re-implementing the scoring.

Tiers:
  T0  calibrate: age dist, standalone baselines, bootstrap CI on the increment, redundancy fraction
  T1  why age is small: age decodability (embedding and hazard), per-disease structure, linear age increment
  T2  why ar_emb is null: predicted-age increment, age-gap test, ar_emb redundancy, prior cross-check
  (T3 = GPU age-only/sex-only retrains, handled outside this script)
  T4  synthesis numbers
"""
import os, sys, json, glob
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
from ukb_disease.benchmark.concordance_benchmark import harrell_counts
from ukb_disease.baseline.cox_valfit import _fit_apply, demo_for, seed_avg
from ukb_disease.baseline.config import resolve_oak_path
from ukb_disease.paths import UKB_ROOT

BASE = str(UKB_ROOT)
OUT = str(Path(__file__).resolve().parent / "output")
os.makedirs(OUT, exist_ok=True)
AGE_TEST_DIR = f"{BASE}/age_sex_disjoint_split/sequence_model_age"      # test_{eids,age,aee,ar_emb}.npy (7416 superset)
AGE_VAL_DIR  = f"{BASE}/age_sex_val_pred/sequence_model_age_val"      # val ar_emb/age/aee, stored as test_*.npy
RES = {}

def jdump():
    json.dump(RES, open(f"{OUT}/probe_results.json", "w"), indent=2, default=float)

# per-phecode Harrell C helpers
def per_phecode_c(haz, T, E, M, panel):
    out = {}
    for j in panel:
        m = M[:, j]
        c, comp = harrell_counts(haz[m, j], T[m, j].astype(float), E[m, j].astype(float))
        out[j] = (c / comp) if comp > 0 else np.nan
    return out

def mean_c(haz, T, E, M, panel):
    v = np.array([per_phecode_c(haz, T, E, M, [j])[j] for j in panel])
    return np.nanmean(v), v

# load TEST hazards (seed-averaged)
print("[load] hazards ...", flush=True)
nd_h, te, tT, tE, tM, _      = seed_avg(f"{BASE}/runs_cox/embedding_wrist_only_disjoint_split_seed*", "test")
wd_h, te2, _, _, _, tdirs    = seed_avg(f"{BASE}/runs_cox/embedding_disjoint_split_seed*", "test")
assert np.array_equal(te, te2), "no-demo/with-demo test eids differ"
names = json.load(open(f"{tdirs[0]}/config.json"))["phecodes"]
P = nd_h.shape[1]

# canonical test panel: >=10 evaluable AND >=1 event (reproduces the benchmark headline panel)
evaln = tM.sum(0); evn = (tE * tM).sum(0)
panel = [j for j in range(P) if evaln[j] >= 10 and evn[j] >= 1]
ndc = per_phecode_c(nd_h, tT, tE, tM, panel)
wdc = per_phecode_c(wd_h, tT, tE, tM, panel)
nd_mean = np.nanmean(list(ndc.values())); wd_mean = np.nanmean(list(wdc.values()))
delta = wd_mean - nd_mean
RES["anchor"] = {"n_panel": len(panel), "nodemo_mean_c": nd_mean, "withdemo_mean_c": wd_mean,
                 "age_sex_delta": delta}
print(f"[ANCHOR] panel={len(panel)}  no-demo={nd_mean:.4f}  with-demo={wd_mean:.4f}  Δ(age/sex)={delta:+.4f}")
jdump()

# covariates aligned to TEST eids
tdemo = demo_for(te)
tage = tdemo["age"].to_numpy(float); tsex = tdemo["sex"].to_numpy(float)

def age_artifact(d, eids):
    dd = resolve_oak_path(d)
    df = pd.DataFrame({"eid": np.load(f"{dd}/test_eids.npy"),
                       "age": np.load(f"{dd}/test_age.npy"),
                       "aee": np.load(f"{dd}/test_aee.npy"),
                       "ar_emb": np.load(f"{dd}/test_ar_emb.npy")}).groupby("eid").mean()
    df = df.reindex(eids)
    predage = (df["age"] + df["aee"]).to_numpy(float)
    return df["ar_emb"].to_numpy(float), predage, df["age"].to_numpy(float)
test_ar_emb, tpredage, tart_age = age_artifact(AGE_TEST_DIR, te)
print(f"[check] corr(acc_dem age, artifact age) = {np.corrcoef(tage, np.nan_to_num(tart_age, nan=np.nanmean(tart_age)))[0,1]:.4f}")

# ================= TIER 0 =================
# T0.1 age distribution
fa = np.isfinite(tage)
RES["T0.1_age_dist"] = {"mean": float(np.mean(tage[fa])), "sd": float(np.std(tage[fa])),
                        "min": float(np.min(tage[fa])), "p1": float(np.percentile(tage[fa],1)),
                        "p99": float(np.percentile(tage[fa],99)), "max": float(np.max(tage[fa])),
                        "frac_female": float(np.nanmean(tsex)),
                        "note": "this cohort spans a narrow adult age range (p1-p99 about 30y)"}

# T0.2 standalone baselines (1-covariate val-fit so each covariate is oriented per phecode).
#      Also raw age-only Harrell C (rank by age, higher=older=risk).
def std_age_only_c():
    v = []
    for j in panel:
        m = tM[:, j]
        c, comp = harrell_counts(tage[m], tT[m, j].astype(float), tE[m, j].astype(float))
        if comp > 0: v.append(c/comp)
    return float(np.nanmean(v))
RES["T0.2_raw_age_only_meanC"] = std_age_only_c()
print(f"[T0.2] raw age-only mean-C (rank by age) = {RES['T0.2_raw_age_only_meanC']:.4f}")

# load VAL hazards (no-demo + with-demo)
print("[load] val hazards ...", flush=True)
ndv_h, ve, vT, vE, vM, _ = seed_avg(f"{BASE}/runs_cox/embedding_wrist_only_disjoint_split_seed*", "val")
wdv_h, ve2, vT2, vE2, vM2, _ = seed_avg(f"{BASE}/runs_cox_val_pred/embedding_val_pred_seed*", "val")
assert np.array_equal(ve, ve2), "no-demo/with-demo VAL eids differ"
vdemo = demo_for(ve); vage = vdemo["age"].to_numpy(float); vsex = vdemo["sex"].to_numpy(float)
val_ar_emb, vpredage, _ = age_artifact(AGE_VAL_DIR, ve)

# val-fit panel: >=10 events in BOTH val and test (cox_valfit convention)
evv = (vE * vM).sum(0)
panel_vf = [j for j in range(P) if evn[j] >= 10 and evv[j] >= 10]
print(f"[val-fit] panel = {len(panel_vf)} phecodes (>=10 events in val AND test)")

def valfit_meanc(vh, th, vcovs, tcovs, pen_c0=10.0):
    """val-fit [hazard, *covs]; return (mean test-C over panel_vf, per-phecode array)."""
    cs = []
    for j in panel_vf:
        mv = vM[:, j]; mt = tM[:, j]
        Tv = vT[mv, j].astype(float); Ev = vE[mv, j].astype(float)
        Tt = tT[mt, j].astype(float); Et = tE[mt, j].astype(float)
        pen = max(0.1, pen_c0 / max(1, int(Ev.sum())))
        Xv = np.column_stack([vh[mv, j]] + [c[mv] for c in vcovs])
        Xt = np.column_stack([th[mt, j]] + [c[mt] for c in tcovs])
        try:
            lp = _fit_apply(Xv, Tv, Ev, Xt, pen)
            c, comp = harrell_counts(lp, Tt, Et)
            cs.append((c/comp) if comp > 0 else np.nan)
        except Exception:
            cs.append(np.nan)
    return float(np.nanmean(cs)), np.array(cs)

# 1-covariate val-fit standalone baselines (no hazard); oriented mean-C
def valfit_standalone(vx, tx, pen_c0=10.0):
    cs = []
    for j in panel_vf:
        mv = vM[:, j]; mt = tM[:, j]
        Tv = vT[mv, j].astype(float); Ev = vE[mv, j].astype(float)
        Tt = tT[mt, j].astype(float); Et = tE[mt, j].astype(float)
        pen = max(0.1, pen_c0 / max(1, int(Ev.sum())))
        try:
            lp = _fit_apply(vx[mv][:, None], Tv, Ev, tx[mt][:, None], pen)
            c, comp = harrell_counts(lp, Tt, Et)
            cs.append((c/comp) if comp > 0 else np.nan)
        except Exception:
            cs.append(np.nan)
    return float(np.nanmean(cs))
RES["T0.2_standalone_valfit"] = {
    "age":     valfit_standalone(vage, tage),
    "sex":     valfit_standalone(vsex, tsex),
    "predage": valfit_standalone(vpredage, tpredage),
    "ar_emb":    valfit_standalone(val_ar_emb, test_ar_emb),
}
print(f"[T0.2] standalone val-fit mean-C: {RES['T0.2_standalone_valfit']}")
jdump()

# ================= TIER 1 / TIER 2: val-fit battery =================
def base_meanc(vh, th):
    return valfit_meanc(vh, th, [], [])
arms = {}
# no-demo base (hazard has NO age inside)
b_nd, _ = base_meanc(ndv_h, nd_h); arms["nodemo_base"] = b_nd
arms["nodemo_+age"]      = valfit_meanc(ndv_h, nd_h, [vage], [tage])[0]
arms["nodemo_+sex"]      = valfit_meanc(ndv_h, nd_h, [vsex], [tsex])[0]
arms["nodemo_+age+sex"]  = valfit_meanc(ndv_h, nd_h, [vage, vsex], [tage, tsex])[0]
arms["nodemo_+predage"]  = valfit_meanc(ndv_h, nd_h, [vpredage], [tpredage])[0]
arms["nodemo_+age+ar_emb"] = valfit_meanc(ndv_h, nd_h, [vage, val_ar_emb], [tage, test_ar_emb])[0]
arms["nodemo_+ar_emb"]     = valfit_meanc(ndv_h, nd_h, [val_ar_emb], [test_ar_emb])[0]
# with-demo base (cross-check against the prior result)
b_wd, _ = base_meanc(wdv_h, wd_h); arms["withdemo_base"] = b_wd
arms["withdemo_+age+sex"]      = valfit_meanc(wdv_h, wd_h, [vage, vsex], [tage, tsex])[0]
arms["withdemo_+age+sex+ar_emb"] = valfit_meanc(wdv_h, wd_h, [vage, vsex, val_ar_emb], [tage, tsex, test_ar_emb])[0]
RES["valfit_arms"] = arms
RES["valfit_deltas"] = {
    "nodemo_age_increment":      arms["nodemo_+age"]      - arms["nodemo_base"],
    "nodemo_sex_increment":      arms["nodemo_+sex"]      - arms["nodemo_base"],
    "nodemo_agesex_increment":   arms["nodemo_+age+sex"]  - arms["nodemo_base"],
    "nodemo_predage_increment":  arms["nodemo_+predage"]  - arms["nodemo_base"],
    "nodemo_ar_emb_over_age":      arms["nodemo_+age+ar_emb"] - arms["nodemo_+age"],
    "withdemo_agesex_increment": arms["withdemo_+age+sex"] - arms["withdemo_base"],
    "withdemo_ar_emb_over_agesex": arms["withdemo_+age+sex+ar_emb"] - arms["withdemo_+age+sex"],
}
print("[val-fit arms]"); [print(f"   {k:26s} {v:.4f}") for k, v in arms.items()]
print("[val-fit deltas]"); [print(f"   {k:26s} {v:+.4f}") for k, v in RES["valfit_deltas"].items()]
jdump()

# ================= TIER 1.1 / 2.3: decodability =================
def load_day_mean(split, eids):
    d = resolve_oak_path(f"{BASE}/day_mean_disjoint_split/{split}")
    idx = pd.read_parquet(f"{d}/index.parquet")
    ar = np.load(f"{d}/ar.npy", mmap_mode="r"); ha = np.load(f"{d}/ha.npy", mmap_mode="r")
    msk = np.load(f"{d}/valid_day_mask.npy", mmap_mode="r")
    n = ar.shape[0]; M = np.zeros((n, 512), dtype=np.float32)
    for i in range(n):
        v = np.asarray(msk[i]).astype(bool)
        if v.any():
            M[i, :256] = np.asarray(ar[i])[v].mean(0); M[i, 256:] = np.asarray(ha[i])[v].mean(0)
    df = pd.DataFrame(M, index=idx["eid"].to_numpy()).groupby(level=0).mean()
    return df.reindex(eids).to_numpy(np.float32)

def decode(Xv, yv, Xt, yt, alpha=10.0):
    okv = np.isfinite(yv) & np.isfinite(Xv).all(1); okt = np.isfinite(yt) & np.isfinite(Xt).all(1)
    r = Ridge(alpha=alpha).fit(Xv[okv], yv[okv]); pred = r.predict(Xt[okt])
    ss_res = np.sum((yt[okt]-pred)**2); ss_tot = np.sum((yt[okt]-yt[okt].mean())**2)
    return {"R2": float(1-ss_res/ss_tot), "MAE": float(np.mean(np.abs(yt[okt]-pred))),
            "corr": float(np.corrcoef(pred, yt[okt])[0,1])}

print("[load] day_mean embeddings ...", flush=True)
emb_v = load_day_mean("val", ve); emb_t = load_day_mean("test", te)
RES["T1.1_age_decode"] = {
    "from_embedding_512d": decode(emb_v, vage, emb_t, tage),
    "from_nodemo_hazard390": decode(ndv_h, vage, nd_h, tage),
    "from_withdemo_hazard390": decode(wdv_h, vage, wd_h, tage),
    "note": "fit on VAL, scored on TEST. Age-clock upper bound corr~0.80/R2~0.64.",
}
RES["T2.3_ar_emb_decode"] = {
    "ar_emb_from_embedding_512d": decode(emb_v, val_ar_emb, emb_t, test_ar_emb),
    "ar_emb_from_nodemo_hazard390": decode(ndv_h, val_ar_emb, nd_h, test_ar_emb),
    "predage_from_embedding_512d": decode(emb_v, vpredage, emb_t, tpredage),
}
print(f"[T1.1] age decode: {RES['T1.1_age_decode']}")
print(f"[T2.3] ar_emb decode: {RES['T2.3_ar_emb_decode']}")
jdump()

# ================= TIER 1.2: per-disease structure =================
rows = []
for j in panel:
    if np.isfinite(ndc[j]) and np.isfinite(wdc[j]):
        rows.append({"phecode": names[j], "n_event": int(evn[j]),
                     "nodemo_c": ndc[j], "withdemo_c": wdc[j], "age_delta": wdc[j]-ndc[j],
                     "age_only_c": harrell_counts(tage[tM[:,j]], tT[tM[:,j],j].astype(float),
                                                  tE[tM[:,j],j].astype(float))[0] /
                                   max(1, harrell_counts(tage[tM[:,j]], tT[tM[:,j],j].astype(float),
                                                  tE[tM[:,j],j].astype(float))[1])})
pdf = pd.DataFrame(rows); pdf.to_csv(f"{OUT}/per_disease_age_contribution.csv", index=False)
def sp(a, b):
    ok = np.isfinite(pdf[a]) & np.isfinite(pdf[b])
    r, p = spearmanr(pdf.loc[ok, a], pdf.loc[ok, b]); return {"rho": float(r), "p": float(p)}
RES["T1.2_per_disease"] = {
    "n_diseases": int(len(pdf)),
    "age_delta_vs_age_only_c": sp("age_delta", "age_only_c"),
    "age_delta_vs_nodemo_c": sp("age_delta", "nodemo_c"),
    "age_delta_vs_n_event": sp("age_delta", "n_event"),
    "frac_age_helps": float((pdf["age_delta"] > 0).mean()),
}
print(f"[T1.2] per-disease: {RES['T1.2_per_disease']}")
jdump()

# ================= TIER 0.3: subject-cluster bootstrap CI on the +0.0045 =================
print("[boot] subject bootstrap on Δ(age/sex) ...", flush=True)
rng = np.random.default_rng(0); N = len(te); n_boot = 200
# precompute per-phecode test arrays
cols_j = {j: (np.where(tM[:, j])[0]) for j in panel}
deltas = []
for b in range(n_boot):
    samp = rng.integers(0, N, N)
    in_samp = np.zeros(N, dtype=bool);
    # use bincount multiplicity for weighting via repeated indices -> rebuild per-phecode
    mult = np.bincount(samp, minlength=N)
    dd = []
    for j in panel:
        idx = cols_j[j]; w = mult[idx]
        keep = w > 0
        if keep.sum() < 3: continue
        ii = idx[keep]; wj = w[keep]
        T = tT[ii, j].astype(float); E = tE[ii, j].astype(float)
        # expand by multiplicity (cheap; panels small)
        rep = np.repeat(np.arange(len(ii)), wj)
        Tr, Er = T[rep], E[rep]
        cn, cmpn = harrell_counts(nd_h[ii, j][rep], Tr, Er)
        cw, cmpw = harrell_counts(wd_h[ii, j][rep], Tr, Er)
        if cmpn > 0 and cmpw > 0:
            dd.append((cw/cmpw) - (cn/cmpn))
    if dd: deltas.append(np.mean(dd))
deltas = np.array(deltas)
RES["T0.3_bootstrap_delta"] = {"n_boot": int(len(deltas)), "mean": float(deltas.mean()),
                               "ci2.5": float(np.percentile(deltas, 2.5)),
                               "ci97.5": float(np.percentile(deltas, 97.5)),
                               "frac_positive": float((deltas > 0).mean())}
print(f"[T0.3] Δ(age/sex) bootstrap: {RES['T0.3_bootstrap_delta']}")

# T0.4 redundancy fraction
std_age = RES["T0.2_standalone_valfit"]["age"]
RES["T0.4_redundancy"] = {
    "standalone_age_over_chance": std_age - 0.5,
    "incremental_agesex_over_nodemo": RES["valfit_deltas"]["nodemo_agesex_increment"],
    "deep_agesex_increment": delta,
    "redundancy_fraction_vs_standalone": float(1 - max(0.0, delta) / max(1e-9, std_age - 0.5)),
}
print(f"[T0.4] redundancy: {RES['T0.4_redundancy']}")
jdump()
print("\n[DONE] results -> " + f"{OUT}/probe_results.json")
