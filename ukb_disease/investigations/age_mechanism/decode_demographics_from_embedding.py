#!/usr/bin/env python3
"""Decode age, sex and BMI from the frozen 512-d disease-prediction embedding, val-fit then test.

Companion to the probe: the probe reports the age decode (R^2~0.62) but only stores the summary.
This script reproduces that decode and additionally decodes sex (logistic to ROC) and BMI (ridge),
and saves the per-subject predictions so the Extended Data "why demographics add little" figure can
plot predicted-vs-true scatters and a sex ROC.

Same recipe as the probe: 512-d day-mean embedding (256 AR + 256 HA), fit on VAL, score TEST, ridge
alpha=10 for the continuous targets. Output schema mirrors the demographics-recovery npz so the
figure builder can reuse the same plotting.

Run:
  python3 -m ukb_disease.investigations.age_mechanism.decode_demographics_from_embedding
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score

from ukb_disease.paths import UKB_ROOT
from ukb_disease.baseline.cox_valfit import seed_avg, demo_for
from ukb_disease.baseline.config import resolve_oak_path
from ukb_disease.baseline import bmi_lib

BASE = str(UKB_ROOT)
OUT_NPZ = f"{UKB_ROOT}/benchmark/output/disjoint_split_v1/embedding_decode.npz"


def load_day_mean(split, eids):
    """512-d day-mean embedding (256 AR + 256 HA), averaged over valid days, aligned to eids.
    Matches the probe loader so the decode reproduces the probe number."""
    d = resolve_oak_path(f"{BASE}/day_mean_disjoint_split/{split}")
    idx = pd.read_parquet(f"{d}/index.parquet")
    ar = np.load(f"{d}/ar.npy", mmap_mode="r")
    ha = np.load(f"{d}/ha.npy", mmap_mode="r")
    msk = np.load(f"{d}/valid_day_mask.npy", mmap_mode="r")
    n = ar.shape[0]
    M = np.zeros((n, 512), dtype=np.float32)
    for i in range(n):
        v = np.asarray(msk[i]).astype(bool)
        if v.any():
            M[i, :256] = np.asarray(ar[i])[v].mean(0)
            M[i, 256:] = np.asarray(ha[i])[v].mean(0)
    df = pd.DataFrame(M, index=idx["eid"].to_numpy()).groupby(level=0).mean()
    return df.reindex(eids).to_numpy(np.float32)


def _metrics(true, pred):
    r = float(np.corrcoef(pred, true)[0, 1])
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return {"r": r, "r2": float(1 - ss_res / ss_tot), "mae": float(np.mean(np.abs(true - pred)))}


def decode_continuous(Xv, yv, Xt, yt, alpha=10.0):
    """Ridge val-fit -> test; return per-subject (true, pred) on finite test rows + metrics."""
    okv = np.isfinite(yv) & np.isfinite(Xv).all(1)
    okt = np.isfinite(yt) & np.isfinite(Xt).all(1)
    r = Ridge(alpha=alpha).fit(Xv[okv], yv[okv])
    pred = r.predict(Xt[okt])
    true = yt[okt]
    return true, pred, _metrics(true, pred)


def decode_sex(Xv, yv, Xt, yt):
    """Logistic val-fit -> test on standardised embedding; return ROC (fpr, tpr, auc)."""
    okv = np.isfinite(yv) & np.isfinite(Xv).all(1)
    okt = np.isfinite(yt) & np.isfinite(Xt).all(1)
    sc = StandardScaler().fit(Xv[okv])
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xv[okv]), yv[okv].astype(int))
    prob = clf.predict_proba(sc.transform(Xt[okt]))[:, 1]
    yb = yt[okt].astype(int)
    fpr, tpr, _ = roc_curve(yb, prob)
    return fpr, tpr, float(roc_auc_score(yb, prob))


def main():
    print("[load] val/test eids ...", flush=True)
    _, te, _, _, _, _ = seed_avg(f"{BASE}/runs_cox/embedding_wrist_only_disjoint_split_seed*", "test")
    _, ve, _, _, _, _ = seed_avg(f"{BASE}/runs_cox/embedding_wrist_only_disjoint_split_seed*", "val")

    print("[load] 512-d day-mean embeddings ...", flush=True)
    emb_v = load_day_mean("val", ve)
    emb_t = load_day_mean("test", te)

    vdemo = demo_for(ve); tdemo = demo_for(te)
    vage = vdemo["age"].to_numpy(float); vsex = vdemo["sex"].to_numpy(float)
    tage = tdemo["age"].to_numpy(float); tsex = tdemo["sex"].to_numpy(float)
    vbmi, _ = bmi_lib.load_bmi(ve); tbmi, _ = bmi_lib.load_bmi(te)

    age_true, age_pred, age_m = decode_continuous(emb_v, vage, emb_t, tage)
    bmi_true, bmi_pred, bmi_m = decode_continuous(emb_v, vbmi, emb_t, tbmi)
    sex_fpr, sex_tpr, sex_auc = decode_sex(emb_v, vsex, emb_t, tsex)

    print(f"[age] r={age_m['r']:.3f}  R2={age_m['r2']:.3f}  MAE={age_m['mae']:.2f}  (probe anchor R2~0.62)")
    print(f"[bmi] r={bmi_m['r']:.3f}  R2={bmi_m['r2']:.3f}  MAE={bmi_m['mae']:.2f}  (reported R2~0.48)")
    print(f"[sex] AUROC={sex_auc:.3f}  (reported ~0.99)")

    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez(
        OUT_NPZ,
        age_pred=age_pred, age_true=age_true,
        age_r=age_m["r"], age_r2=age_m["r2"], age_mae=age_m["mae"],
        bmi_pred=bmi_pred, bmi_true=bmi_true,
        bmi_r=bmi_m["r"], bmi_r2=bmi_m["r2"], bmi_mae=bmi_m["mae"],
        sex_fpr=sex_fpr, sex_tpr=sex_tpr, sex_auc=sex_auc,
        n_age=len(age_true), n_bmi=len(bmi_true),
    )
    print(f"[save] {OUT_NPZ}")


if __name__ == "__main__":
    main()
