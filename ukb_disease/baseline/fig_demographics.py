"""Precompute predicted-vs-true arrays for the demographics-recovery ED figure
(sex ROC, age scatter, BMI scatter). Writes a single npz.

- AGE: the calibrated age model (circ4_ccc_l1), per-subject, in years.
- BMI: the BMI candidate (l3/both/ridge/log/alpha=10), train-fit then test (reuses bmi_lib/bmi_probe).
- SEX: a logistic on the same l3/both wrist features (train-fit then test), same cohort as BMI.

Run: python3 -m ukb_disease.baseline.fig_demographics
"""
from __future__ import annotations
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from ukb_disease.baseline import bmi_lib as L
from ukb_disease.baseline import bmi_probe as P
from ukb_disease.paths import UKB_ROOT

OUT = f"{UKB_ROOT}/benchmark/output/disjoint_split_v1/demographics_recovery.npz"
AGE = f"{UKB_ROOT}/age_campaign/test/test_circ4_ccc_l1_s42"


def _metrics(pred, true):
    r = float(np.corrcoef(pred, true)[0, 1])
    return r, r * r, float(np.mean(np.abs(pred - true)))


def main():
    # AGE: aggregate per subject (years)
    ap = np.load(f"{AGE}/test_predictions.npy").astype(float)
    at = np.load(f"{AGE}/test_age.npy").astype(float)
    try:
        ae = np.load(f"{AGE}/test_eids.npy")
        import pandas as pd
        g = pd.DataFrame({"eid": ae, "p": ap, "t": at}).groupby("eid").mean()
        age_pred, age_true = g["p"].values, g["t"].values
    except FileNotFoundError:
        age_pred, age_true = ap, at
    age_r, age_r2, age_mae = _metrics(age_pred, age_true)

    # BMI + SEX on the same l3/both wrist features (train-fit then test)
    tr = L.load_dataset("train", "l3", "both")
    te = L.load_dataset("test", "l3", "both", allow_test=True)
    sc = StandardScaler().fit(tr["X"])
    Xtr, Xte = sc.transform(tr["X"]), sc.transform(te["X"])

    fwd, inv = P._make_transform("log", tr["bmi"])
    head = P._make_head("ridge", 10.0, 0)
    head.fit(Xtr, fwd(tr["bmi"]))
    bmi_pred, bmi_true = inv(head.predict(Xte)), te["bmi"]
    bmi_r, bmi_r2, bmi_mae = _metrics(bmi_pred, bmi_true)

    clf = LogisticRegression(max_iter=3000, C=1.0).fit(Xtr, tr["sex"].astype(int))
    sex_prob, sex_true = clf.predict_proba(Xte)[:, 1], te["sex"].astype(int)
    sex_auc = float(roc_auc_score(sex_true, sex_prob))
    fpr, tpr, _ = roc_curve(sex_true, sex_prob)

    np.savez(OUT,
             age_pred=age_pred, age_true=age_true, age_r=age_r, age_r2=age_r2, age_mae=age_mae,
             bmi_pred=bmi_pred, bmi_true=bmi_true, bmi_r=bmi_r, bmi_r2=bmi_r2, bmi_mae=bmi_mae,
             sex_fpr=fpr, sex_tpr=tpr, sex_auc=sex_auc)
    print(f"AGE n={len(age_pred):5d}  r={age_r:.4f}  r2={age_r2:.4f}  MAE={age_mae:.2f}")
    print(f"BMI n={len(bmi_pred):5d}  r={bmi_r:.4f}  r2={bmi_r2:.4f}  MAE={bmi_mae:.2f}")
    print(f"SEX n={len(sex_true):5d}  AUROC={sex_auc:.4f}")
    print(f"wrote -> {OUT}")


if __name__ == "__main__":
    main()
