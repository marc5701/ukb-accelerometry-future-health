"""Confound audit for the same-distribution prevalent-disease screener.

Question: is the wrist-only prevalent-PD AUROC physiological, or a
data-collection artifact (wear-time, nonwear/missingness, recording date /
device batch)? We answer it three ways, all built on a single out-of-fold (OOF)
prediction workhorse whose control geometry matches the same_dist_cv
run (~8000 pooled eval controls, train controls capped at 30000), so the OOF
pooled AUROC for the embedding reproduces the headline.

  View 1  Confound-only vs embedding vs combined detector. The same OOF logistic
          (val-selected C, per-fold standardization) is fit on three feature
          blocks: the embedding, the confound block, and their concatenation.
          If confound-only AUROC is near chance and combined ~= embedding, the
          signal is physiological, not an artifact.
  View 2  Direct association. SMD and single-feature AUROC of each confound vs
          the prevalent cases, characterizing the confound magnitude regardless
          of any detector.
  View 3  Within-wear-stratum AUROC of the embedding. Holding wear-time roughly
          fixed (tertiles of mean valid-days), does the embedding still separate
          prevalent cases? If it were just wear-time, the AUROC would collapse.

Confound block (all non-physiological, full coverage):
  nvd_sum, nvd_mean, nvd_max : wear-time (valid days, summed/mean/max over recs)
  n_rec                      : number of recordings
  cal_year, cal_sin, cal_cos : recording calendar (device-batch / season)

The OOF predictions for the embedding are also dumped per subject so the
near-onset sensitivity stage can reuse them without refitting.

Run:
  python3 -u -m ukb_disease.screening.confound_audit \
      --out-dir confound_audit_out
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from ukb_disease.paths import UKB_ROOT
from ukb_disease.baseline.config import (
    SEVEN_DAY_YEARS,
    CANONICAL_DAY_LMOMENT_ROOT,
    resolve_oak_path,
)
from ukb_disease.baseline.make_cv_folds import fold_of_eid
from ukb_disease.baseline.outcomes_processing import load_phecodes
from ukb_disease.screening.run_screening import load_demographics

POOL = f"{UKB_ROOT}/screening_prevalent_v1/pooled.npz"
DENSE = f"{UKB_ROOT}/dense_baseline_panel_nm/all.parquet"

# The motor cluster where wrist genuinely beats the demographic baseline, plus a
# roughly matched neuro target and two negative controls (where demographics win,
# so a real confound there would be age/sex, not wear/device).
DEFAULT_DISEASES = [
    "NS_324.11",   # Parkinson's; primary case (embedding beats demographics)
    "NS_326.1",    # multiple sclerosis
    "NS_324.341",  # extrapyramidal / dystonia
    "NS_356.2",    # neuropathy
    "NS_325",      # abnormality of gait; embedding roughly matches demographics
    "NS_333.1",    # essential tremor; demographics win
    "NS_328.1",    # dementia; demographics win
]


# --------------------------------------------------------------------------- features
def build_confound_matrix(eids: np.ndarray):
    """Per-subject non-physiological confound block aligned to `eids`.

    Returns (C float32 (n, k), colnames). Wear-time + recording-count from the
    canonical day cache index; calendar (year, season) from first_day_code.
    """
    root = resolve_oak_path(CANONICAL_DAY_LMOMENT_ROOT)
    recs = []
    for sp in ("train", "val", "test"):
        ix = pd.read_parquet(f"{root}/{sp}/index.parquet")
        recs.append(ix[["eid", "n_valid_days", "first_day_code"]])
    rec = pd.concat(recs, ignore_index=True)
    g = rec.groupby("eid")
    conf = pd.DataFrame({
        "nvd_sum": g["n_valid_days"].sum(),
        "nvd_mean": g["n_valid_days"].mean(),
        "nvd_max": g["n_valid_days"].max(),
        "n_rec": g.size().astype(float),
        "fdc": g["first_day_code"].min(),
    })
    dt = pd.to_datetime(conf["fdc"])
    conf["cal_year"] = dt.dt.year.astype(float)
    mo = dt.dt.month.astype(float)
    conf["cal_sin"] = np.sin(2 * np.pi * mo / 12)
    conf["cal_cos"] = np.cos(2 * np.pi * mo / 12)
    conf = conf.drop(columns=["fdc"])
    C = conf.reindex(eids)
    cols = list(C.columns)
    return C.to_numpy(np.float32), cols


def _standardize(train, *others):
    mu = np.nanmean(train, axis=0)
    sd = np.nanstd(train, axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)

    def tf(x):
        x = np.where(np.isfinite(x), x, mu)
        return ((x - mu) / sd).astype(np.float32)

    return (tf(train), *[tf(o) for o in others])


def _fit_logistic(Xtr, ytr, Xte, C=1.0):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=C, penalty="l2", solver="lbfgs",
                             max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def _fit_logistic_val(Xtr, ytr, Xva, yva, Xte, c_grid):
    """Pick C by val AUROC, refit on train, predict test (mirrors run_screening)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    best_c, best = c_grid[0], -1.0
    if len(np.unique(yva)) >= 2:
        for C in c_grid:
            try:
                p = _fit_logistic(Xtr, ytr, Xva, C=C)
                a = roc_auc_score(yva, p)
            except Exception:
                a = -1.0
            if a > best:
                best, best_c = a, C
    clf = LogisticRegression(C=best_c, penalty="l2", solver="lbfgs",
                             max_iter=2000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


# --------------------------------------------------------------------------- OOF workhorse
def fixed_control_universe(prevalent, control_idx, cv, n_folds, cv_eval_controls, seed=0):
    """Deterministically pick a fixed pooled control set (~cv_eval_controls), <=
    cv_eval_controls//n_folds per fold, so every fold's controls are scored OOF
    exactly once. Matches the headline eval-control budget."""
    per = max(1, cv_eval_controls // n_folds)
    keep = []
    for k in range(n_folds):
        ck = control_idx[cv[control_idx] == k]
        rng = np.random.default_rng(seed * 100 + k)
        if len(ck) > per:
            ck = rng.choice(ck, size=per, replace=False)
        keep.append(ck)
    return np.sort(np.concatenate(keep))


def oof_predictions(t, cv, X, *, n_folds=5, c_grid=(0.01, 0.1, 1.0), val_frac=0.15,
                    cv_fixed_c=0.1, cv_min_val_pos=8, cv_eval_controls=8000,
                    max_train_controls=30000, ctrl_universe=None, seed=0):
    """Per-subject OOF logistic predictions over (all prevalent + a fixed control set).

    Geometry matches screen_one_same_dist_cv (val-selected C, per-fold standardize,
    train controls capped at max_train_controls) so the pooled OOF AUROC reproduces
    the headline. Returns (idx, y, pred): subject indices into `X`, labels, OOF probs.
    Pass a shared `ctrl_universe` so embedding/confound/combined score the SAME rows.
    """
    notna = ~np.isnan(t)
    prevalent = np.where(notna & (t <= SEVEN_DAY_YEARS))[0]
    control_all = np.where(~notna)[0]
    if ctrl_universe is None:
        ctrl_universe = fixed_control_universe(
            prevalent, control_all, cv, n_folds, cv_eval_controls, seed=seed)
    uni = np.concatenate([prevalent, ctrl_universe])
    y_uni = np.concatenate([np.ones(len(prevalent), int), np.zeros(len(ctrl_universe), int)])
    pred = np.full(len(uni), np.nan)
    pos_set = set(prevalent.tolist())

    for k in range(n_folds):
        in_k = cv[uni] == k
        te_idx_local = np.where(in_k)[0]
        if te_idx_local.size == 0:
            continue
        # train pool = universe NOT in fold k, plus we may draw extra train controls
        tp_pos = prevalent[cv[prevalent] != k]
        tp_ctrl = control_all[cv[control_all] != k]
        if len(tp_pos) < 2:
            continue
        rng = np.random.default_rng(1000 + k)
        # validation carve from the train pool (C selection)
        n_vp = int(round(len(tp_pos) * val_frac))
        vp = rng.choice(tp_pos, size=n_vp, replace=False) if 0 < n_vp < len(tp_pos) else np.empty(0, int)
        tr_pos = np.setdiff1d(tp_pos, vp)
        if max_train_controls and len(tp_ctrl) > max_train_controls:
            tr_ctrl = rng.choice(tp_ctrl, size=max_train_controls, replace=False)
        else:
            tr_ctrl = tp_ctrl
        n_vc = int(round(len(tr_ctrl) * val_frac))
        vc = rng.choice(tr_ctrl, size=n_vc, replace=False) if 0 < n_vc < len(tr_ctrl) else np.empty(0, int)
        tr_ctrl = np.setdiff1d(tr_ctrl, vc)

        tr_idx = np.concatenate([tr_pos, tr_ctrl])
        y_tr = np.concatenate([np.ones(len(tr_pos), int), np.zeros(len(tr_ctrl), int)])
        te_global = uni[te_idx_local]

        use_val = len(vp) >= cv_min_val_pos and len(vc) > 0
        if use_val:
            val_idx = np.concatenate([vp, vc])
            y_va = np.concatenate([np.ones(len(vp), int), np.zeros(len(vc), int)])
            Xtr, Xva, Xte = _standardize(X[tr_idx], X[val_idx], X[te_global])
            p = _fit_logistic_val(Xtr, y_tr, Xva, y_va, Xte, list(c_grid))
        else:
            Xtr, Xte = _standardize(X[tr_idx], X[te_global])
            p = _fit_logistic(Xtr, y_tr, Xte, C=cv_fixed_c)
        pred[te_idx_local] = p

    ok = np.isfinite(pred)
    return uni[ok], y_uni[ok], pred[ok], ctrl_universe


def boot_auroc(y, s, n_boot=1000, seed=0):
    from sklearn.metrics import roc_auc_score
    a = roc_auc_score(y, s)
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    bs = []
    for _ in range(n_boot):
        ip = rng.choice(pos, size=len(pos), replace=True)
        ino = rng.choice(neg, size=len(neg), replace=True)
        ii = np.concatenate([ip, ino])
        try:
            bs.append(roc_auc_score(y[ii], s[ii]))
        except Exception:
            pass
    lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan)
    return float(a), float(lo), float(hi)


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=f"{UKB_ROOT}/confound_audit_out")
    ap.add_argument("--diseases", nargs="+", default=DEFAULT_DISEASES)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--dump-oof", default="NS_324.11",
                    help="phecode whose embedding OOF preds to dump for Stage 2")
    args = ap.parse_args()
    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    t0 = time.time()

    z = np.load(resolve_oak_path(POOL), allow_pickle=True)
    eids = z["eids"].astype(np.int64)
    Xemb = z["X"].astype(np.float32)
    cv = np.array([fold_of_eid(int(e), args.n_folds) for e in eids], dtype=np.int64)
    demo = load_demographics(eids)
    C, ccols = build_confound_matrix(eids)
    phe = load_phecodes().reindex(index=eids)
    print(f"[load] X={Xemb.shape} confound={C.shape} cols={ccols} ({time.time()-t0:.1f}s)", flush=True)

    rows = []
    assoc_rows = []
    strat_rows = []
    for col in args.diseases:
        if col not in phe.columns:
            print(f"  [skip] {col} not in panel", flush=True)
            continue
        t = phe[col].to_numpy(np.float64)
        notna = ~np.isnan(t)
        prevalent = notna & (t <= SEVEN_DAY_YEARS)
        ctrl = ~notna
        npos = int(prevalent.sum())
        tc = time.time()

        # shared fixed control universe so all three blocks score identical rows
        prevalent_idx = np.where(prevalent)[0]
        control_idx = np.where(ctrl)[0]
        ctrl_uni = fixed_control_universe(
            prevalent_idx, control_idx, cv, args.n_folds, cv_eval_controls=8000)

        res = {}
        for tag, Xb in [("emb", Xemb), ("confound", C),
                        ("emb_confound", np.concatenate([Xemb, C], axis=1)),
                        ("demo", demo)]:
            idx, y, s, _ = oof_predictions(t, cv, Xb, n_folds=args.n_folds,
                                           ctrl_universe=ctrl_uni)
            a, lo, hi = boot_auroc(y, s, n_boot=args.n_boot)
            res[tag] = (a, lo, hi, idx, y, s)
            rows.append({"phecode": col, "n_pos": npos, "block": tag,
                         "auroc": a, "auroc_lo": lo, "auroc_hi": hi,
                         "n_eval": len(y), "n_eval_pos": int(y.sum())})
        ea = res["emb"][0]; ca = res["confound"][0]
        comb = res["emb_confound"][0]; da = res["demo"][0]
        print(f"[{col}] n={npos}: emb={ea:.3f} confound={ca:.3f} "
              f"emb+conf={comb:.3f} (d={comb-ea:+.3f}) demo={da:.3f} "
              f"({time.time()-tc:.0f}s)", flush=True)

        # View 2: direct association of each confound feature
        for j, name in enumerate(ccols):
            a_ = C[prevalent, j]; b_ = C[ctrl, j]
            a_ = a_[np.isfinite(a_)]; b_ = b_[np.isfinite(b_)]
            smd = (a_.mean() - b_.mean()) / (np.sqrt((a_.var() + b_.var()) / 2) + 1e-9)
            assoc_rows.append({"phecode": col, "feature": name, "smd": float(smd)})

        # View 3: within wear-stratum AUROC of the embedding (tertiles of nvd_mean)
        from sklearn.metrics import roc_auc_score
        idx, y, s, _ = res["emb"][3], res["emb"][4], res["emb"][5], None
        nvd = C[idx, ccols.index("nvd_mean")]
        qs = np.nanpercentile(nvd, [33.3, 66.7])
        strata = np.digitize(nvd, qs)  # 0,1,2
        for g in (0, 1, 2):
            mm = strata == g
            if mm.sum() > 10 and len(np.unique(y[mm])) == 2:
                strat_rows.append({
                    "phecode": col, "wear_tertile": int(g),
                    "n": int(mm.sum()), "n_pos": int(y[mm].sum()),
                    "auroc": float(roc_auc_score(y[mm], s[mm])),
                })

        if col == args.dump_oof:
            # De-identified positional row index in place of participant ids.
            row_id = np.arange(len(res["emb"][3]), dtype=np.int64)
            np.savez(os.path.join(out, "oof_emb_dump.npz"),
                     idx=res["emb"][3], y=res["emb"][4], pred=res["emb"][5],
                     eids=row_id, t=t[res["emb"][3]], phecode=col)
            print(f"  [dump] OOF emb preds for {col} -> oof_emb_dump.npz", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "confound_detector_auroc.csv"), index=False)
    pd.DataFrame(assoc_rows).to_csv(os.path.join(out, "confound_association.csv"), index=False)
    pd.DataFrame(strat_rows).to_csv(os.path.join(out, "wear_stratified_auroc.csv"), index=False)
    print(f"\n[done] wrote 3 CSVs to {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
