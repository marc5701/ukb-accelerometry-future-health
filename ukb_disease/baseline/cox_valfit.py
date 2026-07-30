"""Out-of-sample combiner: fit on the held-out VALIDATION set, score the TEST set.

Fitting the per-phecode combiner [hazard, age, sex, ar_emb] in-sample on test is optimistically
biased. This module instead fits the per-phecode combiner on the VALIDATION set (held out from
the deep model's training and from test), then applies the fitted coefficients to the TEST set
and reports the test C-index. No test data is used for fitting, so there is no in-sample
optimism, and the base scores (val hazards) are out-of-sample to the deep model, giving
age/ar_emb/features a fair, unbiased comparison.

Val hazards come from `train_rolling_window_cox --predict-only` (regenerated from the frozen checkpoints).
Arms: hazard-only (no combiner) vs val-fit [hazard, age, sex] vs val-fit [+extra covariates].
Also prints the in-sample test number for each arm as the apparent contrast.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError

from ukb_disease.baseline.config import resolve_oak_path
from ukb_disease.benchmark.concordance_benchmark import harrell_counts
from ukb_disease.paths import UKB_ROOT

DEM = f"{UKB_ROOT}/metadata/acc_dem.csv"


def seed_avg(run_glob: str, prefix: str):
    """Average <prefix>_hazards over seed dirs; return (haz[N,P], eids, T, E, M) using the first
    dir's arrays (asserted identical across seeds)."""
    dirs = sorted(glob.glob(resolve_oak_path(run_glob)))
    if not dirs:
        raise SystemExit(f"no dirs match {run_glob}")
    hz, base, used = [], None, []
    for d in dirs:
        if not os.path.exists(f"{d}/{prefix}_hazards.npy"):
            print(f"[valfit] skip {os.path.basename(d)} (no {prefix}_hazards yet)")
            continue
        used.append(d)
        hz.append(np.load(f"{d}/{prefix}_hazards.npy"))
        cur = dict(eids=np.load(f"{d}/{prefix}_eids.npy"),
                   T=np.load(f"{d}/{prefix}_event_times.npy"),
                   E=np.load(f"{d}/{prefix}_is_event.npy"),
                   M=np.load(f"{d}/{prefix}_eval_mask.npy"))
        if base is None:
            base = cur
        else:
            assert np.array_equal(cur["eids"], base["eids"]), f"{prefix} eids differ in {d}"
    print(f"[valfit] {prefix}: {len(used)} seeds {[os.path.basename(u) for u in used]} shape {hz[0].shape}")
    return np.stack(hz).mean(0), base["eids"], base["T"], base["E"], base["M"], used


def demo_for(eids: np.ndarray) -> pd.DataFrame:
    dem = pd.read_csv(resolve_oak_path(DEM)).set_index("eid")
    out = dem.reindex(eids)[["age_at_first_run", "sex"]].rename(columns={"age_at_first_run": "age"})
    return out.fillna(out.mean(numeric_only=True))


def features_for(parquet, cols, eids):
    if not cols:
        return None
    fp = pd.read_parquet(resolve_oak_path(parquet)).set_index("eid").reindex(eids)
    return fp[cols]


def _fit_apply(Xv, Tv, Ev, Xt, penalizer):
    """Fit ridge Cox on val (Xv), z-scored by val stats; return test linear predictor on Xt.
    NaN-robust: nan-aware mean/std, missing values -> 0 after centering (NaN->0 convention)."""
    mu, sd = np.nanmean(Xv, 0), np.nanstd(Xv, 0)
    sd[(sd == 0) | ~np.isfinite(sd)] = 1.0
    cols = [f"x{i}" for i in range(Xv.shape[1])]
    df = pd.DataFrame(np.nan_to_num((Xv - mu) / sd), columns=cols)
    df["T"], df["E"] = Tv, Ev
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=0.0)
    cph.fit(df, "T", "E")
    beta = cph.params_.reindex(cols).to_numpy()
    return np.nan_to_num((Xt - mu) / sd) @ beta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-glob", default=f"{UKB_ROOT}/runs_cox_val_pred/embedding_val_pred_seed*")
    ap.add_argument("--test-glob", default=f"{UKB_ROOT}/runs_cox/embedding_disjoint_split_seed*")
    ap.add_argument("--feat-parquet", default=None)
    ap.add_argument("--extra-cols", nargs="*", default=[],
                    help="extra covariate columns from --feat-parquet (e.g. sleep features)")
    ap.add_argument("--ar_emb-val-dir", default=None,
                    help="age-val_pred dir (test_ar_emb.npy/test_eids.npy = val ar_emb) for the ar_emb arm")
    ap.add_argument("--ar_emb-test-dir", default=None,
                    help="age dir with test_ar_emb.npy/test_eids.npy (test ar_emb)")
    ap.add_argument("--min-events", type=int, default=10)
    ap.add_argument("--penalizer-c0", type=float, default=10.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out_dir = resolve_oak_path(args.out)
    os.makedirs(out_dir, exist_ok=True)

    vh, ve, vT, vE, vM, _ = seed_avg(args.val_glob, "val")
    th, te, tT, tE, tM, tdirs = seed_avg(args.test_glob, "test")
    names = json.load(open(f"{tdirs[0]}/config.json"))["phecodes"]
    P = th.shape[1]

    vdemo, tdemo = demo_for(ve), demo_for(te)
    vage, vsex = vdemo["age"].to_numpy(float), vdemo["sex"].to_numpy(float)
    tage, tsex = tdemo["age"].to_numpy(float), tdemo["sex"].to_numpy(float)
    def feat_arr(eids):
        if not args.extra_cols:
            return None
        return features_for(args.feat_parquet, args.extra_cols, eids).to_numpy(float)
    vfeat = feat_arr(ve)
    tfeat = feat_arr(te)

    def ar_emb_arr(d, eids):
        if not d:
            return None
        dd = resolve_oak_path(d)
        a = pd.DataFrame({"eid": np.load(f"{dd}/test_eids.npy"),
                          "ar_emb": np.load(f"{dd}/test_ar_emb.npy")}).groupby("eid").mean()
        return a.reindex(eids)["ar_emb"].fillna(0.0).to_numpy(float)
    val_ar_emb = ar_emb_arr(args.ar_emb_val_dir, ve)
    test_ar_emb = ar_emb_arr(args.ar_emb_test_dir, te)

    # arms: each is a set of extra blocks added to [hazard, age, sex]
    ARMS = {"age_sex": ()}
    if val_ar_emb is not None:
        ARMS["age_sex_ar_emb"] = ("ar_emb",)
    if vfeat is not None:
        ARMS["age_sex_feat"] = ("feat",)
    if val_ar_emb is not None and vfeat is not None:
        ARMS["full"] = ("ar_emb", "feat")

    def cols(blocks, h, age, sex, ar_emb, feat, m, j):
        c = [h[m, j], age[m], sex[m]]
        if "ar_emb" in blocks:
            c.append(ar_emb[m])
        if "feat" in blocks:
            c += [feat[m, k] for k in range(feat.shape[1])]
        return np.column_stack(c)

    ev_v, ev_t = (vE * vM).sum(0), (tE * tM).sum(0)
    panel = [j for j in range(P) if ev_t[j] >= args.min_events and ev_v[j] >= args.min_events]
    print(f"[valfit] panel (>= {args.min_events} ev in val AND test) = {len(panel)} phecodes; "
          f"arms = {list(ARMS)}")

    rows = []
    for j in panel:
        mv, mt = vM[:, j], tM[:, j]
        Tv, Ev = vT[mv, j].astype(float), vE[mv, j].astype(float)
        Tt, Et = tT[mt, j].astype(float), tE[mt, j].astype(float)
        pen = max(0.1, args.penalizer_c0 / int(Ev.sum()))
        rec = {"phecode": names[j], "n_ev_val": int(Ev.sum()), "n_ev_test": int(Et.sum())}
        c, comp = harrell_counts(th[mt, j], Tt, Et)
        rec["c_hazard"] = c / comp if comp > 0 else np.nan
        for arm, blocks in ARMS.items():
            Xv = cols(blocks, vh, vage, vsex, val_ar_emb, vfeat, mv, j)
            Xt = cols(blocks, th, tage, tsex, test_ar_emb, tfeat, mt, j)
            try:
                lp = _fit_apply(Xv, Tv, Ev, Xt, pen)
                c, comp = harrell_counts(lp, Tt, Et)
                rec[f"c_valfit_{arm}"] = c / comp if comp > 0 else np.nan
            except (ConvergenceError, ValueError, np.linalg.LinAlgError):
                rec[f"c_valfit_{arm}"] = np.nan
        rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(f"{out_dir}/per_phecode_valfit.csv", index=False)
    summ = {"n_panel": len(panel), "min_events": args.min_events,
            "extra_cols": args.extra_cols}
    for col in [c for c in df.columns if c.startswith("c_")]:
        summ[f"mean_{col}"] = float(df[col].mean())
    for arm in ["age_sex", "age_sex_ar_emb", "age_sex_feat", "full"]:
        k = f"mean_c_valfit_{arm}"
        if k in summ:
            summ[f"delta_{arm}_vs_hazard"] = summ[k] - summ["mean_c_hazard"]
    json.dump(summ, open(f"{out_dir}/summary.json", "w"), indent=2)
    print("\n[valfit] ===== SUMMARY (test C-index, combiner FIT ON VAL) =====")
    for k, v in summ.items():
        print(f"  {k:32s} {v:+.4f}" if isinstance(v, float) else f"  {k:32s} {v}")


if __name__ == "__main__":
    main()
