"""Consolidate the prevalent-disease screening results into one figure bundle for
the paper, and fill the single missing artifact: the demographics-arm out-of-fold
predictions for Parkinson's (and the spotlight cluster), needed to draw the
demographics ROC curve in the main figure.

Additive and non-destructive. It reuses the confound_audit OOF workhorse so every
out-of-fold prediction reproduces the headline (pooled OOF PD AUROC ~0.916 vs
0.917). It reads the already-computed scratch CSVs (632-disease selectivity panel,
confound audit, wear strata, near-onset, frozen-model comparison) and copies them
into one self-contained bundle directory, then writes SUMMARY.json with the
headline scalars and a self-verification flag.

Run:
  python3 -m ukb_disease.screening.build_figure_bundle \
      --diseases NS_324.11
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from ukb_disease.baseline.config import SEVEN_DAY_YEARS, resolve_oak_path
from ukb_disease.paths import UKB_ROOT
from ukb_disease.baseline.make_cv_folds import fold_of_eid
from ukb_disease.baseline.outcomes_processing import load_phecodes
from ukb_disease.screening.run_screening import load_demographics
from ukb_disease.screening.confound_audit import (
    POOL,
    boot_auroc,
    fixed_control_universe,
    oof_predictions,
)

# Motor / sleep / mood cluster where the wrist outperforms demographics.
# Parkinson's is the flagship (its OOF feeds the main-figure ROC); the rest are
# dumped so an MS curve or an operating point is available without recompute.
SPOTLIGHT = [
    "NS_324.11",   # Parkinson's            (flagship: ROC panel)
    "NS_326.1",    # multiple sclerosis
    "NS_324.341",  # extrapyramidal / dystonia
    "NS_356.2",    # neuropathy
    "NS_325",      # abnormality of gait
    "MB_286.2",    # major depression
    "RE_474",      # COPD
]

# (bundle name, source path relative to scratch_root or confound_dir). The figure
# build reads ONLY from the bundle, so this is the single canonical provenance.
def _copy_specs(scratch_root: str, confound_dir: str):
    return [
        ("per_disease_sd.csv",       os.path.join(scratch_root, "sd_vs_transfer_logistic.csv")),
        ("confound_audit.csv",       os.path.join(confound_dir, "confound_detector_auroc.csv")),
        ("wear_strata.csv",          os.path.join(confound_dir, "wear_stratified_auroc.csv")),
        ("nearonset.csv",            os.path.join(confound_dir, "nearonset_NS_324.11.csv")),
        ("nearonset.json",           os.path.join(confound_dir, "nearonset_NS_324.11.json")),
        ("embedding_compare.csv",          os.path.join(confound_dir, "embedding_screening_auroc.csv")),
        ("confound_association.csv", os.path.join(confound_dir, "confound_association.csv")),
    ]


def sens_at_spec(y: np.ndarray, s: np.ndarray, spec: float = 0.95) -> float:
    """Sensitivity at a fixed specificity, from a control-score threshold."""
    neg = s[y == 0]
    if neg.size == 0:
        return float("nan")
    thr = np.quantile(neg, spec)               # `spec` of controls fall below thr
    pos = s[y == 1]
    return float((pos >= thr).mean()) if pos.size else float("nan")


# Forest-plot outcome set: wrist wins (movement / mood / respiratory / neuro) plus
# a few age-driven contrasts (metabolic / cardiac). Sex-confounded GU/gyn outcomes
# and tiny/odd small-n outcomes are excluded. The figure sorts rows by the
# recomputed (embedding - demographics) margin.
FOREST_DISEASES = [
    # Ordered so an 11-way shard split (FOREST_DISEASES[i::11]) puts at most one
    # large-n outcome per shard. The figure re-sorts rows by the embedding-minus-
    # demographics margin, so list order here only affects sharding. Names are the
    # authoritative per_disease_sd PhecodeString.
    "RE_475",      # asthma                 5073
    "EM_202.2",    # type 2 diabetes        2662  (contrast)
    "CV_416.2",    # atrial fibrillation    2389  (contrast)
    "MB_286.2",    # major depression       1673
    "EM_236.1",    # obesity                1641  (contrast)
    "RE_474",      # COPD                    890
    "NS_333.1",    # sleep apnea             838  (contrast; sleep-relevant)
    "MB_288",      # anxiety disorders       577
    "NS_330.1",    # epilepsy                467
    "EM_202.1",    # type 1 diabetes         383
    "MS_705.1",    # rheumatoid arthritis    352
    "CV_431.11",   # ischemic stroke         305  (contrast)
    "NS_326.1",    # multiple sclerosis      240
    "CV_424",      # heart failure           141  (contrast)
    "NS_324.11",   # Parkinson's             132
    "NS_325",      # abnormality of gait     111  (contrast)
    "MB_286.1",    # bipolar disorder        106
    "NS_356.2",    # aphasia and dysphasia    73
    "NS_350.5",    # repeated falls           47  (contrast)
    "NS_324.341",  # essential tremor         27
    "NS_328.1",    # dementia                 21  (contrast)
    "NS_328.11",   # Alzheimer's disease      16  (contrast; thin, wide CI)
]


def compute_forest_arms(eids, Xemb, demo, cv, phe, diseases, *, n_folds=5, n_boot=1000):
    """For each disease, AUROC + bootstrap CI for three arms on a shared control universe:
    embedding, demographics, and embedding+demographics (concatenated). Returns a long DataFrame
    (phecode, n_pos, arm, auroc, lo, hi). Disease names are joined by the figure from the
    published panel CSV, so only phecode + n are kept here."""
    rows = []
    for col in diseases:
        if col not in phe.columns:
            print(f"  [skip] {col} not in panel", flush=True)
            continue
        t = phe[col].to_numpy(np.float64)
        notna = ~np.isnan(t)
        prevalent_idx = np.where(notna & (t <= SEVEN_DAY_YEARS))[0]
        control_idx = np.where(~notna)[0]
        ctrl_uni = fixed_control_universe(
            prevalent_idx, control_idx, cv, n_folds, cv_eval_controls=8000)
        npos = int(prevalent_idx.size)
        tc = time.time()
        arms = {"emb": Xemb, "demo": demo,
                "emb_demo": np.concatenate([Xemb, demo], axis=1)}
        for arm, Xb in arms.items():
            idx, y, s, _ = oof_predictions(t, cv, Xb, n_folds=n_folds, ctrl_universe=ctrl_uni)
            a, lo, hi = boot_auroc(y, s, n_boot=n_boot)
            rows.append({"phecode": col, "n_pos": npos, "arm": arm,
                         "auroc": a, "lo": lo, "hi": hi})
        e = {r["arm"]: r["auroc"] for r in rows if r["phecode"] == col}
        print(f"[{col}] n={npos}: emb={e.get('emb',float('nan')):.3f} "
              f"demo={e.get('demo',float('nan')):.3f} emb+demo={e.get('emb_demo',float('nan')):.3f} "
              f"({time.time()-tc:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scratch-root", default=str(UKB_ROOT),
                    help="dir holding sd_vs_transfer_logistic.csv")
    ap.add_argument("--confound-dir", default=None,
                    help="dir holding the confound_audit_out CSVs (default <scratch-root>/confound_audit_out)")
    ap.add_argument("--out-dir", default=f"{UKB_ROOT}/figures_out/screening_fig_bundle")
    ap.add_argument("--pooled-cache", default=POOL)
    ap.add_argument("--diseases", nargs="+", default=SPOTLIGHT)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=1000)
    # forest mode: compute emb / demo / emb+demo arms (with CIs) for FOREST_DISEASES,
    # sharded by disease across an array job, then merge into forest_arms.csv.
    ap.add_argument("--forest", action="store_true", help="compute the 3-arm forest table")
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--merge-forest", action="store_true",
                    help="concat forest_arms_shard*.csv -> forest_arms.csv and exit")
    args = ap.parse_args()

    out = args.out_dir
    os.makedirs(out, exist_ok=True)

    # --- merge mode: consolidate forest shards and exit (no heavy load) ---
    if args.merge_forest:
        import glob
        shards = sorted(glob.glob(os.path.join(out, "forest_arms_shard*.csv")))
        if not shards:
            print(f"[merge] no forest_arms_shard*.csv in {out}"); return
        df = pd.concat([pd.read_csv(s) for s in shards], ignore_index=True)
        df = df.drop_duplicates(subset=["phecode", "arm"], keep="last")
        df.to_csv(os.path.join(out, "forest_arms.csv"), index=False)
        print(f"[merge] {len(shards)} shards -> forest_arms.csv "
              f"({df.phecode.nunique()} outcomes x {df.arm.nunique()} arms)", flush=True)
        return

    confound_dir = args.confound_dir or os.path.join(args.scratch_root, "confound_audit_out")
    oof_dir = os.path.join(out, "oof")
    os.makedirs(oof_dir, exist_ok=True)
    t0 = time.time()

    # --- load the canonical inputs (identical geometry to the published run) ---
    z = np.load(resolve_oak_path(args.pooled_cache), allow_pickle=True)
    eids = z["eids"].astype(np.int64)
    Xemb = z["X"].astype(np.float32)
    cv = np.array([fold_of_eid(int(e), args.n_folds) for e in eids], dtype=np.int64)
    demo = load_demographics(eids)
    phe = load_phecodes().reindex(index=eids)
    print(f"[load] X={Xemb.shape} demo={np.shape(demo)} ({time.time()-t0:.1f}s)", flush=True)

    # --- forest mode: 3-arm table for this shard of FOREST_DISEASES, then exit ---
    if args.forest:
        mine = FOREST_DISEASES[args.shard_idx::args.n_shards]
        print(f"[forest] shard {args.shard_idx}/{args.n_shards}: {mine}", flush=True)
        df = compute_forest_arms(eids, Xemb, demo, cv, phe, mine,
                                 n_folds=args.n_folds, n_boot=args.n_boot)
        shard_path = os.path.join(out, f"forest_arms_shard{args.shard_idx:02d}.csv")
        df.to_csv(shard_path, index=False)
        print(f"[forest] wrote {shard_path} ({len(df)} rows, {time.time()-t0:.0f}s)", flush=True)
        return

    # --- dump wrist + demographics OOF on a SHARED control universe per disease ---
    summary_oof = {}
    for col in args.diseases:
        if col not in phe.columns:
            print(f"  [skip] {col} not in panel", flush=True)
            continue
        t = phe[col].to_numpy(np.float64)
        notna = ~np.isnan(t)
        prevalent_idx = np.where(notna & (t <= SEVEN_DAY_YEARS))[0]
        control_idx = np.where(~notna)[0]
        ctrl_uni = fixed_control_universe(
            prevalent_idx, control_idx, cv, args.n_folds, cv_eval_controls=8000)
        npos = int(prevalent_idx.size)
        tc = time.time()

        res = {}
        for tag, Xb in [("wrist", Xemb), ("demo", demo)]:
            idx, y, s, _ = oof_predictions(t, cv, Xb, n_folds=args.n_folds,
                                           ctrl_universe=ctrl_uni)
            a, lo, hi = boot_auroc(y, s, n_boot=args.n_boot)
            sn = sens_at_spec(y, s, 0.95)
            # Write a de-identified positional row index, not the participant id.
            pd.DataFrame({"row": np.arange(len(idx)), "y": y, "pred": s}).to_csv(
                os.path.join(oof_dir, f"{col}_{tag}.csv"), index=False)
            res[tag] = dict(auroc=a, auroc_lo=lo, auroc_hi=hi, sens95=sn,
                            n_eval=int(len(y)), n_pos=int(y.sum()))
        summary_oof[col] = {"n_prevalent": npos, **{f"{k}_{m}": v
                            for k, d in res.items() for m, v in d.items()}}
        print(f"[{col}] n={npos}: wrist={res['wrist']['auroc']:.3f} "
              f"(sens95={res['wrist']['sens95']:.3f}) demo={res['demo']['auroc']:.3f} "
              f"({time.time()-tc:.0f}s)", flush=True)

    # --- consolidate the already-computed result CSVs into the bundle ---
    copied, missing = [], []
    for name, src in _copy_specs(args.scratch_root, confound_dir):
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(out, name))   # copy, never move/delete
            copied.append(name)
        else:
            missing.append(src)
    print(f"[consolidate] copied {len(copied)} files; missing {missing}", flush=True)

    # --- headline scalars + self-verification ---
    pd_oof = summary_oof.get("NS_324.11", {})
    pd_wrist = pd_oof.get("wrist_auroc", float("nan"))
    pd_demo = pd_oof.get("demo_auroc", float("nan"))
    verify_pass = bool(abs(pd_wrist - 0.917) < 0.02 and abs(pd_demo - 0.700) < 0.04)

    summary = {
        "built_by": "ukb_disease.screening.build_figure_bundle",
        "pooled_cache": resolve_oak_path(args.pooled_cache),
        "n_subjects": int(eids.size),
        "n_folds": args.n_folds,
        "n_boot": args.n_boot,
        "spotlight_oof": summary_oof,
        "pd_wrist_auroc": pd_wrist,
        "pd_demo_auroc": pd_demo,
        "pd_wrist_sens95": pd_oof.get("wrist_sens95", float("nan")),
        "copied": copied,
        "missing_inputs": missing,
        "verify_target_pd_wrist": 0.917,
        "verify_pass": verify_pass,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(os.path.join(out, "SUMMARY.json"), "w") as f:
        json.dump(summary, f, indent=2)

    flag = "PASS" if verify_pass else "FAIL"
    print(f"\n[done] bundle -> {out}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"[verify] PD wrist OOF AUROC={pd_wrist:.4f} (target 0.917), demo={pd_demo:.4f} "
          f"-> verify_pass={flag}", flush=True)


if __name__ == "__main__":
    main()
