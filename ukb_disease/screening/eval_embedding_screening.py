"""Evaluate whether the frozen sequence (Embedding) representation beats or adds to the
L-moment embedding for same-distribution prevalent-disease screening.

Reuses the confound_audit.oof_predictions workhorse (reproduces the L-moment PD
headline). For PD plus the motor cluster, scores four feature blocks on a shared
fixed control universe so they are directly comparable:
  L-moment       : the 1536-d mean-pool embedding (reference)
  Embedding rep        : 4-seed concat of the 256-d penultimate CLS rep (1024-d),
                   purely wrist-derived. The honest Embedding feature.
  L-moment+embedding_rep: concat; does the sequence rep add to the mean-pool?
  Embedding PD hazard  : the leaky upper bound (Embedding head trained on PD labels; most
                   prevalent PD is in train). Reported and flagged, never headline.

Each block is evaluated on the full positive set and on the held-out (bucket in
{val,test}) positives only, the single contamination-free comparison (thin n).

All blocks are evaluated on the same inner-joined subject set (Embedding drops subjects
without 3 consecutive valid days), so the L-moment number here is recomputed on
that common set and may differ slightly from the full-cohort value.

Run (after extract_embedding_features writes the npz):
  python3 -u -m ukb_disease.screening.eval_embedding_screening \
      --embedding embedding_features.npz --out-dir confound_audit_out
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
from ukb_disease.baseline.config import SEVEN_DAY_YEARS, resolve_oak_path
from ukb_disease.baseline.make_cv_folds import fold_of_eid
from ukb_disease.baseline.outcomes_processing import load_phecodes
from ukb_disease.screening.confound_audit import POOL, oof_predictions, boot_auroc

# Motor cluster. NS_324.341 is not in the 390-phecode panel but is still evaluable
# via the rep (disease-agnostic); only its per-disease hazard column is unavailable.
DISEASES = ["NS_324.11", "NS_326.1", "NS_356.2", "NS_324.341"]


def eval_block(t, cv, X, ctrl_uni, n_boot):
    idx, y, s, _ = oof_predictions(t, cv, X, ctrl_universe=ctrl_uni)
    a, lo, hi = boot_auroc(y, s, n_boot=n_boot)
    return a, lo, hi, int(y.sum()), len(y)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embedding", default=f"{UKB_ROOT}/embedding_features.npz")
    ap.add_argument("--out-dir", default=f"{UKB_ROOT}/confound_audit_out")
    ap.add_argument("--diseases", nargs="+", default=DISEASES)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    z = np.load(resolve_oak_path(POOL), allow_pickle=True)
    pe = z["eids"].astype(np.int64)
    Xemb_full = z["X"].astype(np.float32)
    e = np.load(args.embedding, allow_pickle=True)
    ee = e["eids"].astype(np.int64)
    seeds = list(e["seeds"])
    rep_concat_full = np.concatenate([e[f"rep_seed{s}"] for s in seeds], axis=1).astype(np.float32)
    pdhaz_full = e["pd_hazard_mean"].astype(np.float32).reshape(-1, 1)
    bucket_e = {int(x): b for x, b in zip(ee, e["bucket"])}

    # inner-join to the common subject set
    common = np.intersect1d(pe, ee)
    pmap = pd.Series(range(len(pe)), index=pe)
    emap = pd.Series(range(len(ee)), index=ee)
    pi = pmap.loc[common].to_numpy()
    ei = emap.loc[common].to_numpy()
    eids = common
    Xemb = Xemb_full[pi]
    rep = rep_concat_full[ei]
    pdhaz = pdhaz_full[ei]
    cv = np.array([fold_of_eid(int(x), 5) for x in eids], dtype=np.int64)
    bucket = np.array([bucket_e[int(x)] for x in eids], dtype=object)
    print(f"[load] pooled={len(pe)} embedding={len(ee)} common={len(common)} "
          f"rep={rep.shape} ({time.time()-t0:.1f}s)", flush=True)
    n_drop = len(pe) - len(common)
    print(f"[load] dropped {n_drop} subjects without 3 consecutive valid days", flush=True)

    phe = load_phecodes().reindex(index=eids)
    blocks = {"L-moment": Xemb, "Embedding_rep": rep,
              "L-moment+embedding_rep": np.concatenate([Xemb, rep], axis=1),
              "Embedding_pd_hazard_LEAKY": pdhaz}

    rows = []
    for col in args.diseases:
        if col not in phe.columns:
            print(f"  [skip] {col} not in panel", flush=True)
            continue
        t = phe[col].to_numpy(np.float64)
        prevalent = (~np.isnan(t)) & (t <= SEVEN_DAY_YEARS)
        npos = int(prevalent.sum())
        # how many prevalent are in Embedding's train bucket (leakage magnitude)
        in_train = int(np.sum([bucket[i] == "train" for i in np.where(prevalent)[0]]))
        held = int(npos - in_train)
        print(f"\n[{col}] prevalent={npos} (train-bucket {in_train}, held-out val+test {held})", flush=True)

        # shared control universe (computed once on full common set)
        from ukb_disease.screening.confound_audit import fixed_control_universe
        ctrl_idx = np.where(np.isnan(t))[0]
        pos_idx = np.where(prevalent)[0]
        ctrl_uni = fixed_control_universe(pos_idx, ctrl_idx, cv, 5, cv_eval_controls=8000)

        for bname, X in blocks.items():
            if bname == "Embedding_pd_hazard_LEAKY" and col != "NS_324.11":
                continue  # PD-hazard column only meaningful for PD
            # full positive set
            a, lo, hi, ny, ne = eval_block(t, cv, X, ctrl_uni, args.n_boot)
            rows.append({"disease": col, "block": bname, "pos_set": "full",
                         "n_pos": ny, "auroc": a, "lo": lo, "hi": hi})
            print(f"   {bname:22s} full     AUROC={a:.3f} [{lo:.3f},{hi:.3f}] n_pos={ny}", flush=True)
            # held-out positives: drop train-bucket prevalent cases from the universe
            # entirely (they are NOT controls), recompute a fresh control universe on
            # the kept rows, then evaluate. The only Embedding-contamination-free number.
            if held >= 5:
                keep = ~(prevalent & np.array([b == "train" for b in bucket]))
                th, cvh, Xh = t[keep], cv[keep], X[keep]
                ph = (~np.isnan(th)) & (th <= SEVEN_DAY_YEARS)
                cu = fixed_control_universe(np.where(ph)[0], np.where(np.isnan(th))[0],
                                            cvh, 5, cv_eval_controls=8000)
                a2, lo2, hi2, ny2, _ = eval_block(th, cvh, Xh, cu, args.n_boot)
                rows.append({"disease": col, "block": bname, "pos_set": "held_out",
                             "n_pos": ny2, "auroc": a2, "lo": lo2, "hi": hi2})
                print(f"   {bname:22s} held-out AUROC={a2:.3f} [{lo2:.3f},{hi2:.3f}] n_pos={ny2}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "embedding_screening_auroc.csv"), index=False)
    print(f"\n[done] {time.time()-t0:.0f}s -> embedding_screening_auroc.csv", flush=True)


if __name__ == "__main__":
    main()
