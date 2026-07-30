"""Per-disease confusion matrices for a set of target diseases (supplementary).

For each target disease X, among participants who truly have X (ever diagnosed = incident +
prevalent, for statistical power), which OTHER diseases does the model most over-rank them on?
Risk is put on a common footing (age/sex/age^2-residualised cohort percentile), so the confusion
reflects disease-specific misdirection rather than the shared age/frailty axis. Each confusion
carries a subject-bootstrap 95% interval.

Run:
  python3 -m ukb_disease.paper_additions.confusion_targets --out <output_dir>
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ukb_disease.baseline.config import resolve_oak_path
from ukb_disease.baseline.outcomes_processing import load_covariates
from ukb_disease.paper_additions.io_utils import load_seed_avg
from ukb_disease.paper_additions.disease_correlation import _residualize, _phecode_meta
from ukb_disease.genetics.cox_crossfit_prs import FEATURES
from ukb_disease.paper_experiments import io

TARGETS = [  # (display name, phecode)
    ("Parkinson's disease", "NS_324.11"),
    ("Alzheimer's disease", "NS_328.11"),
    ("Heart failure", "CV_424"),
    ("Type 2 diabetes", "EM_202.2"),
    ("Renal failure", "GU_582"),
    ("All-cause mortality", "time_to_death"),
    ("Pulmonary embolism", "CV_440.3"),
    ("Sleep apnea", "NS_333.1"),
    ("Persistent atrial fibrillation", "CV_416.212"),
    ("COPD", "RE_474"),
]
TOPK = 6
N_BOOT = 2000
RNG = np.random.default_rng(20260707)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["raw", "resid"], default="raw",
                    help="raw = the model's actual risk ranking; resid = age/sex/age^2-residualised")
    ap.add_argument("--min-confused-events", type=int, default=30,
                    help="only diseases with >= this many incident cases can appear as a confusion")
    args = ap.parse_args()
    out = resolve_oak_path(args.out); os.makedirs(out, exist_ok=True)

    sa = load_seed_avg("embedding_disjoint_split")
    eids, H, events, mask, phe = (sa["eids"], sa["hazards"].astype(np.float64),
                                  sa["events"], sa["mask"], np.asarray(sa["phecodes"]))
    P = len(phe); idx = {p: i for i, p in enumerate(phe)}
    cov = load_covariates().reindex(eids)
    age, sex = cov["age"].to_numpy(), cov["sex"].to_numpy()
    feat = pd.read_parquet(FEATURES, columns=["eid", "eur_genetic"]).set_index("eid").reindex(eids)
    eur = feat["eur_genetic"].fillna(0).to_numpy() == 1
    keep = eur & ~(np.isnan(age) | np.isnan(sex))
    H, events, mask = H[keep], events[keep], mask[keep]
    age, sex, eids = age[keep], sex[keep], eids[keep]
    N = keep.sum()
    print(f"[targets] EUR+demo test subjects = {N}")

    meta = _phecode_meta()
    names = np.array([meta["name"].get(p, p) for p in phe])
    cats = np.array([meta["cat"].get(p, "Other") for p in phe])

    # common-scale risk -> cohort percentile. raw = the model's actual output ranking; resid =
    # after removing the age/sex/age^2 axis. Confusions restricted to adequately-powered diseases.
    src = _residualize(H, age, sex) if args.mode == "resid" else H
    pct = np.apply_along_axis(lambda c: (rankdata(c) - 0.5) / len(c), 0, src)
    n_inc = np.array([int(events[mask[:, p], p].sum()) for p in range(P)])
    wellpow = n_inc >= args.min_confused_events

    # "ever diagnosed" cohort per target (incident + prevalent) from the timing table
    want = set(["eid"] + [ph for _, ph in TARGETS if ph != "time_to_death"])
    tim = (pd.read_csv(resolve_oak_path(io.MORTALITY_PHECODES_CSV),
                       usecols=lambda c: c in want).set_index("eid").reindex(eids))

    rows = []
    for disp, ph in TARGETS:
        if ph == "time_to_death":
            has = events[:, idx[ph]] > 0.5            # died during follow-up
        elif ph in tim.columns:
            has = ~np.isnan(tim[ph].to_numpy(float))  # ever diagnosed (incident + prevalent)
        else:
            has = events[:, idx[ph]] > 0.5
        n_has = int(has.sum())
        xj = idx[ph]
        # exclude the target itself and any column with the identical disease name
        excl = (np.arange(P) == xj) | (names == names[xj]) | (~wellpow)
        sub = pct[has]                                # (n_has, P)
        lift = sub.mean(0) - 0.5
        # subject bootstrap CI on each column's lift
        bl = np.empty((N_BOOT, P))
        for b in range(N_BOOT):
            bi = RNG.integers(0, n_has, n_has)
            bl[b] = sub[bi].mean(0) - 0.5
        lo = np.percentile(bl, 2.5, axis=0); hi = np.percentile(bl, 97.5, axis=0)
        order = np.argsort(-lift)
        top = [o for o in order if not excl[o]][:TOPK]
        print(f"\n{disp} [{ph}] n_ever={n_has}  self-lift={lift[xj]:+.2f}")
        for rank, y in enumerate(top, 1):
            rows.append({"target": disp, "target_phecode": ph, "n_ever": n_has, "rank": rank,
                         "confused_phecode": phe[y], "confused_name": names[y],
                         "confused_category": cats[y], "same_category": bool(cats[y] == cats[xj]),
                         "lift": float(lift[y]), "lo": float(lo[y]), "hi": float(hi[y]),
                         "ci_excludes_0": bool(lo[y] > 0)})
            print(f"   {rank}. {names[y][:34]:34s} [{cats[y][:4]}] lift={lift[y]:+.2f} "
                  f"[{lo[y]:+.2f},{hi[y]:+.2f}]{' *' if lo[y]>0 else ''}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/confusion_targets.csv", index=False)
    print(f"\nwrote {out}/confusion_targets.csv")


if __name__ == "__main__":
    main()
