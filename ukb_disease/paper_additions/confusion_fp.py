"""False-positive / misdiagnosis structure for the target diseases.

For each target disease X, take the participants the model flags for X at a 95%-specificity operating
point but who do not actually have X (ever diagnosed), the false positives, and ask, corrected for
prevalence, what those people actually have. Reports the precision context per disease (PPV, and how
many false positives per 100 flagged). This is the "if the model calls it Parkinson's but it isn't,
what is it?" view.

Prevalence correction = enrichment (rate among false positives / cohort base rate). The naive version
is dominated by rare-disease artifacts (a rare disease held by 1-2 FPs gives an astronomical lift), so
candidate "actual diseases" are restricted to those with >= MIN_EVENTS incident cases AND held by
>= MIN_FP_COUNT of the false positives, and each enrichment carries a subject-bootstrap 95% interval.

Run:
  python3 -m ukb_disease.paper_additions.confusion_fp --out confusion_fp
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from ukb_disease.baseline.config import resolve_oak_path
from ukb_disease.baseline.outcomes_processing import load_covariates
from ukb_disease.paper_additions.io_utils import load_seed_avg
from ukb_disease.paper_additions.disease_correlation import _phecode_meta
from ukb_disease.genetics.cox_crossfit_prs import FEATURES
from ukb_disease.paper_experiments import io
from ukb_disease.paper_additions.confusion_targets import TARGETS

SPEC = 0.95          # per-disease specificity operating point (defines "predicted positive")
TOPK = 6             # top-k enriched actual diseases per target (bars)
MIN_EVENTS = 30      # candidate actual disease needs >= this many incident cases (well-powered)
MIN_FP_COUNT = 10    # ... and needs to be held by >= this many of the false positives (stable)
N_BOOT = 2000
RNG = np.random.default_rng(20260707)


def _ever_matrix(phe, events, tim):
    """(N, P) bool: ever diagnosed (prevalent OR incident) per phecode, aligned to `phe`."""
    N, P = events.shape
    ever = np.zeros((N, P), bool)
    for j, ph in enumerate(phe):
        if ph in tim.columns and ph != "time_to_death":
            ever[:, j] = ~np.isnan(tim[ph].to_numpy(float))   # prevalent + incident
        else:
            ever[:, j] = events[:, j] > 0.5                   # mortality / missing -> incident
    return ever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--spec", type=float, default=SPEC)
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
    H, events, mask, eids = H[keep], events[keep], mask[keep], eids[keep]
    N = int(keep.sum())
    print(f"[fp] EUR+demo test subjects = {N}  (spec operating point = {args.spec:.0%})")

    meta = _phecode_meta()
    names = np.array([meta["name"].get(p, p) for p in phe])
    cats = np.array([meta["cat"].get(p, "Other") for p in phe])
    ucats = sorted(set(cats.tolist()))

    # full ever-diagnosed matrix (all 390 phecodes) -> "what people actually have"
    tim = pd.read_csv(resolve_oak_path(io.MORTALITY_PHECODES_CSV)).set_index("eid").reindex(eids)
    ever = _ever_matrix(phe, events, tim)
    base = ever.mean(0)                                        # cohort prevalence per disease
    n_inc = np.array([int(events[mask[:, p], p].sum()) for p in range(P)])
    wellpow = n_inc >= MIN_EVENTS

    # category-level ever-diagnosed (any disease in a category) + base rate
    cat_ever = np.zeros((N, len(ucats)), bool)
    for c, cn in enumerate(ucats):
        cols = np.where(cats == cn)[0]
        cat_ever[:, c] = ever[:, cols].any(1)
    cat_base = cat_ever.mean(0)

    rows, summ_rows = [], []
    cat_mat = np.full((len(TARGETS), len(ucats)), np.nan)     # 10 predicted x 14 actual-category enrichment
    for ti, (disp, ph) in enumerate(TARGETS):
        xj = idx[ph]
        ever_x = ever[:, xj]
        s = H[:, xj]
        neg = ~ever_x                                         # never had X = eligible to be a false positive
        thr = np.quantile(s[neg], args.spec)
        flagged = s >= thr
        fp = flagged & neg
        tp = flagged & ever_x
        n_flag, n_fp, n_tp = int(flagged.sum()), int(fp.sum()), int(tp.sum())
        ppv = n_tp / max(1, n_flag)
        fp_per_100 = 100.0 * (1.0 - ppv)
        print(f"\n{disp} [{ph}] flagged={n_flag} FP={n_fp} TP={n_tp} PPV={ppv:.3f} -> ~{round(fp_per_100)} FP/100")
        summ_rows.append({"target": disp, "target_phecode": ph, "n_flagged": n_flag, "n_fp": n_fp,
                          "n_tp": n_tp, "ppv": float(ppv), "fp_per_100": float(fp_per_100)})

        # category-level enrichment among the false positives (for the matrix form)
        cat_mat[ti] = cat_ever[fp].mean(0) / (cat_base + 1e-12)

        # per-disease enrichment among the false positives (for the bars form)
        rate_fp = ever[fp].mean(0)
        enr = rate_fp / (base + 1e-12)
        cnt_fp = ever[fp].sum(0)
        elig = wellpow & (np.arange(P) != xj) & (names != names[xj]) & (cnt_fp >= MIN_FP_COUNT)
        # subject-bootstrap CI on enrichment (resample the FP group; base rate fixed)
        bl = np.empty((N_BOOT, P))
        for b in range(N_BOOT):
            bi = RNG.integers(0, n_fp, n_fp)
            bl[b] = ever[fp][bi].mean(0) / (base + 1e-12)
        lo = np.percentile(bl, 2.5, axis=0); hi = np.percentile(bl, 97.5, axis=0)
        # rank by the LOWER confidence bound (conservative): this demotes rare conditions whose
        # high point-enrichment rests on few counts and a wide interval, and promotes the
        # well-powered comorbidities, so the list is not dominated by unstable rare hits.
        top = [o for o in np.argsort(-lo) if elig[o]][:TOPK]
        for rank, y in enumerate(top, 1):
            rows.append({"target": disp, "target_phecode": ph, "n_flagged": n_flag, "n_fp": n_fp,
                         "ppv": float(ppv), "fp_per_100": float(fp_per_100), "rank": rank,
                         "actual_phecode": phe[y], "actual_name": names[y], "actual_category": cats[y],
                         "same_category": bool(cats[y] == cats[xj]),
                         "enrichment": float(enr[y]), "lo": float(lo[y]), "hi": float(hi[y]),
                         "ci_excludes_1": bool(lo[y] > 1.0),
                         "pct_of_fp": float(100.0 * rate_fp[y]), "base_rate": float(base[y]),
                         "cnt_fp": int(cnt_fp[y])})
            print(f"   {rank}. {names[y][:34]:34s} [{cats[y][:4]}] enr={enr[y]:4.1f} "
                  f"[{lo[y]:.1f},{hi[y]:.1f}]  {100*rate_fp[y]:.0f}% of FPs")

    pd.DataFrame(rows).to_csv(f"{out}/confusion_fp.csv", index=False)
    pd.DataFrame(summ_rows).to_csv(f"{out}/confusion_fp_summary.csv", index=False)
    np.save(f"{out}/confusion_fp_category_matrix.npy", cat_mat)
    with open(f"{out}/confusion_fp_category_labels.json", "w") as f:
        json.dump({"targets": [d for d, _ in TARGETS], "categories": ucats}, f)
    print(f"\nwrote {out}/confusion_fp.csv (+ summary, category matrix)")


if __name__ == "__main__":
    main()
