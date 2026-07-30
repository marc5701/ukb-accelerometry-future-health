"""Stage 2: near-onset sensitivity for prevalent-PD screening.

The same-distribution detector reaches high AUROC on prevalent PD, but those
cases are established disease (median time-since-diagnosis ~2.9 yr, up to ~17 yr)
and usually medicated. The screening-relevant question is whether detection holds
for earlier, less-medicated cases. We re-score PD out-of-fold (the same geometry,
reusing confound_audit.oof_predictions), save per-subject predictions, then split
the prevalent positives by years_since_diagnosis ( = -t for t < 0) and measure
AUROC / sens@95-spec of each subset against the same fixed control pool.

If early cases (<=2 yr since dx) are still detected well, the wrist signal is a
genuine early marker, not just established/medicated disease.

Run (CPU):
  python3 -u -m ukb_disease.screening.stage2_nearonset --out-dir <out_dir>
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


def sens_at_spec(y, s, spec=0.95):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= (1 - spec)
    return float(tpr[ok].max()) if ok.any() else float("nan")


def subset_auroc(y, s, pos_mask_in_pos, n_boot=2000, seed=0):
    """AUROC of (a subset of positives) vs ALL controls. `pos_mask_in_pos` selects
    among the positive rows; controls (y==0) are always all included."""
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    keep_pos = pos[pos_mask_in_pos]
    idx = np.concatenate([keep_pos, neg])
    yy, ss = y[idx], s[idx]
    a, lo, hi = boot_auroc(yy, ss, n_boot=n_boot, seed=seed)
    return a, lo, hi, sens_at_spec(yy, ss), int(len(keep_pos))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=f"{UKB_ROOT}/figures_out/confound_audit_out")
    ap.add_argument("--phecode", default="NS_324.11")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    z = np.load(resolve_oak_path(POOL), allow_pickle=True)
    eids = z["eids"].astype(np.int64)
    Xemb = z["X"].astype(np.float32)
    cv = np.array([fold_of_eid(int(e), args.n_folds) for e in eids], dtype=np.int64)
    phe = load_phecodes().reindex(index=eids)
    t = phe[args.phecode].to_numpy(np.float64)
    print(f"[load] {time.time()-t0:.1f}s; computing OOF for {args.phecode}", flush=True)

    idx, y, pred, _ = oof_predictions(t, cv, Xemb, n_folds=args.n_folds)
    a_all, lo_all, hi_all = boot_auroc(y, pred, n_boot=args.n_boot)
    print(f"[oof] pooled AUROC={a_all:.3f} [{lo_all:.3f},{hi_all:.3f}] "
          f"n_pos={int(y.sum())} n_ctrl={int((y==0).sum())}  (headline ref 0.917)", flush=True)

    # per-subject dump (de-identified: positional row index, not participant ids)
    tt = t[idx]
    ysd = np.where(tt < 0, -tt, 0.0)  # years since dx for the prevalent positives
    dump = pd.DataFrame({"row_index": np.arange(len(idx)), "y": y, "pred": pred,
                         "t": tt, "years_since_dx": np.where(y == 1, ysd, np.nan)})
    dump.to_csv(os.path.join(args.out_dir, f"oof_{args.phecode}.csv"), index=False)

    ysd_pos = ysd[y == 1]
    print(f"[dx] years_since_dx among {len(ysd_pos)} positives: "
          f"min={ysd_pos.min():.2f} med={np.median(ysd_pos):.2f} max={ysd_pos.max():.2f}", flush=True)

    rows = []
    # 1) clinically meaningful absolute bins
    clin = [("recent <=2yr", ysd_pos <= 2.0),
            ("mid 2-5yr", (ysd_pos > 2.0) & (ysd_pos <= 5.0)),
            ("long >5yr", ysd_pos > 5.0),
            ("all", np.ones_like(ysd_pos, bool))]
    for name, mask in clin:
        if mask.sum() < 5:
            continue
        a, lo, hi, sn, npos = subset_auroc(y, pred, mask, n_boot=args.n_boot)
        rows.append({"split": "clinical", "bin": name, "n_pos": npos,
                     "auroc": a, "auroc_lo": lo, "auroc_hi": hi, "sens95": sn})
        print(f"   [clin] {name:14s} n={npos:3d} AUROC={a:.3f} [{lo:.3f},{hi:.3f}] sens95={sn:.3f}", flush=True)

    # 2) tertiles of years_since_dx (balanced n)
    qs = np.quantile(ysd_pos, [1/3, 2/3])
    tert = np.digitize(ysd_pos, qs)
    for g in (0, 1, 2):
        mask = tert == g
        if mask.sum() < 5:
            continue
        rng = ysd_pos[mask]
        a, lo, hi, sn, npos = subset_auroc(y, pred, mask, n_boot=args.n_boot)
        rows.append({"split": "tertile", "bin": f"T{g} [{rng.min():.1f},{rng.max():.1f}]yr",
                     "n_pos": npos, "auroc": a, "auroc_lo": lo, "auroc_hi": hi, "sens95": sn})
        print(f"   [tert] T{g} [{rng.min():.1f},{rng.max():.1f}]yr n={npos:3d} "
              f"AUROC={a:.3f} [{lo:.3f},{hi:.3f}] sens95={sn:.3f}", flush=True)

    # 3) does the OOF score scale with years_since_dx among positives? (severity check)
    from scipy.stats import spearmanr
    rho, pval = spearmanr(ysd_pos, pred[y == 1])
    print(f"[scale] Spearman(score, years_since_dx | positives) rho={rho:+.3f} p={pval:.3f}  "
          f"(near 0 => detection not driven by disease duration)", flush=True)

    summ = {"phecode": args.phecode, "auroc_all": a_all, "auroc_all_ci": [lo_all, hi_all],
            "n_pos": int(y.sum()), "spearman_score_vs_dx": float(rho),
            "spearman_p": float(pval), "bins": rows}
    with open(os.path.join(args.out_dir, f"nearonset_{args.phecode}.json"), "w") as f:
        json.dump(summ, f, indent=2)
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, f"nearonset_{args.phecode}.csv"), index=False)
    print(f"\n[done] {time.time()-t0:.0f}s -> nearonset_{args.phecode}.{{csv,json}}, oof_{args.phecode}.csv", flush=True)


if __name__ == "__main__":
    main()
