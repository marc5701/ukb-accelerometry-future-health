"""Per-disease Harrell-C plus subject-level bootstrap 95% CI for ALL evaluable
outcomes, plus per-category mean-C 95% CI. Feeds Figure 1 panels b (highlight CIs)
and c (category-mean CIs).

Point estimates are taken verbatim from per_disease_c.json (the same load_model plus
harrell_counts harness that produced the plotted values), so the CI is an additive
overlay that stays consistent with the figure. The bootstrap is a single subject
resample per iteration applied across every disease (a cluster bootstrap over the
shared test cohort), so category means get a proper CI that accounts for the shared
subjects.

Run (quick smoke):
  python3 -m ukb_disease.benchmark.per_disease_ci --nboot 20
Run (full):
  python3 -m ukb_disease.benchmark.per_disease_ci --nboot 1000
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from ukb_disease.paths import UKB_ROOT
from ukb_disease.benchmark.concordance_benchmark import load_model, harrell_counts, MEANC_MIN_EVAL
from ukb_disease.benchmark.per_disease_c import phecode_names

RUNS = f"{UKB_ROOT}/runs_cox"
PREFIX = "embedding_disjoint_split"
PHECODE_MAP_CSV = f"{UKB_ROOT}/metadata/Phecode_mapX_filtered.csv"
DEFAULT_OUT = f"{UKB_ROOT}/benchmark/output/disjoint_split_v1"


def category_map():
    """phecode -> PhecodeCategory (raw key, same source as figlib.phecode_map)."""
    m = pd.read_csv(PHECODE_MAP_CSV)[["Phecode", "PhecodeCategory"]].drop_duplicates("Phecode")
    cat = {str(k): v for k, v in zip(m.Phecode, m.PhecodeCategory)}
    cat["time_to_death"] = "Mortality"
    return cat


def load_point_estimates(out_dir):
    """388 plotted outcomes + point C, verbatim from per_disease_c.json."""
    j = json.load(open(os.path.join(out_dir, "per_disease_c.json")))
    per = j["embedding_disjoint_split"]["per_disease"] if "embedding_disjoint_split" in j else j["per_disease"]
    return per  # {phecode: {c, n_eval, n_event, is_mortality}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-dir", default=DEFAULT_OUT, help="dir holding per_disease_c.json (read)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="dir to write the CI files")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    per = load_point_estimates(args.json_dir)
    catmap = category_map()

    mean_h, _by_seed, base, seeds = load_model(RUNS, PREFIX)
    T, E, M = base["T"], base["E"], base["M"]
    N, P = T.shape
    names = phecode_names(RUNS, PREFIX, P)
    col = {n: j for j, n in enumerate(names)}

    # Align the plotted diseases to hazard columns; keep the figure's exact set/order.
    phes, js, point_c, cats, n_ev, n_evt = [], [], [], [], [], []
    for phe, rec in per.items():
        if phe not in col:
            print(f"  WARN {phe} not in hazard columns; skipped")
            continue
        c = catmap.get(phe, "Other")
        if not isinstance(c, str) or c == "" or (isinstance(c, float) and np.isnan(c)):
            c = "Other"
        phes.append(phe); js.append(col[phe]); point_c.append(float(rec["c"]))
        cats.append(c); n_ev.append(int(rec["n_eval"])); n_evt.append(int(rec["n_event"]))
    D = len(phes)
    js = np.asarray(js)
    print(f"[per_disease_ci] {D} diseases aligned; N={N} subjects; nboot={args.nboot}", flush=True)

    # Sanity: recomputed point C must match the JSON (same harness).
    for k in ("NS_324.11", "CV_424", "time_to_death"):
        if k in col:
            j = col[k]; m = M[:, j]
            conc, comp = harrell_counts(mean_h[m, j], T[m, j], E[m, j])
            print(f"  check {k}: recomputed C={conc/comp:.5f} vs json {per[k]['c']}")

    rng = np.random.default_rng(args.seed)
    boot_c = np.full((args.nboot, D), np.nan)
    all_idx = np.arange(N)
    t0 = time.time()
    for b in range(args.nboot):
        bi = rng.choice(all_idx, N, replace=True)
        Mb = M[bi]                     # (N, P) resampled masks
        for di in range(D):
            j = js[di]
            sel = bi[Mb[:, j]]
            if sel.size < MEANC_MIN_EVAL:
                continue
            e = E[sel, j].astype(float)
            if e.sum() < 1:
                continue
            conc, comp = harrell_counts(mean_h[sel, j], T[sel, j], e)
            if comp > 0:
                boot_c[b, di] = conc / comp
        if (b + 1) % max(1, args.nboot // 10) == 0:
            print(f"  boot {b + 1}/{args.nboot}  ({time.time() - t0:.0f}s)", flush=True)

    lo = np.nanpercentile(boot_c, 2.5, axis=0)
    hi = np.nanpercentile(boot_c, 97.5, axis=0)

    diseases = {}
    for di, phe in enumerate(phes):
        diseases[phe] = dict(phecode=phe, category=cats[di],
                             C=round(point_c[di], 4),
                             CI_lo=round(float(lo[di]), 4), CI_hi=round(float(hi[di]), 4),
                             n_event=n_evt[di], n_eval=n_ev[di])
    json.dump(dict(nboot=args.nboot, seeds=list(map(int, seeds)), n_diseases=D, diseases=diseases),
              open(os.path.join(args.out, "per_disease_c_ci.json"), "w"), indent=2)

    # Per-category mean-C CI: point mean == figure catmean; CI from per-boot category means.
    cat_arr = np.asarray(cats)
    pc = np.asarray(point_c)
    rows = []
    for c in sorted(set(cats)):
        mask = cat_arr == c
        pt = float(pc[mask].mean())
        bm = np.nanmean(boot_c[:, mask], axis=1)
        clo, chi = np.nanpercentile(bm, [2.5, 97.5])
        rows.append(dict(category=c, mean_C=round(pt, 4),
                         CI_lo=round(float(clo), 4), CI_hi=round(float(chi), 4),
                         n_diseases=int(mask.sum())))
    pd.DataFrame(rows).sort_values("mean_C").to_csv(
        os.path.join(args.out, "per_category_c_ci.csv"), index=False)

    print(f"[per_disease_ci] wrote per_disease_c_ci.json ({D} diseases) + "
          f"per_category_c_ci.csv ({len(rows)} categories) -> {args.out}  "
          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
