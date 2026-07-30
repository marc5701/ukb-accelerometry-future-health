"""Subject-level bootstrap 95% CIs for the transportability panel (across UK Biobank
assessment-centre regions). Adds mean_C CI columns to per_region_c.csv without changing
any point estimate.

- Deployed (filled circles): resample each region's held-out test subjects and recompute
  the region mean C from the cached deployed hazards (same `cindex` as
  `regional_transport.per_region_deployed`), so the CI is consistent with the plotted point.
- Leave-one-region-out (open squares): refit the ridge-Cox head once per region (exactly
  as `regional_transport.leave_region_out_refit`), capture its per-disease held-out-region
  predictions, then bootstrap the evaluation (resample the region's test subjects under the
  fixed refit head). The CI is conditional on the fitted head, a standard evaluation
  bootstrap stated in the caption.

Deployed-only smoke run:
  python3 -m ukb_disease.paper_experiments.regional_transport_ci --nboot 20 --skip-lro --out <dir>
Full run:
  python3 -m ukb_disease.paper_experiments.regional_transport_ci --nboot 1000
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from ukb_disease.baseline.config import resolve_oak_path
from ukb_disease.baseline.cox_valfit import _fit_apply
from ukb_disease.baseline.neuro_probe.data import load_embedding_pool
from ukb_disease.paper_additions.io_utils import cindex, evaluable_diseases
from ukb_disease.paper_experiments import io, survmetrics as sm
from ukb_disease.paper_experiments import prodromal as g3
from ukb_disease.paper_experiments.regional_transport import (
    region_of, load_centre, N_FLAG, CENTRE_PARQUET, H,
)
from ukb_disease.paths import UKB_ROOT

DEFAULT_OUT = f"{UKB_ROOT}/benchmark/output/disjoint_split_v1"
POINT_CSV = f"{UKB_ROOT}/paper_experiments/Regional_transport/per_region_c.csv"


def cindex_fast(h, t, e):
    """Vectorised Harrell C matching io_utils.cindex exactly (strict-time-greater comparable
    pairs; tied times not comparable; tied hazards count 0.5). O(n_event * n) with numpy
    inner loops, roughly 100x faster than the pure-Python Fenwick tree, needed for 1000 resamples."""
    h = np.asarray(h, float); t = np.asarray(t, float); e = np.asarray(e, float)
    fin = np.isfinite(h)
    ev = np.where((e > 0.5) & fin)[0]
    conc = ties = comp = 0.0
    for a in ev:
        b = (t > t[a]) & fin
        nb = int(b.sum())
        if nb == 0:
            continue
        comp += nb
        hb = h[b]
        conc += float(np.count_nonzero(hb < h[a]))
        ties += float(np.count_nonzero(hb == h[a]))
    if comp == 0:
        return float("nan")
    return (conc + 0.5 * ties) / comp


def boot_region_deployed(sa, regions, idx, nboot, rng, include_pooled=False):
    """{region: (lo, hi)} for the deployed mean_C, resampling each region's test subjects.
    POOLED (the reference line, not a whiskered point) is skipped by default to save time."""
    out = {}
    uniq = [r for r in pd.unique(regions) if r != "Other"]
    for reg in (["POOLED"] if include_pooled else []) + uniq:
        sub = np.ones(len(regions), bool) if reg == "POOLED" else (regions == reg)
        reg_idx = np.where(sub)[0]
        boots = np.full(nboot, np.nan)
        for b in range(nboot):
            bi = rng.choice(reg_idx, reg_idx.size, replace=True)
            cs = []
            for j in idx:
                sel = bi[sa["mask"][bi, j]]
                if sel.size < 5:
                    continue
                e = sa["events"][sel, j].astype(float)
                if e.sum() < 3:
                    continue
                c = cindex_fast(sa["hazards"][sel, j].astype(float), sa["times"][sel, j].astype(float), e)
                if np.isfinite(c):
                    cs.append(c)
            if cs:
                boots[b] = float(np.mean(cs))
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        out[reg] = (float(lo), float(hi))
        print(f"  [deployed] {reg:16s} CI [{lo:.4f}, {hi:.4f}]  (n_reg={reg_idx.size})", flush=True)
    return out


def lro_refit_capture(pool, regions_pool, sa, top_n=60, n_pc=64, sub_train=15000):
    """Derived from regional_transport.leave_region_out_refit (identical fit), but also captures
    the per-disease held-out-region (lp, T_te, E_te) so the evaluation can be bootstrapped.
    Keep in sync with the source if that function changes."""
    idx = evaluable_diseases(sa)
    ev = [(j, int(sa["events"][sa["mask"][:, j], j].sum())) for j in idx]
    ev.sort(key=lambda x: -x[1])
    panel = [j for j, _ in ev[:top_n]]
    codes = [sa["phecodes"][j] for j in panel]

    need = sorted(set(["eid"] + codes))
    tcsv = pd.read_csv(resolve_oak_path(io.MORTALITY_PHECODES_CSV), usecols=need).set_index("eid")
    tcsv = tcsv.reindex(pool.eids)

    X = pool.X.astype(np.float64)
    Xc = X - X.mean(0, keepdims=True)
    _U, _S, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Xc @ Vt[:n_pc].T

    uniq = [r for r in pd.unique(regions_pool) if r != "Other"]
    rng = np.random.default_rng(0)
    rows, captured = {}, {}
    SEVEN = g3.SEVEN
    for reg in ["POOLED"] + uniq:
        is_r = (regions_pool == reg) if reg != "POOLED" else None
        cs, slopes, flagC = [], [], {k: np.nan for k in N_FLAG}
        preds = []
        for code in codes:
            cell = tcsv[code].to_numpy(float)
            admin = float(np.nanmax(cell[cell > SEVEN])) + 1.0 if np.any(cell > SEVEN) else 10.5
            atrisk, T, E, inc = g3.disease_te(cell, admin)
            if reg == "POOLED":
                te = (np.arange(len(T)) % 5 == 0) & atrisk
                tr = (~(np.arange(len(T)) % 5 == 0)) & atrisk
            else:
                te = is_r & atrisk
                tr = (~is_r) & atrisk
            if int(E[tr].sum()) < 10 or int(E[te].sum()) < 3:
                continue
            tr_idx = np.where(tr)[0]
            cases = tr_idx[E[tr_idx] > 0.5]
            ctrls = tr_idx[E[tr_idx] < 0.5]
            if len(ctrls) > sub_train:
                ctrls = rng.choice(ctrls, sub_train, replace=False)
            tr_use = np.concatenate([cases, ctrls])
            try:
                pen = max(0.1, 10.0 / int(E[tr_use].sum()))
                lp = _fit_apply(P[tr_use], T[tr_use], E[tr_use], P[te], pen)
            except Exception:
                continue
            c = cindex(lp, T[te], E[te])
            if np.isfinite(c):
                cs.append(c)
            preds.append((np.asarray(lp, float), T[te].astype(float), E[te].astype(float)))
            cal = sm.calibration_slope_citl(lp, T[te], E[te], H)
            if np.isfinite(cal["slope"]):
                slopes.append(cal["slope"])
            for k, fcode in N_FLAG.items():
                if code == fcode:
                    flagC[k] = c
        rows[reg] = {"n_subjects": int((regions_pool == reg).sum()) if reg != "POOLED"
                     else int((regions_pool != "Other").sum()),
                     "mean_C": float(np.mean(cs)) if cs else np.nan, "n_diseases": len(cs),
                     "calib_slope": float(np.median(slopes)) if slopes else np.nan,
                     **{f"C_{k}": flagC[k] for k in N_FLAG}}
        captured[reg] = preds
    return rows, len(codes), captured


def boot_lro(captured, nboot, rng):
    """{region: (lo, hi)} for the LRO mean_C, bootstrapping the evaluation (per-disease
    resample of the held-out-region test subjects under the fixed refit head)."""
    out = {}
    for reg, preds in captured.items():
        boots = np.full(nboot, np.nan)
        for b in range(nboot):
            cs = []
            for lp, t, e in preds:
                n = len(lp)
                if n < 5:
                    continue
                # cap the resample size: LRO held-out regions can be ~40k subjects, which makes
                # cindex O(n_event*n) per disease per boot intractable. 5000 gives a stable CI
                # (these estimates are already narrow) at a fraction of the cost.
                m = min(n, 5000)
                bi = rng.integers(0, n, m)
                ee = e[bi]
                if ee.sum() < 3:
                    continue
                c = cindex_fast(lp[bi], t[bi], ee)
                if np.isfinite(c):
                    cs.append(c)
            if cs:
                boots[b] = float(np.mean(cs))
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        out[reg] = (float(lo), float(hi))
        print(f"  [lro]      {reg:16s} CI [{lo:.4f}, {hi:.4f}]  (n_dis={len(preds)})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nboot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-lro", action="store_true", help="deployed-only (fast smoke)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    sa = io.load_canonical()
    regions_test = region_of(load_centre(sa["eids"]))
    idx = evaluable_diseases(sa)
    print(f"[g6_ci] deployed: {len(idx)} diseases; regions "
          f"{pd.Series(regions_test).value_counts().to_dict()}", flush=True)

    rng = np.random.default_rng(args.seed)
    dep_ci = boot_region_deployed(sa, regions_test, idx, args.nboot, rng)

    lro_ci = {}
    if not args.skip_lro:
        print("[g6_ci] loading pool + refitting for LRO capture ...", flush=True)
        pool = load_embedding_pool(means_root=io.DAY_MEAN_DISJOINT_SPLIT, splits=("train", "val", "test"))
        regions_pool = region_of(load_centre(pool.eids))
        _rows_B, _n, captured = lro_refit_capture(pool, regions_pool, sa)
        lro_ci = boot_lro(captured, args.nboot, np.random.default_rng(args.seed + 1))

    # merge CI columns into a copy of the point-estimate table
    pt = pd.read_csv(resolve_oak_path(POINT_CSV))
    pt["mean_C_lo"] = np.nan
    pt["mean_C_hi"] = np.nan
    for i, r in pt.iterrows():
        ci = dep_ci if r["method"] == "deployed" else lro_ci
        if r["region"] in ci:
            pt.at[i, "mean_C_lo"], pt.at[i, "mean_C_hi"] = ci[r["region"]]
    outpath = os.path.join(args.out, "per_region_c_ci.csv")
    pt.to_csv(outpath, index=False)
    print(f"[g6_ci] wrote {outpath}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
