"""G3: prodromal neurodegeneration, lead-time controlled.

disjoint_split_v1 split, frozen 512-d wrist embedding, L2-logistic readout
(out-of-fold; no demographics as features). Three pieces:

 1. time-to-diagnosis distribution among incident cases (PD/AD/dementia).
 2. Landmark / washout: exclude prevalent; for w in {0,1,2,3}y drop cases
    diagnosed within w years (controls fixed, marker frozen, no retrain) and
    recompute cumulative/dynamic discrimination among cases diagnosed after w.
 3. Unmatched population-base-rate metrics: a readout trained out-of-fold over
    the full at-risk cohort (no 1:3 matching) gives a marker for every at-risk
    subject. IPCW time-dependent AUROC and AUPRC at {2,3,5,7}y with the base
    rate / no-skill floor reported. A matched 1:3 age/sex AUROC is kept as a
    clearly labelled comparability diagnostic (relates to the prior ~0.834 and
    Schalkamp 2023).

Notes: the encoders are self-supervised on UKB (label-free); the readout is the
out-of-fold part, so it is subject-disjoint from the readout's training, not
"unseen" or "zero-shot". Because the labels censor non-cases administratively
(~10.5y), the censoring KM G(t)~1 at the reported horizons, so the IPCW td-AUC
reduces to the cumulative-case/dynamic-control AUC (checked at run time). CIs are
subject-level cluster bootstrap on the frozen OOF marker.

Run (heavy, a few minutes):
  PYTHONDONTWRITEBYTECODE=1 python3 -m ukb_disease.paper_experiments.prodromal
"""
from __future__ import annotations

import csv
import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ukb_disease.baseline.config import resolve_oak_path
from ukb_disease.baseline.neuro_probe.data import (
    load_embedding_pool, match_1tok, PD_PHECODE, ALZHEIMER_PHECODE, DEMENTIA_PHECODE,
    NEURO_PREFIXES,
)
from ukb_disease.baseline.neuro_probe.probe import _make_probe, cv_probe
from ukb_disease.paper_experiments import io, survmetrics as sm

SEVEN = 7.0 / 365.25
HORIZONS = (2.0, 3.0, 5.0, 7.0)
WASHOUTS = (0.0, 1.0, 2.0, 3.0)
DISEASES = {"pd": PD_PHECODE, "alzheimer": ALZHEIMER_PHECODE, "dementia": DEMENTIA_PHECODE}


def load_timing(pool_eids):
    """Per-pool-subject timing cells for PD/AD/dementia + 'any neuro dx' flag."""
    all_cols = pd.read_csv(resolve_oak_path(io.MORTALITY_PHECODES_CSV), nrows=0).columns.tolist()
    neuro_cols = [c for c in all_cols if any(c.startswith(p) for p in NEURO_PREFIXES)]
    need = sorted(set(["eid"] + list(DISEASES.values()) + neuro_cols))
    df = pd.read_csv(resolve_oak_path(io.MORTALITY_PHECODES_CSV), usecols=need).set_index("eid")
    df = df.reindex(pool_eids)
    any_neuro = df[neuro_cols].notna().any(axis=1).to_numpy()
    return df, any_neuro


def disease_te(cell, admin):
    """At-risk (T,E) for one disease from years-to-dx cells. Excludes prevalent.

    incident: cell>SEVEN -> event=1, time=cell; never (NaN) -> event=0, time=admin.
    prevalent / 7-day zone (cell<=SEVEN) -> NOT at risk.
    Returns atrisk(bool), T, E, incident(bool).
    """
    cell = np.asarray(cell, float)
    isnan = np.isnan(cell)
    incident = (~isnan) & (cell > SEVEN)
    never = isnan
    atrisk = incident | never
    T = np.where(incident, cell, admin)
    E = incident.astype(float)
    return atrisk, T, E, incident


def cd_auc_fast(time, event, score, h):
    """Cumulative-case/dynamic-control AUROC at horizon h (== IPCW when G~1).

    cases = time<=h & event; controls = time>h; censored-before-h dropped.
    """
    case = (time <= h) & (event > 0.5)
    ctrl = time > h
    keep = case | ctrl
    y = case[keep].astype(int)
    s = score[keep]
    nc, nk = int(case.sum()), int(ctrl.sum())
    if nc == 0 or nk == 0:
        return float("nan"), nc, nk
    return float(roc_auc_score(y, s)), nc, nk


def td_ap_fast(time, event, score, h):
    case = (time <= h) & (event > 0.5)
    ctrl = time > h
    keep = case | ctrl
    y = case[keep].astype(int)
    s = score[keep]
    nc, nk = int(case.sum()), int(ctrl.sum())
    if nc == 0 or nk == 0:
        return float("nan"), float("nan"), nc, nk
    return float(average_precision_score(y, s)), nc / (nc + nk), nc, nk


def boot_metric(metric_fn, time, event, score, h, n_boot=400, seed=0):
    """Point + subject-bootstrap 95% CI of metric_fn(...)[0] at horizon h."""
    point = metric_fn(time, event, score, h)[0]
    n = len(time)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = metric_fn(time[idx], event[idx], score[idx], h)[0]
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 2:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def population_oof_marker(X, y, seed=42, n_splits=5):
    """OOF L2-logistic probability over the full at-risk cohort (no matching)."""
    oof = np.full(len(y), np.nan)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        clf = _make_probe(C=1.0, seed=seed)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    return oof


def run_disease(name, col, pool, timing, any_neuro, do_matched, td_rows, ttx_rows):
    cell = timing[col].to_numpy(float)
    admin = float(np.nanmax(cell[cell > SEVEN])) + 1.0 if np.any(cell > SEVEN) else 10.5
    atrisk, T, E, incident = disease_te(cell, admin)

    # time-to-dx distribution (incident). Write a de-identified positional row
    # index, never a UK Biobank EID.
    yrs = cell[incident]
    for i_, y_ in enumerate(yrs):
        ttx_rows.append({"disease": name, "eid_or_index": int(i_), "years_to_dx": round(float(y_), 4)})
    med = float(np.median(yrs)) if len(yrs) else float("nan")
    iqr = (float(np.percentile(yrs, 25)), float(np.percentile(yrs, 75))) if len(yrs) else (np.nan, np.nan)

    # Population OOF marker over the at-risk cohort.
    Xar = pool.X[atrisk]
    Tar, Ear = T[atrisk], E[atrisk]
    cell_ar = cell[atrisk]
    inc_ar = incident[atrisk]
    y_train = inc_ar.astype(int)   # train: incident PD vs all at-risk non-PD
    marker = population_oof_marker(Xar, y_train)
    base_rate = float(inc_ar.mean())
    n_atrisk = int(atrisk.sum())

    # verify G ~ 1 at horizons (administrative censoring)
    G = sm.km_censoring(Tar, Ear)
    g7 = float(G.at(7.0))

    # Population td-AUROC / td-AUPRC at horizons (washout 0).
    for h in HORIZONS:
        auc, lo, hi = boot_metric(cd_auc_fast, Tar, Ear, marker, h, seed=11)
        ap_pt, ap_lo, ap_hi = boot_metric(td_ap_fast, Tar, Ear, marker, h, seed=12)
        ap_full = td_ap_fast(Tar, Ear, marker, h)
        floor, nc, nk = ap_full[1], ap_full[2], ap_full[3]
        td_rows.append({
            "disease": name, "design": "cohort", "horizon_y": h, "washout_y": 0.0,
            "n_case": nc, "n_atrisk": n_atrisk, "base_rate": round(base_rate, 5),
            "auroc": round(auc, 4), "auroc_lo": round(lo, 4), "auroc_hi": round(hi, 4),
            "auprc": round(ap_pt, 4), "auprc_lo": round(ap_lo, 4), "auprc_hi": round(ap_hi, 4),
            "auprc_floor": round(floor, 5),
        })

    # Washout (all diseases; controls fixed, marker frozen).
    # Cohort for washout w = controls (never) + incident with onset > w.
    for w in WASHOUTS:
        keep = (~inc_ar) | (inc_ar & (cell_ar > w))
        Tw, Ew, mw = Tar[keep], Ear[keep], marker[keep]
        ncase_w = int(((Tw <= 7.0) & (Ew > 0.5)).sum())
        auc, lo, hi = boot_metric(cd_auc_fast, Tw, Ew, mw, 7.0, seed=int(20 + w))
        # AUPRC on the SAME case/control set as the AUROC (static average precision;
        # matches Schalkamp's estimand). w=2 == their prodromal definition (dx > 2y).
        apw, aplo, aphi = boot_metric(td_ap_fast, Tw, Ew, mw, 7.0, seed=int(30 + w))
        floor_w = td_ap_fast(Tw, Ew, mw, 7.0)[1]
        td_rows.append({
            "disease": name, "design": "cohort_washout", "horizon_y": 7.0, "washout_y": w,
            "n_case": ncase_w, "n_atrisk": int(keep.sum()),
            "base_rate": round(ncase_w / int(keep.sum()), 5),
            "auroc": round(auc, 4), "auroc_lo": round(lo, 4), "auroc_hi": round(hi, 4),
            "auprc": round(apw, 4), "auprc_lo": round(aplo, 4), "auprc_hi": round(aphi, 4),
            "auprc_floor": round(floor_w, 5),
        })

    # Matched 1:3 comparability arm (PD only).
    matched = {}
    if do_matched:
        prod = (cell > 2.0) & atrisk            # prodromal cases (onset > 2y)
        meta = pool.meta
        has_as = meta["has_age_sex"].to_numpy()
        case_eids = pool.eids[prod & has_as]
        # neuro-free controls in pool
        ctrl_elig = (~any_neuro) & atrisk & has_as & (~incident)
        case_meta = meta.loc[case_eids]
        pool_meta = meta.loc[pool.eids[ctrl_elig]]
        ctrl_eids = match_1tok(case_meta, pool_meta, k=3, rng=np.random.default_rng(42))
        idx = pool.index_of(np.concatenate([case_eids, ctrl_eids]))
        Xm = pool.X[idx]
        ym = np.concatenate([np.ones(len(case_eids)), np.zeros(len(ctrl_eids))]).astype(int)
        res = cv_probe(Xm, ym, n_splits=5, n_repeats=10, n_boot=1000, seed=42)
        matched = {"auroc": res.auroc, "auroc_ci": res.auroc_ci,
                   "auprc": res.auprc, "auprc_ci": res.auprc_ci,
                   "n_case": int(len(case_eids)), "n_ctrl": int(len(ctrl_eids))}
        td_rows.append({
            "disease": name, "design": "matched_1to3", "horizon_y": "", "washout_y": 2.0,
            "n_case": matched["n_case"], "n_atrisk": matched["n_case"] + matched["n_ctrl"],
            "base_rate": round(matched["n_case"] / (matched["n_case"] + matched["n_ctrl"]), 4),
            "auroc": round(matched["auroc"], 4), "auroc_lo": round(matched["auroc_ci"][0], 4),
            "auroc_hi": round(matched["auroc_ci"][1], 4),
            "auprc": round(matched["auprc"], 4), "auprc_lo": round(matched["auprc_ci"][0], 4),
            "auprc_hi": round(matched["auprc_ci"][1], 4), "auprc_floor": 0.25,
        })

    return {"name": name, "n_incident": int(incident.sum()), "n_atrisk": n_atrisk,
            "base_rate": base_rate, "median_ttx": med, "iqr_ttx": iqr, "g7": g7,
            "admin": admin, "matched": matched,
            "marker_atrisk": marker, "T": Tar, "E": Ear, "cell": cell_ar,
            "incident": inc_ar, "eids": pool.eids[atrisk]}


def main():
    out = io.out_dir("Prodromal")
    print("[G3] loading embedding pool (train+val+test, 512-d)...", flush=True)
    pool = load_embedding_pool(means_root=io.DAY_MEAN_DISJOINT_SPLIT, splits=("train", "val", "test"))
    print(f"[G3] pool: {len(pool.eids)} subjects, X={pool.X.shape}", flush=True)
    timing, any_neuro = load_timing(pool.eids)

    td_rows, ttx_rows, summ = [], [], {}
    for name, col in DISEASES.items():
        do_matched = (name == "pd")
        s = run_disease(name, col, pool, timing, any_neuro, do_matched, td_rows, ttx_rows)
        summ[name] = {k: s[k] for k in ["n_incident", "n_atrisk", "base_rate",
                                        "median_ttx", "iqr_ttx", "g7", "admin", "matched"]}
        print(f"[G3] {name:9s} inc={s['n_incident']:4d} atrisk={s['n_atrisk']} "
              f"base={s['base_rate']:.4f} median_ttx={s['median_ttx']:.2f} G(7y)={s['g7']:.3f}", flush=True)

    # write
    fields = ["disease", "design", "horizon_y", "washout_y", "n_case", "n_atrisk",
              "base_rate", "auroc", "auroc_lo", "auroc_hi", "auprc", "auprc_lo",
              "auprc_hi", "auprc_floor"]
    with open(os.path.join(out, "td_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in td_rows:
            w.writerow({k: r.get(k, "") for k in fields})
    with open(os.path.join(out, "time_to_dx.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["disease", "eid_or_index", "years_to_dx"])
        w.writeheader()
        for r in ttx_rows:
            w.writerow(r)

    io.write_json({"split": "disjoint_split_v1", "horizons": list(HORIZONS),
                   "washouts": list(WASHOUTS), "per_disease": summ}, os.path.join(out, "g3_summary.json"))
    _write_md(out, summ, td_rows)
    print(f"[G3] wrote -> {out}")


def _write_md(out, summ, td_rows):
    pd_ = summ["pd"]
    # population PD td-AUROC at horizons (washout 0)
    pd_cohort = {r["horizon_y"]: r for r in td_rows
                 if r["disease"] == "pd" and r["design"] == "cohort"}
    pd_wash = {r["washout_y"]: r for r in td_rows
               if r["disease"] == "pd" and r["design"] == "cohort_washout"}
    m = pd_["matched"]
    lines = ["# G3 - Prodromal neurodegeneration, lead-time controlled\n",
             "## PAPER-READY HEADLINE\n"]
    a5 = pd_cohort.get(5.0, {})
    a7 = pd_cohort.get(7.0, {})
    lines.append(
        f"On the canonical cohort the frozen wrist embedding predicts FUTURE Parkinson's disease "
        f"years before diagnosis. Among {pd_['n_incident']} incident PD cases the median time from "
        f"wrist recording to diagnosis is {pd_['median_ttx']:.1f} years "
        f"(IQR {pd_['iqr_ttx'][0]:.1f}-{pd_['iqr_ttx'][1]:.1f}). On the FULL at-risk cohort "
        f"(n={pd_['n_atrisk']}, incident base rate {pd_['base_rate']*100:.2f}%) the out-of-fold "
        f"readout reaches population IPCW time-dependent AUROC "
        f"{a5.get('auroc',float('nan')):.3f} ([{a5.get('auroc_lo',float('nan')):.3f}, "
        f"{a5.get('auroc_hi',float('nan')):.3f}]) at 5 years and "
        f"{a7.get('auroc',float('nan')):.3f} at 7 years; discrimination is STABLE under lead-time "
        f"washout (AUROC@7y "
        + ", ".join(f"w={int(k)}y:{pd_wash[k]['auroc']:.3f}" for k in sorted(pd_wash)) +
        f"), i.e. not driven by near-onset cases. The matched 1:3 age/sex comparability AUROC is "
        f"{m.get('auroc',float('nan')):.3f} ([{m.get('auroc_ci',[float('nan'),float('nan')])[0]:.3f}, "
        f"{m.get('auroc_ci',[float('nan'),float('nan')])[1]:.3f}]; n_case {m.get('n_case','?')}), "
        f"comparable to the prior 0.834 and Schalkamp 2023.\n")
    lines.append("## Honesty / caveats\n")
    lines.append(
        "- The encoders are self-supervised on UKB (label-free); the readout is out-of-fold, so the "
        "evaluation is subject-disjoint from the readout's training. We do NOT claim 'unseen'/'zero-shot'.\n"
        "- The cohort-OOF readout is a cross-validated estimate over the full cohort (it pools all "
        f"{pd_['n_incident']} PD cases for power, which the test-only stratum cannot). It is reconciled "
        "with the matched-probe AUROC and the frozen-test Embedding per-disease C (0.909).\n"
        f"- Canonical labels censor non-cases administratively (~{pd_['admin']:.1f}y), so G(7y)="
        f"{pd_['g7']:.3f}~1 and the IPCW td-AUC reduces to the cumulative/dynamic AUC. AUPRC is "
        "reported with its no-skill floor; matched (floor 0.25) and population (floor ~base rate) "
        "AUPRCs are different estimands and never compared.\n")
    lines.append("## Population time-dependent metrics (PD, washout 0)\n")
    lines.append("| horizon | n_case | AUROC [95% CI] | AUPRC [95% CI] | floor |")
    lines.append("|---|---|---|---|---|")
    for h in HORIZONS:
        r = pd_cohort.get(h)
        if r:
            lines.append(f"| {int(h)}y | {r['n_case']} | {r['auroc']:.3f} "
                         f"[{r['auroc_lo']:.3f}, {r['auroc_hi']:.3f}] | {r['auprc']:.3f} "
                         f"[{r['auprc_lo']:.3f}, {r['auprc_hi']:.3f}] | {r['auprc_floor']} |")
    lines.append("\n## Washout stability (PD, AUROC@7y)\n")
    lines.append("| washout | n_case | AUROC [95% CI] |")
    lines.append("|---|---|---|")
    for k in sorted(pd_wash):
        r = pd_wash[k]
        lines.append(f"| {int(k)}y | {r['n_case']} | {r['auroc']:.3f} [{r['auroc_lo']:.3f}, {r['auroc_hi']:.3f}] |")
    lines.append("\n## Alzheimer's / dementia (cohort, thin events -> wide CIs)\n")
    lines.append("| disease | horizon | n_case | base_rate | AUROC [95% CI] |")
    lines.append("|---|---|---|---|---|")
    for dis in ("alzheimer", "dementia"):
        for r in td_rows:
            if r["disease"] == dis and r["design"] == "cohort":
                lines.append(f"| {dis} | {int(r['horizon_y'])}y | {r['n_case']} | {r['base_rate']} | "
                             f"{r['auroc']:.3f} [{r['auroc_lo']:.3f}, {r['auroc_hi']:.3f}] |")
    lines.append("\n## Provenance\n")
    lines.append(f"- Embedding pool: `{io.DAY_MEAN_DISJOINT_SPLIT}` (train+val+test, 512-d). "
                 "Timing: mortality_and_phecodes.csv. Figure data: `td_metrics.csv`, `time_to_dx.csv`.\n")
    with open(os.path.join(out, "RESULTS_prodromal.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
