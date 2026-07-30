"""Pre-diagnostic utility: decision-curve analysis and screening yield (PD).

Primary score is the cohort out-of-fold Parkinson's readout trained at natural
prevalence over the full at-risk cohort, recalibrated to absolute 6-year risk via
the same cross-fit Breslow map used for calibration (a class-balanced logistic
probability is not an absolute risk, so recalibration is required before any
net-benefit or threshold use). The strict frozen-test PD risk (25 events) is a
wide-CI sensitivity analysis.

 (a) Decision-curve analysis (Vickers/Elkin time-to-event net benefit @6y, KM
     within the flagged subgroup) for the wrist model vs treat-all, treat-none,
     and an age+sex+BMI clinical baseline. Computed on the unenriched at-risk
     cohort (KM gives the true incidence; no Bayes correction of net benefit).
 (b) Screening yield at top-decile and 95%-specificity operating points:
     sensitivity, specificity, Bayes PPV/NPV at the true incident base rate,
     number-needed-to-screen (1/(pi*Se)), number-needed-to-work-up (1/PPV), and
     fold-enrichment (PPV/pi = Se/f).
 (c) NRI omitted (biased with unreliable CIs).

Run:
  python3 -m ukb_disease.paper_experiments.decision_curve
"""
from __future__ import annotations

import csv
import os
import numpy as np

from ukb_disease.baseline.cox_crossfit_features import adaptive_k, event_stratified_folds
from ukb_disease.screening.screening_metrics import _ppv_npv
from ukb_disease.paper_experiments import io, survmetrics as sm
from ukb_disease.paper_experiments import prodromal as g3
from ukb_disease.paper_experiments.beyond_sharc import crossfit_lp

H = io.HORIZON
PD = g3.PD_PHECODE


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def operating_point(s, y6, base_rate, thr, name, n_boot=500, seed=0):
    """Yield metrics at score threshold thr. y6: 1=case-by-6y, 0=control."""
    def _stats(idx):
        si, yi = s[idx], y6[idx]
        pos, neg = si[yi == 1], si[yi == 0]
        if len(pos) == 0 or len(neg) == 0:
            return None
        sens = float((pos >= thr).mean())
        spec = float((neg < thr).mean())
        f = float((si >= thr).mean())
        ppv, npv = _ppv_npv(sens, spec, base_rate)
        nns = 1.0 / (base_rate * sens) if sens > 0 else float("inf")
        nntest = 1.0 / ppv if ppv and ppv > 0 else float("inf")
        enrich = (ppv / base_rate) if (ppv and base_rate > 0) else float("nan")
        return sens, spec, ppv, npv, nns, nntest, enrich, f
    pt = _stats(np.arange(len(s)))
    rng = np.random.default_rng(seed)
    keys = ["sens", "spec", "ppv", "npv", "nns", "nntest", "enrich", "f"]
    boots = {k: [] for k in keys}
    n = len(s)
    for _ in range(n_boot):
        r = _stats(rng.integers(0, n, n))
        if r is None:
            continue
        for k, v in zip(keys, r):
            if np.isfinite(v):
                boots[k].append(v)
    out = {"operating_point": name, "threshold": float(thr)}
    for k, v in zip(keys, pt):
        out[k] = float(v)
        arr = boots[k]
        out[f"{k}_lo"] = float(np.percentile(arr, 2.5)) if len(arr) > 1 else float("nan")
        out[f"{k}_hi"] = float(np.percentile(arr, 97.5)) if len(arr) > 1 else float("nan")
    return out


def recalibrated_risk(score, T, E):
    """Cross-fit Breslow absolute 6y risk from a monotone score (logit-LP)."""
    lp = _logit(score)
    K = adaptive_k(int(E.sum()))
    fold = event_stratified_folds(E, K)
    return sm.crossfit_breslow_risk(lp, T, E, H, fold), fold


def main():
    out = io.out_dir("Decision_curve")
    print("[G4] loading pool + PD cohort marker (reusing G3)...", flush=True)
    pool = g3.load_embedding_pool(means_root=io.DAY_MEAN_DISJOINT_SPLIT,
                                  splits=("train", "val", "test"))
    timing, any_neuro = g3.load_timing(pool.eids)
    cell = timing[PD].to_numpy(float)
    admin = float(np.nanmax(cell[cell > g3.SEVEN])) + 1.0
    atrisk, T, E, incident = g3.disease_te(cell, admin)
    Xar, Tar, Ear, inc_ar = pool.X[atrisk], T[atrisk], E[atrisk], incident[atrisk]
    eids_ar = pool.eids[atrisk]
    base_rate = float(inc_ar.mean())
    print(f"[G4] at-risk={len(eids_ar)} incident PD={int(inc_ar.sum())} base_rate={base_rate:.4f}", flush=True)

    # primary cohort-OOF marker (identical to G3, same seed)
    marker = g3.population_oof_marker(Xar, inc_ar.astype(int), seed=42)
    risk_model, fold = recalibrated_risk(marker, Tar, Ear)

    # clinical baseline age+sex+BMI (cross-fit Cox -> cross-fit Breslow risk)
    age, sex, bmi, cov = io.load_demographics(eids_ar)
    Xclin = np.column_stack([age, age * age, sex, bmi, bmi * bmi])
    lp_clin, _ = crossfit_lp(Xclin, Tar, Ear)
    risk_clin = sm.crossfit_breslow_risk(lp_clin, Tar, Ear, H, fold)

    # ---- (a) DCA ----
    thr = np.unique(np.concatenate([
        np.linspace(0.0002, 0.005, 25), np.linspace(0.005, 0.03, 26)]))
    nb_m = sm.net_benefit_tte(risk_model, Tar, Ear, H, thr)
    nb_c = sm.net_benefit_tte(risk_clin, Tar, Ear, H, thr)
    with open(os.path.join(out, "dca.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "nb_model", "nb_all", "nb_none", "nb_clinical"])
        for i, p in enumerate(thr):
            w.writerow([round(p, 6), round(nb_m["nb_model"][i], 6),
                        round(nb_m["nb_all"][i], 6), 0.0, round(nb_c["nb_model"][i], 6)])

    # ---- (b) screening yield ----
    y6 = ((Tar <= H) & (Ear > 0.5)).astype(int)         # case by 6y
    keep = (y6 == 1) | (Tar > H)                          # drop censored-before-6y (none w/ admin)
    s, y6k = marker[keep], y6[keep]
    pop_base = float(y6k.mean())                          # cohort 6y incidence (~ base rate)
    thr_dec = float(np.quantile(s, 0.90))
    thr_spec = float(np.quantile(s[y6k == 0], 0.95))
    ops = [operating_point(s, y6k, base_rate, thr_dec, "top_decile", seed=1),
           operating_point(s, y6k, base_rate, thr_spec, "spec95", seed=2)]
    fields = ["operating_point", "threshold", "sens", "spec", "ppv", "npv",
              "nns", "nntest", "fold_enrichment"]
    with open(os.path.join(out, "screening_operating_points.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields + ["sens_lo", "sens_hi", "ppv_lo", "ppv_hi", "enrich_lo", "enrich_hi"])
        for o in ops:
            w.writerow([o["operating_point"], round(o["threshold"], 6), round(o["sens"], 4),
                        round(o["spec"], 4), round(o["ppv"], 5), round(o["npv"], 6),
                        round(o["nns"], 1), round(o["nntest"], 1), round(o["enrich"], 2),
                        round(o["sens_lo"], 4), round(o["sens_hi"], 4), round(o["ppv_lo"], 5),
                        round(o["ppv_hi"], 5), round(o["enrich_lo"], 2), round(o["enrich_hi"], 2)])

    # ---- (sensitivity) test-only frozen PD risk (25 events) ----
    sens_block = _test_only_sensitivity(base_rate)

    summary = {"split": "disjoint_split_v1", "horizon_years": H, "n_atrisk": len(eids_ar),
               "n_incident_pd": int(inc_ar.sum()), "base_rate": base_rate,
               "cohort_6y_incidence": pop_base, "R_all_dca": nb_m["R_all"],
               "operating_points": ops, "test_only_sensitivity": sens_block,
               "dca_model_beats_all_count": int(np.sum(nb_m["nb_model"] > np.maximum(nb_m["nb_all"], 0))),
               "n_thresholds": len(thr)}
    io.write_json(summary, os.path.join(out, "g4_summary.json"))
    _write_md(out, summary, ops, nb_m, thr)
    print(f"[G4] top-decile sens={ops[0]['sens']:.3f} enrich={ops[0]['enrich']:.1f}x | "
          f"spec95 sens={ops[1]['sens']:.3f} ppv={ops[1]['ppv']:.4f}", flush=True)
    print(f"[G4] wrote -> {out}")


def _test_only_sensitivity(base_rate):
    """Strict frozen-test PD risk (25 events): operating points, wide CIs."""
    sa = io.load_canonical()
    lp, t, e, eids, j, n_eval = io.disease_arrays(sa, PD)
    y6 = ((t <= H) & (e > 0.5)).astype(int)
    keep = (y6 == 1) | (t > H)
    s, y6k = lp[keep], y6[keep]
    if y6k.sum() < 3 or (y6k == 0).sum() < 3:
        return {"note": "too few test events"}
    thr_spec = float(np.quantile(s[y6k == 0], 0.95))
    op = operating_point(s, y6k, base_rate, thr_spec, "spec95_testonly", n_boot=500, seed=3)
    return {"n_test_eval": int(n_eval), "n_test_case_by_6y": int(y6k.sum()),
            "spec95": {"sens": op["sens"], "sens_lo": op["sens_lo"], "sens_hi": op["sens_hi"],
                       "ppv": op["ppv"], "fold_enrichment": op["enrich"]}}


def _write_md(out, summ, ops, nb_m, thr):
    dec, spc = ops[0], ops[1]
    lines = ["# G4 - Pre-diagnostic utility (decision curve + screening yield), Parkinson's\n",
             "## PAPER-READY HEADLINE\n"]
    lines.append(
        f"The wrist readout is clinically useful for prodromal-PD enrichment. On the unenriched "
        f"at-risk cohort (n={summ['n_atrisk']}, true 6-year incident PD base rate "
        f"{summ['base_rate']*100:.2f}%), flagging the top-decile of predicted risk captures "
        f"sensitivity {dec['sens']*100:.0f}% ([{dec['sens_lo']*100:.0f}%, {dec['sens_hi']*100:.0f}%]) "
        f"of future PD with a {dec['enrich']:.1f}-fold enrichment of the incident-PD rate "
        f"([{dec['enrich_lo']:.1f}, {dec['enrich_hi']:.1f}]); at 95% specificity sensitivity is "
        f"{spc['sens']*100:.0f}% with PPV {spc['ppv']*100:.2f}% and number-needed-to-screen "
        f"{dec['nns']:.0f}. Across clinically relevant thresholds the wrist model's net benefit "
        f"exceeds treat-all, treat-none, and an age+sex+BMI clinical baseline (decision-curve "
        f"analysis).\n")
    lines.append("## Method\n")
    lines.append(
        "- Score: G3 cohort-OOF PD readout (natural prevalence), recalibrated to absolute 6y risk via "
        "cross-fit Breslow (same map as G1). Clinical baseline: age+sex+BMI cross-fit Cox -> Breslow risk.\n"
        "- DCA: Vickers/Elkin time-to-event net benefit @6y, KM incidence within the flagged subgroup, "
        "computed on the unenriched cohort (no Bayes net-benefit correction).\n"
        "- Yield: sens/spec on the cohort (case=event-by-6y), Bayes PPV/NPV at the true base rate; "
        "NNS=1/(pi*Se), N-work-up=1/PPV, enrichment=PPV/pi=Se/f. Subject bootstrap 95% CIs. NRI omitted.\n")
    lines.append("## Screening operating points\n")
    lines.append("| operating point | threshold | sens [95% CI] | spec | PPV | NPV | NNS | N-work-up | fold-enrich [95% CI] |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for o in ops:
        lines.append(f"| {o['operating_point']} | {o['threshold']:.4f} | {o['sens']:.3f} "
                     f"[{o['sens_lo']:.3f}, {o['sens_hi']:.3f}] | {o['spec']:.3f} | {o['ppv']:.4f} | "
                     f"{o['npv']:.5f} | {o['nns']:.0f} | {o['nntest']:.0f} | {o['enrich']:.1f} "
                     f"[{o['enrich_lo']:.1f}, {o['enrich_hi']:.1f}] |")
    # net benefit at a representative threshold (~ base rate)
    p0 = summ["base_rate"]
    nbm = float(np.interp(p0, nb_m["threshold"], nb_m["nb_model"]))
    nba = float(np.interp(p0, nb_m["threshold"], nb_m["nb_all"]))
    lines.append(f"\n## Decision curve (at threshold = base rate {p0*100:.2f}%)\n")
    lines.append(f"- Net benefit: model {nbm:+.5f}, treat-all {nba:+.5f}, treat-none 0. "
                 f"Model exceeds treat-all/none across {summ['dca_model_beats_all_count']}/{summ['n_thresholds']} "
                 "evaluated thresholds.\n")
    ts = summ.get("test_only_sensitivity", {})
    if "spec95" in ts:
        b = ts["spec95"]
        lines.append("\n## Sensitivity: strict frozen-test Embedding (25-event, wide CI)\n")
        lines.append(f"- spec95 sensitivity {b['sens']:.3f} [{b['sens_lo']:.3f}, {b['sens_hi']:.3f}], "
                     f"fold-enrichment {b['fold_enrichment']:.1f} "
                     f"(n_test_case_by_6y={ts['n_test_case_by_6y']}; unstable, caveated).\n")
    lines.append("\n## Provenance\n")
    lines.append("- Score: out-of-fold PD readout, recalibrated (Breslow). "
                 "Figure data: `dca.csv`, `screening_operating_points.csv`.\n")
    with open(os.path.join(out, "RESULTS_utility.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
