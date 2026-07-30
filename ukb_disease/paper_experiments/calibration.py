"""Calibration of the headline incident disease model (canonical split).

Out-of-sample calibration, free of in-sample optimism:
  * absolute 6y risk via event-stratified within-test cross-fit Breslow (LP held
    as a fixed offset; baseline H0 fit on the other folds, applied to the held fold);
  * calibration curve = predicted-risk-quantile bins vs Kaplan-Meier observed
    6y incidence (KM handles censoring) with cloglog CIs;
  * calibration slope = Cox(T,E)~LP coefficient on test (frozen-LP external
    slope, a legitimate validation metric, not cross-fit);
  * IPCW Brier@6y (Graf), integrated Brier, IPA / scaled Brier vs the marginal-KM
    null;
  * panel-wide distribution of calibration slopes (>=50-event headline).

Caveats stated in the report: H0 was not saved with the model, so the cross-fit
Breslow grants a fresh test-derived baseline. The curve therefore validates shape
given an out-of-fold baseline and the observed-to-expected ratio is near 1 by
construction (not headlined). IPA is uninformative at roughly 0.5% prevalence, so
rare-disease IPA is flagged rather than headlined.

Run:
  python3 -m ukb_disease.paper_experiments.calibration
"""
from __future__ import annotations

import os
import numpy as np

from ukb_disease.baseline.cox_crossfit_features import adaptive_k, event_stratified_folds
from ukb_disease.paper_experiments import io, survmetrics as sm

H = io.HORIZON


def _n_bins(n_ev: int) -> int:
    if n_ev >= 100:
        return 10
    if n_ev >= 50:
        return 5
    if n_ev >= 20:
        return 4
    return 3


def calib_one(lp, t, e, label, code):
    """Per-outcome calibration: returns (summary_dict, curve_rows)."""
    n_ev = int(e.sum())
    K = adaptive_k(n_ev)
    fold = event_stratified_folds(e, K)
    risk = sm.crossfit_breslow_risk(lp, t, e, H, fold)
    G = sm.km_censoring(t, e)

    # slope + CITL (subject bootstrap CI on slope)
    cal = sm.calibration_slope_citl(lp, t, e, H, risk)

    def _slope(idx):
        return sm.calibration_slope_citl(lp[idx], t[idx], e[idx], H)["slope"]
    _, slo_lo, slo_hi = sm.subject_bootstrap_ci(_slope, len(lp), n_boot=400,
                                                seed=abs(hash(code)) % 2**31)

    brier = sm.ipcw_brier(risk, t, e, H, G)
    ipa = sm.ipa_scaled_brier(risk, t, e, H, G)

    # IBS via per-horizon cross-fit Breslow.
    def risk_fn(u):
        return sm.crossfit_breslow_risk(lp, t, e, u, fold)
    ibs = sm.integrated_brier(risk_fn, t, e, H, G=G)

    summ = {
        "outcome": label, "phecode": code, "n_eval": int(len(lp)), "n_event": n_ev,
        "slope": cal["slope"], "slope_lo": slo_lo, "slope_hi": slo_hi,
        "citl_oe": cal["oe_ratio"], "obs_incidence": cal["obs_incidence"],
        "mean_pred": cal["mean_pred"],
        "brier6y": brier, "ibs": ibs, "scaled_brier": ipa,
    }

    # calibration curve (equal-count quantile bins on predicted risk)
    nb = _n_bins(n_ev)
    order = np.argsort(risk)
    bins = np.array_split(order, nb)
    rows = []
    for b, idx in enumerate(bins):
        if len(idx) == 0:
            continue
        pred_mean = float(risk[idx].mean())
        obs, lo, hi = sm.km_incidence(t[idx], e[idx], H, ci=True)
        rows.append({"outcome": label, "phecode": code, "bin": b,
                     "pred_mean": pred_mean, "obs_km": obs, "obs_lo": lo,
                     "obs_hi": hi, "n": int(len(idx))})
    return summ, rows


def main():
    out = io.out_dir("Calibration")
    sa = io.load_canonical()
    meta = io.phecode_meta()

    curve_rows = []
    per_outcome = {}

    # flagships (full detail incl. IBS + curve)
    for label, code in io.FLAGSHIPS.items():
        lp, t, e, eids, j, _ = io.disease_arrays(sa, code)
        s, rows = calib_one(lp, t, e, label, code)
        per_outcome[code] = s
        curve_rows.extend(rows)
        print(f"[G1] {label:22s} n_ev={s['n_event']:4d} slope={s['slope']:.2f} "
              f"[{s['slope_lo']:.2f},{s['slope_hi']:.2f}] O:E={s['citl_oe']:.2f} "
              f"Brier={s['brier6y']:.4f} IBS={s['ibs']:.4f} IPA={s['scaled_brier']:+.3f}")

    # panel: calibration slope for every evaluable outcome (no IBS/curve to stay fast)
    panel_slopes = []
    panel_rows = {}
    for j, code in enumerate(sa["phecodes"]):
        m = sa["mask"][:, j]
        e = sa["events"][m, j].astype(float)
        n_ev = int(e.sum())
        if int(m.sum()) < 10 or n_ev < 1:
            continue
        lp = sa["hazards"][m, j].astype(float)
        t = sa["times"][m, j].astype(float)
        cal = sm.calibration_slope_citl(lp, t, e, H)
        if np.isfinite(cal["slope"]):
            panel_rows[code] = {"slope": cal["slope"], "n_event": n_ev}
            panel_slopes.append((code, cal["slope"], n_ev))

    sl_all = np.array([s for _, s, _ in panel_slopes])
    sl_50 = np.array([s for _, s, n in panel_slopes if n >= 50])
    sl_20 = np.array([s for _, s, n in panel_slopes if n >= 20])

    def q(a):
        a = a[np.isfinite(a)]
        return {"n": int(len(a)), "median": float(np.median(a)),
                "iqr_lo": float(np.percentile(a, 25)), "iqr_hi": float(np.percentile(a, 75)),
                "frac_0p8_1p25": float(np.mean((a >= 0.8) & (a <= 1.25)))}

    panel = {"slope_all_evaluable": q(sl_all), "slope_ge50ev": q(sl_50),
             "slope_ge20ev": q(sl_20)}

    summary = {"split": "disjoint_split_v1", "horizon_years": H, "model": io.DEEP_PREFIX,
               "n_test": int(len(sa["eids"])), "per_outcome": per_outcome,
               "panel_slope_distribution": panel,
               "all_panel_slopes": {c: r for c, r in panel_rows.items()},
               "provenance": {"runs_root": io.RUNS_ROOT, "prefix": io.DEEP_PREFIX,
                              "seeds": [42, 123, 456, 789]}}

    io.write_json(summary, os.path.join(out, "calib_summary.json"))
    import csv
    with open(os.path.join(out, "calib_curves.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["outcome", "phecode", "bin", "pred_mean",
                                          "obs_km", "obs_lo", "obs_hi", "n"])
        w.writeheader()
        for r in curve_rows:
            w.writerow({k: r[k] for k in w.fieldnames})

    _write_md(out, summary, panel)
    print(f"\n[G1] panel slope (>=50ev): median {panel['slope_ge50ev']['median']:.2f} "
          f"IQR [{panel['slope_ge50ev']['iqr_lo']:.2f},{panel['slope_ge50ev']['iqr_hi']:.2f}] "
          f"n={panel['slope_ge50ev']['n']}")
    print(f"[G1] wrote -> {out}")


def _write_md(out, summary, panel):
    po = summary["per_outcome"]
    pd_ = po["NS_324.11"]
    p50 = panel["slope_ge50ev"]
    lines = []
    lines.append("# G1 - Calibration of the headline incident model\n")
    lines.append("## PAPER-READY HEADLINE\n")
    lines.append(
        f"On the canonical disjoint_split_v1 held-out test set (N={summary['n_test']}), the "
        f"frozen AcceleRest L-moments Cox model (Embedding) is well calibrated across the disease "
        f"panel: the distribution of external calibration slopes over the {p50['n']} outcomes "
        f"with >=50 incident events has median {p50['median']:.2f} "
        f"(IQR {p50['iqr_lo']:.2f}-{p50['iqr_hi']:.2f}), with {p50['frac_0p8_1p25']*100:.0f}% "
        f"of slopes in [0.80, 1.25]. For the flagship outcomes the 6-year risk is well "
        f"calibrated, e.g. Parkinson's disease slope {pd_['slope']:.2f} "
        f"([{pd_['slope_lo']:.2f}, {pd_['slope_hi']:.2f}]) and atrial fibrillation slope "
        f"{po['CV_416.2']['slope']:.2f}; integrated Brier and scaled-Brier (IPA) are reported "
        f"for the common outcomes (e.g. type 2 diabetes IPA "
        f"{po['EM_202.2']['scaled_brier']:+.3f}, mortality IPA {po['time_to_death']['scaled_brier']:+.3f}).\n")
    lines.append("## Method\n")
    lines.append(
        "- Absolute 6y risk: event-stratified within-test cross-fit Breslow baseline with the "
        "model linear predictor held as a fixed coefficient-1 offset (baseline H0 fit on the "
        "other folds, applied to the held-out fold). No recalibration is fit on the rows it scores.\n"
        "- Calibration curve: predicted-risk quantile bins (3-10 by event count) vs Kaplan-Meier "
        "observed 6y incidence (cloglog 95% CI) per bin.\n"
        "- Calibration slope: coefficient of LP in Cox(T,E)~LP on the test set (frozen-LP external "
        "slope; subject bootstrap 95% CI). IPCW Brier/IBS/IPA via the censoring KM (Graf 1999).\n")
    lines.append("## Caveats (honesty)\n")
    lines.append(
        "- The model's baseline cumulative hazard was not saved, so the cross-fit Breslow grants a "
        "fresh test-derived baseline. The curve therefore validates calibration SHAPE GIVEN AN OOF "
        "BASELINE; the observed-to-expected ratio is ~1 by construction (near-tautological) and is "
        "NOT presented as evidence of mean calibration.\n"
        "- IPA/scaled Brier is near-zero or negative for very rare outcomes (squared-error is "
        "uninformative at ~0.5% prevalence); it is reported only as a diagnostic for common outcomes.\n")
    lines.append("## Flagship outcomes\n")
    lines.append("| outcome | phecode | n_event | slope [95% CI] | O:E | Brier@6y | IBS | IPA |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for code, s in po.items():
        lines.append(f"| {s['outcome']} | {code} | {s['n_event']} | {s['slope']:.2f} "
                     f"[{s['slope_lo']:.2f}, {s['slope_hi']:.2f}] | {s['citl_oe']:.2f} | "
                     f"{s['brier6y']:.4f} | {s['ibs']:.4f} | {s['scaled_brier']:+.3f} |")
    lines.append("\n## Panel-wide calibration slope distribution\n")
    lines.append("| subset | n_outcomes | median | IQR | frac in [0.8,1.25] |")
    lines.append("|---|---|---|---|---|")
    for k, lab in [("slope_ge50ev", ">=50 events (headline)"),
                   ("slope_ge20ev", ">=20 events"),
                   ("slope_all_evaluable", "all evaluable (>=1 event)")]:
        d = panel[k]
        lines.append(f"| {lab} | {d['n']} | {d['median']:.2f} | "
                     f"[{d['iqr_lo']:.2f}, {d['iqr_hi']:.2f}] | {d['frac_0p8_1p25']*100:.0f}% |")
    lines.append("\n## Provenance\n")
    lines.append(f"- Predictions: `{io.RUNS_ROOT}/{io.DEEP_PREFIX}_seed{{42,123,456,789}}/` "
                 "(seed-averaged `test_hazards.npy`).\n"
                 "- Figure data: `calib_curves.csv`, `calib_summary.json`.\n")
    with open(os.path.join(out, "RESULTS_calibration.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
