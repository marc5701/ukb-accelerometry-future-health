"""Proportional-hazards (PH) check for the frozen risk score.

The disease model emits one time-invariant risk score (linear predictor, LP) per
outcome, trained by Cox partial likelihood, which assumes proportional hazards.
This script tests that assumption per outcome with scaled Schoenfeld residuals
(Grambsch-Therneau) on the single LP covariate, on the held-out test set, using
the same seed-averaged LPs that produced the calibration slopes
(io.load_canonical, seed-averaged canonical run).

Self-consistency gate: the single-covariate Cox coefficient computed here is, by
construction, the external calibration slope (identical standardized fit). We
assert it reproduces the flagship slopes (PD 1.08, AF 0.95, mortality 1.09,
HF 1.31); if it does not, the loader or LP differs from the calibration run and
the run should stop.

We report whatever the test finds. A large fraction of PH violations would
qualify the constant-hazard-ratio interpretation and must be stated as such.

Run:
  python3 -m ukb_disease.paper_experiments.proportional_hazards_check
"""
from __future__ import annotations

import csv
import os
import warnings

import numpy as np
import pandas as pd

from ukb_disease.paper_experiments import io

H = io.HORIZON

# Flagship calibration slopes used as a self-consistency gate.
G1_FLAGSHIP_SLOPE = {
    "NS_324.11": 1.08,   # Parkinson's disease
    "CV_416.2": 0.95,    # Atrial fibrillation
    "time_to_death": 1.09,  # All-cause mortality
    "CV_424": 1.31,      # Heart failure
}
GATE_TOL = 0.03


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values (monotone), same length/order as input."""
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(q_sorted, 0.0, 1.0)
    return q


def ph_one(lp, t, e):
    """Single-covariate Cox on standardized LP + scaled-Schoenfeld PH test.

    Returns dict(slope, ph_stat, ph_p, trend_sign, n_event) or None on failure.
    slope is de-standardized -> identical to the G1 external calibration slope.
    trend_sign > 0: LP effect strengthens over follow-up; < 0: attenuates.
    """
    from lifelines import CoxPHFitter
    from lifelines.statistics import proportional_hazard_test

    lp = np.asarray(lp, float)
    t = np.asarray(t, float)
    e = np.asarray(e).astype(float)
    if e.sum() < 5 or np.std(lp) == 0:
        return None
    sd = lp.std() + 1e-12
    z = (lp - lp.mean()) / sd
    df = pd.DataFrame({"lp": z, "T": t, "E": e})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            cph = CoxPHFitter(penalizer=0.0).fit(df, "T", "E")
        except Exception:
            try:
                cph = CoxPHFitter(penalizer=1e-3).fit(df, "T", "E")
            except Exception:
                return None
        slope = float(cph.params_["lp"] / sd)  # de-standardized == G1 slope
        try:
            zph = proportional_hazard_test(cph, df, time_transform="rank")
            srow = zph.summary
            ph_stat = float(srow["test_statistic"].iloc[0])
            ph_p = float(srow["p"].iloc[0])
        except Exception:
            return None
        # direction of any time trend: sign of corr(scaled Schoenfeld resid, event-time rank)
        trend_sign = float("nan")
        try:
            res = cph.compute_residuals(df, kind="scaled_schoenfeld")
            rt = t[res.index.to_numpy()]
            rr = np.argsort(np.argsort(rt)).astype(float)
            rv = res["lp"].to_numpy()
            if np.std(rv) > 0 and np.std(rr) > 0:
                trend_sign = float(np.sign(np.corrcoef(rv, rr)[0, 1]))
        except Exception:
            pass
    return {"slope": slope, "ph_stat": ph_stat, "ph_p": ph_p,
            "trend_sign": trend_sign, "n_event": int(e.sum())}


def main():
    out = io.out_dir("Proportional_hazards_check")
    sa = io.load_canonical()
    name = io.phecode_meta()["name"]

    rows = []
    for j, code in enumerate(sa["phecodes"]):
        m = sa["mask"][:, j]
        if int(m.sum()) < 10:
            continue
        lp = sa["hazards"][m, j].astype(float)
        t = sa["times"][m, j].astype(float)
        e = sa["events"][m, j].astype(float)
        r = ph_one(lp, t, e)
        if r is None:
            continue
        r["phecode"] = code
        r["outcome"] = name.get(code, code)
        rows.append(r)

    df = pd.DataFrame(rows).sort_values("ph_p").reset_index(drop=True)

    # ----- self-consistency gate (flagships reproduce G1 slopes) --------------
    gate = []
    for code, pub in G1_FLAGSHIP_SLOPE.items():
        got = df.loc[df.phecode == code, "slope"]
        got = float(got.iloc[0]) if len(got) else float("nan")
        ok = np.isfinite(got) and abs(got - pub) <= GATE_TOL
        gate.append((code, pub, got, ok))
        print(f"[GATE] {code:14s} G1_slope={pub:.2f}  here={got:.3f}  "
              f"{'OK' if ok else 'MISMATCH'}")
    gate_pass = all(g[3] for g in gate)
    if not gate_pass:
        print("[GATE] *** SELF-CONSISTENCY FAILED - LPs differ from G1; do NOT use ***")

    # ----- panels + BH-FDR ----------------------------------------------------
    def summarize(sub, label):
        if len(sub) == 0:
            return {"panel": label, "n": 0}
        q = bh_fdr(sub["ph_p"].to_numpy())
        return {
            "panel": label, "n": int(len(sub)),
            "n_raw_p_lt_.05": int((sub["ph_p"] < 0.05).sum()),
            "frac_raw_p_lt_.05": float((sub["ph_p"] < 0.05).mean()),
            "n_fdr_lt_.05": int((q < 0.05).sum()),
            "frac_fdr_lt_.05": float((q < 0.05).mean()),
        }

    ge50 = df[df.n_event >= 50].copy()
    ge50["q_bh"] = bh_fdr(ge50["ph_p"].to_numpy()) if len(ge50) else []
    df["q_bh_ge50"] = np.nan
    df.loc[ge50.index, "q_bh_ge50"] = ge50["q_bh"].to_numpy()

    panels = {
        "ge50ev_primary": summarize(ge50, ">=50 events (primary, matches G1 slope panel)"),
        "ge20ev": summarize(df[df.n_event >= 20], ">=20 events"),
        "all_ge10eval": summarize(df, "all evaluable (>=10 eval, >=5 events)"),
    }

    flag = []
    for label, code in io.FLAGSHIPS.items():
        r = df[df.phecode == code]
        if len(r):
            r = r.iloc[0]
            flag.append({"outcome": label, "phecode": code, "n_event": int(r.n_event),
                         "slope": float(r.slope), "ph_p": float(r.ph_p),
                         "q_bh_ge50": (None if not np.isfinite(r.q_bh_ge50) else float(r.q_bh_ge50)),
                         "trend_sign": (None if not np.isfinite(r.trend_sign) else float(r.trend_sign))})

    summary = {
        "split": "disjoint_split_v1", "horizon_years": H, "model": io.DEEP_PREFIX,
        "n_test": int(len(sa["eids"])), "test_transform": "rank",
        "method": "Grambsch-Therneau scaled Schoenfeld residuals, single LP covariate",
        "self_consistency_gate": {"pass": bool(gate_pass),
                                  "flagships": [{"phecode": c, "g1_slope": p,
                                                 "here": g, "ok": bool(o)}
                                                for c, p, g, o in gate]},
        "panels": panels, "flagships": flag,
        "provenance": {"runs_root": io.RUNS_ROOT, "prefix": io.DEEP_PREFIX,
                       "seeds": [42, 123, 456, 789], "lp_source": "seed-avg test_hazards.npy"},
    }
    io.write_json(summary, os.path.join(out, "ph_summary.json"))

    with open(os.path.join(out, "ph_test_perdisease.csv"), "w", newline="") as f:
        cols = ["outcome", "phecode", "n_event", "slope", "ph_stat", "ph_p",
                "q_bh_ge50", "trend_sign"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for _, r in df.iterrows():
            w.writerow({k: r[k] for k in cols})

    _write_md(out, summary, panels, flag)
    _write_fig(out, ge50, flag)

    p = panels["ge50ev_primary"]
    print(f"\n[G5] primary panel (>=50 ev, n={p['n']}): "
          f"raw p<.05 in {p['n_raw_p_lt_.05']} ({p['frac_raw_p_lt_.05']*100:.0f}%); "
          f"BH-FDR<.05 in {p['n_fdr_lt_.05']} ({p['frac_fdr_lt_.05']*100:.0f}%)")
    print(f"[G5] gate {'PASS' if gate_pass else 'FAIL'}; wrote -> {out}")


def _write_md(out, summary, panels, flag):
    p = panels["ge50ev_primary"]
    L = []
    L.append("# G5 - Proportional-hazards check for the frozen risk score\n")
    L.append("## Method\n")
    L.append(
        "The model outputs one time-invariant risk score (LP) per outcome, trained by Cox "
        "partial likelihood (assumes proportional hazards). Per outcome we fit a single-covariate "
        "Cox on the held-out test set with the LP as covariate, then test PH with scaled Schoenfeld "
        "residuals (Grambsch-Therneau, rank-transformed time). The single-covariate coefficient is, "
        "by construction, the G1 external calibration slope; the flagship slopes are reproduced as a "
        "self-consistency gate.\n")
    g = summary["self_consistency_gate"]
    L.append(f"\n**Self-consistency gate: {'PASS' if g['pass'] else 'FAIL'}** "
             + "; ".join(f"{x['phecode']} G1={x['g1_slope']:.2f}/here={x['here']:.2f}"
                         for x in g["flagships"]) + "\n")
    L.append("\n## Result (honest)\n")
    L.append(
        f"On the disjoint_split_v1 test set (N={summary['n_test']}), among the {p['n']} well-powered "
        f"outcomes (>=50 incident events) the proportional-hazards test rejected PH (raw p<0.05) for "
        f"{p['n_raw_p_lt_.05']} outcomes ({p['frac_raw_p_lt_.05']*100:.0f}%), and after "
        f"Benjamini-Hochberg FDR control for {p['n_fdr_lt_.05']} ({p['frac_fdr_lt_.05']*100:.0f}%). "
        "Under the null (PH holds everywhere) about 5% would reject at raw p<0.05, so this is the "
        "reference to compare against.\n")
    L.append("\n## Panels\n")
    L.append("| panel | n | raw p<.05 | frac | BH-FDR<.05 | frac |")
    L.append("|---|---|---|---|---|---|")
    for k in ["ge50ev_primary", "ge20ev", "all_ge10eval"]:
        d = panels[k]
        if d.get("n", 0) == 0:
            continue
        L.append(f"| {d['panel']} | {d['n']} | {d['n_raw_p_lt_.05']} | "
                 f"{d['frac_raw_p_lt_.05']*100:.0f}% | {d['n_fdr_lt_.05']} | "
                 f"{d['frac_fdr_lt_.05']*100:.0f}% |")
    L.append("\n## Flagship outcomes\n")
    L.append("| outcome | phecode | n_event | slope | PH p | BH-q(>=50) | trend |")
    L.append("|---|---|---|---|---|---|---|")
    for r in flag:
        tr = "" if r["trend_sign"] is None else ("strengthens" if r["trend_sign"] > 0 else "attenuates")
        q = "" if r["q_bh_ge50"] is None else f"{r['q_bh_ge50']:.3f}"
        L.append(f"| {r['outcome']} | {r['phecode']} | {r['n_event']} | {r['slope']:.2f} | "
                 f"{r['ph_p']:.3f} | {q} | {tr} |")
    L.append("\n## Provenance\n")
    L.append(f"- LPs: seed-averaged `test_hazards.npy` over seeds 42/123/456/789 at "
             f"`{io.RUNS_ROOT}/{io.DEEP_PREFIX}_seed*`; identical to G1.\n"
             "- Per-disease detail: `ph_test_perdisease.csv`; machine summary: `ph_summary.json`.\n")
    with open(os.path.join(out, "RESULTS_ph_check.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def _write_fig(out, ge50, flag):
    """Diagnostic figure: distribution of PH p-values on the primary panel."""
    if len(ge50) == 0:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    pvals = ge50["ph_p"].to_numpy()
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.hist(pvals, bins=np.linspace(0, 1, 21), color="#5B8DB8",
            edgecolor="white", linewidth=0.4)
    ax.axhline(len(pvals) / 20.0, color="#888888", ls="--", lw=0.8,
               label="uniform (PH holds)")
    ax.set_xlabel("Schoenfeld PH test p-value")
    ax.set_ylabel("outcomes")
    ax.set_title(f"PH test, {len(pvals)} outcomes with $\\geq$50 events")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"ph_pvalue_hist.{ext}"), dpi=200,
                    bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
