"""Frailty-adjusted false-positive / misdiagnosis structure (companion to confusion_fp.py).

confusion_fp.py shows, for each target disease X, what the model's false positives (flagged at 95%
specificity but never diagnosed with X) actually have, as a raw enrichment
`rate_in_FPs / cohort_base_rate`. This companion tests whether that raw quantity is dominated by a
general frailty axis, and whether removing it makes the disease-specific pattern (e.g. Parkinson's to
constipation / orthostatic hypotension) emerge.

Context that shaped this analysis:
  - The model's PC1 ("shared axis") explains ~76% of the hazard variance but is only weakly tied to
    realized sickness: it correlates ~0.22 with realized disease burden on the full test cohort (0.29
    on this EUR+demo subset) and ~0.03 with age (it is computed on age/sex-residualized hazards). So
    PC1 is the model's shared prediction axis, not a frailty/age axis; matching on it tests robustness
    to the model's dominant axis, not to realized sickness.
  - The actual confound is that false positives carry ~2x more real diagnoses than true negatives
    (e.g. heart-failure FPs ~5 vs ~2.6 prevalent conditions). So the descriptive de-confounder is
    realized prevalent comorbidity burden; the inferential claim rests on a logistic model that also
    adjusts age+sex.

Estimators, in order of rigor for a "beyond frailty" claim:
  (1) PRIMARY INFERENCE = age+sex+burden-adjusted logistic odds ratio for false-positive status
      (CI + p-value; BH-FDR across each pre-specified prodrome family).
  (2) DESCRIPTIVE effect size = burden-matched enrichment E* = rate_Y(FP) / expected_Y, expected via
      direct standardization of true-negative rates to the FP distribution over prevalent-burden
      deciles. Its CI resamples the FP group only (denominator held fixed) so it is approximate, a
      lower bound on width; significance is taken from the OR, not from E*.
  (3) ROBUSTNESS to the model's shared axis = PC1-decile-matched E* (age-orthogonal; agreement with (2)
      shows the pattern is not merely the model's shared prediction axis, not that age is controlled).

Collider note: prev_burden is a count over prevalent diagnoses, so in the pooled/prevalent framings the
outcome Y is itself part of the matching variable, biasing E* toward 1 (conservative, mechanical). The
incident framing (Y is a future event, not in the burden count) is the collider-free comparison and the
honest test of whether burden matching removes real signal.

Three framings of the comorbidity matrix Y (the FP definition is identical across all three):
  - pooled     : ever diagnosed (prevalent OR incident)
  - prevalent  : diagnosed at/before accelerometry (t <= 7/365.25 yr)
  - incident   : diagnosed after accelerometry (t > 7/365.25 yr)      (usually under-powered; flagged)

For Parkinson's and Alzheimer's a pre-specified prodrome code list (fixed from the literature before
looking at any result) is always reported with CI and an under-powered flag, so a null cannot be hidden.

Run:
  python3 -m ukb_disease.paper_additions.false_positives_burden_adjusted --out false_positives_burden_adjusted
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ukb_disease.baseline.config import resolve_oak_path
from ukb_disease.baseline.outcomes_processing import load_covariates
from ukb_disease.paper_additions.io_utils import load_seed_avg
from ukb_disease.paper_additions.disease_correlation import _phecode_meta, _residualize
from ukb_disease.genetics.cox_crossfit_prs import FEATURES
from ukb_disease.paper_experiments import io
from ukb_disease.paper_additions.confusion_targets import TARGETS

SPEC = 0.95            # per-disease specificity operating point (defines "predicted positive")
TOPK = 6              # data-driven top-k enriched actual diseases per target (bars), ranked by primary E*
MIN_CASE = 30         # candidate actual disease needs >= this many cases in the framing (well-powered)
MIN_FP_COUNT = 10     # ... and needs to be held by >= this many of the false positives (stable)
FRAILTY_MIN_EVENTS = 50   # incident-event floor for the diseases that DEFINE the PC1 axis (== shared_axis)
N_DECILE = 10         # strata for direct standardization
N_BOOT = 2000
PREV_CUT = 7.0 / 365.25    # <= this (years) counts as prevalent-effective (baseline outcomes policy)
RNG = np.random.default_rng(20260707)

FRAMINGS = ("pooled", "prevalent", "incident")

# ---- pre-specified clinical prodrome / associated-feature codes (fixed BEFORE computing) ----
PRODROME = {
    "Parkinson's disease": [
        "GI_529.5",    # Constipation
        "CV_446.2",    # Orthostatic hypotension
        "CV_446",      # Hypotension
        "NS_354",      # Dizziness and giddiness
        "NS_325",      # Abnormality of gait and mobility
        "NS_350.5",    # Repeated falls
        "GU_594.3",    # Urinary incontinence and enuresis
        "GU_594.1",    # Retention of urine
        "MB_286.2",    # Major depressive disorder
        "MB_288",      # Anxiety and anxiety disorders
    ],
    "Alzheimer's disease": [
        "NS_329.1",    # Memory loss
        "NS_328.1",    # Dementias
        "NS_329.8",    # Altered mental status, unspecified
        "MB_286.2",    # Major depressive disorder
        "NS_350.5",    # Repeated falls
        "NS_325",      # Abnormality of gait and mobility
    ],
}


def _framed_matrices(phe, events, tim):
    """Return three (N, P) bool matrices: EVER (prevalent|incident), PREV (t<=7d), INC (t>7d).

    Uses the signed time-to-event in `tim` (NaN = never, t<0 prevalent, t in [0,7d] effective-prevalent,
    t>7d incident) exactly as the baseline outcomes policy defines them. For phecodes absent from `tim`
    (e.g. time_to_death) fall back to the incident-event indicator, with no prevalent component.
    """
    N, P = events.shape
    EVER = np.zeros((N, P), bool)
    PREV = np.zeros((N, P), bool)
    INC = np.zeros((N, P), bool)
    for j, ph in enumerate(phe):
        if ph in tim.columns and ph != "time_to_death":
            t = tim[ph].to_numpy(float)
            has = ~np.isnan(t)
            EVER[:, j] = has
            PREV[:, j] = has & (t <= PREV_CUT)
            INC[:, j] = has & (t > PREV_CUT)
        else:
            inc = events[:, j] > 0.5
            EVER[:, j] = inc
            INC[:, j] = inc
    return {"pooled": EVER, "prevalent": PREV, "incident": INC}


def _pc1_score(H, age, sex, n_event, exclude_j):
    """Per-subject PC1 score = leading PC of age/sex-residualized, standardized hazards over the
    well-powered disease basis, EXCLUDING column `exclude_j` (leave-target-out; no circularity).
    Mirrors the shared-axis decomposition."""
    W = np.where(n_event >= FRAILTY_MIN_EVENTS)[0]
    W = W[W != exclude_j]
    Hres = _residualize(H[:, W], age, sex)
    Xc = Hres - Hres.mean(axis=0, keepdims=True)
    Xs = Xc / (Xc.std(axis=0, keepdims=True) + 1e-9)
    _, _, Vt = np.linalg.svd(Xs, full_matrices=False)
    loading = Vt[0].copy()
    if loading.sum() < 0:
        loading = -loading
    return Xs @ loading, len(W)


def _deciles(x, pool):
    """Integer stratum 0..N_DECILE-1 for every subject, by deciles of x within `pool`."""
    edges = np.quantile(x[pool], np.linspace(0.1, 0.9, N_DECILE - 1))
    return np.digitize(x, edges)


def _tn_rate_by_stratum(M, tn, strat):
    """(N_DECILE, P) true-negative Y-rate within each stratum; NaN where a stratum has no TN."""
    Q = np.full((N_DECILE, M.shape[1]), np.nan)
    for d in range(N_DECILE):
        sel = tn & (strat == d)
        if sel.any():
            Q[d] = M[sel].mean(axis=0)
    return Q


def _matched_expected(fp_strata, Q, n_fp):
    """Direct-standardization expected rate (P,): sum_d w_d Q[d] over strata with FP weight and defined Q,
    renormalized so empty-TN strata do not bias the estimate."""
    w = np.bincount(fp_strata, minlength=N_DECILE).astype(float) / max(1, n_fp)
    valid = np.isfinite(Q).all(axis=1) & (w > 0)
    wv = w[valid]
    if wv.sum() <= 0:
        return np.full(Q.shape[1], np.nan)
    wv = wv / wv.sum()
    return (wv[:, None] * Q[valid]).sum(axis=0)


def _adjusted_or(y, fp, burden, age, sex, neg):
    """PRIMARY INFERENCE. Age+sex+burden-adjusted odds ratio for FP-status: logistic
    y ~ 1 + fp + burden_z + age_z + sex, fit on never-X subjects (`neg`). This is the estimator that
    directly removes age (which the burden/PC1 matching does not). Returns (or, lo, hi, p) or NaNs."""
    import statsmodels.api as sm
    m = neg & np.isfinite(burden) & np.isfinite(age) & np.isfinite(sex)
    yy = y[m].astype(float)
    if yy.sum() < 5 or (1 - yy).sum() < 5:
        return np.nan, np.nan, np.nan, np.nan
    def _z(v):
        v = v[m].astype(float); return (v - v.mean()) / (v.std() + 1e-9)
    X = np.column_stack([np.ones(int(m.sum())), fp[m].astype(float), _z(burden), _z(age), sex[m].astype(float)])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.Logit(yy, X).fit(disp=0, maxiter=200, method="lbfgs")
        b = res.params[1]; ci = res.conf_int()[1]
        return float(np.exp(b)), float(np.exp(ci[0])), float(np.exp(ci[1])), float(res.pvalues[1])
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def _bh_fdr(pvals):
    """Benjamini-Hochberg FDR-adjusted q-values for a 1-D array of p-values (NaNs -> NaN, ignored)."""
    p = np.asarray(pvals, float)
    ok = np.where(np.isfinite(p))[0]
    q = np.full(p.shape, np.nan)
    if len(ok) == 0:
        return q
    order = ok[np.argsort(p[ok])]
    n = len(order)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * n / (rank + 1))
        q[i] = prev
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--spec", type=float, default=SPEC)
    args = ap.parse_args()
    out = resolve_oak_path(args.out); os.makedirs(out, exist_ok=True)

    # ---- load like confusion_fp.py (EUR + demographics-complete test set) ----
    sa = load_seed_avg("embedding_disjoint_split")
    eids, H, events, mask, phe = (sa["eids"], sa["hazards"].astype(np.float64),
                                  sa["events"], sa["mask"], np.asarray(sa["phecodes"]))
    P = len(phe); idx = {p: i for i, p in enumerate(phe)}
    cov = load_covariates().reindex(eids)
    age = cov["age"].to_numpy(); sex = cov["sex"].to_numpy()
    feat = pd.read_parquet(FEATURES, columns=["eid", "eur_genetic"]).set_index("eid").reindex(eids)
    eur = feat["eur_genetic"].fillna(0).to_numpy() == 1
    keep = eur & ~(np.isnan(age) | np.isnan(sex))
    H, events, mask, eids, age, sex = H[keep], events[keep], mask[keep], eids[keep], age[keep], sex[keep]
    N = int(keep.sum())
    print(f"[burden_adjusted] EUR+demo test subjects = {N}  (spec = {args.spec:.0%})")

    meta = _phecode_meta()
    names = np.array([meta["name"].get(p, p) for p in phe])
    cats = np.array([meta["cat"].get(p, "Other") for p in phe])

    tim = pd.read_csv(resolve_oak_path(io.MORTALITY_PHECODES_CSV)).set_index("eid").reindex(eids)
    MATS = _framed_matrices(phe, events, tim)                      # pooled / prevalent / incident (N,P)
    n_event = np.array([int(events[mask[:, p], p].sum()) for p in range(P)])   # incident, eval-masked
    prev_burden = MATS["prevalent"].sum(axis=1).astype(float)      # baseline realized comorbidity count

    # diagnostics reported in the summary
    pc1_full, n_frail = _pc1_score(H, age, sex, n_event, exclude_j=-1)
    rho_pc1_burden = float(spearmanr(pc1_full, MATS["pooled"].sum(1))[0])
    rho_pc1_age = float(spearmanr(pc1_full, age)[0])
    print(f"[burden_adjusted] PC1 basis={n_frail} diseases;  PC1~burden rho={rho_pc1_burden:.3f}  PC1~age rho={rho_pc1_age:.3f}")

    bar_rows, prod_rows, summ = [], [], []
    robust = {fr: {"ebur": [], "raw": [], "epc1": []} for fr in FRAMINGS}  # honest (non-circular) robustness
    for disp, ph in TARGETS:
        xj = idx[ph]
        ever_x = MATS["pooled"][:, xj]
        s = H[:, xj]
        neg = ~ever_x                                            # never had X -> eligible false positive
        thr = np.quantile(s[neg], args.spec)
        flagged = s >= thr
        fp = flagged & neg
        tn = neg & ~flagged
        tp = flagged & ever_x
        n_flag, n_fp, n_tp = int(flagged.sum()), int(fp.sum()), int(tp.sum())
        ppv = n_tp / max(1, n_flag); fp_per_100 = 100.0 * (1.0 - ppv)
        print(f"\n{disp} [{ph}] flagged={n_flag} FP={n_fp} TP={n_tp} PPV={ppv:.3f} ~{round(fp_per_100)} FP/100"
              f"  burden FP={prev_burden[fp].mean():.1f} TN={prev_burden[tn].mean():.1f}")
        summ.append({"target": disp, "target_phecode": ph, "n_flagged": n_flag, "n_fp": n_fp,
                     "n_tp": n_tp, "ppv": float(ppv), "fp_per_100": float(fp_per_100),
                     "burden_fp": float(prev_burden[fp].mean()), "burden_tn": float(prev_burden[tn].mean())})

        pc1, _ = _pc1_score(H, age, sex, n_event, exclude_j=xj)   # leave-X-out
        st_bur = _deciles(prev_burden, neg)                       # PRIMARY matching strata
        st_pc1 = _deciles(pc1, neg)                               # ROBUSTNESS matching strata
        or_cache = {}                                             # keyed by (framing, y): M differs per framing

        for fr in FRAMINGS:
            M = MATS[fr]
            base = M.mean(axis=0)
            rate_fp = M[fp].mean(axis=0)
            cnt_fp = M[fp].sum(axis=0)
            n_case = M.sum(axis=0)
            raw_enr = rate_fp / (base + 1e-12)

            Qb = _tn_rate_by_stratum(M, tn, st_bur)
            Qp = _tn_rate_by_stratum(M, tn, st_pc1)
            e_bur = rate_fp / (_matched_expected(st_bur[fp], Qb, n_fp) + 1e-12)   # PRIMARY
            e_pc1 = rate_fp / (_matched_expected(st_pc1[fp], Qp, n_fp) + 1e-12)   # ROBUSTNESS

            # subject bootstrap of FP group -> CI on raw, primary (burden), robustness (pc1)
            raw_bl = np.empty((N_BOOT, P)); eb_bl = np.empty((N_BOOT, P)); ep_bl = np.empty((N_BOOT, P))
            vb = np.isfinite(Qb).all(1); vp = np.isfinite(Qp).all(1)
            for b in range(N_BOOT):
                bi = RNG.integers(0, n_fp, n_fp)
                rfb = M[fp][bi].mean(axis=0)
                raw_bl[b] = rfb / (base + 1e-12)
                for Q, valid, st, dest in ((Qb, vb, st_bur, eb_bl), (Qp, vp, st_pc1, ep_bl)):
                    wb = np.bincount(st[fp][bi], minlength=N_DECILE).astype(float) / n_fp
                    ok = valid & (wb > 0)
                    exp = (wb[ok, None] * Q[ok]).sum(0) / wb[ok].sum() if wb[ok].sum() > 0 else np.full(P, np.nan)
                    dest[b] = rfb / (exp + 1e-12)
            for A in (eb_bl, ep_bl):
                A[~np.isfinite(A)] = np.nan
            raw_lo, raw_hi = np.percentile(raw_bl, [2.5, 97.5], axis=0)
            eb_lo, eb_hi = np.nanpercentile(eb_bl, [2.5, 97.5], axis=0)
            ep_lo, ep_hi = np.nanpercentile(ep_bl, [2.5, 97.5], axis=0)

            elig = (n_case >= MIN_CASE) & (np.arange(P) != xj) & (names != names[xj]) & (cnt_fp >= MIN_FP_COUNT)
            # honest robustness: attenuation E*/raw over ALL eligible Y (not the top-k), per framing
            em = elig & np.isfinite(e_bur) & np.isfinite(raw_enr) & (raw_enr > 0)
            robust[fr]["ebur"].extend(e_bur[em].tolist())
            robust[fr]["raw"].extend(raw_enr[em].tolist())
            robust[fr]["epc1"].extend(e_pc1[em].tolist())
            order = [o for o in np.argsort(-e_bur) if elig[o] and np.isfinite(e_bur[o])][:TOPK]

            def _or(y):
                key = (fr, y)
                if key not in or_cache:
                    or_cache[key] = _adjusted_or(M[:, y], fp, prev_burden, age, sex, neg)
                return or_cache[key]

            def _row(y, rank, prereg):
                orr = _or(y)
                return {"target": disp, "target_phecode": ph, "framing": fr, "rank": rank,
                        "actual_phecode": phe[y], "actual_name": names[y], "actual_category": cats[y],
                        "same_category": bool(cats[y] == cats[xj]),
                        "raw_enr": float(raw_enr[y]), "raw_lo": float(raw_lo[y]), "raw_hi": float(raw_hi[y]),
                        "e_star": float(e_bur[y]), "es_lo": float(eb_lo[y]), "es_hi": float(eb_hi[y]),
                        "es_excludes_1": bool(eb_lo[y] > 1.0),
                        "e_star_pc1": float(e_pc1[y]), "es_pc1_lo": float(ep_lo[y]), "es_pc1_hi": float(ep_hi[y]),
                        "or": orr[0], "or_lo": orr[1], "or_hi": orr[2], "or_p": orr[3],
                        "or_sig": bool(np.isfinite(orr[1]) and orr[1] > 1.0),  # age+sex+burden-adjusted survival
                        "cnt_fp": int(cnt_fp[y]), "n_case": int(n_case[y]), "base_rate": float(base[y]),
                        "pct_of_fp": float(100.0 * rate_fp[y]),
                        "underpowered": bool(cnt_fp[y] < MIN_FP_COUNT), "prespecified": prereg}

            for rank, y in enumerate(order, 1):
                bar_rows.append(_row(y, rank, False))
                print(f"   {fr:9s} {rank}. {names[y][:28]:28s} E*={e_bur[y]:4.1f}[{eb_lo[y]:.1f},{eb_hi[y]:.1f}] "
                      f"pc1={e_pc1[y]:4.1f} raw={raw_enr[y]:4.1f}  {100*rate_fp[y]:.0f}%FP  n={int(cnt_fp[y])}")

            for code in PRODROME.get(disp, []):
                if code in idx:
                    prod_rows.append(_row(idx[code], 0, True))

    prod = pd.DataFrame(prod_rows)
    # BH-FDR across each pre-specified prodrome family (per target x framing) on the adjusted-OR p-values
    prod["or_q"] = np.nan
    for (tg, fr), g in prod.groupby(["target", "framing"]):
        prod.loc[g.index, "or_q"] = _bh_fdr(g["or_p"].to_numpy())
    prod["or_fdr_sig"] = (prod["or_q"] < 0.10) & prod["or_lo"].gt(1.0)
    pd.DataFrame(bar_rows).to_csv(f"{out}/burden_adjusted_bars.csv", index=False)
    prod.to_csv(f"{out}/prodrome_check.csv", index=False)

    # HONEST (non-circular) robustness: attenuation of E* vs raw over ALL eligible Y, per framing.
    # Median ratio ~1 (esp. in the collider-free INCIDENT framing) => burden matching removes little signal.
    robustness = {}
    for fr in FRAMINGS:
        eb = np.array(robust[fr]["ebur"]); rw = np.array(robust[fr]["raw"]); ep = np.array(robust[fr]["epc1"])
        ok = np.isfinite(eb) & np.isfinite(rw) & (rw > 0)
        robustness[fr] = {
            "n_eligible": int(ok.sum()),
            "median_Estar_over_raw": float(np.median(eb[ok] / rw[ok])) if ok.any() else None,
            "spearman_Estar_raw": float(spearmanr(eb[ok], rw[ok])[0]) if ok.sum() > 2 else None,
            "median_PC1_over_raw": float(np.median(ep[ok] / rw[ok])) if ok.any() else None,
        }

    def _prod_survival(tg):
        g = prod[(prod.target == tg) & (prod.framing == "pooled")]
        pw = g[~g.underpowered]
        return {"n_codes": int(len(g)), "n_powered": int(len(pw)),
                "n_Estar_excl1": int((pw.es_lo > 1.0).sum()),        # burden-matched CI excludes 1
                "n_OR_excl1": int((pw.or_lo > 1.0).sum()),           # age+sex+burden-adjusted OR CI excl 1
                "n_OR_fdr_sig": int(pw.or_fdr_sig.sum()),            # + BH-FDR<0.10
                "OR_survivors": pw[pw.or_lo > 1.0].actual_name.tolist(),
                "constipation_OR": g[g.actual_name == "Constipation"][["or", "or_lo", "or_hi", "cnt_fp"]]
                    .round(2).to_dict("records"),
                "orthostatic_OR": g[g.actual_name == "Orthostatic hypotension"][["or", "or_lo", "or_hi",
                    "cnt_fp", "underpowered"]].round(2).to_dict("records")}

    summary = {
        "n_subjects": N, "pc1_basis_diseases": int(n_frail),
        "pc1_burden_spearman": rho_pc1_burden, "pc1_age_spearman": rho_pc1_age,
        "spec": args.spec, "n_boot": N_BOOT, "n_decile": N_DECILE,
        "min_case": MIN_CASE, "min_fp_count": MIN_FP_COUNT,
        "primary_inference": "age+sex+burden-adjusted logistic OR (BH-FDR per prodrome family)",
        "descriptive_effect": "prevalent-burden-matched enrichment E* (FP-resample CI; approximate)",
        "shared_axis_robustness": "leave-target-out PC1-decile match (model prediction axis, not age)",
        "robustness_Estar_vs_raw": robustness,
        "PD_prodrome_survival": _prod_survival("Parkinson's disease"),
        "AD_prodrome_survival": _prod_survival("Alzheimer's disease"),
        "targets": summ,
    }
    with open(f"{out}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[burden_adjusted] PC1~burden={rho_pc1_burden:.3f} PC1~age={rho_pc1_age:.3f}")
    for fr in FRAMINGS:
        r = robustness[fr]
        print(f"[burden_adjusted] robustness {fr:9s}: median E*/raw={r['median_Estar_over_raw']:.2f} "
              f"(n_elig={r['n_eligible']})  <- incident is the collider-free test")
    pdps = summary["PD_prodrome_survival"]
    print(f"[burden_adjusted] PD prodrome (powered={pdps['n_powered']}): E*-excl1={pdps['n_Estar_excl1']} "
          f"OR-excl1={pdps['n_OR_excl1']} OR-FDR={pdps['n_OR_fdr_sig']}  survivors={pdps['OR_survivors']}")
    print(f"[burden_adjusted] constipation OR={pdps['constipation_OR']}  OH OR={pdps['orthostatic_OR']}")
    print(f"[burden_adjusted] wrote {out}/burden_adjusted_bars.csv (+ prodrome_check.csv, summary.json)")


if __name__ == "__main__":
    main()
