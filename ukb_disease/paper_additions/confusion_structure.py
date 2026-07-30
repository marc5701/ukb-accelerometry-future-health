"""Misdiagnosis / confusion structure (supplementary).

For subjects who truly have disease X, when the model over-ranks a DIFFERENT disease,
which one? The misdirection is clinically coherent: it lands on same-category or
related diseases, not random phecodes, and this holds for disease-specific structure
beyond the shared physiological axis. The model has learned disease-relevant
physiology, not one frailty score.

Design (frozen per-disease hazards, European-ancestry test subjects):
  1. Put every disease's risk on a common scale: residualize each disease-hazard on
     [1, age_z, age_z^2, sex] (reuse disease_correlation._residualize) then convert to a
     cohort percentile. Percentiling removes the per-Cox scale/base-rate artifact.
  2. Directed confusion matrix  M[X,Y] = mean_{true X}(pct_Y) - 0.5  (lift over the median
     subject). Diagonal is the model's discriminative signal for X (sanity check).
     Off-diagonal is cross-activation. Directed and case-conditioned, unlike the symmetric
     shared-axis correlation used in the main text.
  3. Coherence statistic: share of each disease's positive off-diagonal confusion mass that
     lands in its own clinical category (PhecodeCategory) vs a label-permutation null (the
     "related, not random" number). Reported (a) as-is, (b) after projecting out the shared
     PC1 axis, and (c) restricting true-X to subjects who never developed Y (non-comorbid
     robustness: genuine signal overlap, not just correct comorbidity prediction).
  4. False-positive companion: among top-decile-for-X subjects who never get X, what do they
     actually have, reported as same-category lift over base rate.

Run:
  python3 -m ukb_disease.paper_additions.confusion_structure
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ukb_disease.paths import UKB_ROOT
from ukb_disease.baseline.config import resolve_oak_path
from ukb_disease.baseline.outcomes_processing import load_covariates
from ukb_disease.paper_additions.io_utils import load_seed_avg
from ukb_disease.paper_additions.disease_correlation import _residualize, _phecode_meta
from ukb_disease.genetics.cox_crossfit_prs import FEATURES

OUT = f"{UKB_ROOT}/paper_additions/output/confusion"
MIN_TRUE = 10      # row floor: >=10 incident cases (standard minimum for a stable estimate).
#                    The category matrix is essentially unchanged from >=1 to >=20
#                    (corr 0.94-1.0, max cell shift ~4 pp), so this is a mild stability
#                    floor, not a cherry-pick; keeps 314 of 390 diseases.
TOPK = 5           # "top-k confused-with" for the legible table
N_PERM = 2000      # permutation null for coherence
RNG = np.random.default_rng(20260707)


def _percentile_cols(mat: np.ndarray) -> np.ndarray:
    """Column-wise cohort percentile in [0,1] (ties averaged)."""
    return np.apply_along_axis(lambda c: (rankdata(c) - 0.5) / len(c), 0, mat)


def _load_eur_mask(eids: np.ndarray) -> np.ndarray:
    feat = pd.read_parquet(FEATURES, columns=["eid", "eur_genetic"]).set_index("eid").reindex(eids)
    return feat["eur_genetic"].fillna(0).to_numpy() == 1


def _coherence(M: np.ndarray, same_cat: np.ndarray) -> float:
    """Share of positive off-diagonal confusion mass on same-category diseases.

    M: (P,P) directed lift, same_cat: (P,P) bool (category(X)==category(Y)). Diagonal
    excluded. Weighted by positive lift so it reflects where the model actually over-flags.
    """
    P = M.shape[0]
    off = ~np.eye(P, dtype=bool)
    w = np.where((M > 0) & off, M, 0.0)
    tot = w.sum()
    return float((w * same_cat).sum() / tot) if tot > 0 else np.nan


def main() -> None:
    Path(OUT).mkdir(parents=True, exist_ok=True)
    sa = load_seed_avg("embedding_disjoint_split")
    eids, H, events, mask, phe = (sa["eids"], sa["hazards"].astype(np.float64),
                                  sa["events"], sa["mask"], sa["phecodes"])
    P = len(phe)
    phe = np.asarray(phe)

    cov = load_covariates().reindex(eids)
    age, sex = cov["age"].to_numpy(), cov["sex"].to_numpy()
    eur = _load_eur_mask(eids)
    keep = eur & ~(np.isnan(age) | np.isnan(sex))
    print(f"[B] test subjects={len(eids)}  EUR+demo={int(keep.sum())}")

    H, events, mask = H[keep], events[keep], mask[keep]
    age, sex = age[keep], sex[keep]
    truth = (events > 0.5)                     # (N,P) truly has disease X
    n_true = truth.sum(axis=0).astype(int)

    meta = _phecode_meta()
    names = np.array([meta["name"].get(p, p) for p in phe])
    cats = np.array([meta["cat"].get(p, "Other") for p in phe])
    same_cat = (cats[:, None] == cats[None, :])

    # common-scale risk: residualize on age/sex(+age^2), then cohort percentile
    Hr = _residualize(H, age, sex)
    pct = _percentile_cols(Hr)                 # (N,P) in [0,1], column mean = 0.5

    # directed confusion matrix M[X,Y] = mean_{true X} pct_Y - 0.5
    M = np.full((P, P), np.nan)
    for x in range(P):
        tx = truth[:, x]
        if tx.sum() >= 1:
            M[x] = pct[tx].mean(axis=0) - 0.5
    rows_ok = np.where(n_true >= MIN_TRUE)[0]   # diseases with a stable confusion profile
    print(f"[B] diseases with >= {MIN_TRUE} true cases (row floor): {len(rows_ok)}/{P}")

    # shared-axis (PC1) removed variant, on the SAME residualized hazards
    Xs = (Hr - Hr.mean(0)) / (Hr.std(0) + 1e-9)
    _, _, Vt = np.linalg.svd(Xs, full_matrices=False)
    Xs_beyond = Xs - np.outer(Xs @ Vt[0], Vt[0])
    pct_beyond = _percentile_cols(Xs_beyond)
    M_beyond = np.full((P, P), np.nan)
    for x in range(P):
        tx = truth[:, x]
        if tx.sum() >= 1:
            M_beyond[x] = pct_beyond[tx].mean(axis=0) - 0.5

    # coherence (observed) on the row-floor diseases, with permutation null
    def coherence_with_null(Mat):
        sub = Mat[np.ix_(rows_ok, rows_ok)]
        sc = same_cat[np.ix_(rows_ok, rows_ok)]
        obs = _coherence(sub, sc)
        cats_sub = cats[rows_ok]
        null = np.empty(N_PERM)
        for b in range(N_PERM):
            perm = RNG.permutation(cats_sub)
            null[b] = _coherence(sub, perm[:, None] == perm[None, :])
        z = (obs - null.mean()) / (null.std() + 1e-12)
        p = float((null >= obs).mean())
        return obs, float(null.mean()), float(null.std()), float(z), p

    coh_obs, coh_null, coh_null_sd, coh_z, coh_p = coherence_with_null(M)
    cohb_obs, cohb_null, _, cohb_z, cohb_p = coherence_with_null(M_beyond)

    # non-comorbid robustness: for each (X,Y), true-X who never developed Y
    M_nc = np.full((P, P), np.nan)
    for x in rows_ok:
        tx = truth[:, x]
        base = pct[tx]                                   # (n_x, P)
        ty_among = truth[tx]                             # (n_x, P) did each true-X also get Y?
        for y in range(P):
            sel = ~ty_among[:, y]                        # true-X who did NOT get Y
            if sel.sum() >= 5:
                M_nc[x, y] = base[sel, y].mean() - 0.5
    sub_nc = M_nc[np.ix_(rows_ok, rows_ok)]
    sc_nc = same_cat[np.ix_(rows_ok, rows_ok)]
    coh_nc = _coherence(np.where(np.isnan(sub_nc), 0.0, sub_nc), sc_nc)

    # top-k coherence (legible: share of top-k confusions that are same-category)
    base_top = [[o for o in np.argsort(-M[x]) if o != x][:TOPK] for x in rows_ok]
    base_top_b = [[o for o in np.argsort(-M_beyond[x]) if o != x][:TOPK] for x in rows_ok]
    topk_obs = float(np.mean([np.mean(cats[ix] == cats[rows_ok[i]]) for i, ix in enumerate(base_top)]))
    topk_obs_b = float(np.mean([np.mean(cats[ix] == cats[rows_ok[i]]) for i, ix in enumerate(base_top_b)]))
    topk_null = np.empty(N_PERM)
    for b in range(N_PERM):
        cc = RNG.permutation(cats)
        topk_null[b] = np.mean([np.mean(cc[ix] == cc[rows_ok[i]]) for i, ix in enumerate(base_top)])
    topk_z = (topk_obs - topk_null.mean()) / (topk_null.std() + 1e-12)
    topk_p = float((topk_null >= topk_obs).mean())

    # Robust category-level confusion (pools per-cell noise). Computed on BOTH the raw
    #   residualized risk (shows the dominant frailty gradient) and the beyond-shared-axis
    #   risk (isolates disease-specific structure -> the coherent-diagonal headline).
    ucats = sorted(set(cats.tolist()))
    K = len(ucats); kidx = {c: i for i, c in enumerate(ucats)}
    kcodes = np.array([kidx[c] for c in cats])
    Gcol = np.zeros((P, K)); Gcol[np.arange(P), kcodes] = 1.0
    rr = np.arange(len(rows_ok))

    def _own_other(Mat, kc):
        M_ro = Mat[rows_ok]; diagv = Mat[rows_ok, rows_ok]
        tot_sum = M_ro.sum(1) - diagv; tot_cnt = P - 1
        G = np.zeros((P, K)); G[np.arange(P), kc] = 1.0
        sum_toward = M_ro @ G
        cnt = np.tile(G.sum(0), (len(rows_ok), 1)).astype(float)
        krow = kc[rows_ok]
        sum_toward[rr, krow] -= diagv; cnt[rr, krow] -= 1.0
        own_c = cnt[rr, krow]; oth_c = tot_cnt - own_c
        own = np.where(own_c > 0, sum_toward[rr, krow] / np.where(own_c > 0, own_c, 1), np.nan)
        oth = np.where(oth_c > 0, (tot_sum - sum_toward[rr, krow]) / np.where(oth_c > 0, oth_c, 1), np.nan)
        return own, oth

    def _cat_matrix(Mat):
        M_ro = Mat[rows_ok]
        Grow = np.zeros((len(rows_ok), K)); Grow[rr, kcodes[rows_ok]] = 1.0
        numer = Grow.T @ M_ro @ Gcol
        denom = Grow.sum(0)[:, None] * Gcol.sum(0)[None, :]
        for a in range(K):
            ra = rows_ok[kcodes[rows_ok] == a]
            numer[a, a] -= Mat[ra, ra].sum(); denom[a, a] -= len(ra)
        return numer / np.where(denom > 0, denom, np.nan)

    def _om_stat(Mat):
        own, oth = _own_other(Mat, kcodes)
        obs = float(np.nanmean(own - oth)); frac = float(np.nanmean(own > oth))
        null = np.array([np.nanmean(np.subtract(*_own_other(Mat, RNG.permutation(kcodes))))
                         for _ in range(N_PERM)])
        z = float((obs - null.mean()) / (null.std() + 1e-12)); p = float((null >= obs).mean())
        return {"own": own, "oth": oth, "obs": obs, "frac": frac, "z": z, "p": p}

    om_raw = _om_stat(M); om_bey = _om_stat(M_beyond)
    own_m, oth_m = om_raw["own"], om_raw["oth"]
    om_obs, om_z, om_p, frac_own_gt = om_raw["obs"], om_raw["z"], om_raw["p"], om_raw["frac"]
    Cc = _cat_matrix(M); Cc_beyond = _cat_matrix(M_beyond)

    # PPV / enrichment operating-point view of the category confusion matrix Cc.
    #   Precision/PPV reading: at a per-disease 95%-specificity flag on the raw hazard, the
    #   category-level enrichment E[i,j] = P(true cat i | flagged cat j) / prevalence(i). Raw
    #   enrichment is dominated by the shared frailty/comorbidity axis (every cell enriched);
    #   double-centring (remove row+col marginals) isolates the disease-specific residual, whose
    #   positive diagonal means flags are specific to their own category beyond frailty. Report
    #   how many categories keep a positive diagonal.
    SPEC_OP = 0.95
    flagd = np.zeros((H.shape[0], P), bool)
    for j in range(P):
        neg = mask[:, j] & (~truth[:, j])
        if neg.sum() >= 20:
            flagd[:, j] = mask[:, j] & (H[:, j] >= np.quantile(H[neg, j], SPEC_OP))

    def _cat_or(m):
        o = np.zeros((m.shape[0], K), bool)
        for c in range(K):
            cols = np.where(kcodes == c)[0]
            if len(cols):
                o[:, c] = m[:, cols].any(1)
        return o

    TrueC, FlagC = _cat_or(truth), _cat_or(flagd)
    base_c = TrueC.mean(0)
    L_enr = np.full((K, K), np.nan)
    for j in range(K):
        fj = FlagC[:, j]
        if fj.sum() >= 10:
            for i in range(K):
                q = TrueC[fj, i].mean()
                if base_c[i] > 0 and q > 0:
                    L_enr[i, j] = np.log2(q / base_c[i])
    D_enr = L_enr - np.nanmean(L_enr, 1)[:, None] - np.nanmean(L_enr, 0)[None, :] + np.nanmean(L_enr)
    enr_diag = np.array([L_enr[i, i] for i in range(K)])
    dc_diag = np.array([D_enr[i, i] for i in range(K)])
    ppv_diag_pos_raw = int(np.nansum(enr_diag > 0))
    ppv_diag_pos_beyond = int(np.nansum(dc_diag > 0))
    ppv_diag_total = int(np.sum(np.isfinite(dc_diag)))

    # per-disease top-k confused-with (legible table)
    topk_rows = []
    for x in rows_ok:
        order = np.argsort(-M[x])
        conf = [o for o in order if o != x][:TOPK]
        for rank, y in enumerate(conf, 1):
            topk_rows.append({
                "true_phecode": phe[x], "true_name": names[x], "true_category": cats[x],
                "n_true": int(n_true[x]), "rank": rank,
                "confused_with_phecode": phe[y], "confused_with_name": names[y],
                "confused_with_category": cats[y], "same_category": bool(same_cat[x, y]),
                "lift": float(M[x, y]),
            })

    # false-positive companion: top-decile-for-X non-cases -> their true diagnoses
    fp_rows = []
    base_rate = truth.mean(axis=0)                        # per-disease base rate
    for x in rows_ok:
        thr = np.quantile(pct[:, x], 0.90)
        fp = (pct[:, x] >= thr) & (~truth[:, x])          # high-risk-for-X but not X
        if fp.sum() < 10:
            continue
        dx_rate = truth[fp].mean(axis=0)                  # what FPs actually have
        lift = dx_rate / (base_rate + 1e-9)
        order = np.argsort(-lift)
        top = [o for o in order if o != x][:TOPK]
        # same-category share of the FP true-diagnosis lift
        sc_mass = float(lift[top][cats[top] == cats[x]].sum())
        tot_mass = float(lift[top].sum())
        for rank, y in enumerate(top, 1):
            fp_rows.append({
                "true_phecode": phe[x], "true_name": names[x], "true_category": cats[x],
                "n_fp": int(fp.sum()), "rank": rank,
                "fp_actual_phecode": phe[y], "fp_actual_name": names[y],
                "fp_actual_category": cats[y], "same_category": bool(cats[y] == cats[x]),
                "lift_over_baserate": float(lift[y]),
            })

    summary = {
        "n_test": int(len(eids)), "n_eur_demo": int(keep.sum()),
        "n_diseases": P, "row_floor_min_true": MIN_TRUE, "n_rows_ok": int(len(rows_ok)),
        "coherence_observed": coh_obs, "coherence_null_mean": coh_null,
        "coherence_null_sd": coh_null_sd, "coherence_z": coh_z, "coherence_p_perm": coh_p,
        "coherence_beyond_sharedaxis_observed": cohb_obs,
        "coherence_beyond_sharedaxis_null_mean": cohb_null,
        "coherence_beyond_sharedaxis_z": cohb_z, "coherence_beyond_sharedaxis_p": cohb_p,
        "coherence_noncomorbid": float(coh_nc),
        "coherence_topk_observed": topk_obs, "coherence_topk_beyond_axis": topk_obs_b,
        "coherence_topk_null_mean": float(topk_null.mean()), "coherence_topk_z": float(topk_z),
        "coherence_topk_p_perm": topk_p,
        "own_minus_other_category_lift_raw": om_obs, "own_minus_other_z_raw": float(om_z),
        "own_minus_other_p_perm_raw": om_p, "frac_diseases_own_gt_other_raw": frac_own_gt,
        "own_minus_other_category_lift_beyond": om_bey["obs"], "own_minus_other_z_beyond": om_bey["z"],
        "own_minus_other_p_perm_beyond": om_bey["p"], "frac_diseases_own_gt_other_beyond": om_bey["frac"],
        "mean_diagonal_lift": float(np.nanmean(np.diag(M)[rows_ok])),
        "ppv_spec_operating_point": SPEC_OP,
        "ppv_enrich_diag_positive_raw": ppv_diag_pos_raw,
        "ppv_enrich_diag_positive_beyond": ppv_diag_pos_beyond,
        "ppv_enrich_diag_total": ppv_diag_total,
    }
    np.save(f"{OUT}/confusion_ppv_enrichment_log2.npy", L_enr)
    np.save(f"{OUT}/confusion_ppv_enrichment_log2_doublecentered.npy", D_enr)
    np.save(f"{OUT}/confusion_M.npy", M)
    np.save(f"{OUT}/confusion_M_beyond_axis.npy", M_beyond)
    np.save(f"{OUT}/confusion_M_noncomorbid.npy", M_nc)
    np.save(f"{OUT}/confusion_category_matrix.npy", Cc)
    np.save(f"{OUT}/confusion_category_matrix_beyond.npy", Cc_beyond)
    with open(f"{OUT}/confusion_categories.json", "w") as f:
        json.dump(ucats, f)
    pd.DataFrame({"phecode": phe[rows_ok], "name": names[rows_ok], "category": cats[rows_ok],
                  "n_true": n_true[rows_ok], "own_cat_lift": own_m, "other_cat_lift": oth_m}
                 ).to_csv(f"{OUT}/confusion_own_vs_other.csv", index=False)
    pd.DataFrame({"phecode": phe, "name": names, "category": cats,
                  "n_true": n_true}).to_csv(f"{OUT}/confusion_diseases.csv", index=False)
    pd.DataFrame(topk_rows).to_csv(f"{OUT}/confusion_topk.csv", index=False)
    pd.DataFrame(fp_rows).to_csv(f"{OUT}/confusion_false_positives.csv", index=False)
    with open(f"{OUT}/confusion_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"[B] diagonal lift (discriminative signal) mean={summary['mean_diagonal_lift']:+.3f}")
    print(f"[B] coherence {coh_obs:.3f} vs null {coh_null:.3f} (z={coh_z:.1f}, p={coh_p:.4f}); "
          f"beyond-axis {cohb_obs:.3f} (z={cohb_z:.1f}); non-comorbid {coh_nc:.3f}")
    print(f"[B] top-{TOPK} coherence obs={topk_obs:.3f} (beyond={topk_obs_b:.3f}) "
          f"null={topk_null.mean():.3f} z={topk_z:.1f} p={topk_p:.4f}")
    print(f"[B] own-vs-other category lift RAW={om_obs:+.4f} z={om_z:.1f} p={om_p:.4f} "
          f"frac={frac_own_gt:.2f} | BEYOND-axis={om_bey['obs']:+.4f} z={om_bey['z']:.1f} "
          f"p={om_bey['p']:.4f} frac={om_bey['frac']:.2f}")
    print(f"[B] PPV/enrichment (spec {SPEC_OP:.0%}): diagonal enriched raw {ppv_diag_pos_raw}/{ppv_diag_total}, "
          f"beyond frailty gradient {ppv_diag_pos_beyond}/{ppv_diag_total}")
    print(f"[B] wrote -> {OUT}")


if __name__ == "__main__":
    main()
