"""Shared Actigraphic Risk Component (SHARC): characterization and hypothesis tests.

SHARC = PC1 of the age/sex/age^2-residualized, standardized, per-disease six-year
predicted hazards, on the 5,253 held-out test subjects, over the 101 well-powered
diseases (>= 50 incident events). Reproduces the exact loading block of the shared
axis decomposition (so PC1 equals the shared axis), then joins raw activity stats,
demographics and mortality and runs analyses A1-A4 plus B.

CPU-only, seconds to a minute. Writes to sharc/ (plus figures/).

Run:
  python3 -m ukb_disease.paper_additions.sharc.compute_sharc
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, normaltest, skew, kurtosis

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from ukb_disease.paths import UKB_ROOT
from ukb_disease.baseline.outcomes_processing import load_covariates
from ukb_disease.baseline.config import resolve_oak_path
from ukb_disease.paper_additions.io_utils import load_seed_avg
from ukb_disease.paper_additions.disease_correlation import (
    _phecode_meta, _residualize, MIN_EVENTS,
)

OUT = os.path.dirname(os.path.abspath(__file__))
FIG = f"{OUT}/figures"
CIRC_CSV = f"{UKB_ROOT}/circadian_covariates_v3/all.csv"
ACC_DEM = f"{UKB_ROOT}/metadata/acc_dem.csv"
BMI_CSV = f"{UKB_ROOT}/metadata/BMI_yes.csv"

rng = np.random.default_rng(0)


def _multivar_r2(y: np.ndarray, X: np.ndarray) -> tuple[float, np.ndarray]:
    """OLS R^2 of y ~ X (with intercept) + standardized coefs (X pre-standardized)."""
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    Xs = StandardScaler().fit_transform(X[ok])
    ys = (y[ok] - y[ok].mean())
    reg = LinearRegression().fit(Xs, ys)
    r2 = reg.score(Xs, ys)
    return float(r2), reg.coef_


def main() -> None:
    Path(FIG).mkdir(parents=True, exist_ok=True)
    report = {}

    # ============================================================= LOAD (exact)
    sa = load_seed_avg("embedding_disjoint_split")
    eids, H, events, mask, phe = (sa["eids"], sa["hazards"], sa["events"],
                                  sa["mask"], sa["phecodes"])
    H = H.astype(np.float64)

    cov = load_covariates().reindex(eids)
    age = cov["age"].to_numpy()
    sex = cov["sex"].to_numpy()
    ok = ~(np.isnan(age) | np.isnan(sex))

    n_event = np.array([int(events[mask[:, p], p].sum()) for p in range(len(phe))])
    W = np.where(n_event >= MIN_EVENTS)[0]
    phe_W = [phe[p] for p in W]
    meta = _phecode_meta()
    names = [meta["name"].get(p, p) for p in phe_W]
    cats = np.array([meta["cat"].get(p, "Other") for p in phe_W])
    n_dis = len(W)

    Hw_res = _residualize(H[ok][:, W], age[ok], sex[ok])
    Xc = Hw_res - Hw_res.mean(axis=0, keepdims=True)
    Xs = Xc / (Xc.std(axis=0, keepdims=True) + 1e-9)

    U, svals, Vt = np.linalg.svd(Xs, full_matrices=False)
    var_explained = (svals ** 2) / np.sum(svals ** 2)
    loading = Vt[0].copy()
    if loading.sum() < 0:
        loading = -loading
    score1 = Xs @ loading                       # SHARC, one row per subject in eids[ok]
    Xs_shared = np.outer(score1, loading)
    Xs_beyond = Xs - Xs_shared                  # 5253 x 101 disease-specific residual (rows x diseases)

    eids_ok = eids[ok]
    burden = (events[ok][:, W] > 0.5).sum(axis=1).astype(float)

    # mortality hazard column (predicted all-cause death hazard)
    mort_idx = phe.index("time_to_death") if "time_to_death" in phe else None
    mort_haz = H[ok, mort_idx] if mort_idx is not None else None

    # ============================================================= SANITY GATE
    gate = {
        "pc1_var_explained": float(var_explained[0]),
        "frac_loadings_positive": float((loading > 0).mean()),
        "n_pos_loadings": int((loading > 0).sum()),
        "n_diseases": int(n_dis),
        "spearman_score1_mortality_hazard": float(spearmanr(score1, mort_haz)[0])
        if mort_haz is not None else float("nan"),
        "spearman_score1_burden": float(spearmanr(score1, burden)[0]),
        "n_ok_demo": int(ok.sum()),
    }
    print("=== SANITY GATE ===")
    print(json.dumps(gate, indent=2))
    ref = dict(pc1_var_explained=0.762, frac_loadings_positive=0.97,
               spearman_score1_mortality_hazard=0.449, spearman_score1_burden=0.222)
    gate_ok = (abs(gate["pc1_var_explained"] - ref["pc1_var_explained"]) < 0.01
               and abs(gate["frac_loadings_positive"] - ref["frac_loadings_positive"]) < 0.02
               and abs(gate["spearman_score1_mortality_hazard"] - ref["spearman_score1_mortality_hazard"]) < 0.02
               and abs(gate["spearman_score1_burden"] - ref["spearman_score1_burden"]) < 0.02)
    gate["MATCHED"] = bool(gate_ok)
    report["sanity_gate"] = gate
    if not gate_ok:
        print("!!! SANITY GATE FAILED, STOPPING")
        with open(f"{OUT}/report.json", "w") as f:
            json.dump(report, f, indent=2)
        return
    print("SANITY GATE: MATCHED\n")

    # ============================================================= JOIN DATA
    circ = pd.read_csv(resolve_oak_path(CIRC_CSV)).set_index("eid")
    enmo_cols = [c for c in circ.columns if c.startswith("enmo_")]
    print("enmo_* columns:", enmo_cols)
    report["enmo_cols"] = enmo_cols

    dem = pd.read_csv(resolve_oak_path(ACC_DEM)).set_index("eid")
    bmi = pd.read_csv(resolve_oak_path(BMI_CSV), usecols=["eid", "21001-0.0"]).set_index("eid")
    bmi = bmi.rename(columns={"21001-0.0": "bmi"})

    idx = pd.Index(eids_ok, name="eid")
    A = circ.reindex(idx)[enmo_cols]
    dem_j = dem.reindex(idx)
    bmi_j = bmi.reindex(idx)["bmi"]

    df = pd.DataFrame({"eid": eids_ok, "SHARC": score1, "burden": burden,
                       "age": age[ok], "sex": sex[ok]})
    df = df.set_index("eid")
    for c in enmo_cols:
        df[c] = A[c].to_numpy()
    df["ttd"] = dem_j["time_to_death"].to_numpy()     # NaN = alive/censored
    df["bmi"] = bmi_j.to_numpy()
    if mort_haz is not None:
        df["mort_haz"] = mort_haz

    n_enmo_cov = int(df[enmo_cols].notna().all(axis=1).sum())
    print(f"subjects with full enmo coverage: {n_enmo_cov}/{len(df)}")
    print(f"subjects with BMI: {int(df['bmi'].notna().sum())}/{len(df)}")
    report["coverage"] = {"n_sharc": int(len(df)), "n_full_enmo": n_enmo_cov,
                          "n_bmi": int(df["bmi"].notna().sum()),
                          "n_ttd_nonnan": int(df["ttd"].notna().sum())}

    # ============================================================= A1 distribution
    sharc = df["SHARC"].to_numpy()
    nt = normaltest(sharc)
    a1 = {
        "mean": float(np.mean(sharc)), "sd": float(np.std(sharc)),
        "skew": float(skew(sharc)), "kurtosis_excess": float(kurtosis(sharc)),
        "normaltest_stat": float(nt.statistic), "normaltest_p": float(nt.pvalue),
    }
    # percentile scale 0-100
    df["SHARC_pct"] = 100.0 * pd.Series(sharc, index=df.index).rank(pct=True)

    # 6-year mortality event: died within 6y (ttd in (0, 6]); alive/censored otherwise
    ttd = df["ttd"].to_numpy()
    died6 = np.where(np.isfinite(ttd) & (ttd > 0) & (ttd <= 6.0), 1.0, 0.0)
    df["died6"] = died6
    dec = pd.qcut(df["SHARC"], 10, labels=False, duplicates="drop")
    df["SHARC_decile"] = dec
    decile_tbl = df.groupby("SHARC_decile").agg(
        n=("SHARC", "size"), mort6=("died6", "mean"), burden=("burden", "mean"),
        SHARC_mean=("SHARC", "mean")).reset_index()
    a1["decile_table"] = decile_tbl.to_dict("records")
    a1["mort6_top_vs_bottom_decile"] = (
        float(decile_tbl["mort6"].iloc[-1]), float(decile_tbl["mort6"].iloc[0]))
    a1["burden_top_vs_bottom_decile"] = (
        float(decile_tbl["burden"].iloc[-1]), float(decile_tbl["burden"].iloc[0]))
    a1["overall_mort6_rate"] = float(died6.mean())
    report["A1_distribution"] = a1

    # figure: histogram
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(sharc, bins=60, color="#4C72B0", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("SHARC (PC1 score)"); ax.set_ylabel("subjects")
    ax.text(0.02, 0.95, f"skew={a1['skew']:.2f} kurt={a1['kurtosis_excess']:.2f}\n"
            f"normaltest p={a1['normaltest_p']:.1e}", transform=ax.transAxes,
            va="top", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/A1_hist.png", dpi=150); plt.close(fig)

    # figure: decile gradient
    fig, ax1 = plt.subplots(figsize=(5, 3.2))
    ax1.plot(decile_tbl["SHARC_decile"] + 1, decile_tbl["mort6"] * 100, "o-", color="#C44E52", label="6y mortality %")
    ax1.set_xlabel("SHARC decile (1=lowest)"); ax1.set_ylabel("6-year mortality (%)", color="#C44E52")
    ax2 = ax1.twinx()
    ax2.plot(decile_tbl["SHARC_decile"] + 1, decile_tbl["burden"], "s--", color="#4C72B0", label="mean burden")
    ax2.set_ylabel("mean incident disease burden", color="#4C72B0")
    fig.tight_layout(); fig.savefig(f"{FIG}/A1_decile_gradient.png", dpi=150); plt.close(fig)

    # ============================================================= A2 SHARC vs activity
    # univariate
    a2_uni = []
    for c in enmo_cols:
        v = df[c].to_numpy()
        m = np.isfinite(v) & np.isfinite(sharc)
        rho = spearmanr(sharc[m], v[m])[0]
        r = pearsonr(sharc[m], v[m])[0]
        a2_uni.append({"stat": c, "spearman": float(rho), "pearson": float(r),
                       "n": int(m.sum())})
    a2_uni_df = pd.DataFrame(a2_uni).sort_values("spearman", key=abs, ascending=False)

    Xact = df[enmo_cols].to_numpy()
    r2_act, coefs_act = _multivar_r2(sharc, Xact)
    r2_agesex, _ = _multivar_r2(sharc, df[["age", "sex"]].to_numpy())
    r2_all, coefs_all = _multivar_r2(sharc, df[enmo_cols + ["age", "sex"]].to_numpy())

    # standardized coefs (activity-only model): which stats dominate
    dom = sorted(zip(enmo_cols, coefs_act.tolist()), key=lambda t: -abs(t[1]))

    # SHARC_hat from activity(+age,sex): does the simple approximation retain prediction?
    def fit_hat(cols):
        Xf = df[cols].to_numpy()
        mrow = np.all(np.isfinite(Xf), axis=1)
        Xf_s = StandardScaler().fit_transform(Xf[mrow])
        reg = LinearRegression().fit(Xf_s, sharc[mrow])
        hat = np.full(len(df), np.nan)
        hat[mrow] = reg.predict(Xf_s)
        return hat, mrow

    hat_act, m_act = fit_hat(enmo_cols)
    hat_all, m_all = fit_hat(enmo_cols + ["age", "sex"])

    def retain(hat, mrow):
        r = {}
        if mort_haz is not None:
            mm = mrow & np.isfinite(df["mort_haz"].to_numpy())
            r["spearman_hat_mortalityhaz"] = float(spearmanr(hat[mm], df["mort_haz"].to_numpy()[mm])[0])
            r["spearman_trueSHARC_mortalityhaz"] = float(spearmanr(sharc[mm], df["mort_haz"].to_numpy()[mm])[0])
        r["spearman_hat_burden"] = float(spearmanr(hat[mrow], burden[mrow])[0])
        r["spearman_trueSHARC_burden"] = float(spearmanr(sharc[mrow], burden[mrow])[0])
        r["spearman_hat_died6"] = float(spearmanr(hat[mrow], died6[mrow])[0])
        r["spearman_trueSHARC_died6"] = float(spearmanr(sharc[mrow], died6[mrow])[0])
        return r

    # Raw retain(hat, mortality_hazard) is age-confounded: the predicted mortality
    # hazard is heavily age-driven and activity stats correlate with age, whereas true
    # SHARC is age-residualized (rho_age ~ 0). Partial out age (linear) from both the score
    # and the mortality hazard, then re-correlate for the age-independent retention. This
    # is the number to trust for the "scalable" claim.
    def _resid_lin(y, x):
        Xr = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(Xr, y, rcond=None)
        return y - Xr @ beta

    age_ok = df["age"].to_numpy()
    age_partial = {}
    if mort_haz is not None:
        mm = m_act & np.isfinite(df["mort_haz"].to_numpy()) & np.isfinite(age_ok)
        mh = df["mort_haz"].to_numpy()
        age_partial = {
            "spearman_age_mortalityhaz": float(spearmanr(age_ok[mm], mh[mm])[0]),
            "spearman_age_hat_activity": float(spearmanr(age_ok[mm], hat_act[mm])[0]),
            "spearman_age_trueSHARC": float(spearmanr(age_ok[mm], sharc[mm])[0]),
            "spearman_hat_mortalityhaz_agepartialled":
                float(spearmanr(_resid_lin(hat_act[mm], age_ok[mm]),
                                _resid_lin(mh[mm], age_ok[mm]))[0]),
            "spearman_trueSHARC_mortalityhaz_agepartialled":
                float(spearmanr(_resid_lin(sharc[mm], age_ok[mm]),
                                _resid_lin(mh[mm], age_ok[mm]))[0]),
        }

    a2 = {
        "univariate": a2_uni_df.to_dict("records"),
        "R2_SHARC_activity": r2_act,
        "R2_SHARC_agesex": r2_agesex,
        "R2_SHARC_activity_agesex": r2_all,
        "dominant_activity_stats_std_coef": dom[:6],
        "retain_activity_only": retain(hat_act, m_act),
        "retain_activity_agesex": retain(hat_all, m_all),
        "age_partialled_mortality_retention": age_partial,
    }
    report["A2_activity"] = a2

    # scatter: SHARC vs enmo_mean colored by mortality hazard (or age)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    cvar = df["mort_haz"].to_numpy() if mort_haz is not None else df["age"].to_numpy()
    clabel = "predicted mortality hazard" if mort_haz is not None else "age"
    m = np.isfinite(df["enmo_mean"].to_numpy()) & np.isfinite(cvar)
    sc = ax.scatter(df["enmo_mean"].to_numpy()[m], sharc[m], c=cvar[m], s=4, alpha=0.4,
                    cmap="viridis")
    ax.set_xlabel("enmo_mean (mean activity intensity)"); ax.set_ylabel("SHARC")
    plt.colorbar(sc, ax=ax, label=clabel)
    rho_em = spearmanr(df["enmo_mean"].to_numpy()[m], sharc[m])[0]
    ax.text(0.02, 0.02, f"spearman(SHARC, enmo_mean)={rho_em:.3f}", transform=ax.transAxes, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/A2_scatter_enmomean.png", dpi=150); plt.close(fig)

    # ============================================================= A3 complementary
    # For each disease-specific residual column, regress on activity stats for R^2
    Xact_full = df[enmo_cols].to_numpy()
    mrow = np.all(np.isfinite(Xact_full), axis=1)
    Xact_s = StandardScaler().fit_transform(Xact_full[mrow])
    resid_r2 = []
    for j in range(n_dis):
        y = Xs_beyond[mrow, j]
        reg = LinearRegression().fit(Xact_s, y)
        resid_r2.append(reg.score(Xact_s, y))
    resid_r2 = np.array(resid_r2)
    a3 = {
        "SHARC_activity_R2": r2_act,
        "residual_activity_R2_mean": float(resid_r2.mean()),
        "residual_activity_R2_median": float(np.median(resid_r2)),
        "residual_activity_R2_max": float(resid_r2.max()),
        "residual_activity_R2_p90": float(np.percentile(resid_r2, 90)),
        "n_residuals_R2_gt_0.01": int((resid_r2 > 0.01).sum()),
        "two_part_story_holds": bool(r2_act > 5 * resid_r2.mean()),
        "top5_residual_activity_R2": sorted(
            [{"disease": names[j], "phecode": phe_W[j], "R2": float(resid_r2[j])}
             for j in range(n_dis)], key=lambda d: -d["R2"])[:5],
    }
    report["A3_complementary"] = a3

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(resid_r2, bins=30, color="#8172B3", edgecolor="white", linewidth=0.3)
    ax.axvline(r2_act, color="#C44E52", lw=2, label=f"SHARC~activity R2={r2_act:.2f}")
    ax.set_xlabel("residual (disease-specific) ~ activity R^2"); ax.set_ylabel("diseases")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/A3_residual_r2.png", dpi=150); plt.close(fig)

    # ============================================================= A4 age/sex/bmi + strata
    def sp(colx):
        v = df[colx].to_numpy(); m = np.isfinite(v) & np.isfinite(sharc)
        return float(spearmanr(sharc[m], v[m])[0]), int(m.sum())
    rho_age, _ = sp("age"); rho_sex, _ = sp("sex"); rho_bmi, nbmi = sp("bmi")

    # within-stratum mortality gradient: median-split age x sex (4 strata)
    age_med = np.nanmedian(df["age"].to_numpy())
    strata = {}
    for sxv, sxn in [(0, "female"), (1, "male")]:
        for ah, an in [(False, "age<=med"), (True, "age>med")]:
            sel = (df["sex"].to_numpy() == sxv) & ((df["age"].to_numpy() > age_med) == ah)
            sel = sel & np.isfinite(sharc)
            if sel.sum() < 30:
                continue
            lv = sharc[sel]
            # mortality-hazard gradient
            rho_mh = float(spearmanr(lv, df["mort_haz"].to_numpy()[sel])[0]) if mort_haz is not None else float("nan")
            # 6y mortality rate top vs bottom SHARC half within stratum
            dsel = died6[sel]
            hi = lv > np.median(lv)
            strata[f"{sxn}|{an}"] = {
                "n": int(sel.sum()),
                "spearman_SHARC_mortalityhaz": rho_mh,
                "mort6_upperSHARChalf": float(dsel[hi].mean()),
                "mort6_lowerSHARChalf": float(dsel[~hi].mean()),
                "spearman_SHARC_burden": float(spearmanr(lv, burden[sel])[0]),
            }
    a4 = {"spearman_SHARC_age": rho_age, "spearman_SHARC_sex": rho_sex,
          "spearman_SHARC_bmi": rho_bmi, "n_bmi": nbmi, "age_median": float(age_med),
          "strata": strata}
    report["A4_agesex_bmi"] = a4

    # ============================================================= B disease UMAP
    S_beyond = np.corrcoef(Xs_beyond, rowvar=False)          # 101 x 101
    Dmat = 1.0 - S_beyond
    np.fill_diagonal(Dmat, 0.0)
    Dmat = np.clip((Dmat + Dmat.T) / 2.0, 0, 2)

    method = None
    coords = None
    try:
        import umap  # noqa
        reducer = umap.UMAP(metric="precomputed", n_neighbors=10, min_dist=0.3,
                            random_state=0)
        coords = reducer.fit_transform(Dmat)
        method = "umap-learn (precomputed)"
    except Exception:
        from sklearn.manifold import MDS
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=0,
                  n_init=4, max_iter=500, normalized_stress="auto")
        coords = mds.fit_transform(Dmat)
        method = "sklearn MDS (precomputed, umap-learn unavailable)"

    # silhouette by category (only categories with >=2 members)
    cat_counts = pd.Series(cats).value_counts()
    valid_cats = cat_counts[cat_counts >= 2].index
    smask = np.isin(cats, valid_cats)
    sil = float(silhouette_score(coords[smask], cats[smask])) if smask.sum() > len(valid_cats) else float("nan")

    # permutation test: mean within-category vs between-category residual correlation
    iu = np.triu_indices(n_dis, k=1)
    same = (cats[iu[0]] == cats[iu[1]])
    corr_pairs = S_beyond[iu]
    obs_within = float(corr_pairs[same].mean())
    obs_between = float(corr_pairs[~same].mean())
    obs_diff = obs_within - obs_between
    n_perm = 5000
    perm_diffs = np.empty(n_perm)
    for i in range(n_perm):
        pc = rng.permutation(cats)
        sm = (pc[iu[0]] == pc[iu[1]])
        perm_diffs[i] = corr_pairs[sm].mean() - corr_pairs[~sm].mean()
    perm_p = float((np.sum(perm_diffs >= obs_diff) + 1) / (n_perm + 1))
    b = {
        "method": method,
        "silhouette_by_category": sil,
        "n_valid_categories": int(len(valid_cats)),
        "mean_within_category_resid_corr": obs_within,
        "mean_between_category_resid_corr": obs_between,
        "within_minus_between": obs_diff,
        "permutation_p": perm_p,
        "n_perm": n_perm,
    }
    report["B_umap"] = b

    # figure
    fig, ax = plt.subplots(figsize=(7, 5.5))
    uniq_cats = list(pd.unique(cats))
    cmap = plt.get_cmap("tab20")
    color_for = {c: cmap(i % 20) for i, c in enumerate(uniq_cats)}
    for c in uniq_cats:
        sel = cats == c
        ax.scatter(coords[sel, 0], coords[sel, 1], s=30, color=color_for[c],
                   label=f"{c} ({sel.sum()})", alpha=0.8, edgecolor="k", linewidth=0.3)
    # label the neurodegenerative triad
    flagship = {"NS_324.11": "Parkinson", "NS_328.11": "Alzheimer", "NS_328.1": "dementia"}
    for j, pc in enumerate(phe_W):
        if pc in flagship:
            ax.annotate(flagship[pc], (coords[j, 0], coords[j, 1]), fontsize=8,
                        fontweight="bold", color="red")
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    ax.legend(fontsize=6, ncol=2, loc="best", framealpha=0.8)
    ax.text(0.02, 0.98, f"{method}\nsilhouette={sil:.3f}  perm p={perm_p:.3f}",
            transform=ax.transAxes, va="top", fontsize=7)
    fig.tight_layout(); fig.savefig(f"{FIG}/B_disease_umap.png", dpi=150); plt.close(fig)

    # ============================================================= SAVE
    df.reset_index()[["eid", "SHARC", "SHARC_pct", "SHARC_decile", "burden", "age",
                      "sex", "bmi", "died6"]].to_csv(f"{OUT}/sharc_per_subject.csv", index=False)
    a2_uni_df.to_csv(f"{OUT}/A2_univariate_activity.csv", index=False)
    pd.DataFrame({"phecode": phe_W, "name": names, "category": cats,
                  "pc1_loading": loading, "resid_activity_R2": resid_r2,
                  "umap_x": coords[:, 0], "umap_y": coords[:, 1]}).to_csv(
        f"{OUT}/disease_table.csv", index=False)
    with open(f"{OUT}/report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)

    print("\n=== KEY RESULTS ===")
    print(json.dumps({k: v for k, v in report.items()
                      if k in ("sanity_gate", "coverage")}, indent=2))
    print("A1 decile mort6 (bottom..top):",
          [round(r["mort6"], 4) for r in a1["decile_table"]])
    print("A1 decile burden (bottom..top):",
          [round(r["burden"], 3) for r in a1["decile_table"]])
    print(f"A2 R2 SHARC~activity={r2_act:.4f}  SHARC~age+sex={r2_agesex:.4f}  "
          f"SHARC~act+age+sex={r2_all:.4f}")
    print("A2 dominant activity stats (std coef):", a2["dominant_activity_stats_std_coef"][:5])
    print("A2 retain (activity-only):", json.dumps(a2["retain_activity_only"], indent=2))
    print("A2 retain (activity+age+sex):", json.dumps(a2["retain_activity_agesex"], indent=2))
    print("A2 age-partialled mortality retention:",
          json.dumps(a2["age_partialled_mortality_retention"], indent=2))
    print(f"A3 residual~activity R2 mean={a3['residual_activity_R2_mean']:.4f} "
          f"median={a3['residual_activity_R2_median']:.4f} max={a3['residual_activity_R2_max']:.4f}")
    print(f"A3 two-part story holds: {a3['two_part_story_holds']}")
    print(f"A4 spearman SHARC: age={rho_age:.3f} sex={rho_sex:.3f} bmi={rho_bmi:.3f}")
    print("A4 strata:", json.dumps(strata, indent=2))
    print(f"B method={method} silhouette={sil:.3f} within={obs_within:.3f} "
          f"between={obs_between:.3f} perm_p={perm_p:.4f}")
    print(f"\nWrote outputs to {OUT}")


if __name__ == "__main__":
    main()
