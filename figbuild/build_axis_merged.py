"""Merged shared-axis figure builder.

Combines the two shared-axis panels into one main figure (SHARC plus
disease-specific signatures) and demotes the remaining panels to an Extended
Data figure. Uses the same data loaders and output directories as the other
build scripts, so numbers are preserved by construction.

  fig_disease_corr.pdf  (main figure)   panels: (a) corr-vs-comorbidity scatter,
                        (b) PCA variance, (c) beyond-axis dumbbell, (d) per-organ variance partition
  fig_axis_supp.pdf     (Extended Data)  panels: (a) PC1 loadings, (b) comorbidity matrix,
                        (c) burden by decile, (d) predicted-risk correlation matrix
"""
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import figlib as F
from figlib import OK, save, W2

UKB_ROOT = os.environ.get("UKB_ROOT", "data")
PA = Path(f"{UKB_ROOT}/paper_additions/output")
CORR = PA / "disease_correlation"
SHARED = PA / "shared_axis"
G2DIR = Path(f"{UKB_ROOT}/paper_experiments/output/Beyond_sharc")

# representative outcomes for the disease-specific dumbbell: discrimination BEYOND THE SHARC (shared
# axis), i.e. C_full minus C_general (SHARC alone). Curated to clinically interpretable outcomes led by
# the neurodegeneration flagships (Parkinson's, Alzheimer's, dementia) plus atrial fibrillation and a
# few autonomic/rhythm/vascular outcomes. Age-proxy conditions that top this metric (benign prostatic
# hyperplasia, hernia, cataract, macular degeneration, fractures) are NOT shown: the SHARC is age/sex-
# residualised, so an age-related condition can show a large gain that reflects age, not physiology.
SHOW = ["NS_324.11", "NS_328.11", "NS_328.1", "CV_416.2", "CV_417.3", "CV_446.2",
        "CV_433", "CV_413.3", "MS_703.1", "CA_103"]
SHORT = {"NS_324.11": "Parkinson's", "NS_328.11": "Alzheimer's", "NS_328.1": "Dementia",
         "CV_416.2": "Atrial fibrillation", "CV_417.3": "Bradycardia",
         "CV_446.2": "Orthostatic hypotension", "CV_433": "Cerebrovascular disease",
         "CV_413.3": "Tricuspid valve disorder", "MS_703.1": "Gout",
         "CA_103": "Skin cancer"}


def fig_merged():
    """Merged main figure, written to fig_disease_corr.pdf (filename the manuscript includes)."""
    from matplotlib.patches import Patch
    S = np.load(CORR / "score_corr_residualized.npy")
    Co = np.load(CORR / "comorbidity_phi.npy")
    order = np.load(CORR / "cluster_order.npy")
    summ = json.load(open(CORR / "summary.json"))
    ev = np.load(CORR / "var_explained.npy")
    g2 = pd.read_csv(G2DIR / "per_disease_residual_c.csv").set_index("phecode")
    vp = pd.read_csv(SHARED / "variance_partition.csv")

    n = S.shape[0]

    fig = plt.figure(figsize=(W2, 6.35))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.04],
                  hspace=0.36, wspace=0.52, left=0.185, right=0.915, top=0.955, bottom=0.15)

    # (d) per-organ-system predicted-risk variance partition: shared / group-specific / idiosyncratic.
    # Promoted from the Extended Data figure so the "one shared axis + disease-specific structure"
    # thesis is shown directly; the demoted correlation matrix now lives in fig_axis_supp panel (d).
    axa = fig.add_subplot(gs[1, 1])
    vp2 = vp.sort_values("group_specific_frac")
    yv = np.arange(len(vp2))
    sh, grp, idi = (vp2["shared_frac"].to_numpy(), vp2["group_specific_frac"].to_numpy(),
                    vp2["idiosyncratic_frac"].to_numpy())
    axa.barh(yv, sh, color=OK["blue"])
    axa.barh(yv, grp, left=sh, color=OK["orange"])
    axa.barh(yv, idi, left=sh + grp, color=OK["grey"])
    axa.set_yticks(yv)
    axa.set_yticklabels([f"{F.cat_label(c)} ({m})" for c, m in zip(vp2["category"], vp2["n"])],
                        fontsize=6.4)
    axa.tick_params(length=0, axis="y")
    axa.set_ylim(-0.6, len(vp2) - 0.4)
    axa.set_xlim(0, 1)
    axa.set_xlabel("Share of predicted-risk variance")
    F.panel(axa, "d")
    handles = [Patch(facecolor=OK["blue"], label="SHARC"),
               Patch(facecolor=OK["orange"], label="group-specific"),
               Patch(facecolor=OK["grey"], label="idiosyncratic")]
    axa.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=3,
               fontsize=6.4, frameon=False, handlelength=1.0, handleheight=1.0,
               handletextpad=0.4, columnspacing=1.1, borderaxespad=0.0)

    # (a) scatter: predicted-risk correlation vs comorbidity
    axb = fig.add_subplot(gs[0, 0])
    iu = np.triu_indices(n, k=1)
    x = Co[iu]; y = S[iu]
    axb.scatter(x, y, s=4, alpha=0.18, color=OK["grey"], edgecolors="none", rasterized=True)
    b0 = summ["S_resid_intercept_at_zero_comorbidity"]; b1 = summ["S_resid_slope_per_comorbidity"]
    xs = np.linspace(x.min(), np.percentile(x, 99.5), 50)
    axb.plot(xs, b0 + b1 * xs, color=OK["vermillion"], lw=1.6)
    axb.axhline(b0, ls=":", color=OK["blue"], lw=1.0)
    axb.set_xlabel("Disease co-occurrence (phi)")
    axb.set_ylabel("Predicted-risk correlation")
    F.panel(axb, "a")
    axb.text(0.96, 0.06,
             f"$R^2$ = {summ['S_resid_vs_comorbidity_R2']:.2f}\n"
             f"r = {b0:.2f} at zero co-occurrence",
             transform=axb.transAxes, ha="right", va="bottom", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=OK["grey"], lw=0.6))

    # (b) principal-component variance explained (scree); PC1 == the SHARC, PC2..10 are not
    axc = fig.add_subplot(gs[0, 1])
    pc1 = summ["pc1_var_explained"]; pc13 = summ["pc1_3_var_explained"]
    k = 10
    axc.bar(np.arange(1, k + 1), ev[:k] * 100, color=OK["green"], width=0.7)
    axc.set_xlabel("Principal component")
    axc.set_ylabel("Variance explained (%)")
    F.panel(axc, "b")
    axc.set_xticks(np.arange(1, k + 1))
    axc.text(0.52, 0.92, f"PC1 = {pc1*100:.0f}%\nPC1-3 = {pc13*100:.0f}%",
             transform=axc.transAxes, ha="center", va="top", fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=OK["grey"], lw=0.6))

    # (c) disease-specific signal beyond the SHARC (shared axis) dumbbell: open circle = SHARC alone
    #     (C_general), filled = full model (C_full); labelled gain = delta_incr = C_full - C_general.
    axd = fig.add_subplot(gs[1, 0])
    gg = g2.loc[SHOW].sort_values("delta_incr")
    yy = np.arange(len(gg))
    for yi, (ph, r) in zip(yy, gg.iterrows()):
        col = F.cat_color(r.category)
        axd.plot([r.C_general, r.C_full], [yi, yi], color=col, lw=1.7, alpha=0.5,
                 solid_capstyle="round", zorder=2)
        axd.scatter(r.C_general, yi, s=20, facecolor="white", edgecolor=OK["grey"],
                    linewidth=1.0, zorder=3)
        thin = r.n_ev < 50
        axd.scatter(r.C_full, yi, s=(34 if thin else 28), marker=("D" if thin else "o"),
                    color=col, edgecolor="white", linewidth=0.6, zorder=4)
    axd.axvline(0.5, ls=":", color=OK["grey"], lw=0.9, zorder=1)
    axd.set_yticks(yy)
    axd.set_yticklabels([SHORT[ph] for ph in gg.index], fontsize=6.6)
    for t, ph in zip(axd.get_yticklabels(), gg.index):
        t.set_color(F.cat_color(g2.loc[ph, "category"]))
    axd.set_ylim(-0.7, len(gg) + 0.8)
    axd.set_xlim(0.37, 1.02)
    axd.set_xlabel("Concordance (C-index)")
    axd.tick_params(length=0, axis="y")
    F.panel(axd, "c")
    # label the disease-specific gain (delta = C_full - C_general) beside every disease, flagship in bold
    for yi, (ph, r) in zip(yy, gg.iterrows()):
        axd.annotate(f"+{r.delta_incr:.2f}", (r.C_full, yi),
                     xytext=(4, 0), textcoords="offset points", ha="left", va="center",
                     fontsize=5.8, color=F.cat_color(r.category),
                     fontweight=("bold" if ph == "NS_324.11" else "normal"))
    top = len(gg) - 1
    r_top = gg.iloc[-1]
    axd.annotate("SHARC", (r_top.C_general, top), xytext=(-7, 10), textcoords="offset points",
                 ha="right", va="bottom", fontsize=5.8, color=OK["grey"],
                 arrowprops=dict(arrowstyle="-", lw=0.5, color=OK["grey"], shrinkA=0, shrinkB=2))
    axd.annotate("full model", (r_top.C_full, top), xytext=(0, 10), textcoords="offset points",
                 ha="center", va="bottom", fontsize=5.8, color="#333333",
                 arrowprops=dict(arrowstyle="-", lw=0.5, color="#999999", shrinkA=0, shrinkB=2))

    save(fig, "fig_disease_corr")


def fig_axis_supp():
    """NEW Extended Data figure -> fig_axis_supp.pdf (demoted panels)."""
    S = np.load(CORR / "score_corr_residualized.npy")
    Co = np.load(CORR / "comorbidity_phi.npy")
    order = np.load(CORR / "cluster_order.npy")
    ld = pd.read_csv(SHARED / "loadings.csv")
    summ = json.load(open(SHARED / "summary.json"))
    sub = np.load(SHARED / "pc1_subject.npz")
    score1, burden = sub["score1"], sub["burden"]

    Coo = Co[np.ix_(order, order)].copy()
    n = Coo.shape[0]
    Coo[np.diag_indices(n)] = np.nan

    fig = plt.figure(figsize=(W2, 6.05))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.12],
                  hspace=0.30, wspace=0.40, left=0.16, right=0.85, top=0.965, bottom=0.175)

    # (b) comorbidity heatmap (same clustered order as merged fig panel a); placed top-RIGHT so its
    # colorbar sits on the outer edge and does not collide with the PC1-loading category labels
    axa = fig.add_subplot(gs[0, 1])
    cmap_b = plt.get_cmap("viridis").copy(); cmap_b.set_bad("white")
    vmax = float(np.nanpercentile(np.abs(Coo), 99))
    imb = axa.imshow(Coo, cmap=cmap_b, vmin=0.0, vmax=max(vmax, 0.1),
                     aspect="equal", interpolation="nearest")
    F.panel(axa, "b")
    axa.set_xticks([]); axa.set_yticks([])
    axa.set_xlabel(f"{n} diseases (clustered)")
    cbb = fig.colorbar(imb, ax=axa, fraction=0.046, pad=0.03)
    cbb.set_label("phi coefficient", fontsize=8)

    # (a) PC1 loadings per disease, grouped by organ system; top-LEFT (labels on outer edge)
    axb = fig.add_subplot(gs[0, 0])
    axb.set_axisbelow(True)
    cat_means = ld.groupby("category")["pc1_loading"].mean().sort_values()
    cats_sorted = list(cat_means.index)
    ccol = {c: F.cat_color(c) for c in cats_sorted}
    for yi in range(0, len(cats_sorted), 2):
        axb.axhspan(yi - 0.5, yi + 0.5, color="#f4f4f4", zorder=0)
    rng = np.random.default_rng(0)
    for yi, c in enumerate(cats_sorted):
        vals = ld.loc[ld.category == c, "pc1_loading"].to_numpy()
        jit = (rng.random(len(vals)) - 0.5) * 0.30
        axb.scatter(vals, np.full(len(vals), yi) + jit, s=11, color=ccol[c],
                    alpha=0.85, edgecolors="none", zorder=2)
        axb.scatter([cat_means[c]], [yi], marker="D", s=20, color="k",
                    edgecolors="white", linewidths=0.6, zorder=3)
    axb.axvline(0, ls=":", color=OK["grey"], lw=0.9)
    axb.set_yticks(range(len(cats_sorted)))
    axb.set_yticklabels([F.cat_label(c) for c in cats_sorted], fontsize=6.6)
    axb.set_ylim(-0.7, len(cats_sorted) - 0.3)
    axb.set_xlabel("SHARC loading")
    F.panel(axb, "a")
    axb.text(0.035, 0.97,
             f"{summ['frac_loadings_positive']*100:.0f}% load positively\n"
             f"PC1 = {summ['pc1_var_explained']*100:.0f}% of variance",
             transform=axb.transAxes, ha="left", va="top", fontsize=6.6,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=OK["grey"], lw=0.6))

    # (c) SHARC score vs incident-disease burden, by decile
    axc = fig.add_subplot(gs[1, 0])
    dec = pd.qcut(score1, 10, labels=False, duplicates="drop")
    nd = int(dec.max()) + 1
    mb = np.array([burden[dec == i].mean() for i in range(nd)])
    se = np.array([burden[dec == i].std() / np.sqrt((dec == i).sum()) for i in range(nd)])
    axc.errorbar(np.arange(1, nd + 1), mb, yerr=se, fmt="-o", color=OK["blue"],
                 ms=4, lw=1.4, capsize=2.5)
    axc.set_xlabel("SHARC score, decile")
    axc.set_ylabel("Mean incident diseases")
    F.panel(axc, "c")
    axc.text(0.05, 0.95, f"Spearman r = {summ['burden_spearman']:.2f}",
             transform=axc.transAxes, ha="left", va="top", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=OK["grey"], lw=0.6))

    # (d) predicted-risk correlation heatmap, clustered (same order as panel b), 0-1 colour scale.
    # Demoted from the main figure; the ~5.6% of weakly negative pairs render at the 0 floor.
    axd = fig.add_subplot(gs[1, 1])
    So = S[np.ix_(order, order)].copy()
    So[np.diag_indices(n)] = np.nan
    cmap_d = plt.get_cmap("magma").copy(); cmap_d.set_bad("white")
    imd = axd.imshow(So, cmap=cmap_d, vmin=0.0, vmax=1.0, aspect="equal", interpolation="nearest")
    F.panel(axd, "d")
    axd.set_xticks([]); axd.set_yticks([])
    axd.set_xlabel(f"{n} diseases (clustered)")
    cbd = fig.colorbar(imd, ax=axd, fraction=0.046, pad=0.04)
    cbd.set_label("Spearman r", fontsize=8)

    save(fig, "fig_axis_supp")


if __name__ == "__main__":
    fig_merged()
    fig_axis_supp()
    print("done: fig_disease_corr (merged) + fig_axis_supp (ED)")
