"""Build two main figures for the paper additions:
  fig_disease_corr.pdf  - shared prediction signal vs comorbidity
  fig_controls.pdf      - feature-source ladder + context titration

Imports figlib for the shared style/palette/save. Set UKB_ROOT to point at the
data root.
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


def _load_json(p):
    return json.load(open(p)) if Path(p).exists() else None


# --------------------------------------------------------------------------- #
# Figure: disease correlation (shared signal vs comorbidity)
# --------------------------------------------------------------------------- #
def fig_disease_corr():
    S = np.load(CORR / "score_corr_residualized.npy")
    Co = np.load(CORR / "comorbidity_phi.npy")
    order = np.load(CORR / "cluster_order.npy")
    dis = pd.read_csv(CORR / "diseases.csv")
    summ = json.load(open(CORR / "summary.json"))

    So = S[np.ix_(order, order)].copy()
    Coo = Co[np.ix_(order, order)].copy()
    n = So.shape[0]
    di = np.diag_indices(n)
    So[di] = np.nan; Coo[di] = np.nan  # mask self-correlation (=1) for clean contrast

    fig = plt.figure(figsize=(W2, 6.4))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 0.78],
                  hspace=0.42, wspace=0.32, left=0.07, right=0.90, top=0.96, bottom=0.09)

    # (a) predicted-risk correlation heatmap, clustered
    axa = fig.add_subplot(gs[0, 0])
    cmap_a = plt.get_cmap("magma").copy(); cmap_a.set_bad("white")
    cmap_b = plt.get_cmap("viridis").copy(); cmap_b.set_bad("white")
    im = axa.imshow(So, cmap=cmap_a, vmin=0.2, vmax=1.0, aspect="equal", interpolation="nearest")
    F.panel(axa, "a")
    axa.set_xticks([]); axa.set_yticks([])
    axa.set_xlabel(f"{n} diseases (clustered)")
    cb = fig.colorbar(im, ax=axa, fraction=0.046, pad=0.03)
    cb.set_label("Spearman r", fontsize=8)

    # (b) comorbidity heatmap, same order
    axb = fig.add_subplot(gs[0, 1])
    vmax = float(np.nanpercentile(np.abs(Coo), 99))
    imb = axb.imshow(Coo, cmap=cmap_b, vmin=0.0, vmax=max(vmax, 0.1),
                     aspect="equal", interpolation="nearest")
    F.panel(axb, "b")
    axb.set_xticks([]); axb.set_yticks([])
    axb.set_xlabel(f"{n} diseases (clustered)")
    cbb = fig.colorbar(imb, ax=axb, fraction=0.046, pad=0.03)
    cbb.set_label("phi coefficient", fontsize=8)

    # (c) scatter: shared score correlation vs comorbidity
    axc = fig.add_subplot(gs[1, 0])
    iu = np.triu_indices(n, k=1)
    x = Co[iu]; y = S[iu]
    axc.scatter(x, y, s=4, alpha=0.18, color=OK["grey"], edgecolors="none", rasterized=True)
    b0 = summ["S_resid_intercept_at_zero_comorbidity"]; b1 = summ["S_resid_slope_per_comorbidity"]
    xs = np.linspace(x.min(), np.percentile(x, 99.5), 50)
    axc.plot(xs, b0 + b1 * xs, color=OK["vermillion"], lw=1.6)
    axc.axhline(b0, ls=":", color=OK["blue"], lw=1.0)
    axc.set_xlabel("Disease co-occurrence (phi)")
    axc.set_ylabel("Predicted-risk correlation")
    F.panel(axc, "c")
    axc.text(0.96, 0.06,
             f"$R^2$ = {summ['S_resid_vs_comorbidity_R2']:.2f}\n"
             f"r = {b0:.2f} at zero co-occurrence",
             transform=axc.transAxes, ha="right", va="bottom", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=OK["grey"], lw=0.6))

    # (d) shared-axis variance explained
    axd = fig.add_subplot(gs[1, 1])
    pc1 = summ["pc1_var_explained"]; pc13 = summ["pc1_3_var_explained"]
    ev = np.load(CORR / "var_explained.npy")
    k = 10
    axd.bar(np.arange(1, k + 1), ev[:k] * 100, color=OK["green"], width=0.7)
    axd.set_xlabel("Shared-risk component")
    axd.set_ylabel("Variance explained (%)")
    F.panel(axd, "d")
    axd.set_xticks(np.arange(1, k + 1))
    axd.text(0.96, 0.92, f"PC1 = {pc1*100:.0f}%\nPC1-3 = {pc13*100:.0f}%",
             transform=axd.transAxes, ha="right", va="top", fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=OK["grey"], lw=0.6))

    save(fig, "fig_disease_corr")


# --------------------------------------------------------------------------- #
# Figure: controls (feature-source ladder + context titration)
# --------------------------------------------------------------------------- #
# Publication labels for the ladder arms.
LADDER_LABELS = {
    "random_gaussian": "Random features",
    "random_encoder": "Untrained encoder",
    "summary_stats": "Summary statistics",
    "learned_mean": "Learned (mean)",
    "learned_lmoments": "Learned (distributional)",
}
LADDER_ORDER = ["random_gaussian", "random_encoder", "summary_stats", "learned_mean", "learned_lmoments"]
LADDER_COLORS = {
    "random_gaussian": OK["grey"], "random_encoder": OK["purple"],
    "summary_stats": OK["orange"], "learned_mean": OK["skyblue"],
    "learned_lmoments": OK["blue"],
}


def fig_controls():
    # prefer the full 5-arm ladder (with the untrained-encoder) once it exists
    ladder = (_load_json(PA / "boot_ladder_full" / "bootstrap_results.json")
              or _load_json(PA / "boot_ladder" / "bootstrap_results.json"))
    days_b = _load_json(PA / "boot_titration_days" / "bootstrap_results.json")
    days_c = _load_json(PA / "titration_days" / "curve.json")
    hours_c = _load_json(PA / "titration_hours" / "curve.json")

    fig = plt.figure(figsize=(W2, 3.2))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.25, 1.0, 1.0],
                  wspace=0.42, left=0.22, right=0.985, top=0.94, bottom=0.17)

    # (a) feature-source ladder
    axa = fig.add_subplot(gs[0, 0])
    if ladder:
        arms = [a for a in LADDER_ORDER if a in ladder["marginal"]]
        y = np.arange(len(arms))
        pts = [ladder["marginal"][a]["point"] for a in arms]
        lo = [ladder["marginal"][a]["ci_lo"] for a in arms]
        hi = [ladder["marginal"][a]["ci_hi"] for a in arms]
        cols = [LADDER_COLORS[a] for a in arms]
        axa.barh(y, pts, color=cols, height=0.62,
                 xerr=[np.array(pts) - np.array(lo), np.array(hi) - np.array(pts)],
                 error_kw=dict(ecolor="#333333", elinewidth=1.0, capsize=2.5))
        axa.set_yticks(y); axa.set_yticklabels([LADDER_LABELS[a] for a in arms])
        axa.set_xlim(0.5, 0.725)
        axa.axvline(0.5, ls=":", color=OK["grey"], lw=0.8)
        for yi, p, h in zip(y, pts, hi):
            axa.text(h + 0.003, yi, f"{p:.3f}", va="center", ha="left", fontsize=7.5)
    axa.set_xlabel("Mean concordance")
    F.panel(axa, "a")

    # (b) days-of-wear titration
    axb = fig.add_subplot(gs[0, 1])
    if days_c:
        ks = [d["k_days"] for d in days_c["curve"]]
        mc = [d["mean_c"] for d in days_c["curve"]]
        indist = [d["in_distribution"] for d in days_c["curve"]]
        lo = hi = None
        if days_b:
            lo = [days_b["marginal"].get(f"days_k{k}", {}).get("ci_lo", np.nan) for k in ks]
            hi = [days_b["marginal"].get(f"days_k{k}", {}).get("ci_hi", np.nan) for k in ks]
            axb.fill_between(ks, lo, hi, color=OK["blue"], alpha=0.15, lw=0)
        ks_a = np.array(ks); mc_a = np.array(mc); ind = np.array(indist)
        axb.plot(ks_a[ind], mc_a[ind], "-o", color=OK["blue"], ms=4, lw=1.4, label="deployed")
        if (~ind).any():
            axb.plot(ks_a[~ind], mc_a[~ind], "o", color=OK["grey"], ms=4,
                     mfc="white", label="below window")
        axb.axhline(mc[-1], ls=":", color=OK["grey"], lw=0.8)
        axb.set_xticks(ks)
        axb.legend(fontsize=7, loc="lower right", frameon=False)
    axb.set_xlabel("Days of wear")
    axb.set_ylabel("Mean concordance")
    F.panel(axb, "b")

    # (c) hours-per-day titration
    axc = fig.add_subplot(gs[0, 2])
    hours_b = _load_json(PA / "boot_titration_hours" / "bootstrap_results.json")
    if hours_c:
        hs = [d["hours"] for d in hours_c["curve"]]
        mc = [d["mean_c"] for d in hours_c["curve"]]
        if hours_b:
            lo = [hours_b["marginal"].get(f"hours_H{h}", {}).get("ci_lo", np.nan) for h in hs]
            hi = [hours_b["marginal"].get(f"hours_H{h}", {}).get("ci_hi", np.nan) for h in hs]
            axc.fill_between(hs, lo, hi, color=OK["green"], alpha=0.15, lw=0)
        axc.plot(hs, mc, "-o", color=OK["green"], ms=4, lw=1.4)
        axc.axhline(mc[-1], ls=":", color=OK["grey"], lw=0.8)
        axc.set_xscale("log", base=2); axc.set_xticks(hs); axc.set_xticklabels(hs)
    else:
        axc.text(0.5, 0.5, "hours titration\n(pending)", ha="center", va="center",
                 transform=axc.transAxes, color=OK["grey"], fontsize=8)
    axc.set_xlabel("Hours per day")
    axc.set_ylabel("Mean concordance")
    F.panel(axc, "c")

    save(fig, "fig_controls")


# --------------------------------------------------------------------------- #
# Figure: dissecting the shared disease-risk axis (one axis + group specificity)
# --------------------------------------------------------------------------- #
def fig_shared_axis():
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    ld = pd.read_csv(SHARED / "loadings.csv")
    summ = json.load(open(SHARED / "summary.json"))
    sub = np.load(SHARED / "pc1_subject.npz")
    score1, burden = sub["score1"], sub["burden"]
    g2 = pd.read_csv(G2DIR / "per_disease_residual_c.csv").set_index("phecode")
    g2sum = json.load(open(G2DIR / "beyond_axis_summary.json"))
    vp = pd.read_csv(SHARED / "variance_partition.csv")

    allcats = sorted(ld["category"].unique())
    ccol = {c: F.cat_color(c) for c in allcats}

    fig = plt.figure(figsize=(W2, 6.05))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.16],
                  hspace=0.20, wspace=0.40, left=0.16, right=0.83, top=0.965, bottom=0.175)

    # (a) PC1 loadings per disease, grouped by organ system (sorted by mean loading)
    axa = fig.add_subplot(gs[0, 0])
    axa.set_axisbelow(True)
    cat_means = ld.groupby("category")["pc1_loading"].mean().sort_values()
    cats_sorted = list(cat_means.index)
    for yi in range(0, len(cats_sorted), 2):
        axa.axhspan(yi - 0.5, yi + 0.5, color="#f4f4f4", zorder=0)
    rng = np.random.default_rng(0)
    for yi, c in enumerate(cats_sorted):
        vals = ld.loc[ld.category == c, "pc1_loading"].to_numpy()
        jit = (rng.random(len(vals)) - 0.5) * 0.30
        axa.scatter(vals, np.full(len(vals), yi) + jit, s=12, color=ccol[c],
                    alpha=0.85, edgecolors="none", zorder=2)
        axa.scatter([cat_means[c]], [yi], marker="D", s=22, color="k",
                    edgecolors="white", linewidths=0.6, zorder=3)
    axa.axvline(0, ls=":", color=OK["grey"], lw=0.9)
    axa.set_yticks(range(len(cats_sorted)))
    axa.set_yticklabels([F.cat_label(c) for c in cats_sorted], fontsize=7.2)
    axa.set_ylim(-0.7, len(cats_sorted) - 0.3)
    axa.set_xlabel("PC1 loading (shared-axis weight)")
    F.panel(axa, "a")
    axa.text(0.035, 0.97,
             f"{summ['frac_loadings_positive']*100:.0f}% load positively\n"
             f"PC1 = {summ['pc1_var_explained']*100:.0f}% of variance\n"
             "black diamond = group mean",
             transform=axa.transAxes, ha="left", va="top", fontsize=6.8,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=OK["grey"], lw=0.6))

    # (b) shared-axis score vs incident-disease burden, by decile
    axb = fig.add_subplot(gs[0, 1])
    dec = pd.qcut(score1, 10, labels=False, duplicates="drop")
    nd = int(dec.max()) + 1
    mb = np.array([burden[dec == i].mean() for i in range(nd)])
    se = np.array([burden[dec == i].std() / np.sqrt((dec == i).sum()) for i in range(nd)])
    axb.errorbar(np.arange(1, nd + 1), mb, yerr=se, fmt="-o", color=OK["blue"],
                 ms=4, lw=1.4, capsize=2.5)
    axb.set_xlabel("Shared-axis (PC1) score, decile")
    axb.set_ylabel("Mean incident diseases")
    F.panel(axb, "b")
    axb.text(0.05, 0.95, f"Spearman r = {summ['burden_spearman']:.2f}",
             transform=axb.transAxes, ha="left", va="top", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=OK["grey"], lw=0.6))

    # (c) beyond-axis disease-specific signal: on the shared axis alone each outcome collapses
    # toward chance (open grey), but the wrist recovers strong, disease-specific discrimination
    # (filled, organ-system colour). The recovered gap is what the frailty axis cannot carry.
    axc = fig.add_subplot(gs[1, 0])
    SHOW = ["GU_602.1", "MS_727", "GI_520.11", "NS_324.11", "SO_374.5", "NS_328.11",
            "GU_623.1", "MS_745.3", "SO_371", "CV_413.3", "SO_396", "NS_328.1", "CV_416.2"]
    SHORT = {"GU_602.1": "Prostatic hyperplasia", "MS_727": "Bone disorders",
             "GI_520.11": "Inguinal hernia", "NS_324.11": "Parkinson's", "SO_374.5": "Macular degen.",
             "NS_328.11": "Alzheimer's", "GU_623.1": "Uterine prolapse", "MS_745.3": "Upper-limb fracture",
             "SO_371": "Cataract", "CV_413.3": "Tricuspid valve", "SO_396": "Hearing loss",
             "NS_328.1": "Dementia", "CV_416.2": "Atrial fibrillation"}
    gg = g2.loc[SHOW].sort_values("delta_incr")        # ascending -> largest gain at the top
    yy = np.arange(len(gg))
    for yi, (ph, r) in zip(yy, gg.iterrows()):
        col = F.cat_color(r.category)
        axc.plot([r.C_general, r.C_full], [yi, yi], color=col, lw=1.7, alpha=0.5,
                 solid_capstyle="round", zorder=2)
        axc.scatter(r.C_general, yi, s=20, facecolor="white", edgecolor=OK["grey"],
                    linewidth=1.0, zorder=3)
        thin = r.n_ev < 50
        axc.scatter(r.C_full, yi, s=(34 if thin else 28), marker=("D" if thin else "o"),
                    color=col, edgecolor="white", linewidth=0.6, zorder=4)
    axc.axvline(0.5, ls=":", color=OK["grey"], lw=0.9, zorder=1)
    axc.set_yticks(yy)
    axc.set_yticklabels([SHORT[ph] for ph in gg.index], fontsize=6.6)
    for t, ph in zip(axc.get_yticklabels(), gg.index):
        t.set_color(F.cat_color(g2.loc[ph, "category"]))
    axc.set_ylim(-0.7, len(gg) + 0.8)
    axc.set_xlim(0.37, 0.97)
    axc.set_xlabel("Concordance (C-index)")
    axc.tick_params(length=0, axis="y")
    F.panel(axc, "c")
    # call out Parkinson's gain (the lead outcome); diamonds mark fewer than 50 incident cases
    pk = list(gg.index).index("NS_324.11")
    axc.annotate(f"+{g2.loc['NS_324.11', 'delta_incr']:.2f}", (g2.loc['NS_324.11', 'C_full'], pk),
                 xytext=(5, 0), textcoords="offset points", ha="left", va="center",
                 fontsize=6.2, color=F.cat_color("Neurological"), fontweight="bold")
    # label the two marker types inline above the top row (widest spread) instead of a boxed legend,
    # so nothing sits over the dense dumbbells
    top = len(gg) - 1
    r_top = gg.iloc[-1]
    axc.annotate("shared axis", (r_top.C_general, top), xytext=(-7, 10), textcoords="offset points",
                 ha="right", va="bottom", fontsize=5.8, color=OK["grey"],
                 arrowprops=dict(arrowstyle="-", lw=0.5, color=OK["grey"], shrinkA=0, shrinkB=2))
    axc.annotate("full model", (r_top.C_full, top), xytext=(0, 10), textcoords="offset points",
                 ha="center", va="bottom", fontsize=5.8, color="#333333",
                 arrowprops=dict(arrowstyle="-", lw=0.5, color="#999999", shrinkA=0, shrinkB=2))

    # (d) per-organ-system variance partition: shared / group-specific / idiosyncratic
    axd = fig.add_subplot(gs[1, 1])
    vp2 = vp.sort_values("group_specific_frac")
    y = np.arange(len(vp2))
    sh, grp, idi = (vp2["shared_frac"].to_numpy(), vp2["group_specific_frac"].to_numpy(),
                    vp2["idiosyncratic_frac"].to_numpy())
    axd.barh(y, sh, color=OK["blue"])
    axd.barh(y, grp, left=sh, color=OK["orange"])
    axd.barh(y, idi, left=sh + grp, color=OK["grey"])
    axd.set_yticks(y)
    axd.set_yticklabels([f"{F.cat_label(c)} ({m})" for c, m in zip(vp2["category"], vp2["n"])], fontsize=7.0)
    axd.yaxis.tick_right()
    axd.tick_params(length=0)
    axd.set_ylim(-0.6, len(vp2) - 0.4)
    axd.set_xlim(0, 1)
    axd.set_xlabel("Share of predicted-risk variance")
    F.panel(axd, "d")

    # legend stacked VERTICALLY directly beneath panel (d), centred on its column, so the three
    # variance components read as a key for (d) rather than floating in the middle of the figure
    handles = [Patch(facecolor=OK["blue"], label="shared axis"),
               Patch(facecolor=OK["orange"], label="group-specific"),
               Patch(facecolor=OK["grey"], label="idiosyncratic")]
    axd.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.155), ncol=1,
               fontsize=7.0, frameon=False, handlelength=1.1, handleheight=1.0,
               handletextpad=0.5, labelspacing=0.42, borderaxespad=0.0)

    save(fig, "fig_shared_axis")


# --------------------------------------------------------------------------- #
# Extended Data: residual disease-risk correlation after projecting out the shared axis
# (demoted from the main shared-axis figure to make room for the per-disease residual-C panel)
# --------------------------------------------------------------------------- #
def ed_shared_residual_corr():
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.colors import to_rgba

    ld = pd.read_csv(SHARED / "loadings.csv")
    allcats = sorted(ld["category"].unique())
    ccol = {c: F.cat_color(c) for c in allcats}
    bz = np.load(SHARED / "beyond_by_category.npz", allow_pickle=True)
    Sb, bounds = bz["S_beyond_ordered"], [int(b) for b in bz["bounds"]]
    cat_labels, cat_pos = list(bz["cat_labels"]), list(bz["cat_label_pos"])

    fig = plt.figure(figsize=(F.W2, F.W2 * 0.66))  # double-column (180 mm) Nature width; room for labels + colorbar
    gs = GridSpec(1, 2, figure=fig, width_ratios=[0.05, 1.0], wspace=0.05,
                  left=0.20, right=0.90, top=0.95, bottom=0.06)
    axcs = fig.add_subplot(gs[0, 0])    # organ-system colour sidebar
    axc = fig.add_subplot(gs[0, 1])     # heatmap
    n = Sb.shape[0]
    row_cat, prev = [], 0
    for lbl, b in zip(cat_labels, bounds):
        row_cat += [lbl] * (b - prev); prev = b
    strip = np.array([to_rgba(ccol[c]) for c in row_cat]).reshape(n, 1, 4)
    axcs.imshow(strip, aspect="auto", interpolation="nearest")
    axcs.set_xticks([])
    keep_pos, keep_lab, last = [], [], -1e9
    for p, l in sorted(zip(cat_pos, cat_labels)):
        if p - last >= n * 0.052:
            keep_pos.append(p); keep_lab.append(l); last = p
    axcs.set_yticks(keep_pos)
    axcs.set_yticklabels([F.cat_label(l) for l in keep_lab], fontsize=7.0)
    for t, l in zip(axcs.get_yticklabels(), keep_lab):
        t.set_color(ccol[l])
    axcs.tick_params(length=0)

    M = Sb.copy()
    np.fill_diagonal(M, np.nan)
    vmax = float(np.nanpercentile(np.abs(M), 97))
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("white")
    im = axc.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
    for b in bounds[:-1]:
        axc.axhline(b - 0.5, color="k", lw=0.4)
        axc.axvline(b - 0.5, color="k", lw=0.4)
    axc.set_xticks([]); axc.set_yticks([])
    cb = fig.colorbar(im, ax=axc, fraction=0.045, pad=0.03)
    cb.set_label("residual correlation", fontsize=8)
    save(fig, "ed_shared_residual_corr")


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "corr"):
        fig_disease_corr()
    if which in ("all", "controls"):
        fig_controls()
    if which in ("all", "axis"):
        fig_shared_axis()
    if which in ("all", "axis", "resid"):
        ed_shared_residual_corr()
    print("done")
