"""Build two figures from the paper-experiment outputs:
  fig3_prodromal      -> Figure 5: prodromal neurodegeneration
  fig6_calib_utility  -> Extended Data Fig 5: calibration + clinical utility
Reads the figure-data CSV/JSON under paper_experiments/. Vector PDF -> draft/figures/.
"""
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.gridspec import GridSpec
import figlib as F
from figlib import OK, save

UKB_ROOT = os.environ.get("UKB_ROOT", "data")
PE = Path(f"{UKB_ROOT}/paper_experiments")

C_PD = OK["purple"]        # Parkinson's (protagonist)
C_AD = OK["blue"]          # Alzheimer's
C_DEM = OK["green"]        # dementia
C_MODEL = OK["vermillion"] # wrist model in DCA
C_CLIN = OK["grey"]        # demographic baseline


# =========================================================================== #
# FIG 1: overview (pipeline + phenome-wide breadth + transportability)
# =========================================================================== #
def fig1_overview():
    df = F.load_per_disease().copy()
    catmean = df.groupby("PhecodeCategory").c.mean().sort_values()
    cat_order = list(catmean.index)
    df["catrank"] = df.PhecodeCategory.map({c: i for i, c in enumerate(cat_order)})
    df = df.sort_values(["catrank", "c"]).reset_index(drop=True)
    df["x"] = np.arange(len(df)); N = len(df)

    fig = plt.figure(figsize=(F.W2, 7.05))
    gs = GridSpec(4, 2, height_ratios=[0.58, 2.0, 0.36, 1.82], width_ratios=[1.32, 1.0],
                  hspace=0.48, wspace=0.32, figure=fig)

    # (a) pipeline schematic ------------------------------------------------
    axa = fig.add_subplot(gs[0, :]); axa.axis("off"); axa.set_xlim(0, 1); axa.set_ylim(0, 1)
    F.panel(axa, "a")
    titles = ["Wrist\naccelerometer", "Two frozen\nfoundation models", "Daily\nrepresentation",
              "Survival\nmodel", "Incident\ndisease risk"]
    subs = ["UK Biobank\n~1 week", "AcceleRest +\nActivity model", "512-d per day\n(distributional)",
            "rolling-window\nCox model", "390 outcomes\n6-year horizon"]
    cols = [OK["grey"], OK["blue"], OK["green"], OK["purple"], OK["vermillion"]]
    bw, bh, yc = 0.165, 0.74, 0.46
    centers = np.linspace(0.016 + bw / 2, 1 - 0.016 - bw / 2, 5)
    for x, t, s, col in zip(centers, titles, subs, cols):
        axa.add_patch(FancyBboxPatch((x - bw / 2, yc - bh / 2), bw, bh, boxstyle="round,pad=0.008",
                                     linewidth=1.5, edgecolor=col, facecolor="white"))
        axa.text(x, yc + 0.16, t, ha="center", va="center", fontsize=7.4, fontweight="bold", color="#222")
        axa.text(x, yc - 0.20, s, ha="center", va="center", fontsize=6.0, color="#555")
    for i in range(4):
        axa.add_patch(FancyArrowPatch((centers[i] + bw / 2 + 0.004, yc), (centers[i + 1] - bw / 2 - 0.004, yc),
                      arrowstyle="-|>", mutation_scale=10, lw=1.3, color="#888"))

    # (b) phenome-wide breadth (manhattan) ----------------------------------
    axb = fig.add_subplot(gs[1, :]); F.panel(axb, "b")
    for cat in cat_order:
        sub = df[df.PhecodeCategory == cat]
        axb.scatter(sub.x, sub.c, s=11, color=F.cat_color(cat), edgecolor="none", alpha=0.85)
    axb.axhline(0.5, color="#999", ls=":", lw=0.9)
    def pick(ph=None, name=None):
        r = df[df.phecode == ph] if ph else df[df.PhecodeString.str.contains(name, case=False, na=False)].sort_values("c")
        return r.iloc[-1] if len(r) else None
    # (phecode, label, label_x, label_y): manual leader-line control.
    #   label_x, label_y = where the TEXT sits, in DATA units. The RING end of each line is
    #   locked to the real data point; you only move the text end.
    #     x: 0..N runs across the cloud (N=388); use N+something for the right margin
    #        (bigger = further right = longer line).
    #     y: concordance, 0.45 (bottom) .. 0.99 (top) (bigger = higher up).
    #   To move one line, change its label_x / label_y below. For a perfectly horizontal
    #   leader, set label_y to that disease's concordance (the value shown in parentheses).
    spec = [
        ("NS_324.11",     "Parkinson's disease",  N + 22, 0.965),
        ("NS_328.11",     "Alzheimer's disease",  N + 22, 0.925),
        ("CV_424",        "Heart failure",        N + 22, 0.865),
        ("EM_236.1",      "Obesity",              N + 22, 0.825),
        ("NS_328.1",      "Dementia",             N + 22, 0.795),
        ("time_to_death", "All-cause mortality",  N + 22, 0.765),
    ]
    for ph, lab, lx, ly in spec:
        r = pick(ph=ph) if ph else pick(name=lab)
        if r is None:
            continue
        hx, hc = int(r.x), float(r.c)
        # ring on the real data point (LOCKED)
        axb.scatter([hx], [hc], s=44, facecolor="none", edgecolor="#111", linewidth=1.2, zorder=6)
        # leader line from the ring (xy) to the label text (xytext = your label_x/label_y)
        axb.annotate(f"{lab} ({hc:.2f})", xy=(hx, hc), xytext=(lx, ly),
                     fontsize=6.3, va="center", ha="left", color="#111",
                     arrowprops=dict(arrowstyle="-", lw=0.55, color="#b0b0b0",
                                     shrinkA=3.5, shrinkB=4, relpos=(0, 0.5)))
    axb.set_xticks([]); axb.set_xlim(-3, N + 150); axb.set_ylim(0.45, 0.995)
    axb.set_xlabel("388 incident outcomes, grouped by disease category and ranked by concordance within each", fontsize=7.6)
    axb.set_ylabel("Concordance (6-year incident)")
    # disease-category colour key: a rectangular horizontal legend BELOW the plot, in its own
    # gridspec row, so it never covers data points (identical F.cat_color mapping as panel c)
    axleg = fig.add_subplot(gs[2, :]); axleg.axis("off")
    cat_handles = [Line2D([0], [0], marker="o", ls="", mfc=F.cat_color(c), mec="none", ms=4.5,
                          label=F.cat_label(c)) for c in reversed(cat_order)]
    axleg.legend(handles=cat_handles, loc="upper center", bbox_to_anchor=(0.5, 2.0),
                 ncol=7, fontsize=6.0, frameon=True,
                 facecolor="white", edgecolor="#bbbbbb", framealpha=1.0, handletextpad=0.3,
                 columnspacing=1.0, labelspacing=0.6, borderaxespad=0.1)

    # (c) mean concordance by category --------------------------------------
    axc = fig.add_subplot(gs[3, 0]); F.panel(axc, "c")
    cm = catmean
    axc.barh(range(len(cm)), cm.values, color=[F.cat_color(c) for c in cm.index], edgecolor="white", linewidth=0.3)
    axc.set_yticks(range(len(cm))); axc.set_yticklabels([F.cat_label(c) for c in cm.index], fontsize=6.0)
    axc.axvline(0.5, color="#999", ls=":", lw=0.8)
    axc.set_xlim(0.5, 0.76); axc.set_xlabel("Mean concordance by category")

    # (d) transportability across assessment-centre regions -----------------
    axd = fig.add_subplot(gs[3, 1]); F.panel(axd, "d")
    tr = pd.read_csv(PE / "Regional_transport" / "per_region_c.csv")
    dep = tr[(tr.method == "deployed") & (tr.region != "POOLED")].copy()
    lro = tr[(tr.method == "leave_region_out") & (tr.region != "POOLED")].set_index("region")
    order = ["North England", "South England", "Central England", "Scotland", "Wales"]
    dep = dep.set_index("region").reindex(order).reset_index()
    xpos = np.arange(len(order))
    axd.scatter(xpos, dep.mean_C, s=34, color=OK["blue"], zorder=5, label="Deployed model")
    axd.scatter(xpos, [lro.loc[r].mean_C if r in lro.index else np.nan for r in order],
                s=30, facecolor="none", edgecolor=OK["grey"], linewidth=1.2, marker="s", zorder=5, label="Leave-region-out")
    dpool = float(tr[(tr.method == "deployed") & (tr.region == "POOLED")].mean_C)
    lpool = float(tr[(tr.method == "leave_region_out") & (tr.region == "POOLED")].mean_C)
    axd.axhline(dpool, color=OK["blue"], lw=0.9, ls="-", alpha=0.6)
    axd.axhline(lpool, color=OK["grey"], lw=0.9, ls="--", alpha=0.7)
    axd.set_xticks(xpos); axd.set_xticklabels(["North", "South", "Central", "Scot.", "Wales"], fontsize=6.2)
    axd.set_ylim(0.62, 0.72); axd.set_ylabel("Mean concordance")
    axd.text(4, dep.mean_C.iloc[-1] - 0.012, "n=193", fontsize=5.4, color="#888", ha="center")
    axd.legend(fontsize=5.6, loc="upper center", ncol=2, frameon=False, handletextpad=0.3,
               columnspacing=1.0)

    save(fig, "fig1_overview")


# =========================================================================== #
# Figure 5: prodromal neurodegeneration
# =========================================================================== #
def fig3_prodromal():
    td = pd.read_csv(PE / "Prodromal" / "td_metrics.csv")
    ttd = pd.read_csv(PE / "Prodromal" / "time_to_dx.csv")

    # three neurodegenerative outcomes; Parkinson's is the protagonist (emphasised heavier)
    OUT = [("pd", C_PD, "Parkinson's", "o"),
           ("alzheimer", C_AD, "Alzheimer's", "s"),
           ("dementia", C_DEM, "Dementia", "^")]

    def lw_ms_al(dis):
        return (2.0, 4.6, 0.15) if dis == "pd" else (1.3, 3.2, 0.10)

    fig, axes = plt.subplots(1, 3, figsize=(F.W2, 2.7), layout="constrained")

    # (a) time-to-diagnosis distribution, all three outcomes
    ax = axes[0]; F.panel(ax, "a")
    bins = np.arange(0, 11, 0.7)
    for dis, col, lab, _mk in OUT:
        lead = ttd[ttd.disease == dis].years_to_dx.to_numpy()
        lw = 1.9 if dis == "pd" else 1.3
        ax.hist(lead, bins=bins, histtype="step", color=col, lw=lw, label=lab)
        ax.axvline(float(np.median(lead)), color=col, lw=0.9, ls="--", alpha=0.85)
    ax.set_xlabel("Years from recording\nto diagnosis")
    ax.set_ylabel("Incident cases")
    ax.set_xlim(0, 10)
    ax.legend(fontsize=5.8, loc="upper left", frameon=False, handlelength=1.1, handletextpad=0.4)

    # (b) population time-dependent AUROC vs horizon, PD / AD / dementia
    ax = axes[1]; F.panel(ax, "b")
    coh = td[td.design == "cohort"]
    for dis, col, lab, mk in OUT:
        s = coh[coh.disease == dis].sort_values("horizon_y")
        lw, ms, al = lw_ms_al(dis)
        ax.plot(s.horizon_y, s.auroc, "-", marker=mk, color=col, ms=ms, lw=lw, label=lab)
        ax.fill_between(s.horizon_y, s.auroc_lo, s.auroc_hi, color=col, alpha=al, linewidth=0)
    ax.axhline(0.5, color="#999", ls=":", lw=0.8)
    ax.set_xlabel("Prediction horizon (years)")
    ax.set_ylabel("Population time-dependent AUROC")
    ax.set_ylim(0.45, 1.0); ax.set_xticks([2, 3, 5, 7])
    ax.legend(fontsize=6.0, loc="lower left", frameon=False, handlelength=1.4, handletextpad=0.4)

    # (c) lead-time washout: 7-year AUROC excluding cases within w years, all three outcomes
    ax = axes[2]; F.panel(ax, "c")
    ww = td[td.design == "cohort_washout"]
    for dis, col, lab, mk in OUT:
        s = ww[ww.disease == dis].sort_values("washout_y")
        lw, ms, al = lw_ms_al(dis)
        ax.plot(s.washout_y, s.auroc, "-", marker=mk, color=col, ms=ms, lw=lw)
        ax.fill_between(s.washout_y, s.auroc_lo, s.auroc_hi, color=col, alpha=al, linewidth=0)
    ax.set_xlabel("Cases excluded within\nyears of recording (washout)")
    ax.set_ylabel("7-year AUROC")
    ax.set_ylim(0.6, 0.92); ax.set_xticks([0, 1, 2, 3])

    save(fig, "fig3_prodromal")


# =========================================================================== #
# Extended Data Fig 5: calibration + clinical utility
# =========================================================================== #
def fig6_calib_utility():
    cc = pd.read_csv(PE / "Calibration" / "calib_curves.csv")
    cs = json.load(open(PE / "Calibration" / "calib_summary.json"))
    dca = pd.read_csv(PE / "Decision_curve" / "dca.csv")
    ops = pd.read_csv(PE / "Decision_curve" / "screening_operating_points.csv")

    fig, axes = plt.subplots(1, 5, figsize=(F.W2, 2.5), layout="constrained")

    # (a) calibration curves for the flagship low-incidence outcomes (PD, dementia, heart failure)
    ax = axes[0]; F.panel(ax, "a")
    sel = [("Parkinson's disease", "Parkinson's", C_PD),
           ("All-cause dementia", "Dementia", C_DEM),
           ("Heart failure", "Heart failure", OK["orange"])]
    mx = 0.0
    for name, lab, col in sel:
        s = cc[cc.outcome == name].dropna(subset=["obs_km"])
        ax.plot(s.pred_mean, s.obs_km, "-o", color=col, ms=3.0, lw=1.2, label=lab)
        if len(s):
            mx = max(mx, float(s.pred_mean.max()), float(s.obs_km.max()))
    mx = mx * 1.08
    ax.plot([0, mx], [0, mx], color="#bbb", lw=0.9, ls="--", zorder=0)
    ax.set_xlim(0, mx); ax.set_ylim(0, mx)
    ax.set_xlabel("Predicted 6-year risk")
    ax.set_ylabel("Observed incidence")
    ax.legend(fontsize=5.6, loc="upper left", frameon=False, handlelength=1.1, handletextpad=0.3)

    # (b) calibration-slope distribution across the panel
    ax = axes[1]; F.panel(ax, "b")
    slopes = np.array([v["slope"] for v in cs["all_panel_slopes"].values()
                       if v.get("n_event", 0) >= 50], dtype=float)
    slopes = slopes[np.isfinite(slopes)]
    slc = np.clip(slopes, 0, 2.5)
    ax.hist(slc, bins=np.arange(0, 2.55, 0.15), color=OK["green"], edgecolor="white", linewidth=0.4, alpha=0.9)
    ax.axvspan(0.8, 1.25, color="#cccccc", alpha=0.35, lw=0)
    ax.axvline(1.0, color="#888", lw=0.8, ls=":")
    med = float(np.median(slopes))
    ax.axvline(med, color="#222", lw=1.0, ls="--")
    ax.text(med + 0.05, ax.get_ylim()[1] * 0.93, f"median {med:.2f}", fontsize=6.4, color="#222", ha="left", va="top")
    ax.set_xlabel("Calibration slope")
    ax.set_ylabel("Outcomes")
    ax.set_xlim(0, 2.5)

    # (c) decision-curve analysis for incident Parkinson's
    ax = axes[2]; F.panel(ax, "c")
    d = dca.sort_values("threshold")
    ax.plot(d.threshold * 100, d.nb_model, color=C_MODEL, lw=1.6, label="Wrist model")
    ax.plot(d.threshold * 100, d.nb_clinical, color=C_CLIN, lw=1.2, label="Age, sex, BMI")
    ax.plot(d.threshold * 100, d.nb_all, color="#999", lw=1.0, ls="--", label="Screen all")
    ax.axhline(0.0, color="#999", lw=0.8, ls=":")
    ax.set_xlabel("Threshold probability (%)")
    ax.set_ylabel("Net benefit")
    ax.set_xlim(0, min(1.5, d.threshold.max() * 100))
    ax.set_ylim(-0.0012, 0.0033)   # focus on the relevant window (rare-disease net benefit is small)
    ax.legend(fontsize=5.6, loc="upper right", frameon=False, handlelength=1.3, handletextpad=0.3)

    # (d) screening yield at two operating points
    ax = axes[3]; F.panel(ax, "d")
    o = ops.set_index("operating_point")
    pts = [("top_decile", "Top 10%"), ("spec95", "95% spec.")]
    x = np.arange(len(pts)); wbar = 0.55
    enr = [float(o.loc[k].fold_enrichment) for k, _ in pts]
    ax.bar(x, enr, wbar, color=C_PD, edgecolor="white",
           yerr=[[enr[i] - float(o.loc[pts[i][0]].enrich_lo) for i in range(len(pts))],
                 [float(o.loc[pts[i][0]].enrich_hi) - enr[i] for i in range(len(pts))]],
           error_kw=dict(lw=0.8, capsize=2))
    for i, (k, _) in enumerate(pts):
        # place the sens/NNS label above the upper error-bar cap, not just above the bar, so the
        # whisker never overlaps the text
        ax.text(i, float(o.loc[k].enrich_hi) + 0.4, f"{int(round(100*float(o.loc[k].sens)))}% sens\nNNS {int(round(float(o.loc[k].nns)))}",
                ha="center", va="bottom", fontsize=5.8, color="#333")
    ax.set_xticks(x); ax.set_xticklabels([p[1] for p in pts], fontsize=6.6)
    ax.set_ylabel("Fold-enrichment of incident PD")
    ax.set_ylim(0, max(enr) * 1.45)

    # (e) proportional-hazards check: distribution of Schoenfeld residual test P values
    #     across the 101 well-powered outcomes (>=50 incident events). Absorbed here from the
    #     former standalone ed_ph_check figure. Under proportional hazards the P values are
    #     uniform, so the dashed line is the count expected per bin.
    ax = axes[4]; F.panel(ax, "e")
    phe = pd.read_csv(F.UKB / "paper_experiments" / "output" / "Proportional_hazards_check" / "ph_test_perdisease.csv")
    phe = phe[phe["n_event"] >= 50]
    pph = phe["ph_p"].to_numpy(float)
    NBINS_PH = 20
    ax.hist(pph, bins=np.linspace(0, 1, NBINS_PH + 1), color=OK["blue"],
            edgecolor="white", linewidth=0.5, zorder=2)
    ax.axhline(len(pph) / NBINS_PH, color=OK["grey"], ls="--", lw=1.0, zorder=3)
    ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel("Schoenfeld $P$ value")
    ax.set_ylabel("Outcomes")

    save(fig, "fig6_calib_utility")


if __name__ == "__main__":
    print("building paper-experiment figures...")
    fig1_overview()
    fig3_prodromal()
    fig6_calib_utility()
    print("done.")
