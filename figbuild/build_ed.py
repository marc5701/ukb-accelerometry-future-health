"""Extended Data figures for the UKB disease-prediction paper."""
import os
import json
import traceback
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import figlib as F
from figlib import OK, CONST, save
C = CONST


# ED1: cohort / split flow
def ed1_cohort_flow():
    fig = plt.figure(figsize=(F.W2, 3.0)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    # top: pretraining pool
    top = FancyBboxPatch((0.20, 0.70), 0.60, 0.12, boxstyle="round,pad=0.01",
                         lw=1.5, edgecolor=OK["grey"], facecolor="#f3f3f3")
    ax.add_patch(top)
    ax.text(0.5, 0.76, "Self-supervised pretraining: 108,904 UK Biobank wrist recordings\n"
            "(AcceleRest + Activity foundation models)",
            ha="center", va="center", fontsize=7.6)
    # three split boxes
    splits = [("Train", "87,652 subjects\n96,016 recordings", OK["blue"], 0.18),
              ("Validation", "4,772 subjects\n5,331 recordings", OK["orange"], 0.5),
              ("Test (held out)", "5,272 subjects\n7,460 recordings\nbenchmark N = 5,253", OK["green"], 0.82)]
    for name, sub, col, x in splits:
        b = FancyBboxPatch((x - 0.14, 0.28), 0.28, 0.20, boxstyle="round,pad=0.01",
                           lw=1.8, edgecolor=col, facecolor="white")
        ax.add_patch(b)
        ax.text(x, 0.44, name, ha="center", va="center", fontsize=9, fontweight="bold", color=col)
        ax.text(x, 0.355, sub, ha="center", va="center", fontsize=7.3, color="#333")
        # arrows point at each box with a clear gap (do not touch the box edges)
        ax.add_patch(FancyArrowPatch((0.5, 0.665), (x, 0.515), arrowstyle="-|>",
                     mutation_scale=11, lw=1.2, color="#aaa"))
    # (subject-disjointness, panel size and scoring details are stated in the caption)
    save(fig, "ed1_cohort_flow")


# ED2: panel composition
def ed2_composition():
    comp = F.per_disease_full()["_composition"]
    fig, ax = plt.subplots(figsize=(F.W15, 3.0))
    labels = ["282 shared\n(previous split)", "282 shared\n(this split)", "106 added rare\n(this split)"]
    vals = [comp["shared_meanC_b"], comp["shared_meanC_a"], comp["a_only_meanC"]]
    cols = [OK["grey"], OK["blue"], OK["green"]]
    bars = ax.bar(np.arange(3), vals, 0.6, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.002, f"{v:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.axhline(C["MEAN_C_EMBEDDING"], color=OK["vermillion"], ls="--", lw=1.0)
    # label above the line on the left, over the short bars (clear of the third bar)
    ax.text(0.05, C["MEAN_C_EMBEDDING"] + 0.0025, f"full-panel mean {C['MEAN_C_EMBEDDING']:.3f}",
            fontsize=7, color=OK["vermillion"], ha="left", va="bottom")
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.66, 0.72); ax.set_ylabel("Mean concordance")
    ax.set_title("The headline rises, not falls: shared outcomes are stable (Δ+0.0009);\nthe 106 added rare "
                 "diseases are higher-concordance (0.710)", fontsize=9, fontweight="bold")
    fig.tight_layout(); save(fig, "ed2_composition")


# ED3: age estimation
def ed3_age():
    fig, ax = plt.subplots(figsize=(F.W1, 2.9), layout="constrained")
    steps = ["Activity model\n(daily mean)", "+ AcceleRest +\nL-moments", "+ time-of-day\n(4 dayparts)"]
    vals = [C["AGE_HA_MEAN"], 0.8515, C["AGE_TEST_R"]]
    ax.plot(range(3), vals, "-o", color=OK["blue"], lw=2, ms=7)
    for i, v in enumerate(vals):
        # clear the dot; put the leftmost label below so it doesn't touch the benchmark line
        if i == 0:
            ax.text(i, v - 0.022, f"{v:.3f}", ha="center", va="top", fontsize=8, fontweight="bold")
        else:
            ax.text(i, v + 0.020, f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.axhline(C["AGE_OXFORD"], color=OK["vermillion"], ls="--", lw=1.2)
    # label sits clearly below the dashed line (right side is empty there)
    ax.text(2.45, C["AGE_OXFORD"] - 0.020, f"prior wrist-age benchmark {C['AGE_OXFORD']}",
            fontsize=7, color=OK["vermillion"], ha="right", va="top")
    ax.axhline(C["GBM_ENMO"], color="#999", ls=":", lw=1.0)
    ax.text(0.0, C["GBM_ENMO"] + 0.006, f"classical rest-activity features {C['GBM_ENMO']:.2f}",
            fontsize=6.8, color="#888", ha="left")
    ax.set_xticks(range(3)); ax.set_xticklabels(steps, fontsize=7.4)
    # floor lowered to 0.40 so the classical-baseline line (0.478) sits clearly
    # above the axis bottom and reads as its own value, not the axis floor.
    ax.set_ylim(0.40, 0.90); ax.set_ylabel("Predicted vs chronological age (Pearson r)")

    save(fig, "ed3_age")


# ED4: BMI estimation
def ed4_bmi():
    fig, axes = plt.subplots(1, 2, figsize=(F.W2, 3.0), gridspec_kw={"width_ratios": [1, 1]})
    ax = axes[0]
    labs = ["AcceleRest\nonly", "Activity\nonly", "Both +\nL-moments", "Held-out\ntest"]
    vals = [0.534, 0.688, C["BMI_VAL_R"], C["BMI_TEST_R"]]
    cols = [OK["purple"], OK["blue"], OK["green"], OK["black"]]
    bars = ax.bar(np.arange(4), vals, 0.62, color=cols)
    for i, v in enumerate(vals):
        if i == 1:
            # the Activity-only bar (0.688) sits just under the dashed prior-benchmark
            # line (0.716); put its value inside the bar so the line does not strike the label
            ax.text(i, v - 0.012, f"{v:.3f}", ha="center", va="top", fontsize=7.6,
                    fontweight="bold", color="white")
        else:
            ax.text(i, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold")
    ax.axhline(C["BMI_OXFORD"], color=OK["vermillion"], ls="--", lw=1.2)
    # label in the clear top-left area (above the two short bars), away from the value labels
    ax.text(0.0, 0.81, f"prior benchmark {C['BMI_OXFORD']}",
            fontsize=7, color=OK["vermillion"], ha="left", va="top")
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(labs, fontsize=7.2)
    ax.set_ylim(0.0, 0.85); ax.set_ylabel("BMI prediction (Pearson r)")
    F.panel(ax, "a")

    ax = axes[1]
    labs = ["Raw", "Age/sex-\nresidualized", "Within age/\nsex cells", "Age/sex\nalone"]
    vals = [C["BMI_TEST_R"], C["BMI_PARTIAL"], 0.765, C["BMI_DEMO"]]
    ax.bar(np.arange(4), vals, 0.62, color=[OK["black"], OK["green"], OK["green"], OK["grey"]])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold")
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(labs, fontsize=7.0)
    ax.set_ylim(0, 0.85); ax.set_ylabel("Pearson r")
    F.panel(ax, "b")
    fig.tight_layout(); save(fig, "ed4_bmi")


# ED5: age/sex decomposition + estimation floor
def ed5_agesex():
    """Why demographics add little: age, sex and BMI decoded directly from the frozen
    disease-prediction embedding (linear readout, fit on validation, scored on the held-out test
    set). (a) sex by ROC/AUROC, (b) age and (c) BMI by predicted-vs-true (Pearson r, r^2, MAE).
    Distinct from Fig 3 d-f, which shows the stronger DEDICATED wrist estimators."""
    d = F.load_embedding_decode()
    fig = plt.figure(figsize=(F.W2, 2.5))
    gs = fig.add_gridspec(1, 3, wspace=0.40, left=0.065, right=0.985, top=0.86, bottom=0.16)

    # (a) sex ROC
    axa = fig.add_subplot(gs[0]); fig.text(0.012, 0.91, "a", fontsize=11, fontweight="bold", va="top")
    axa.plot([0, 1], [0, 1], color="#c0c0c0", ls="--", lw=0.9, zorder=2)
    axa.plot(d["sex_fpr"], d["sex_tpr"], color=OK["green"], lw=2.0, zorder=3)
    axa.set_xlim(0, 1); axa.set_ylim(0, 1); axa.set_aspect("equal", adjustable="box")
    axa.set_xlabel("False-positive rate", fontsize=8); axa.set_ylabel("True-positive rate", fontsize=8)
    axa.tick_params(labelsize=6.5)
    axa.text(0.93, 0.10, f"AUROC = {float(d['sex_auc']):.3f}", ha="right", va="bottom",
             fontsize=7.6, fontweight="bold", color="#176217")

    def scatter(ax, x, y, lo, hi, col, xlab, ylab, r, r2, mae, unit):
        # light, white-edged, rasterized markers so the cloud reads as discrete dots and embeds
        # cleanly in the vector PDF; identity line drawn on top so it stays visible.
        ax.scatter(x, y, s=4, c=col, alpha=0.30, edgecolors="white", linewidths=0.15,
                   zorder=3, rasterized=True)
        ax.plot([lo, hi], [lo, hi], color="#333", ls="--", lw=1.0, zorder=4)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(xlab, fontsize=8); ax.set_ylabel(ylab, fontsize=8); ax.tick_params(labelsize=6.5)
        ax.text(0.05, 0.96, f"$r$ = {r:.3f}\n$r^2$ = {r2:.2f}\nMAE = {mae:.2f}{unit}",
                transform=ax.transAxes, va="top", ha="left", fontsize=6.8, color="#333")

    # (b) age
    axb = fig.add_subplot(gs[1]); fig.text(0.357, 0.91, "b", fontsize=11, fontweight="bold", va="top")
    scatter(axb, d["age_true"], d["age_pred"], 40, 85, OK["blue"],
            "Chronological age (yr)", "Predicted age (yr)",
            float(d["age_r"]), float(d["age_r2"]), float(d["age_mae"]), " yr")
    # (c) BMI
    axc = fig.add_subplot(gs[2]); fig.text(0.665, 0.91, "c", fontsize=11, fontweight="bold", va="top")
    scatter(axc, d["bmi_true"], d["bmi_pred"], 15, 55, OK["vermillion"],
            "Measured BMI (kg m$^{-2}$)", "Predicted BMI (kg m$^{-2}$)",
            float(d["bmi_r"]), float(d["bmi_r2"]), float(d["bmi_mae"]), "")

    save(fig, "ed5_agesex")


# ED6: residual orthogonality
def ed6_residual():
    e1, _ = F.load_residual()
    d = e1[e1.phecode != "time_to_death"]
    fig, axes = plt.subplots(1, 2, figsize=(F.W2, 3.0), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    ax.hist(d.dC_wrist, bins=40, color=OK["blue"], alpha=0.85)
    ax.axvline(0, color="#333", lw=1.0)
    ax.axvline(d.dC_wrist.mean(), color=OK["vermillion"], lw=1.5, ls="--")
    ax.text(d.dC_wrist.mean() + 0.002, ax.get_ylim()[1] * 0.9,
            f"mean +{d.dC_wrist.mean():.3f}", color=OK["vermillion"], fontsize=7.5)
    ax.set_xlabel("Wrist gain in C beyond flexible age + sex + BMI (per disease)")
    ax.set_ylabel("Number of diseases")
    F.panel(ax, "a")
    ax = axes[1]
    frac_wrist = float((d.dC_wrist > 0).mean())
    wbd = float(d.dC_wrist.mean())
    dnull = float(d.dC_null.mean())
    vals = [wbd, dnull]
    # 95% disease-panel bootstrap CIs (B=1000), written by the sibling ed_bar_cis.py
    _cis_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ed_bar_cis.json")
    _rci = json.load(open(_cis_json))["figures"]["residual"]
    _rk = ["wrist", "null"]
    assert all(abs(vals[i] - _rci[k]["point"]) < 5e-4 for i, k in enumerate(_rk)), "residual CI/bar mismatch"
    rlo = [vals[i] - _rci[k]["ci_lo"] for i, k in enumerate(_rk)]
    rhi = [_rci[k]["ci_hi"] - vals[i] for i, k in enumerate(_rk)]
    ax.bar([0, 1], vals, 0.5, color=[OK["blue"], OK["grey"]], zorder=2)
    ax.errorbar([0, 1], vals, yerr=[rlo, rhi], fmt="none", ecolor="#222", elinewidth=1.0,
                capsize=3, capthick=1.0, zorder=3)
    for i, k in enumerate(_rk):
        ax.text(i, _rci[k]["ci_hi"] + 0.0010, f"{vals[i]:+.3f}", ha="center", va="bottom",
                fontsize=8.5, fontweight="bold")
    ax.axhline(0, color="#333", lw=0.9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Wrist beyond\nage+sex+BMI", "Matched random\ncontrol"], fontsize=7.2)
    ax.set_ylim(min(_rci["null"]["ci_lo"] - 0.002, -0.003), _rci["wrist"]["ci_hi"] + 0.007)
    ax.set_ylabel("Mean Δ C")
    F.panel(ax, "b")
    fig.tight_layout(); save(fig, "ed6_residual")


# ED7: day/night attribution
def ed7_day_night():
    dn = F.load_day_night_profiles()
    cells = ["HA_day", "HA_night", "AR_day", "AR_night"]
    nice = {"HA_day": "Activity\nday", "HA_night": "Activity\nnight", "AR_day": "Sleep /\nbreathing\nday", "AR_night": "Sleep /\nbreathing\nnight"}
    fig, axes = plt.subplots(1, 2, figsize=(F.W2, 3.1), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    glob = [dn[(dn.cell == c)].dc_mean.mean() for c in cells]
    ax.bar(np.arange(4), glob, 0.6, color=[OK["blue"], OK["skyblue"], OK["purple"], "#c9a8d6"])
    for i, v in enumerate(glob):
        ax.text(i, v + 0.0003, f"{v:.4f}", ha="center", fontsize=7.4, fontweight="bold")
    ax.set_xticks(np.arange(4)); ax.set_xticklabels([nice[c] for c in cells], fontsize=7.2)
    ax.set_ylabel("Mean drop in C when removed")
    F.panel(ax, "a")
    # (b) per-disease heatmap
    ax = axes[1]
    wl = F.load_selected_diseases(); lab2ph = dict(zip(wl.label, wl.phecode))
    dis = [("Parkinson's", "NS_324.11"), ("Alzheimer's", "NS_328.11"), ("Obesity", "EM_236.1"),
           ("All-cause mortality", "time_to_death")]
    for nm in ["Sleep apnea", "Type 2 diabetes"]:
        cand = [v for k, v in lab2ph.items() if nm.split()[0].lower() in k.lower()]
        if cand: dis.append((nm, cand[0]))
    M, names = [], []
    for nm, ph in dis:
        row = [dn[(dn.phecode == ph) & (dn.cell == c)].dc_mean.values for c in cells]
        if all(len(r) for r in row):
            M.append([r[0] for r in row]); names.append(nm)
    M = np.array(M)
    vmin, vmax = -0.01, 0.04
    im = ax.imshow(M, cmap="RdYlBu_r", aspect="auto", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(4)); ax.set_xticklabels([nice[c].replace("\n", " ") for c in cells], fontsize=6.4, rotation=20, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7.2)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            # RdYlBu_r is dark only at the extreme ends (deep blue low, deep red high) and
            # light through the middle (light blue -> yellow -> light orange): use white text
            # only on the genuinely dark cells, dark text everywhere else.
            norm = (M[i, j] - vmin) / (vmax - vmin)
            txtcol = "white" if (norm < 0.20 or norm > 0.80) else "#000"
            ax.text(j, i, f"{M[i,j]:+.3f}", ha="center", va="center", fontsize=5.6, color=txtcol)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=6)
    F.panel(ax, "b")
    fig.tight_layout(); save(fig, "ed7_day_night")


# ED8: negative-results compendium
def ed8_negatives():
    items = sorted(F.NEG_RESULTS, key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(F.W15, 3.4))
    y = np.arange(len(items))
    vals = [v for _, v in items]
    cols = [OK["green"] if v > 0.005 else OK["vermillion"] if v < -0.001 else OK["grey"] for v in vals]
    ax.barh(y, vals, color=cols, edgecolor="white", height=0.7)
    for i, (nm, v) in enumerate(items):
        if v >= 0:
            ax.text(v + 0.0007, i, f"{v:+.4f}", va="center", ha="left", fontsize=6.4, color="#333")
        else:
            ax.text(0.0008, i, f"{v:+.4f}", va="center", ha="left", fontsize=6.4, color="#444")
    ax.axvline(0, color="#333", lw=1.1)
    ax.set_yticks(y); ax.set_yticklabels([nm for nm, _ in items], fontsize=7.2)
    ax.set_xlabel("Δ mean C vs the relevant baseline")
    ax.set_title("Every architecture, loss, and feature lever we tried: only a disease-specific\n"
                 "sampling tweak (Parkinson's) helped; the rest tie or lose", fontsize=8.8, fontweight="bold")
    ax.set_xlim(-0.040, 0.026)
    fig.tight_layout(); save(fig, "ed8_negatives")


# ED9: prodromal-PD AUPRC vs prior single-modality benchmarks
def ed9_pd_benchmark():
    """Wrist embedding vs Schalkamp et al. 2023 accelerometry, two matched regimes.
    (a) average precision of our wrist embedding vs their bespoke accelerometry model for
    PRODROMAL PD (diagnosed >2 yr later) and for DIAGNOSED / prevalent-screening PD; each
    our-vs-theirs pair at a shared base rate. (b) lead-time (washout) stability of our
    prodromal AUPRC vs their prodromal 0.07. Data: investigations/pd_auprc/results.json."""
    from matplotlib.patches import Patch
    d = F.load_pd_auprc()
    pr, dg, sk, wsh = (d["primary_prodromal_cohort"], d["primary_diagnosed_cohort"],
                       d["schalkamp"], d["washout_stability"])
    GREEN, GREY, SKC = OK["green"], "#9a9a9a", "#d0651a"

    fig = plt.figure(figsize=(F.W2, 3.35))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.28,
                          left=0.085, right=0.985, top=0.90, bottom=0.16)

    # (a) AUPRC: wrist embedding vs Schalkamp accelerometry, two regimes
    axa = fig.add_subplot(gs[0, 0]); F.panel(axa, "a")
    skp, skd = sk["accelerometry_prodromal"], sk["accelerometry_diagnosed"]
    groups = [  # (x label, ours, theirs)
        ("Prodromal\n(dx >2 yr later)",
         dict(v=pr["auprc"], ci=pr["auprc_ci"], enr=pr["enrichment"]),
         dict(v=skp["auprc"], sd=skp["sd"], enr=skp["enrichment"])),
        ("Diagnosed\n(prevalent screening)",
         dict(v=dg["auprc"], ci=dg["auprc_ci"], enr=dg["enrichment"]),
         dict(v=skd["auprc"], sd=skd["sd"], enr=skd["enrichment"])),
    ]
    bw = 0.34
    for gi, (glab, ours, theirs) in enumerate(groups):
        xo, xt = gi - 0.19, gi + 0.19
        axa.bar(xo, ours["v"], width=bw, color=GREEN, zorder=2)
        elo, ehi = ours["v"] - ours["ci"][0], ours["ci"][1] - ours["v"]
        axa.errorbar(xo, ours["v"], yerr=[[elo], [ehi]], fmt="none", ecolor="#333", capsize=2.5, lw=1.0, zorder=3)
        axa.text(xo, ours["ci"][1] + 0.018, f"{ours['v']:.2f}", ha="center", va="bottom", fontsize=7.6, fontweight="bold", color="#111")
        axa.text(xo, ours["ci"][1] + 0.062, f"{ours['enr']:.0f}×", ha="center", va="bottom", fontsize=6.1, color=GREEN)
        axa.bar(xt, theirs["v"], width=bw, color=GREY, zorder=2)
        axa.errorbar(xt, theirs["v"], yerr=theirs["sd"], fmt="none", ecolor="#333", capsize=2.5, lw=1.0, zorder=3)
        axa.text(xt, theirs["v"] + theirs["sd"] + 0.018, f"{theirs['v']:.2f}", ha="center", va="bottom", fontsize=7.0, color="#555")
        axa.text(xt, theirs["v"] + theirs["sd"] + 0.062, f"{theirs['enr']:.0f}×", ha="center", va="bottom", fontsize=6.1, color="#8a8a8a")
    axa.set_xticks([0, 1]); axa.set_xticklabels([g[0] for g in groups], fontsize=7.2)
    axa.set_xlim(-0.6, 1.6); axa.set_ylim(0, 0.9)
    axa.set_ylabel("Average precision (AP)")
    axa.legend(handles=[Patch(color=GREEN, label="This work (wrist embedding)"),
                        Patch(color=GREY, label="Schalkamp et al. 2023")],
               loc="upper left", frameon=False, fontsize=6.4, handlelength=1.1,
               handletextpad=0.5, borderaxespad=0.3)

    # (b) lead-time (washout) stability
    axb = fig.add_subplot(gs[0, 1]); F.panel(axb, "b")
    ws = [s["washout_y"] for s in wsh]; ap = [s["auprc"] for s in wsh]
    lo = [s["auprc_ci"][0] for s in wsh]; hi = [s["auprc_ci"][1] for s in wsh]
    skp = sk["accelerometry_prodromal"]
    axb.fill_between([min(ws), max(ws)], skp["auprc"] - skp["sd"], skp["auprc"] + skp["sd"], color=SKC, alpha=0.10, zorder=0)
    axb.axhline(skp["auprc"], color=SKC, ls="--", lw=1.2, zorder=2)
    axb.text(3.1, skp["auprc"] + 0.007, "Schalkamp\nprodromal 0.07", fontsize=6.0, color=SKC, va="bottom", ha="right")
    axb.fill_between(ws, lo, hi, color=GREEN, alpha=0.18, zorder=1)
    axb.plot(ws, ap, "-o", color=GREEN, ms=4, lw=1.5, zorder=3)
    axb.axvline(2.0, color="#bbb", ls=":", lw=0.9, zorder=0)
    axb.text(2.0, 0.285, "matched\ndefinition", fontsize=6.0, color="#888", ha="center", va="top")
    axb.set_xticks([0, 1, 2, 3]); axb.set_xlim(-0.15, 3.2); axb.set_ylim(0, 0.30)
    axb.set_xlabel("Lead-time washout (years)"); axb.set_ylabel("AP")

    save(fig, "ed9_pd_benchmark")


# ED pipeline: the stage-by-stage search over other approaches (sequence model is Extended Data Fig 4)
def ed_pipeline():
    """Pipeline-stage map of the other approaches we tried (the age-pretrained
    sequence model is Extended Data Fig 4). (a) schematic; (b) honest physiology gate."""
    GREEN, RED, GREY = OK["green"], OK["vermillion"], "#6b6b6b"

    fig = plt.figure(figsize=(F.W2, 5.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.45, 1.0], hspace=0.34,
                          left=0.065, right=0.985, top=0.96, bottom=0.085)
    axs = fig.add_subplot(gs[0]); axs.axis("off"); axs.set_xlim(0, 1); axs.set_ylim(0, 1)

    fig.text(0.012, 0.935, "a", fontsize=11, fontweight="bold", va="top", ha="left")
    boxes = ["Raw wrist\nsignal", "Frozen\nencoder", "Daily\naggregation",
             "Temporal\nmodel", "Cox loss"]
    cx = [0.085, 0.275, 0.465, 0.655, 0.845]
    bw, bh, by = 0.135, 0.15, 0.81
    edge = [GREY, GREY, GREEN, GREY, GREY]
    for x, lab, ec in zip(cx, boxes, edge):
        axs.add_patch(FancyBboxPatch((x - bw / 2, by - bh / 2), bw, bh,
                      boxstyle="round,pad=0.008,rounding_size=0.025",
                      lw=1.7, edgecolor=ec, facecolor="white", zorder=3))
        axs.text(x, by, lab, ha="center", va="center", fontsize=7.3,
                 fontweight="bold", color="#222", zorder=4)
    for i in range(len(cx) - 1):
        axs.add_patch(FancyArrowPatch((cx[i] + bw / 2 + 0.004, by), (cx[i + 1] - bw / 2 - 0.004, by),
                      arrowstyle="-|>", mutation_scale=11, lw=1.3, color="#888", zorder=2))
    axs.add_patch(FancyArrowPatch((cx[4] + bw / 2 + 0.004, by), (0.95, by),
                  arrowstyle="-|>", mutation_scale=11, lw=1.3, color="#888", zorder=2))
    axs.text(0.992, by, "390\ndisease\nhazards", ha="right", va="center",
             fontsize=5.6, color="#666", style="italic")

    cards = [
        (1, RED,   [("Fine-tuning", "-0.050"), ("Age-pretrain init", "ties"), ("Input masking", "0.000")]),
        (2, GREEN, [("Distributional", "+0.0016"), ("Per-day mean", "base"), ("Covariance", "-0.007")]),
        (3, RED,   [("State-space", "-0.004"), ("Longer context", "-0.007"), ("Time-of-day", "+0.000")]),
        (4, RED,   [("Discrete-time", "-0.008"), ("Auxiliary heads", "-0.001"), ("Large batch", "-0.007")]),
    ]
    cw, ctop, cbot = 0.182, 0.585, 0.175
    for idx, mark, lines in cards:
        x = cx[idx]
        axs.add_patch(FancyArrowPatch((x, by - bh / 2 - 0.002), (x, ctop + 0.008),
                      arrowstyle="-", lw=0.8, color="#c8c8c8", zorder=1))
        axs.add_patch(FancyBboxPatch((x - cw / 2, cbot), cw, ctop - cbot,
                      boxstyle="round,pad=0.006,rounding_size=0.018", lw=1.1,
                      edgecolor=mark, facecolor=("#eef7ee" if mark == GREEN else "#fafafa"), zorder=2))
        sym = r"$\checkmark$" if mark == GREEN else r"$\times$"
        axs.text(x, ctop - 0.05, sym, ha="center", va="center", fontsize=12.5, color=mark, zorder=4)
        ly = ctop - 0.135
        for nm, dl in lines:
            is_win = dl == "+0.0016"
            axs.text(x - cw / 2 + 0.011, ly, nm, ha="left", va="center", fontsize=5.3,
                     color=("#176217" if is_win else "#222"),
                     fontweight=("bold" if is_win else "normal"), zorder=4)
            if dl == "base":
                dtxt, dcol, dwt = "(baseline)", "#777", "normal"
            elif dl[:1] in "+-":
                dtxt = "$%s$" % dl
                dcol = GREEN if is_win else "#333"
                dwt = "bold" if is_win else "normal"
            else:
                dtxt, dcol, dwt = dl, "#333", "normal"
            axs.text(x + cw / 2 - 0.011, ly, dtxt, ha="right", va="center", fontsize=5.3,
                     color=dcol, fontweight=dwt, zorder=4)
            ly -= 0.088

    axs.add_patch(FancyBboxPatch((0.018, 0.02), 0.93 - 0.018, 0.108,
                  boxstyle="round,pad=0.004,rounding_size=0.012", lw=0.9,
                  edgecolor="#cccccc", facecolor="#fbfbfb", zorder=1))
    axs.text(0.035, 0.095, "Augmentations to the model (both redundant):", fontsize=6.0,
             fontweight="bold", color="#444", ha="left", va="center")
    axs.text(0.05, 0.05, r"$\times$  hand-engineered physiology features", fontsize=5.8,
             color="#555", ha="left", va="center")
    axs.text(0.52, 0.05, r"$\times$  disease-specific sampling (helps Parkinson's only)",
             fontsize=5.8, color="#555", ha="left", va="center")

    ax = fig.add_subplot(gs[1])
    # panel (b) has long multi-word y-tick labels; indent its axes so they fit inside the
    # fixed canvas (the schematic above keeps the gridspec's wider full-width box).
    _p = ax.get_position()
    ax.set_position([0.135, _p.y0, _p.x1 - 0.135, _p.height])
    fig.text(0.012, 0.415, "b", fontsize=11, fontweight="bold", va="top", ha="left")
    _, boot, _ = F.load_crossfit()
    arms = boot["arms"]
    order = [("regularity", "Rest-activity\nregularity"), ("resp", "Respiratory\nburden"),
             ("rem", "REM/deep\narchitecture"), ("sleepqual", "Sleep\nquality"),
             ("all_sleep", "All engineered\nfeatures"), ("null_rar", "Matched random\ncontrol")]
    for i, (k, lab) in enumerate(order):
        a = arms[k]
        mean = a["mean_delta"]; lo, hi = a["mean_delta_ci"]
        col = "#555" if k == "null_rar" else OK["vermillion"]
        ax.plot([lo, hi], [i, i], color=col, lw=2.4, solid_capstyle="round", zorder=3)
        ax.scatter([mean], [i], s=30, color=col, zorder=4, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="#333", lw=1.1)
    ax.set_yticks(np.arange(len(order))); ax.set_yticklabels([o[1] for o in order], fontsize=6.7)
    ax.set_xlim(-0.062, 0.006)
    ax.set_xlabel("Out-of-fold ΔC added to the deep model", fontsize=8.2)
    save(fig, "ed_pipeline")


def ed_demographics():
    """Wrist recovery of the three demographics, held-out test: sex by AUROC (ROC),
    age and BMI by Pearson r with r-squared and MAE (predicted-vs-true)."""
    d = F.load_demographics()
    fig = plt.figure(figsize=(F.W2, 2.5))
    gs = fig.add_gridspec(1, 3, wspace=0.40, left=0.065, right=0.985, top=0.86, bottom=0.16)

    # (a) sex ROC
    axa = fig.add_subplot(gs[0]); fig.text(0.012, 0.91, "a", fontsize=11, fontweight="bold", va="top")
    axa.plot([0, 1], [0, 1], color="#c0c0c0", ls="--", lw=0.9, zorder=2)
    axa.plot(d["sex_fpr"], d["sex_tpr"], color=OK["green"], lw=2.0, zorder=3)
    axa.set_xlim(0, 1); axa.set_ylim(0, 1); axa.set_aspect("equal", adjustable="box")
    axa.set_xlabel("False-positive rate", fontsize=8); axa.set_ylabel("True-positive rate", fontsize=8)
    axa.tick_params(labelsize=6.5)
    axa.text(0.93, 0.10, f"AUROC = {float(d['sex_auc']):.3f}", ha="right", va="bottom",
             fontsize=7.6, fontweight="bold", color="#176217")

    def scatter(ax, x, y, lo, hi, col, xlab, ylab, title, r, r2, mae, unit):
        # light, white-edged markers so the clouds read as discrete dots without
        # burying the reference line; rasterized so they embed cleanly at 300 dpi in
        # the vector PDF (axes/text stay vector) and no PDF viewer can drop them.
        ax.scatter(x, y, s=4, c=col, alpha=0.30, edgecolors="white", linewidths=0.15,
                   zorder=3, rasterized=True)
        # identity reference drawn ON TOP of the cloud so it stays visible in dense regions
        ax.plot([lo, hi], [lo, hi], color="#333", ls="--", lw=1.0, zorder=4)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(xlab, fontsize=8); ax.set_ylabel(ylab, fontsize=8); ax.tick_params(labelsize=6.5)
        ax.text(0.05, 0.96, f"$r$ = {r:.3f}\n$r^2$ = {r2:.2f}\nMAE = {mae:.2f}{unit}",
                transform=ax.transAxes, va="top", ha="left", fontsize=6.8, color="#333")

    # (b) age
    axb = fig.add_subplot(gs[1]); fig.text(0.357, 0.91, "b", fontsize=11, fontweight="bold", va="top")
    scatter(axb, d["age_true"], d["age_pred"], 44, 78, OK["blue"],
            "Chronological age (yr)", "Predicted age (yr)", "Age",
            float(d["age_r"]), float(d["age_r2"]), float(d["age_mae"]), " yr")
    # (c) BMI
    axc = fig.add_subplot(gs[2]); fig.text(0.665, 0.91, "c", fontsize=11, fontweight="bold", va="top")
    scatter(axc, d["bmi_true"], d["bmi_pred"], 15, 46, OK["vermillion"],
            "Measured BMI (kg m$^{-2}$)", "Predicted BMI (kg m$^{-2}$)", "Body-mass index",
            float(d["bmi_r"]), float(d["bmi_r2"]), float(d["bmi_mae"]), "")

    save(fig, "ed_demographics")


def ed_screening(bundle=None):
    """Supporting analyses for the prevalent-disease screening forest (Figure ref).
    (a) selectivity across the phenome; (b) the Parkinson's ROC and operating point;
    (c) the physiological confound + wear audit (with the dementia honest-exception);
    (d) near-onset robustness; (e) the frozen-embedding ceiling."""
    from matplotlib.patches import Patch
    from sklearn.metrics import roc_curve, roc_auc_score
    B = F.load_screening_bundle(bundle)
    sd, conf, wear, no, embedding = B["sd"], B["confound"], B["wear"], B["nearonset"], B["embedding"]
    PD = "NS_324.11"

    fig = plt.figure(figsize=(F.W2, 7.0))
    gs = fig.add_gridspec(2, 6, hspace=0.46, wspace=1.05, left=0.075, right=0.975,
                          top=0.915, bottom=0.075)
    axa = fig.add_subplot(gs[0, 0:3])
    axb = fig.add_subplot(gs[0, 3:6])
    axc = fig.add_subplot(gs[1, 0:2])
    axd = fig.add_subplot(gs[1, 2:4])
    axe = fig.add_subplot(gs[1, 4:6])
    for xy, lab in [((0.012, 0.94), "a"), ((0.505, 0.94), "b"),
                    ((0.012, 0.45), "c"), ((0.345, 0.45), "d"), ((0.675, 0.45), "e")]:
        fig.text(xy[0], xy[1], lab, fontsize=11, fontweight="bold", va="top", ha="left")

    # (a) selectivity across the phenome
    robust = sd[sd.sd_robust_beats_demo == True]
    nominal = sd[(sd.sd_beats_demo == True) & (sd.sd_robust_beats_demo != True)]
    rest = sd[sd.sd_beats_demo != True]
    axa.plot([0.45, 0.95], [0.45, 0.95], color="#777", ls="--", lw=0.9, zorder=1)
    axa.text(0.915, 0.90, "wrist >\ndemographics", fontsize=5.8, color="#888",
             ha="right", va="top", style="italic")
    axa.scatter(rest.demo_auroc, rest.sd_auroc, s=7, color="#cfcfcf", alpha=0.45,
                edgecolors="none", zorder=2)
    axa.scatter(nominal.demo_auroc, nominal.sd_auroc, s=11, color="#9ec3e0", alpha=0.6,
                edgecolors="none", zorder=3)
    szr = np.clip(16 + 1.7 * np.sqrt(robust.n_test_pos.values), 16, 66)
    axa.scatter(robust.demo_auroc, robust.sd_auroc, s=szr,
                color=[F.cat_color(c) for c in robust.category], alpha=0.92,
                edgecolors="white", linewidths=0.4, zorder=4)
    for ph, nm, tp, star in [(PD, "Parkinson's", (0.575, 0.945), True),
                             ("NS_326.1", "Mult. sclerosis", (0.485, 0.85), False),
                             ("RE_474", "COPD", (0.80, 0.85), False),
                             ("MB_286.2", "Depression", (0.485, 0.70), False)]:
        r = sd[sd.phecode == ph]
        if not len(r):
            continue
        x, yv = float(r.demo_auroc.iloc[0]), float(r.sd_auroc.iloc[0])
        if star:
            axa.scatter([x], [yv], marker="*", s=150, color=OK["blue"],
                        edgecolors="white", linewidths=0.5, zorder=6)
        axa.annotate(nm, xy=(x, yv), xytext=tp, fontsize=6.4, color="#111", ha="left",
                     va="center", bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7),
                     arrowprops=dict(arrowstyle="-", lw=0.6, color="#999", shrinkA=2, shrinkB=3))
    axa.set_xlim(0.46, 0.92); axa.set_ylim(0.46, 0.97)
    axa.set_xlabel("Demographic baseline AUROC", fontsize=8.4)
    axa.set_ylabel("Wrist-only AUROC", fontsize=8.4); axa.tick_params(labelsize=7.5)
    legh = [Line2D([0], [0], marker='o', ls='', mfc=F.cat_color(c), mec='white', mew=0.4,
                   ms=5, label=c) for c in ["Neurological", "Mental", "Respiratory"]]
    legh.append(Line2D([0], [0], marker='o', ls='', mfc="#9ec3e0", mec='none', ms=4,
                       label="other nominal win"))
    axa.legend(handles=legh, loc="lower right", frameon=False, fontsize=5.8,
               handletextpad=0.2, labelspacing=0.25, borderaxespad=0.3)
    axa.set_title("Selective across the phenome", fontsize=9.0, fontweight="bold")

    # (b) Parkinson's ROC and operating point
    oof = B["oof"].get(PD, {})
    axb.plot([0, 1], [0, 1], color="#bbb", ls=":", lw=1.0, zorder=1)
    if "demo" in oof:
        d = oof["demo"]; fpr, tpr, _ = roc_curve(d.y, d.pred)
        axb.plot(fpr, tpr, color=OK["orange"], lw=1.5, alpha=0.9, zorder=2,
                 label=f"Demographics  {roc_auc_score(d.y, d.pred):.2f}")
    if "wrist" in oof:
        w = oof["wrist"]; fpr, tpr, _ = roc_curve(w.y, w.pred)
        axb.plot(fpr, tpr, color=OK["blue"], lw=2.1, zorder=4,
                 label=f"Embedding  {C['SCR_PD_AUROC']:.3f}")
        ix = max(0, min(np.searchsorted(fpr, 0.05, side="right") - 1, len(fpr) - 1))
        axb.scatter([fpr[ix]], [tpr[ix]], s=46, color=OK["blue"], edgecolor="white",
                    linewidth=0.8, zorder=5)
        axb.annotate(f"{C['SCR_PD_SENS95']*100:.0f}% sensitivity\nat 95% specificity",
                     xy=(fpr[ix], tpr[ix]), xytext=(0.30, 0.50), fontsize=6.8, va="center",
                     arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#333", shrinkA=2, shrinkB=3))
    axb.set_xlim(0, 1); axb.set_ylim(0, 1.0); axb.set_aspect("equal", adjustable="box")
    axb.set_xlabel("False-positive rate (1 - specificity)", fontsize=8.0)
    axb.set_ylabel("True-positive rate", fontsize=8.0); axb.tick_params(labelsize=7.5)
    axb.legend(loc="lower right", frameon=False, fontsize=6.8, handlelength=1.3,
               title="Prevalent Parkinson's (n=132)", title_fontsize=6.8)
    axb.set_title("Parkinson's detection, wrist-only", fontsize=9.0, fontweight="bold")

    # (c) confound + wear audit (with dementia honest-exception note)
    cp = conf[conf.phecode == PD].set_index("block")
    bars = [("Embedding", "emb", OK["blue"]), ("Emb. + confounds", "emb_confound", "#7fb1d6"),
            ("Demographics", "demo", OK["orange"]), ("Confounds only", "confound", "#9a9a9a")]
    ypos = [5, 4, 3, 2]
    for (lab, key, col), yv in zip(bars, ypos):
        a, lo, hi = cp.loc[key, ["auroc", "auroc_lo", "auroc_hi"]]
        axc.barh(yv, a, height=0.62, color=col, zorder=3)
        axc.plot([lo, hi], [yv, yv], color="#333", lw=1.0, zorder=4)
        axc.text(min(hi + 0.015, 0.99), yv, f"{a:.3f}", va="center", ha="left", fontsize=6.4, color="#222")
    wp = wear[wear.phecode == PD].set_index("wear_tertile")
    for ws, yv, lab in [(0, 0.7, "low wear"), (2, -0.05, "high wear")]:
        if ws in wp.index:
            a = float(wp.loc[ws, "auroc"])
            axc.barh(yv, a, height=0.5, color=OK["blue"], alpha=0.4, zorder=3)
            axc.text(a + 0.015, yv, f"{a:.3f} ({lab})", va="center", ha="left", fontsize=5.8, color="#444")
    axc.axhline(1.45, color="#ddd", lw=0.8)
    axc.text(0.47, 1.45, "within wear strata", fontsize=5.8, color="#888", va="center", ha="left", style="italic")
    axc.axvline(0.5, color="#888", ls=":", lw=1.0, zorder=2)
    axc.text(0.5, 5.8, "chance", fontsize=5.8, color="#666", ha="center", va="bottom")
    axc.set_yticks(ypos); axc.set_yticklabels([b[0] for b in bars], fontsize=6.8)
    axc.set_xlim(0.45, 1.0); axc.set_ylim(-1.4, 6.3)
    axc.set_xlabel("Parkinson's screening AUROC", fontsize=8.0)
    dem = conf[conf.phecode == "NS_328.1"].set_index("block")["auroc"]
    axc.text(0.45, -1.0, f"max confound SMD {C['SCR_MAXSMD']:.2f}. The audit flags a real wear\n"
             f"confound for dementia ({dem.get('confound', float('nan')):.2f}); dementia is not a wrist win\n"
             f"(demographics {dem.get('demo', float('nan')):.2f} > embedding {dem.get('emb', float('nan')):.2f}) and we do not claim it.",
             fontsize=5.4, color="#555", va="top", ha="left")
    axc.set_title("Physiological, not an artifact", fontsize=8.8, fontweight="bold")

    # (d) near-onset robustness
    cl = no[no.split == "clinical"].copy()
    order = ["recent <=2yr", "mid 2-5yr", "long >5yr", "all"]
    cl = cl.set_index("bin").loc[order].reset_index()
    xs = np.arange(len(cl))
    for i, row in cl.iterrows():
        is_all = row["bin"] == "all"
        col = "#1a4e74" if is_all else OK["blue"]
        axd.plot([xs[i], xs[i]], [row.auroc_lo, row.auroc_hi], color=col, lw=1.8,
                 solid_capstyle="round", zorder=3)
        axd.scatter([xs[i]], [row.auroc], s=50 if is_all else 38, color=col,
                    edgecolors="white", linewidths=0.6, zorder=4)
        axd.text(xs[i], row.auroc_hi + 0.006, f"{row.auroc:.3f}", ha="center", va="bottom",
                 fontsize=6.2, fontweight="bold" if is_all else "normal")
        axd.text(xs[i], 0.792, f"sens\n{row.sens95:.2f}", ha="center", va="bottom", fontsize=5.6, color="#666")
    axd.axhline(0.5, color="#bbb", ls=":", lw=0.9)
    sp = B["nearonset_json"]
    axd.text(0.5, 0.985, f"Spearman = +{sp['spearman_score_vs_dx']:.3f}\np = {sp['spearman_p']:.2f}",
             transform=axd.transAxes, fontsize=6.0, ha="center", va="top", color="#333",
             bbox=dict(boxstyle="round,pad=0.25", fc="#f4f4f4", ec="#ddd", lw=0.6))
    axd.set_xticks(xs)
    axd.set_xticklabels(["recent\n$\\leq$2 y", "mid\n2-5 y", "long\n>5 y", "all"], fontsize=6.6)
    axd.set_xlim(-0.5, len(cl) - 0.5); axd.set_ylim(0.78, 1.0)
    axd.set_xlabel("Years since diagnosis", fontsize=8.0)
    axd.set_ylabel("Parkinson's AUROC", fontsize=8.0); axd.tick_params(labelsize=7.2)
    axd.set_title("A genuine early marker", fontsize=8.8, fontweight="bold")

    # (e) frozen-embedding ceiling
    he = embedding[embedding.pos_set == "held_out"].set_index(["disease", "block"])
    groups = [("NS_324.11", "Park."), ("NS_326.1", "MS")]
    reps = [("L-moment", OK["blue"], "L-moment pooling"),
            ("Embedding_rep", OK["green"], "Deep Embedding rep."),
            ("L-moment+embedding_rep", "#9a9a9a", "Both")]
    w = 0.26
    for gi, (ph, gn) in enumerate(groups):
        for ri, (blk, col, _) in enumerate(reps):
            if (ph, blk) not in he.index:
                continue
            a, lo, hi = he.loc[(ph, blk), ["auroc", "lo", "hi"]]
            x = gi + (ri - 1) * w
            axe.bar(x, a, w * 0.92, color=col, zorder=3)
            axe.plot([x, x], [lo, hi], color="#333", lw=0.9, zorder=4)
    axe.axhline(0.5, color="#bbb", ls=":", lw=0.9)
    axe.text(0.5, 0.452, "(Embedding's disease-trained\nhazard 0.949 is leakage)", transform=axe.transAxes,
             fontsize=5.4, color="#a33", ha="center", va="bottom", style="italic")
    axe.set_xticks(range(len(groups))); axe.set_xticklabels([g[1] for g in groups], fontsize=7.4)
    axe.set_ylim(0.45, 1.02); axe.set_ylabel("Held-out AUROC", fontsize=8.0); axe.tick_params(labelsize=7.2)
    axe.legend(handles=[Patch(color=c, label=l) for _, c, l in reps], loc="lower center",
               bbox_to_anchor=(0.5, -0.32), frameon=False, fontsize=5.8, handlelength=1.1, handleheight=0.9, ncol=1)
    axe.set_title("The frozen mean-pool\nis the ceiling", fontsize=8.4, fontweight="bold")

    fig.suptitle("Prevalent-disease screening: selectivity, Parkinson's detection, and validation",
                 fontsize=10.0, fontweight="bold", y=0.975)
    save(fig, "ed_screening")


# main-panel diseases, data-driven from the source-attribution sweep (PD pinned); each a clear,
# well-powered exemplar of an archetype (daytime/nighttime, movement vs sleep-and-breathing model,
# genetics top-up vs wrist-only). Only COPD and sleep apnea lack a matched PRS; every other row
# carries genetics. Easy to swap.
# Previous 8-disease selection for the main-text source-attribution figure (Figure 4), kept so that
# figure can be restored verbatim: set DEFAULT_SA_DISEASES = SA_DISEASES_8_LEGACY (and revert the
# row_h/vmax in build_fig2_source_attribution plus the supp break_frac to 0.05).
SA_DISEASES_8_LEGACY = [
    ("Parkinson's disease", "NS_324.11"),   # movement (HA), night-heavy; genetics ~0
    ("Ischemic stroke", "CV_431.11"),       # movement (HA), day and night; genetics ~0
    ("Asthma", "RE_475"),                   # both models, day and night; modest genetics
    ("Type 2 diabetes", "EM_202.2"),        # daytime movement + genetics + BMI
    ("Coronary heart disease", "CV_404.2"), # genetics-led, broad small wrist signal
    ("Atrial fibrillation", "CV_416.2"),    # genetics + nocturnal sleep-and-breathing
    ("COPD", "RE_474"),                     # movement + nocturnal sleep-and-breathing; no PRS
    ("Sleep apnea", "NS_333.1"),            # sleep-and-breathing model + BMI; no PRS
]

# Current main-text source-attribution figure (Figure 4) selection: the ten focus diseases (same set
# and phecodes as Extended Data Fig. ed_confusion_targets). The builder re-groups rows by organ system, so the input order
# here is not the final row order. Five carry a matched PRS (PD, Alzheimer, type 2 diabetes,
# pulmonary embolism, persistent AF); the other five show 'n/a' in the genetics column.
DEFAULT_SA_DISEASES = [
    ("Parkinson's disease", "NS_324.11"),
    ("Alzheimer's disease", "NS_328.11"),
    ("Heart failure", "CV_424"),
    ("Type 2 diabetes", "EM_202.2"),
    ("Renal failure", "GU_582"),
    ("All-cause mortality", "time_to_death"),
    ("Pulmonary embolism", "CV_440.3"),
    ("Sleep apnea", "NS_333.1"),
    ("Persistent AF", "CV_416.212"),
    ("COPD", "RE_474"),
]

# Source-attribution-figure-only (Figure 4) display override: file sleep apnea under Respiratory in the source-attribution
# heatmap, next to COPD. Both are breathing disorders carried by the sleep-and-breathing model
# with no matched PRS. This is local to the source-attribution heatmaps; the phecode taxonomy
# (NS_333.1 = Neurological), the per-disease longtable, the frozen disease list, and every other
# figure are left unchanged.
SA_CATEGORY_OVERRIDE = {"NS_333.1": "Respiratory"}

# white -> deep red sequential (white = no contribution; the key affordance for "adds nothing").
SA_CMAP_STOPS = ["#ffffff", "#fff0ea", "#fdcab5", "#fc8d6e", "#e6452f", "#a50f15"]
SA_CMAP = LinearSegmentedColormap.from_list("wr", SA_CMAP_STOPS)
SA_CMAP.set_bad("#f4f4f4")


def sa_extended_cmap(break_frac):
    """White->red ramp identical to SA_CMAP over [0, break_frac], then continuing into deeper
    reds to 1.0. With break_frac = (Figure-4 vmax) / (this figure's vmax), every value at or below
    the Figure-4 maximum renders in the exact same colour as Figure 4; only larger values (unique
    to this fuller panel) extend into the darker reds. Keeps one monotonic sequential scale."""
    n = len(SA_CMAP_STOPS)
    base = [(i / (n - 1) * break_frac, c) for i, c in enumerate(SA_CMAP_STOPS)]
    ext = [(break_frac + (1 - break_frac) * 0.55, "#6f0712"), (1.0, "#3f040c")]
    cmap = LinearSegmentedColormap.from_list("wr_ext", base + ext)
    cmap.set_bad("#f4f4f4")
    return cmap
# sub-labels echo the two AcceleRest heads named in the mechanistic interpretability figure
# (HA = movement / Activity; AcceleRest = sleep-and-breathing), each crossed with daytime vs nighttime.
SA_SUBLABELS = ["Activity\n(HA)", "Sleep /\nbreathing\n(AcceleRest)",
                "Activity\n(HA)", "Sleep /\nbreathing\n(AcceleRest)",
                "PRS", "Age", "Sex", "BMI"]
SA_GROUPS = [("Daytime wrist", 0, 1), ("Nighttime wrist", 2, 3),
             ("Genetics", 4, 4), ("Demographics", 5, 7)]


def fig_source_attribution(diseases=None, name="fig_source_attribution", vmax=0.05,
                           row_h=0.52, lab_fs=7.0, fig_w=None, annotate=True,
                           sig_style="asterisk", pdf=True, cmap=None, cbar_ticks=None,
                           center_spacer=0.18, header_ratio=0.06, pad_in=1.55,
                           cbar_label_fs=7.5, col_gap=0.5, row_gap=0.6,
                           group_rows=True, cat_label_fs=7.0, top_margin=0.0, sig_note=True):
    """Per-disease source-attribution heatmap (white->red = per-source contribution to concordance).

    Wrist columns show within-wrist importance (drop in C when that daytime/nighttime x model
    component is removed); genetics and demographics show the increment in C over the full wrist
    model. When annotate is True each cell prints its delta-C, with an asterisk where the 95% CI
    excludes zero; cells with no matched PRS read 'n/a'. With annotate False (the dense ED
    companion) cells are colour-only with significance dots.

    Rows are grouped by organ system (group_rows) with a gap and a category label between blocks,
    and within each block ordered by descending total wrist contribution; the four source-column
    groups are separated by an airy gap (col_gap) with a dark cut line in each gap. The colour bar
    always spans [0, vmax] with vmax >= the largest printed cell, so the top tick never contradicts
    a printed value. diseases: list of (display_label, phecode)."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    if diseases is None:
        diseases = DEFAULT_SA_DISEASES
    sa = F.load_source_attribution()
    srcs = F.SOURCE_ORDER
    rec = {(r.phecode, r.source): r for r in sa.itertuples(index=False)}
    pm = F.phecode_map()

    # organ-system category per disease + canonical (paper-wide) ordering
    CANON = list(F.CAT_COLORS.keys())
    def catname(ph):
        if ph == "time_to_death":
            return "Mortality"
        if ph in SA_CATEGORY_OVERRIDE:
            return F.cat_label(SA_CATEGORY_OVERRIDE[ph])
        c = pm.loc[ph, "PhecodeCategory"] if ph in pm.index else None
        return F.cat_label(c) if isinstance(c, str) else "Other"
    def cat_idx(nm):
        return CANON.index(nm) if nm in CANON else len(CANON)
    _wcells = ("day_activity", "day_sleepbreath", "night_activity", "night_sleepbreath")
    def wtot(ph):                                          # total (clipped) wrist contribution
        s = 0.0
        for c in _wcells:
            r = rec.get((ph, c))
            if r is not None and bool(r.available) and np.isfinite(r.dC):
                s += max(float(r.dC), 0.0)
        return s
    if group_rows:
        diseases = sorted(diseases, key=lambda d: (cat_idx(catname(d[1])), -wtot(d[1])))

    n, k = len(diseases), len(srcs)
    labels = [d[0] for d in diseases]; phs = [d[1] for d in diseases]
    cats = [catname(p) for p in phs]
    M = np.full((n, k), np.nan); Sig = np.zeros((n, k), bool); A = np.zeros((n, k), bool)
    for i, ph in enumerate(phs):
        for j, s in enumerate(srcs):
            r = rec.get((ph, s))
            if r is None:
                continue
            A[i, j] = bool(r.available); Sig[i, j] = bool(r.significant)
            if A[i, j] and np.isfinite(r.dC):
                M[i, j] = float(r.dC)
    if vmax is None:
        pos = M[np.isfinite(M) & (M > 0)]
        vmax = min(0.08, max(0.03, round(float(np.percentile(pos, 92)), 3))) if pos.size else 0.05
    Mc = np.clip(M, 0, vmax); Mc[~A] = np.nan
    cmap = SA_CMAP if cmap is None else cmap
    norm = Normalize(vmin=0.0, vmax=vmax)
    if cbar_ticks is None:                                # default: 6 even ticks up to vmax
        cbar_ticks = [round(vmax / 5.0 * t, 4) for t in range(6)]

    def catcol(ph):
        if ph == "time_to_death":
            return F.cat_color("Mortality")
        if ph in SA_CATEGORY_OVERRIDE:
            return F.cat_color(SA_CATEGORY_OVERRIDE[ph])
        cat = pm.loc[ph, "PhecodeCategory"] if ph in pm.index else None
        return F.cat_color(cat) if isinstance(cat, str) else "#bbbbbb"
    chips = [catcol(p) for p in phs]

    # gap-aware column centres: an airy gap between each of the four source groups
    BOUND = [b for _, _, b in SA_GROUPS[:-1]]             # last column index of each non-final group
    def xcen(j):
        return j + col_gap * sum(1 for b in BOUND if j > b)
    xs = [xcen(j) for j in range(k)]

    # gap-aware row centres: a gap (for the category label) before each new organ-system block
    ys = []; y = top_margin; prev = None
    for i in range(n):
        if prev is not None and cats[i] != prev:
            y += row_gap
        ys.append(y); prev = cats[i]; y += 1.0
    y_bot = ys[-1] + 0.5
    span = y_bot - (-0.5)

    fig = plt.figure(figsize=(fig_w or F.W2, row_h * span + pad_in), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, width_ratios=[0.42, 1.0, 0.028, center_spacer],
                          height_ratios=[header_ratio, 1.0], wspace=0.015, hspace=0.02)
    axh = fig.add_subplot(gs[0, 1]); axh.axis("off")     # two-level group header
    axc = fig.add_subplot(gs[1, 0]); axc.axis("off")     # disease chips + names + category labels
    ax = fig.add_subplot(gs[1, 1])                        # heatmap
    axcb = fig.add_subplot(gs[1, 2])                      # vertical colourbar
    fig.add_subplot(gs[1, 3]).axis("off")                # right spacer balances the name column
    fig.add_subplot(gs[0, 3]).axis("off")                # so the heatmap is centred on the page

    def _txtcol(v):                                       # black on light/mid cells, white on deep
        r, g, b = cmap(norm(min(max(v, 0.0), vmax)))[:3]
        return "#1a1a1a" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.52 else "white"
    for i in range(n):
        for j in range(k):
            xc, yc = xs[j], ys[i]
            if not A[i, j]:                               # no matched PRS -> light grey 'n/a'
                ax.add_patch(Rectangle((xc - 0.5, yc - 0.5), 1, 1, facecolor="#ececef",
                                       edgecolor="#a9a9ad", lw=1.0, zorder=1.5))
                if annotate:
                    ax.text(xc, yc, "n/a", ha="center", va="center", fontsize=5.8,
                            style="italic", color="#9b9ba2", zorder=4)
                continue
            fc = cmap(norm(Mc[i, j])) if np.isfinite(Mc[i, j]) else "#f4f4f4"
            ax.add_patch(Rectangle((xc - 0.5, yc - 0.5), 1, 1, facecolor=fc,   # grey box keeps
                                   edgecolor="#a9a9ad", lw=1.0, zorder=1.5))    # even white cells
            if annotate and np.isfinite(M[i, j]):
                v, sig = M[i, j], bool(Sig[i, j])
                txt = f"{v:+.3f}" if abs(v) >= 5e-4 else "0.000"
                base = _txtcol(v)
                if sig_style == "asterisk":               # plain number, * marks significance
                    ax.text(xc, yc, txt + ("*" if sig else ""), ha="center", va="center",
                            fontsize=6.2, color=base, zorder=4)
                elif sig_style == "outline":              # significant cells get a dark box
                    ax.text(xc, yc, txt, ha="center", va="center", fontsize=6.2, color=base, zorder=4)
                    if sig:
                        ax.add_patch(Rectangle((xc - 0.5, yc - 0.5), 1, 1, fill=False,
                                               edgecolor="#1a1a1a", lw=1.5, zorder=5.5))
                elif sig_style == "bold":                 # original: bold = significant
                    ax.text(xc, yc, txt, ha="center", va="center", fontsize=6.2,
                            fontweight=("bold" if sig else "normal"), color=base, zorder=4)
                else:                                     # "fade": non-significant numbers recede
                    ax.text(xc, yc, txt, ha="center", va="center", fontsize=6.2, color=base,
                            alpha=(1.0 if sig else 0.42), zorder=4)
    if not annotate:                                      # ED companion: keep significance dots
        for i in range(n):
            for j in range(k):
                if Sig[i, j] and A[i, j]:
                    ax.scatter([xs[j]], [ys[i]], s=6.5, marker="o", facecolor="#1a1a1a",
                               edgecolor="white", lw=0.35, zorder=6)

    for b in BOUND:                                        # heavier dark divider centred in each gap
        xd = (xcen(b) + xcen(b + 1)) / 2.0
        ax.plot([xd, xd], [ys[0] - 0.5, y_bot], color="#33333a", lw=1.2, zorder=5)

    ax.set_xticks(xs); ax.set_xticklabels(SA_SUBLABELS, fontsize=6.0)
    ax.xaxis.set_ticks_position("bottom")
    ax.tick_params(axis="x", length=4, width=1.0, color="#222", pad=6)  # leader line to each label
    ax.set_yticks([])
    for sp in ax.spines.values():                         # spines off; frame drawn tight below so
        sp.set_visible(False)                             # it hugs the cells, not the label margin
    ax.set_xlim(-0.5, xs[-1] + 0.5); ax.set_ylim(y_bot, -0.5)
    ax.add_patch(Rectangle((-0.5, ys[0] - 0.5),           # black frame hugging the cell block: the
                           (xs[-1] + 0.5) - (-0.5),        # top-margin whitespace holding the first
                           y_bot - (ys[0] - 0.5),          # category label is left un-boxed
                           fill=False, edgecolor="#222", lw=1.0, zorder=6, clip_on=False))

    # top: two-level group header (a short rule under each name spans its mapped columns), tucked
    # down close to the top row
    axh.set_xlim(-0.5, xs[-1] + 0.5); axh.set_ylim(0, 1)
    for gname, a, b in SA_GROUPS:
        axh.plot([xcen(a) - 0.45, xcen(b) + 0.45], [0.02, 0.02], color="#33333a", lw=1.0,
                 clip_on=False)
        axh.text((xcen(a) + xcen(b)) / 2, 0.12, gname, ha="center", va="bottom", fontsize=7.0,
                 fontweight="bold")

    # left: organ-system chip + disease name (per row); category bracket in its own far-left lane
    # with the label to the RIGHT of the bracket, so neither the label nor a long disease name
    # collides with the coloured bracket.
    axc.set_ylim(y_bot, -0.5); axc.set_xlim(0, 1)
    for i in range(n):
        axc.add_patch(Rectangle((0.95, ys[i] - 0.30), 0.04, 0.60, color=chips[i], lw=0,
                                clip_on=False))
        axc.text(0.90, ys[i], labels[i], ha="right", va="center", fontsize=lab_fs)
    i0 = 0                                                 # walk category blocks for label + bracket
    for i in range(1, n + 1):
        if i == n or cats[i] != cats[i0]:
            top, bot = ys[i0] - 0.5, ys[i - 1] + 0.5
            axc.plot([0.02, 0.02], [top + 0.05, bot - 0.05], color=catcol(phs[i0]), lw=2.4,
                     clip_on=False, zorder=3, solid_capstyle="round")
            axc.text(0.09, ys[i0] - 0.58, cats[i0], ha="left", va="bottom", fontsize=cat_label_fs,
                     fontweight="bold", color="#222", clip_on=False)
            i0 = i

    sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, cax=axcb)
    cb.set_ticks(cbar_ticks)
    cb.set_label("Per-source contribution to concordance (ΔC)", fontsize=cbar_label_fs, labelpad=4)
    cb.ax.tick_params(labelsize=5.8, length=2)
    cb.outline.set_linewidth(1.0)
    # per-source significance criterion. For the main-text figure this note lives in the LaTeX
    # legend instead (sig_note=False); the dense ED companion keeps it beneath the panel because
    # supplements are read standalone.
    if sig_note:
        fig.supxlabel(
            "*, significant contribution; the criterion differs by source: wrist components, "
            "subject-level 95% CI excludes zero (Benjamini-Hochberg FDR);\n"
            "polygenic score, 95% CI excludes zero; age, sex and body-mass index, above the "
            "95th-percentile covariate-permutation null.",
            fontsize=5.3, ha="center", va="bottom", linespacing=1.5)
    save(fig, name, pdf=pdf)


# short, paper-consistent display labels for the supplementary PRS-complete panel
SA_SUPP_LABELS = {
    "CV_431.11": "Ischemic stroke", "CV_401.1": "Hypertension",
    "CV_416.2": "Atrial fibrillation", "CV_416.211": "Paroxysmal AF",
    "CV_416.212": "Persistent AF", "CV_404.2": "Coronary heart disease",
    "CV_404.1": "Myocardial infarction", "CV_404.11": "Acute MI",
    "CV_404": "Ischemic heart disease", "CV_402": "Elevated blood pressure",
    "CV_440.3": "Pulmonary embolism", "CA_103": "Skin cancer (non-melanoma)",
    "NS_328.1": "Dementia", "NS_328.11": "Alzheimer's disease",
    "NS_324.11": "Parkinson's disease", "SO_375.1": "Glaucoma",
    "SO_375.11": "Open-angle glaucoma", "SO_374.5": "Macular degeneration",
    "GI_525.1": "Celiac disease", "GI_522.11": "Crohn's disease",
    "GI_522.12": "Ulcerative colitis", "EM_202.1": "Type 1 diabetes",
}


def fig_source_attribution_supp(name="fig_source_attribution_supp", vmax=0.30, pdf=True):
    """Extended-Data companion: the source-attribution decomposition for every outcome carrying a
    matched PRS (all 26; demographics also computed, so every cell is available -- no 'n/a'). Same
    layout and metric as the main-text source-attribution figure, Figure 4 (delta-C printed in each cell, * = 95% CI excludes
    zero). The colour scale is WIDENED (vmax) relative to Figure 4 to span the larger,
    genetics-driven contributions that appear once the full PRS panel is shown; vmax is set above
    the largest cell so the top tick never contradicts a printed value."""
    import re
    sa = F.load_source_attribution(); pm = F.phecode_map()
    demo_ok = set(sa[(sa.source == "age") & sa.available].phecode)        # demographics computed
    prs = set(sa[(sa.source == "genetics") & sa.available].phecode)       # matched PRS available
    keep = sorted(prs & demo_ok)                                          # all 26 PRS diseases
    val = sa.pivot_table(index="phecode", columns="source", values="dC", observed=True)
    cells = ["day_activity", "day_sleepbreath", "night_activity", "night_sleepbreath"]

    def wrist_tot(ph):
        return float(val.loc[ph, cells].clip(lower=0).sum()) if ph in val.index else 0.0

    def lab(ph):
        if ph in SA_SUPP_LABELS:
            return SA_SUPP_LABELS[ph]
        s = pm.loc[ph, "PhecodeString"] if ph in pm.index else ph
        return re.sub(r"\s*[\[(][^\])]*[\])]", "", str(s)).rstrip("*").strip()

    keep.sort(key=lambda p: -wrist_tot(p))                # group_rows re-sorts by category below
    dis = [(lab(p), p) for p in keep]
    # colour scale matches Figure 4 exactly up to its maximum (0.07), then extends into darker reds
    # so the diseases shared with Figure 4 render in the same colour, not washed out.
    # sig_note=False: the per-source significance criterion moves to the LaTeX caption (per review),
    # matching how main-text Figure 4 handles the same note.
    fig_source_attribution(dis, name=name, row_h=0.235, lab_fs=6.0, annotate=True,
                           vmax=vmax, pdf=pdf, cmap=sa_extended_cmap(0.07 / vmax),
                           cbar_ticks=[0.0, 0.07, 0.15, vmax],
                           header_ratio=0.04, pad_in=0.75, cbar_label_fs=10.0,
                           row_gap=0.33, cat_label_fs=5.6, top_margin=0.0, sig_note=False)


def build_fig2_source_attribution():
    """Main-text source-attribution figure, Figure 4 (the ten focus diseases in DEFAULT_SA_DISEASES). Fitted to roughly the same
    page height as the previous 8-row version by lowering row_h, and with vmax raised to 0.07 so the
    largest printed cell (persistent AF nocturnal sleep/breathing, +0.062) stays within the colour bar.
    The ED companion fig_source_attribution_supp matches this vmax via sa_extended_cmap(0.07/vmax)."""
    fig_source_attribution(name="fig_source_attribution", vmax=0.07,
                           cbar_ticks=[0.0, 0.02, 0.04, 0.06, 0.07], row_h=0.39, sig_note=False)


def ed_against_chance(name="ed_against_chance"):
    """Per-disease incident-prediction behaviour for the >=10-incident-case panel (the rich
    Extended Data table): (a) AUPRC vs the chance floor, (b) top-5% positive-predictive-value
    enrichment, (c) F1 at the top-5% budget vs its random baseline, (d) wrist vs demographic
    concordance, (e) incident concordance vs prevalent-disease screening AUROC. Coloured by
    organ system; Parkinson's disease, Alzheimer's disease and dementia labelled."""
    import pandas as pd
    df = pd.read_csv(F.UKB / "paper_additions" / "rich_per_disease" / "rich_per_disease.csv")
    FLAG = {"NS_324.11": "PD", "NS_328.11": "AD", "NS_328.1": "Dem.", "time_to_death": "Mort."}
    def col(r): return F.cat_color(F.cat_label(str(r["category"])))
    cols = df.apply(col, axis=1)

    def letter(ax, s):
        ax.text(-0.18, 1.06, s, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")

    def annotate(ax, x, y, dx=6, dy=3):
        for _, r in df[df.phecode.isin(FLAG)].iterrows():
            if pd.notna(r[x]) and pd.notna(r[y]):
                ax.annotate(FLAG[r.phecode], (r[x], r[y]), textcoords="offset points",
                            xytext=(dx, dy), fontsize=6.0, fontweight="bold", color="#111", zorder=6)

    fig = plt.figure(figsize=(F.W2, 4.7))
    gs = fig.add_gridspec(2, 3, wspace=0.42, hspace=0.42,
                          left=0.07, right=0.985, top=0.94, bottom=0.09)

    # (a) AUPRC vs chance (log-log; diagonal = no-skill floor = incidence)
    axa = fig.add_subplot(gs[0, 0]); letter(axa, "a")
    axa.scatter(df.prevalence, df.auprc, s=9, c=cols, alpha=0.65, edgecolors="white",
                linewidths=0.2, zorder=3, rasterized=True)
    lim = [min(df.prevalence.min(), df.auprc.min()) * 0.7, 1.0]
    axa.plot(lim, lim, color="#333", ls="--", lw=1.0, zorder=4)
    axa.set_xscale("log"); axa.set_yscale("log"); axa.set_xlim(lim); axa.set_ylim(lim)
    axa.set_xlabel("Incidence (chance AUPRC)", fontsize=8)
    axa.set_ylabel("AUPRC", fontsize=8); axa.tick_params(labelsize=6.5)
    annotate(axa, "prevalence", "auprc")

    # (b) PPV@top-5% enrichment over the base rate (log y; chance = 1)
    axb = fig.add_subplot(gs[0, 1]); letter(axb, "b")
    b = df[df.ppv5_fold > 0]
    axb.scatter(b.prevalence, b.ppv5_fold, s=9, c=b.apply(col, axis=1), alpha=0.65,
                edgecolors="white", linewidths=0.2, zorder=3, rasterized=True)
    axb.axhline(1.0, color="#333", ls="--", lw=1.0, zorder=4)
    axb.set_xscale("log"); axb.set_yscale("log")
    axb.set_xlabel("Incidence", fontsize=8)
    axb.set_ylabel("PPV enrichment (top-5% / chance)", fontsize=8); axb.tick_params(labelsize=6.5)
    annotate(axb, "prevalence", "ppv5_fold")

    # (c) F1 at top-5% vs the random-ranking F1 curve (chance)
    axc = fig.add_subplot(gs[0, 2]); letter(axc, "c")
    axc.scatter(df.prevalence, df.f1_5, s=9, c=cols, alpha=0.65, edgecolors="white",
                linewidths=0.2, zorder=3, rasterized=True)
    pgrid = np.logspace(np.log10(df.prevalence.min()), np.log10(df.prevalence.max()), 100)
    f1_rand = 2 * pgrid * 0.05 / (pgrid + 0.05)          # random top-5%: PPV=incidence, recall=0.05
    axc.plot(pgrid, f1_rand, color="#333", ls="--", lw=1.0, zorder=4)
    axc.set_xscale("log")
    axc.set_xlabel("Incidence", fontsize=8)
    axc.set_ylabel("F1 at top-5%", fontsize=8); axc.tick_params(labelsize=6.5)
    annotate(axc, "prevalence", "f1_5")

    # (d) wrist vs demographic concordance (diagonal = equality)
    axd = fig.add_subplot(gs[1, 0]); letter(axd, "d")
    axd.scatter(df.c_demo, df.c_model, s=9, c=cols, alpha=0.65, edgecolors="white",
                linewidths=0.2, zorder=3, rasterized=True)
    dlim = [0.45, 0.97]
    axd.plot(dlim, dlim, color="#333", ls="--", lw=1.0, zorder=4)
    axd.set_xlim(dlim); axd.set_ylim(dlim); axd.set_aspect("equal", adjustable="box")
    axd.set_xlabel("Demographics $C$ (age, sex, BMI)", fontsize=8)
    axd.set_ylabel("Wrist embedding $C$", fontsize=8); axd.tick_params(labelsize=6.5)
    annotate(axd, "c_demo", "c_model")

    # (e) incident concordance vs prevalent-disease screening AUROC
    axe = fig.add_subplot(gs[1, 1]); letter(axe, "e")
    e = df[df.prev_auroc.notna()]
    axe.scatter(e.prev_auroc, e.c_model, s=9, c=e.apply(col, axis=1), alpha=0.65,
                edgecolors="white", linewidths=0.2, zorder=3, rasterized=True)
    axe.axvline(0.5, color="#bbb", ls=":", lw=0.8, zorder=2)
    axe.set_xlim(0.45, 0.97); axe.set_ylim(0.55, 0.97)
    axe.set_xlabel("Prevalent screening AUROC", fontsize=8)
    axe.set_ylabel("Incident $C$", fontsize=8); axe.tick_params(labelsize=6.5)
    annotate(axe, "prev_auroc", "c_model")

    # (f) category legend
    axl = fig.add_subplot(gs[1, 2]); axl.axis("off")
    present = [c for c in F.CAT_COLORS if c in set(df.category.map(F.cat_label))]
    # de-duplicate alias keys mapping to the same display label
    seen, handles = set(), []
    for c in present:
        lbl = F.cat_label(c)
        if lbl in seen:
            continue
        seen.add(lbl)
        handles.append(Line2D([0], [0], marker="o", ls="", markersize=5,
                              markerfacecolor=F.cat_color(c), markeredgecolor="white",
                              markeredgewidth=0.2, label=lbl))
    axl.legend(handles=handles, loc="center", fontsize=6.2, frameon=False,
               ncol=1, handletextpad=0.4, labelspacing=0.5)

    save(fig, name)
    # source data
    scols = ["phecode", "name", "category", "n_event", "prevalence", "auprc", "auprc_fold",
             "ppv5", "ppv5_fold", "f1_5", "c_model", "c_demo", "dc", "prev_auroc"]
    df[scols].to_csv(F.DRAFT_FIGS.parent / "ed_against_chance_source_data.csv", index=False)


def _conc_dir():
    return F.UKB / "paper_additions" / "concentration_figs"


def ed_concentration(name="ed_concentration"):
    """Where the risk concentrates: the diseases with the highest positive predictive value
    in the top-5% highest-risk group, and the highest AUPRC, at the six-year horizon. Bars are
    coloured by organ system; the tick is each outcome's incidence (the chance level), and the
    bracketed multiplier is enrichment over that chance level. 95% bootstrap intervals."""
    import pandas as pd
    ppv = pd.read_csv(_conc_dir() / "concentration_ppv.csv").sort_values("ppv").reset_index(drop=True)
    au = pd.read_csv(_conc_dir() / "concentration_auprc.csv").sort_values("auprc").reset_index(drop=True)

    fig = plt.figure(figsize=(F.W2, 7.2))
    gs = fig.add_gridspec(2, 1, hspace=0.14, left=0.30, right=0.90, top=0.99, bottom=0.055)

    def barpanel(ax, df, val, lab_fn, xlabel):
        y = np.arange(len(df))
        colors = [F.cat_color(F.cat_label(str(c))) for c in df.category]
        ax.barh(y, df[val], xerr=[df[val] - df[f"{val}_lo"], df[f"{val}_hi"] - df[val]],
                color=colors, edgecolor="black", linewidth=0.35, capsize=1.5, height=0.8,
                error_kw={"linewidth": 0.6})
        ax.scatter(df.prevalence, y, marker="|", s=42, color="#111", linewidth=1.1, zorder=6)
        ax.set_yticks(y); ax.set_yticklabels([str(n).split("[")[0].strip().rstrip("*").strip()[:36] for n in df.name],
                                             fontsize=5.4)
        pad = df[f"{val}_hi"].max() * 0.015
        for yi, (_, r) in zip(y, df.iterrows()):
            ax.text(r[f"{val}_hi"] + pad, yi, lab_fn(r), va="center", ha="left",
                    fontsize=5.1, color="#222")
        ax.set_xlim(0, df[f"{val}_hi"].max() * 1.42); ax.set_ylim(-0.6, len(df) - 0.4)
        ax.set_xlabel(xlabel, fontsize=7.2); ax.tick_params(labelsize=6.0)
        ax.spines[["top", "right"]].set_visible(False)

    axa = fig.add_subplot(gs[0]); fig.text(0.008, 0.983, "a", fontsize=11, fontweight="bold", va="top")
    barpanel(axa, ppv, "ppv", lambda r: f"{r.ppv*100:.0f}%  ({r.enrichment:.1f}x)",
             "Fraction of the top-5% highest-risk who develop the disease (6-year horizon)")
    axb = fig.add_subplot(gs[1]); fig.text(0.008, 0.505, "b", fontsize=11, fontweight="bold", va="top")
    barpanel(axb, au, "auprc", lambda r: f"{r.auprc:.3f}  ({r.lift:.1f}x)",
             "Area under the precision-recall curve (6-year horizon)")

    cats = sorted(set(F.cat_label(str(c)) for c in pd.concat([ppv, au]).category))
    handles = [Line2D([0], [0], marker="s", ls="", markersize=5, markerfacecolor=F.cat_color(c),
                      markeredgecolor="black", markeredgewidth=0.3, label=c) for c in cats]
    handles.append(Line2D([0], [0], marker="|", ls="", color="#111", markersize=7,
                          markeredgewidth=1.3, label="Incidence (chance)"))
    axb.legend(handles=handles, fontsize=5.4, loc="lower right", frameon=True, ncol=1,
               handletextpad=0.4, labelspacing=0.32)
    save(fig, name)


def ed_horizon(name="ed_horizon"):
    """Mean AUPRC and mean top-5% PPV across diseases as a function of the detection horizon;
    both stay above the incidence (chance) baseline and the gap widens as events accumulate."""
    import pandas as pd
    hz = pd.read_csv(_conc_dir() / "horizon_sweep.csv")
    fig = plt.figure(figsize=(F.W2, 2.9))
    gs = fig.add_gridspec(1, 2, wspace=0.26, left=0.075, right=0.985, top=0.9, bottom=0.17)

    def panel(ax, col, ylab, lettx):
        ax.plot(hz.horizon, hz[col], "-o", color=F.C_EMBEDDING, lw=2.0, ms=5, label="Wrist embedding")
        ax.plot(hz.horizon, hz.mean_prevalence, "--", color="#777", lw=1.4, label="Incidence (chance)")
        for x, yv in zip(hz.horizon, hz[col]):
            ax.annotate(f"{yv:.3f}", (x, yv), textcoords="offset points", xytext=(0, 5),
                        ha="center", fontsize=5.2)
        ax.set_xlabel("Detection horizon (years from recording)", fontsize=7.6)
        ax.set_ylabel(ylab, fontsize=7.6); ax.set_xticks(hz.horizon); ax.tick_params(labelsize=6.3)
        ax.set_ylim(0, max(hz[col].max(), hz.mean_prevalence.max()) * 1.28)
        ax.grid(alpha=0.25); ax.legend(fontsize=6.3, loc="upper left")

    fig.text(0.01, 0.95, "a", fontsize=11, fontweight="bold", va="top")
    panel(fig.add_subplot(gs[0]), "mean_auprc", "Mean AUPRC across diseases", "a")
    fig.text(0.52, 0.95, "b", fontsize=11, fontweight="bold", va="top")
    panel(fig.add_subplot(gs[1]), "mean_ppv_top5", "Mean PPV at top-5%", "b")
    save(fig, name)


def ed_six_disease(name="ed_six_disease"):
    """Precision, recall and F1 at the top-5% operating point across the detection horizon for
    six headline diseases (identified in the caption)."""
    import pandas as pd
    df = pd.read_csv(_conc_dir() / "six_disease_horizon.csv")
    order = ["All-cause mortality", "Parkinson's disease", "Heart failure",
             "Essential hypertension", "Type 2 diabetes", "Atrial fibrillation"]
    fig = plt.figure(figsize=(F.W2, 4.8))
    gs = fig.add_gridspec(2, 3, wspace=0.30, hspace=0.52, left=0.065, right=0.985, top=0.91, bottom=0.10)
    letters = "abcdef"
    for i, dis in enumerate(order):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        d = df[df.label == dis].sort_values("horizon")
        ax.plot(d.horizon, d.precision, "-o", color=F.OK["blue"], lw=1.4, ms=3, label="Precision")
        ax.plot(d.horizon, d.recall, "-s", color=F.OK["vermillion"], lw=1.4, ms=3, label="Recall")
        ax.plot(d.horizon, d.f1, "-^", color=F.OK["green"], lw=1.4, ms=3, label="F1")
        ax.set_xticks(range(1, 9)); ax.set_ylim(0, 1.08); ax.tick_params(labelsize=5.8)
        ax.text(0.5, 1.04, f"({letters[i]}) {dis}", transform=ax.transAxes, ha="center",
                va="bottom", fontsize=6.4, fontweight="bold")
        ax.text(0.96, 0.96, f"n={int(d.n_pos.max())}", transform=ax.transAxes, ha="right",
                va="top", fontsize=5.4, color="#666")
        if i // 3 == 1:
            ax.set_xlabel("Horizon (years)", fontsize=7)
        if i % 3 == 0:
            ax.set_ylabel("Metric at top-5%", fontsize=7)
        if i == 0:
            ax.legend(fontsize=5.2, loc="upper left", frameon=False, handlelength=1.3,
                      labelspacing=0.25)
        ax.grid(alpha=0.22)
    save(fig, name)


def ed_pr(name="ed_pr"):
    """Per-disease precision versus recall at the top-5% operating point (six-year horizon).
    Each point is one outcome; bubble area scales with incident cases, colour is organ system,
    dashed contours are lines of equal F1, and the ten highest-F1 outcomes are labelled."""
    import pandas as pd
    df = pd.read_csv(_conc_dir() / "pr_at_horizon.csv")
    fig = plt.figure(figsize=(F.W2, 4.6))
    ax = fig.add_axes([0.075, 0.115, 0.545, 0.85])

    rmax = min(1.0, df.recall.max() * 1.15)
    pmax = min(1.0, df.precision.max() * 1.28)
    for f1v in [0.05, 0.1, 0.2, 0.3, 0.4]:
        r = np.linspace(1e-3, 1, 400)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = f1v * r / (2 * r - f1v)
        ok = (2 * r - f1v > 0) & (p > 0) & (p <= 1.02)
        ax.plot(r[ok], p[ok], ls=":", color="#c8c8c8", lw=0.7, zorder=1)
        ridx = np.where(ok & (p <= pmax))[0]
        if len(ridx):
            k = ridx[0]
            ax.text(r[k], min(p[k], pmax) * 0.99, f"F1={f1v:g}", fontsize=5.0, color="#999",
                    ha="left", va="top", zorder=1)

    sizes = 12 + 300 * (df.n_pos / df.n_pos.max())
    colors = [F.cat_color(F.cat_label(str(c))) for c in df.category]
    ax.scatter(df.recall, df.precision, s=sizes, c=colors, alpha=0.75, edgecolors="white",
               linewidths=0.3, zorder=3)
    # Number the ten highest-F1 outcomes ON the points; names go in a side key so no labels overlap.
    top = df.nlargest(10, "f1")
    top_sizes = 12 + 300 * (top.n_pos / df.n_pos.max())
    ax.scatter(top.recall, top.precision, s=top_sizes, facecolors="none", edgecolors="#111",
               linewidths=0.6, zorder=5)
    top = top.reset_index(drop=True)
    for i, (_, r) in enumerate(top.iterrows(), 1):
        ax.annotate(str(i), (r.recall, r.precision), textcoords="offset points", xytext=(0, 0),
                    ha="center", va="center", fontsize=5.0, fontweight="bold", color="#111", zorder=6)
    ax.set_xlim(0, rmax); ax.set_ylim(0, pmax)
    ax.set_xlabel("Recall at top-5%", fontsize=8); ax.set_ylabel("Precision at top-5% (PPV)", fontsize=8)
    ax.tick_params(labelsize=6.5)

    # numbered key (upper right margin), one line per labelled outcome, never overlapping
    fig.text(0.655, 0.955, "Ten highest-F1 outcomes", fontsize=6.0, fontweight="bold", va="top")
    key = "\n".join(f"{i}.  {str(r['name']).split('[')[0].strip().rstrip('*').strip()}"
                    for i, (_, r) in enumerate(top.iterrows(), 1))
    fig.text(0.655, 0.915, key, fontsize=5.3, va="top", linespacing=1.5)

    # category legend (lower right margin)
    cats = sorted(set(F.cat_label(str(c)) for c in df.category))
    handles = [Line2D([0], [0], marker="o", ls="", markersize=4.5, markerfacecolor=F.cat_color(c),
                      markeredgecolor="white", markeredgewidth=0.3, label=c) for c in cats]
    ax.legend(handles=handles, fontsize=5.2, loc="upper left", bbox_to_anchor=(1.03, 0.52),
              frameon=False, labelspacing=0.32, title="Organ system", title_fontsize=5.8)
    save(fig, name)


def main():
    # ed2_composition is no longer built: it compared the current panel to a previous split, which
    # the paper no longer references. The ED figure files keep their on-disk names (ed3..ed9);
    # LaTeX renumbers them sequentially since ed2_composition is no longer included.
    # ed_screening is no longer built: its prevalent-screening supporting figure was cut from the
    # manuscript (screening trimmed to the main fig_screening). ed_screening() is kept defined below
    # for recoverability.
    # ed_demographics is no longer built as a standalone figure: its three panels (sex ROC, age and
    # BMI recovery) became the bottom row of main-text Figure 3 (fig4_wrist_vs_clinic). The function
    # is kept defined as the source we ported from.
    for b in [ed1_cohort_flow, ed3_age, ed4_bmi, ed5_agesex, ed6_residual,
              build_fig2_source_attribution, fig_source_attribution_supp,
              ed9_pd_benchmark, ed_pipeline,
              ed_concentration, ed_horizon, ed_six_disease, ed_pr]:
        try:
            b()
        except Exception as e:
            print(f"  !! {b.__name__} FAILED: {e}")
            traceback.print_exc()
    print("ED figures done")


if __name__ == "__main__":
    main()
