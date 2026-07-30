"""Build the 7 main-text figures for the UKB disease-prediction paper.
Each is built from the canonical disjoint_split_v1 artifacts. Vector PDF -> draft/figures/.
"""
import os
import json
import traceback
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
import figlib as F
from figlib import OK, CONST, save

UKB_ROOT = os.environ.get("UKB_ROOT", "data")
C = CONST

# =========================================================================== #
# FIG 1: study overview / pipeline schematic
# =========================================================================== #
def fig1_overview():
    # No descriptive strip in the figure (it dominated the bounding box and shrank
    # the schematic); the study facts live in the caption instead.
    fig = plt.figure(figsize=(F.W2, 1.95))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    titles = ["Wrist\naccelerometer", "Two frozen\nfoundation models", "Daily\nrepresentation",
              "Survival\nmodel", "Incident\ndisease risk"]
    subs = ["UK Biobank\n~1 week", "AcceleRest\n+ Activity model", "512-d per day\n(distributional)",
            "rolling-window\nCox model", "390 outcomes\n6-year horizon"]
    cols = [OK["grey"], OK["blue"], OK["green"], OK["purple"], OK["vermillion"]]
    # boxes sized wide enough for the longest bold title ("foundation models");
    # margins/centers leave a visible gap between boxes for the connector arrows.
    bw, bh, yc = 0.168, 0.56, 0.50
    margin = 0.016
    centers = np.linspace(margin + bw/2, 1 - margin - bw/2, 5)
    for x, title, sub, col in zip(centers, titles, subs, cols):
        box = FancyBboxPatch((x - bw/2, yc - bh/2), bw, bh, boxstyle="round,pad=0.008",
                             linewidth=1.6, edgecolor=col, facecolor="white")
        ax.add_patch(box)
        ax.text(x, yc + 0.115, title, ha="center", va="center", fontsize=8.0,
                fontweight="bold", color="#222")
        ax.text(x, yc - 0.135, sub, ha="center", va="center", fontsize=6.6, color="#555")
    for i in range(4):
        x0, x1 = centers[i], centers[i + 1]
        ax.add_patch(FancyArrowPatch((x0 + bw/2 + 0.003, yc), (x1 - bw/2 - 0.003, yc),
                     arrowstyle="-|>", mutation_scale=11, lw=1.4, color="#888"))

    save(fig, "fig1_overview")


# =========================================================================== #
# FIG 2: phenome-wide disease prediction (Manhattan)
# =========================================================================== #
def fig2_phewas():
    df = F.load_per_disease().copy()
    # order categories by mean C for a readable left->right gradient; each dot is one
    # outcome, ranked by concordance within its category (x carries no other meaning).
    catmean = df.groupby("PhecodeCategory").c.mean().sort_values()
    cat_order = list(catmean.index)
    df["catrank"] = df.PhecodeCategory.map({c: i for i, c in enumerate(cat_order)})
    df = df.sort_values(["catrank", "c"]).reset_index(drop=True)
    df["x"] = np.arange(len(df))
    N = len(df)

    fig, ax = plt.subplots(figsize=(F.W2, 3.7), layout="constrained")
    for cat in cat_order:
        sub = df[df.PhecodeCategory == cat]
        ax.scatter(sub.x, sub.c, s=14, color=F.cat_color(cat), edgecolor="none", alpha=0.85)
    for cat in cat_order[:-1]:                                   # faint category separators
        ax.axvline(df[df.PhecodeCategory == cat].x.max() + 0.5, color="#eee", lw=0.6, zorder=0)
    ax.axhline(0.5, color="#999", ls=":", lw=1.0)
    ax.text(3, 0.508, "chance = 0.50", fontsize=7, color="#999", ha="left", va="bottom")

    # highlighted outcomes: ring each dot, leader line to a de-cluttered right-margin label
    def pick(ph=None, name=None):
        r = df[df.phecode == ph] if ph else df[df.PhecodeString.str.contains(name, case=False, na=False)].sort_values("c")
        return r.iloc[-1] if len(r) else None
    spec = [("NS_324.11", "Parkinson's disease", None), ("NS_328.11", "Alzheimer's disease", None),
            ("NS_328.1", "Dementia", None), ("EM_236.1", "Obesity", None),
            (None, "Heart failure", "Heart failure"), ("time_to_death", "All-cause mortality", None)]
    hl = []
    for ph, lab, nm in spec:
        r = pick(ph=ph, name=nm)
        if r is not None:
            hl.append((lab, int(r.x), float(r.c)))
    hl.sort(key=lambda t: t[2])                                  # by concordance ascending
    gap, lys = 0.027, []                                        # greedy de-clutter of label heights
    for _, _, hc in hl:
        lys.append(hc if not lys else max(hc, lys[-1] + gap))
    label_x = N + 6
    for (lab, hx, hc), ly in zip(hl, lys):
        ax.scatter([hx], [hc], s=44, facecolor="none", edgecolor="#111", linewidth=1.2, zorder=6)
        ax.annotate(f"{lab}  ({hc:.2f})", xy=(hx, hc), xytext=(label_x, ly),
                    fontsize=6.7, va="center", ha="left", color="#111",
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="#aaa", shrinkA=2, shrinkB=3))

    ax.set_xticks([]); ax.set_xlim(-3, N + 108); ax.set_ylim(0.45, 0.99)
    ax.set_xlabel("388 incident outcomes, grouped by disease category and ranked by concordance within each",
                  fontsize=8.3, labelpad=3)
    ax.set_ylabel("Concordance index (6-year incident)")
    handles = [Line2D([0], [0], marker='o', ls='', mfc=F.cat_color(c), mec='none', ms=5,
                      label=F.cat_label(c)) for c in cat_order]
    fig.legend(handles=handles, loc="outside lower center", ncol=5,
               frameon=False, fontsize=6.2, handletextpad=0.2, columnspacing=0.9)
    save(fig, "fig2_phewas")


# =========================================================================== #
# FIG 3: per-disease highlights (forest) + breadth across categories
# =========================================================================== #
def fig3_forest_breadth():
    # embedding per-disease C + 1,000-resample bootstrap CI. Point matches per_disease_c.json,
    # the phenome-wide Manhattan figure, and the longtable; the CI is embedding's own bootstrap (one consistent source).
    ci = F.load_embedding_ci()
    wl = F.load_selected_diseases(); cat_by_ph = dict(zip(wl.phecode, wl.category))
    pdf = F.load_per_disease(); cat_by_ph2 = dict(zip(pdf.phecode, pdf.PhecodeCategory))
    rows = []
    for ph, e in ci.items():
        cat = cat_by_ph.get(ph) or cat_by_ph2.get(ph) or ("Mortality" if ph == "time_to_death" else "Other")
        rows.append((e["label"], e["C"], e["CI_lo"], e["CI_hi"], int(e["n_event"]), cat))
    rows = sorted(rows, key=lambda r: r[1])
    _short = {"Dementias": "Dementia", "Abnormality of gait": "Abnormal gait/mobility",
              "Major depressive": "Major depression"}
    labels = [_short.get(r[0], r[0]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(F.W2, 4.3), gridspec_kw={"width_ratios": [1.05, 1]},
                             layout="constrained")
    # (a) forest
    ax = axes[0]
    F.panel(ax, "a")
    y = np.arange(len(rows))
    for i, (lab, c, lo, hi, n, cat) in enumerate(rows):
        col = F.cat_color(cat)
        if not np.isnan(lo):
            ax.plot([lo, hi], [i, i], color=col, lw=2.0, solid_capstyle="round")
        ax.scatter([c], [i], s=34, color=col, zorder=5, edgecolor="white", linewidth=0.6)
        # n labels in a dedicated column right of the longest CI (avoids the PD CI overrun)
        ax.text(1.045, i, f"n={n}", ha="left", va="center", fontsize=6.0, color="#999")
    ax.axvline(0.5, color="#999", ls=":", lw=1.0)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.2)
    ax.set_xlim(0.45, 1.14); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Concordance index (95% CI)")

    # (b) breadth by category
    ax = axes[1]
    F.panel(ax, "b")
    df = F.load_per_disease()
    d = df[~df.is_mortality]
    g = d.groupby("PhecodeCategory").agg(meanC=("c", "mean"), n=("c", "size"))
    g = g[g.n >= 3].sort_values("meanC")
    yy = np.arange(len(g))
    ax.barh(yy, g.meanC.values, color=[F.cat_color(c) for c in g.index], edgecolor="white", height=0.72)
    for i, (cat, rr) in enumerate(g.iterrows()):
        ax.text(rr.meanC + 0.003, i, f"{rr.meanC:.3f}", va="center", ha="left", fontsize=6.4, color="#333")
    ax.axvline(0.5, color="#999", ls=":", lw=1.0)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{F.cat_label(c)}  (n={int(g.loc[c,'n'])})" for c in g.index], fontsize=7.0)
    ax.set_xlim(0.5, max(0.74, g.meanC.max() + 0.04))
    ax.set_xlabel("Mean concordance in category")
    save(fig, "fig3_forest_breadth")


# =========================================================================== #
# FIG 4: the wrist signal carries it and rivals the clinic
# =========================================================================== #
def fig4_wrist_vs_clinic():
    cl, cs = F.load_clinical()
    dm = F.load_demographics()                          # predicted-vs-true demographic recovery (bottom row)

    # 2x3 layout. Top row = the wrist is primary: it beats the clinic (b) and the officially
    # distributed UKB activity summaries (c) at predicting disease, and adding demographics barely
    # moves concordance (a). Bottom row = the SAME wrist signal recovers sex, age and body-mass
    # index on the held-out test set (ported from the former Extended Data demographics figure; data
    # and numbers are byte-identical). An explicit gridspec is used (not constrained layout) because
    # the bottom-row scatters set an equal aspect, which fights constrained layout.
    # near-square cells so the equal-aspect bottom-row scatters fill their cells and the two
    # rows read as the same size (a taller canvas leaves the square panels floating in whitespace).
    fig = plt.figure(figsize=(F.W2, 4.8))
    gs = fig.add_gridspec(2, 3, wspace=0.36, hspace=0.30,
                          left=0.070, right=0.985, top=0.955, bottom=0.085)

    # (a) four-arm mean concordance across the 388-outcome panel: the wrist embedding alone, the
    # wrist embedding with age, sex and BMI added (deep joint), an age/sex/BMI clinical baseline with
    # no wrist signal, and a ridge model on the 100 UKB activity-summary fields. The two wrist arms
    # share a blue family; the demographic baseline is vermillion, the activity baseline grey. All-
    # cause mortality (wrist-only 0.771 > with-demographics 0.767) is stated in the caption, not drawn,
    # because the wrist+age+sex+BMI mortality concordance is not separately computed. Values are figlib
    # CONST, from the clinical_baseline and summary_baseline output summary.json files.
    ax = fig.add_subplot(gs[0, 0])
    F.panel(ax, "a")
    x = np.array([0.0, 1.0, 2.0, 3.0]); w = 0.60
    vals = [C["NODEMO_C"], C["WRIST_DEMOBMI_C"], C["CLIN_ONLY_C"], C["SUMMARY_ONLY_C"]]
    cols = [OK["blue"], OK["skyblue"], OK["vermillion"], OK["grey"]]
    # 95% disease-panel bootstrap CIs (B=1000; clinical_baseline/clinical_baseline_bar_cis.py)
    _ci = json.load(open(f"{UKB_ROOT}/investigations/clinical_baseline/output/"
                         "clinical_baseline_bar_cis.json"))["arms"]
    _ord = ["wrist_only", "wrist_demobmi", "clinical", "activity_summary"]
    assert all(abs(vals[i] - _ci[k]["point"]) < 1e-4 for i, k in enumerate(_ord)), "CI/CONST mismatch"
    ylo = [vals[i] - _ci[k]["ci_lo"] for i, k in enumerate(_ord)]
    yhi = [_ci[k]["ci_hi"] - vals[i] for i, k in enumerate(_ord)]
    ax.bar(x, vals, w, color=cols, zorder=2)
    ax.errorbar(x, vals, yerr=[ylo, yhi], fmt="none", ecolor="#222", elinewidth=1.0,
                capsize=3, capthick=1.0, zorder=3)
    for xi, k in zip(x, _ord):
        ax.text(xi, _ci[k]["ci_hi"] + 0.004, f"{_ci[k]['point']:.3f}", ha="center",
                va="bottom", fontsize=6.0)
    ax.axhline(0.5, color="#999", ls=":", lw=1.0)  # chance; labelled in the caption (as in panels b, c)
    ax.set_xticks(x)
    ax.set_xticklabels(["Wrist\nonly", "Wrist +\nage/sex/BMI", "Age/sex/BMI\nonly", "Activity\nsummary"],
                       fontsize=5.2)
    ax.set_xlim(-0.62, 3.62); ax.set_ylim(0.48, 0.73); ax.set_ylabel("Concordance index")

    # (b) embedding vs clinical scatter, with leader-line labels for a few key diseases
    ax = fig.add_subplot(gs[0, 1])
    F.panel(ax, "b")
    sub = cl[cl.phecode != "time_to_death"].dropna(subset=["auroc_clinical", "auroc_embedding"])
    above = sub.auroc_embedding >= sub.auroc_clinical
    # encode the two groups by BOTH colour and marker so meaning never rests on colour alone
    ax.scatter(sub.auroc_clinical[above], sub.auroc_embedding[above], s=8, marker="o",
               color=OK["green"], alpha=0.55, label="wrist ≥ clinical")
    ax.scatter(sub.auroc_clinical[~above], sub.auroc_embedding[~above], s=10, marker="^",
               color=OK["vermillion"], alpha=0.6, label="clinical > wrist")
    ax.plot([0.4, 1.0], [0.4, 1.0], color="#666", lw=0.9, ls="--")
    ax.set_xlim(0.4, 1.0); ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("Clinical (age+sex+BMI) AUROC"); ax.set_ylabel("Wrist embedding AUROC")
    # leader-line labels (clinical AUROC, embedding AUROC) for a few key diseases;
    # the +0.053 AUROC / 300-of-384 statistic is reported in the caption.
    def _pt(name):
        m = cl.merge(F.phecode_map(), left_on="phecode", right_index=True, how="left")
        m = m[m.PhecodeString.str.contains(name, case=False, na=False)].sort_values("auroc_embedding")
        return (m.iloc[-1].auroc_clinical, m.iloc[-1].auroc_embedding) if len(m) else None
    annot = [("Parkinson's", "Parkinson", (0.66, 0.975)),
             ("Heart failure", "Heart failure", (0.70, 0.92)),
             ("Obesity", "Obesity", (0.86, 0.72))]
    for lab, key, tpos in annot:
        p = _pt(key)
        if p:
            ax.annotate(lab, xy=p, xytext=tpos, fontsize=6.4, color="#111", ha="left", va="center",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7),
                        arrowprops=dict(arrowstyle="-", lw=0.6, color="#888", shrinkA=2, shrinkB=3))
    ax.legend(loc="lower right", frameon=False, fontsize=6.0, handlelength=1.0)

    # (c) embedding vs UKB-distributed summarized activity features (mirror of b; was panel d).
    # Per-disease 6-yr AUROC: wrist embedding (y) vs a ridge-Cox on the official UKB
    # accelerometer summary metrics (x), fit on validation and scored on test on the SAME
    # cohort/panel as (b). Points above the line = wrist wins; the win-rate / +ΔAUROC is
    # reported in the caption.
    ax = fig.add_subplot(gs[0, 2])
    F.panel(ax, "c")
    sd, _ = F.load_summary_activity("full")
    sub2 = sd[sd.phecode != "time_to_death"].dropna(subset=["auroc_summary", "auroc_emb"])
    above2 = sub2.auroc_emb >= sub2.auroc_summary
    ax.scatter(sub2.auroc_summary[above2], sub2.auroc_emb[above2], s=8, marker="o",
               color=OK["green"], alpha=0.55, label="wrist ≥ summary")
    ax.scatter(sub2.auroc_summary[~above2], sub2.auroc_emb[~above2], s=10, marker="^",
               color=OK["vermillion"], alpha=0.6, label="summary > wrist")
    ax.plot([0.4, 1.0], [0.4, 1.0], color="#666", lw=0.9, ls="--")
    ax.set_xlim(0.4, 1.0); ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("UKB activity-summary AUROC"); ax.set_ylabel("Wrist embedding AUROC")
    # leader-line labels for the same key diseases as (b); the stat is reported in the caption
    def _pt_d(name):
        m = sd.merge(F.phecode_map(), left_on="phecode", right_index=True, how="left")
        m = m[m.PhecodeString.str.contains(name, case=False, na=False)].sort_values("auroc_emb")
        return (m.iloc[-1].auroc_summary, m.iloc[-1].auroc_emb) if len(m) else None
    for lab, key, tpos in [("Parkinson's", "Parkinson", (0.58, 1.00)),
                           ("Heart failure", "Heart failure", (0.44, 0.90)),
                           ("Obesity", "Obesity", (0.47, 0.69))]:
        p = _pt_d(key)
        if p:
            ax.annotate(lab, xy=p, xytext=tpos, fontsize=6.4, color="#111", ha="left", va="center",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7),
                        arrowprops=dict(arrowstyle="-", lw=0.6, color="#888", shrinkA=2, shrinkB=3))
    ax.legend(loc="lower right", frameon=False, fontsize=6.0, handlelength=1.0)

    # ---- bottom row: the same wrist signal recovers sex, age and BMI (held-out test set) ----
    # Dedicated wrist estimators; distinct from, and stronger than, the demographic information
    # that is merely linearly decodable from the disease-prediction embedding (that decode is
    # quoted in the caption/text). Ported verbatim from the former ED demographics figure.
    # (d) sex ROC
    axd = fig.add_subplot(gs[1, 0]); F.panel(axd, "d")
    axd.plot([0, 1], [0, 1], color="#c0c0c0", ls="--", lw=0.9, zorder=2)
    axd.plot(dm["sex_fpr"], dm["sex_tpr"], color=OK["green"], lw=2.0, zorder=3)
    axd.set_xlim(0, 1); axd.set_ylim(0, 1); axd.set_aspect("equal", adjustable="box")
    axd.set_xlabel("False-positive rate"); axd.set_ylabel("True-positive rate")
    axd.text(0.93, 0.10, f"AUROC = {float(dm['sex_auc']):.3f}", ha="right", va="bottom",
             fontsize=7.2, fontweight="bold", color="#176217")

    def _recovery_scatter(ax, xt, yt, lo, hi, col, xlab, ylab, r, r2, mae, unit):
        # light, white-edged markers so the cloud reads as discrete dots without burying the
        # identity line; rasterized so it embeds cleanly in the vector PDF (axes/text stay vector).
        ax.scatter(xt, yt, s=4, c=col, alpha=0.30, edgecolors="white", linewidths=0.15,
                   zorder=3, rasterized=True)
        ax.plot([lo, hi], [lo, hi], color="#333", ls="--", lw=1.0, zorder=4)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)
        ax.text(0.05, 0.96, f"$r$ = {r:.3f}\n$r^2$ = {r2:.2f}\nMAE = {mae:.2f}{unit}",
                transform=ax.transAxes, va="top", ha="left", fontsize=6.8, color="#333")

    # (e) age
    axe = fig.add_subplot(gs[1, 1]); F.panel(axe, "e")
    _recovery_scatter(axe, dm["age_true"], dm["age_pred"], 44, 78, OK["blue"],
                      "Chronological age (yr)", "Predicted age (yr)",
                      float(dm["age_r"]), float(dm["age_r2"]), float(dm["age_mae"]), " yr")
    # (f) BMI
    axf = fig.add_subplot(gs[1, 2]); F.panel(axf, "f")
    _recovery_scatter(axf, dm["bmi_true"], dm["bmi_pred"], 15, 46, OK["vermillion"],
                      "Measured BMI (kg m$^{-2}$)", "Predicted BMI (kg m$^{-2}$)",
                      float(dm["bmi_r"]), float(dm["bmi_r2"]), float(dm["bmi_mae"]), "")

    save(fig, "fig4_wrist_vs_clinic")


# =========================================================================== #
# FIG 5: mechanistic interpretability
# =========================================================================== #
def fig5_interpretability():
    mp = F.load_modality_profiles()
    wl = F.load_selected_diseases()
    lab2ph = dict(zip(wl.label, wl.phecode))
    # surrogate gaps per disease
    gaps = [("Obesity", 0.148), ("Parkinson's", 0.117), ("Alzheimer's", 0.105),
            ("COPD", 0.091), ("Dementia", 0.082), ("Type 2 diabetes", 0.076),
            ("Depression", 0.066), ("Sleep apnea", 0.045), ("Mortality", 0.036),
            ("Atrial fibrillation", 0.018)]
    gaps = sorted(gaps, key=lambda x: x[1])

    fig, axes = plt.subplots(1, 2, figsize=(F.W2, 3.6), gridspec_kw={"width_ratios": [1, 1.1]},
                             layout="constrained")
    # (a) surrogate gap lollipop
    ax = axes[0]
    F.panel(ax, "a")
    y = np.arange(len(gaps))
    for i, (lab, g) in enumerate(gaps):
        ax.plot([0, g], [i, i], color="#bbb", lw=1.2, zorder=1)
        ax.scatter([g], [i], s=34, color=OK["green"], zorder=3)
        ax.text(g + 0.004, i, f"+{g:.3f}", va="center", fontsize=6.2, color="#333")
    ax.set_yticks(y); ax.set_yticklabels([g[0] for g in gaps], fontsize=7.2)
    ax.set_xlim(0, 0.18); ax.set_xlabel("Gain over named sleep/activity features (ΔC)")

    # (b) modality heatmap AR vs HA
    ax = axes[1]
    F.panel(ax, "b")
    dis = [("Parkinson's", "NS_324.11"), ("Alzheimer's", "NS_328.11"), ("Dementia", "NS_328.1"),
           ("Sleep apnea", "NS_333.1"), ("Obesity", "EM_236.1"), ("Type 2 diabetes", "EM_202.2"),
           ("Heart failure", "CV_424"), ("Atrial fibrillation", "CV_416.2"), ("All-cause mortality", "time_to_death")]
    rows, names = [], []
    for nm, ph in dis:
        if ph is None:
            cand = [v for k, v in lab2ph.items() if nm.split()[0].lower() in k.lower()]
            ph = cand[0] if cand else None
        ar = mp[(mp.phecode == ph) & (mp.cell == "AR")]
        ha = mp[(mp.phecode == ph) & (mp.cell == "HA")]
        if len(ar) and len(ha):
            rows.append([ar.dc_mean.values[0], ha.dc_mean.values[0]]); names.append(nm)
    M = np.array(rows)
    im = ax.imshow(M, cmap="RdYlBu_r", aspect="auto", vmin=-0.02, vmax=0.16, interpolation="nearest")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Sleep /\nbreathing\n(AcceleRest)", "Movement\n(Activity)"], fontsize=7.0)
    ax.set_yticks(np.arange(len(names))); ax.set_yticklabels(names, fontsize=7.2)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]:+.3f}", ha="center", va="center", fontsize=6.0,
                    color="#000" if M[i, j] < 0.10 else "white")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Drop in C when removed", fontsize=7); cb.ax.tick_params(labelsize=6)
    save(fig, "fig5_interpretability")


# =========================================================================== #
# FIG 6: the frozen representation is saturated (what we tried)
# =========================================================================== #
def fig6_saturation():
    """Modelling-ablations figure. The learned daily representation is necessary and the model
    needs little input (feature-source ladder + days/hours titration), and an age-pretrained
    sequence model over the un-aggregated patch embeddings does not beat pooling them (a matched
    bar under the frozen-embedding ceiling and a per-disease scatter across the phenome).
    Absorbs the former standalone controls figure; the schematic pathway panel was dropped on review.
    (a) feature-source ladder; (b) days-of-wear titration; (c) hours-per-day titration;
    (d) sequence vs a mean of the same patches under the ceiling; (e) the tie disease-by-disease."""
    SEQ, BASE = OK["orange"], OK["blue"]
    PA = F.UKB / "paper_additions" / "output"

    def _lj(p):
        return json.load(open(p)) if p.exists() else None

    ladder = (_lj(PA / "boot_ladder_full" / "bootstrap_results.json")
              or _lj(PA / "boot_ladder" / "bootstrap_results.json"))
    days_b = _lj(PA / "boot_titration_days" / "bootstrap_results.json")
    days_c = _lj(PA / "titration_days" / "curve.json")
    hours_b = _lj(PA / "boot_titration_hours" / "bootstrap_results.json")
    hours_c = _lj(PA / "titration_hours" / "curve.json")

    LADDER_LABELS = {"random_gaussian": "Random features", "random_encoder": "Untrained encoder",
                     "summary_stats": "Summary statistics", "learned_mean": "Learned (mean)",
                     "learned_lmoments": "Learned (distributional)"}
    LADDER_ORDER = ["random_gaussian", "random_encoder", "summary_stats", "learned_mean", "learned_lmoments"]
    LADDER_COLORS = {"random_gaussian": OK["grey"], "random_encoder": OK["purple"],
                     "summary_stats": OK["orange"], "learned_mean": OK["skyblue"],
                     "learned_lmoments": OK["blue"]}

    fig = plt.figure(figsize=(F.W2, 6.0))
    outer = fig.add_gridspec(2, 1, height_ratios=[0.82, 1.2], hspace=0.32,
                             left=0.165, right=0.985, top=0.96, bottom=0.08)
    gs0 = outer[0].subgridspec(1, 3, width_ratios=[1.28, 1.0, 1.0], wspace=0.5)
    # d and e each take a full half of the row (no side spacers) so d's model labels have room;
    # both are given an equal square box below so they render the same size with aligned axes.
    gs1 = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.14)

    # ---- (a) feature-source ladder: the learned representation is necessary ----
    axa = fig.add_subplot(gs0[0, 0])
    if ladder:
        arms = [a for a in LADDER_ORDER if a in ladder["marginal"]]
        y = np.arange(len(arms))
        pts = [ladder["marginal"][a]["point"] for a in arms]
        clo = [ladder["marginal"][a]["ci_lo"] for a in arms]
        chi = [ladder["marginal"][a]["ci_hi"] for a in arms]
        cols = [LADDER_COLORS[a] for a in arms]
        axa.barh(y, pts, color=cols, height=0.62,
                 xerr=[np.array(pts) - np.array(clo), np.array(chi) - np.array(pts)],
                 error_kw=dict(ecolor="#333333", elinewidth=1.0, capsize=2.5))
        axa.set_yticks(y); axa.set_yticklabels([LADDER_LABELS[a] for a in arms], fontsize=6.6)
        axa.set_xlim(0.5, 0.725)
        axa.axvline(0.5, ls=":", color=OK["grey"], lw=0.8)
        for yi, p, h in zip(y, pts, chi):
            axa.text(h + 0.003, yi, f"{p:.3f}", va="center", ha="left", fontsize=6.4)
    axa.set_xlabel("Mean concordance")
    F.panel(axa, "a")

    # ---- (b) days-of-wear titration ----
    axb = fig.add_subplot(gs0[0, 1])
    if days_c:
        ks = [d["k_days"] for d in days_c["curve"]]
        mc = [d["mean_c"] for d in days_c["curve"]]
        indist = [d["in_distribution"] for d in days_c["curve"]]
        if days_b:
            lo = [days_b["marginal"].get(f"days_k{k}", {}).get("ci_lo", np.nan) for k in ks]
            hi = [days_b["marginal"].get(f"days_k{k}", {}).get("ci_hi", np.nan) for k in ks]
            axb.fill_between(ks, lo, hi, color=OK["blue"], alpha=0.15, lw=0)
        ks_a = np.array(ks); mc_a = np.array(mc); ind = np.array(indist)
        axb.plot(ks_a[ind], mc_a[ind], "-o", color=OK["blue"], ms=4, lw=1.4, label="deployed")
        if (~ind).any():
            axb.plot(ks_a[~ind], mc_a[~ind], "o", color=OK["grey"], ms=4, mfc="white", label="below window")
        axb.axhline(mc[-1], ls=":", color=OK["grey"], lw=0.8)
        axb.set_xticks(ks)
        axb.legend(fontsize=6.2, loc="lower right", frameon=False)
    axb.set_xlabel("Days of wear")
    axb.set_ylabel("Mean concordance")
    F.panel(axb, "b")

    # ---- (c) hours-per-day titration ----
    axc = fig.add_subplot(gs0[0, 2])
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
    axc.set_xlabel("Hours per day")
    axc.set_ylabel("Mean concordance")
    F.panel(axc, "c")

    # ---- (d) sequence vs a mean of the same patches, under the main-model ceiling ----
    axd = fig.add_subplot(gs1[0, 0])
    # bar heights + 95% disease-panel bootstrap CIs (B=1000; ed_bar_cis.py)
    _bar_cis = Path(__file__).resolve().parent / "ed_bar_cis.json"
    _aci = json.load(open(_bar_cis))["figures"]["agepretrain"]
    _ak = ["main", "sequence", "mean"]
    vals = [_aci[k]["point"] for k in _ak]
    alo = [_aci[k]["point"] - _aci[k]["ci_lo"] for k in _ak]
    ahi = [_aci[k]["ci_hi"] - _aci[k]["point"] for k in _ak]
    labs = ["Main model\n(distrib. pooling)", "Sequence\nmodel", "Mean of\nsame patches"]
    cols = [BASE, SEQ, "#a9cbe8"]
    axd.bar([0, 1, 2], vals, width=0.62, color=cols, zorder=3)
    axd.errorbar([0, 1, 2], vals, yerr=[alo, ahi], fmt="none", ecolor="#222", elinewidth=1.0,
                 capsize=3, capthick=1.0, zorder=4)
    for i, k in enumerate(_ak):
        axd.text(i, _aci[k]["ci_hi"] + 0.0015, f"{vals[i]:.3f}", ha="center", va="bottom",
                 fontsize=6.0, fontweight="bold")
    axd.axhline(0.5, color="#aaa", ls=":", lw=0.9)
    axd.text(2.45, 0.505, "chance", fontsize=5.0, color="#888", ha="right", va="bottom")
    # main-model ceiling: extend the headline level across the per-patch bars
    axd.plot([0.45, 2.42], [vals[0], vals[0]], color=BASE, ls="--", lw=0.85, zorder=2)
    axd.text(2.42, vals[0] + 0.0012, "ceiling", ha="right", va="bottom", fontsize=5.0, color=BASE, style="italic")
    # the sequence model only ties a simple mean of the same patches (subject-level paired test)
    axd.plot([1, 1, 2, 2], [0.676, 0.679, 0.679, 0.676], color="#555", lw=0.9, zorder=5)
    axd.text(1.5, 0.680, "ties, $P=0.11$", ha="center", va="bottom", fontsize=5.4, color="#555")
    axd.set_ylim(0.49, 0.72); axd.set_xticks([0, 1, 2]); axd.set_xticklabels(labs, fontsize=6.4)
    axd.tick_params(labelsize=6.2); axd.set_ylabel("Mean concordance", fontsize=7.0)
    axd.set_xlim(-0.6, 2.6)
    axd.set_box_aspect(1)
    F.panel(axd, "d")

    # ---- (e) the tie holds across the phenome: per-disease ----
    axe = fig.add_subplot(gs1[0, 1])
    pdd = F.load_seq_vs_mean()
    nev = pdd["n_event"].values.astype(float)
    sizes = np.clip(2.0 + 0.95 * np.sqrt(nev), 3, 58)
    pw = np.sqrt(nev / nev.max())
    rgba = np.tile(np.array([0.31, 0.39, 0.49, 1.0]), (len(nev), 1))
    rgba[:, 3] = np.clip(0.16 + 0.72 * pw, 0.14, 0.9)
    lo, hi = 0.42, 0.93
    axe.plot([lo, hi], [lo, hi], color="#444", ls="--", lw=0.9, zorder=2)
    axe.scatter(pdd["mean_c"], pdd["seq_c"], s=sizes, c=rgba, linewidths=0.0, zorder=3)
    axe.set_xlim(lo, hi); axe.set_ylim(lo, hi); axe.set_box_aspect(1)
    axe.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9]); axe.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    axe.tick_params(labelsize=6.2)
    axe.set_xlabel("Mean of same patches (C)", fontsize=7.0)
    axe.set_ylabel("Sequence model (C)", fontsize=7.0)
    axe.text(0.045, 0.975, "388 outcomes\n$r=0.78$ overall,\n0.95 best-powered\n(size $\\propto$ events)",
             transform=axe.transAxes, fontsize=5.0, va="top", ha="left", color="#333")
    F.panel(axe, "e")

    save(fig, "fig6_saturation")


# =========================================================================== #
# FIG 7: honest, leakage-free evaluation
# =========================================================================== #
def fig7_honest_eval():
    B = F.benchmark()
    bm = B["models"]; e3 = bm["embedding_disjoint_split"]; p4 = bm["sequence_model_disjoint_split"]
    pr = B["paired_vs_reference"]["embedding_disjoint_split"]
    fav_c, fav_r = pr["diseases_favor_candidate"], pr["diseases_favor_reference"]
    # four aligned subplots (same top/bottom): honest-number bars on the left, then the
    # full three-way benchmark on the right (mean C, uncensored C, paired head-to-head).
    fig = plt.figure(figsize=(F.W2, 3.0))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.5, 0.72, 0.72, 0.72],
                          left=0.075, right=0.99, top=0.9, bottom=0.21, wspace=0.62)
    ax0 = fig.add_subplot(gs[0]); ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2]); ax3 = fig.add_subplot(gs[3])
    F.panel(ax0, "a"); F.panel(ax1, "b")

    # (a) joint-Cox optimism
    labs = ["Held-out\ndeep model", "In-sample\njoint model", "Honest\ncross-fit"]
    # in-sample = the SAME combiner reproduced in the honest harness (0.7075), so the
    # gap to its honest cross-fit (0.6685) is exactly the documented +0.039 optimism.
    vals = [C["RAW_HAZARD"], 0.7075, C["JOINT_HONEST"]]
    ax0.bar(np.arange(3), vals, 0.6, color=[OK["green"], OK["grey"], OK["blue"]])
    for i, v in enumerate(vals):
        ax0.text(i, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=7.2, fontweight="bold")
    ax0.axhline(C["RAW_HAZARD"], color=OK["green"], ls=":", lw=1.0)
    ax0.annotate("in-sample\noptimism +0.039", xy=(1.30, 0.7075), xytext=(1.5, 0.762),
                 fontsize=6.2, ha="center", va="center", color="#a33",
                 arrowprops=dict(arrowstyle="-|>", lw=1.0, color="#a33", shrinkA=3, shrinkB=2))
    ax0.set_xticks(np.arange(3)); ax0.set_xticklabels(labs, fontsize=7.0)
    ax0.set_ylim(0.55, 0.79); ax0.set_ylabel("Mean concordance")

    # (b) three views: mean C, uncensored C, paired head-to-head
    ax1.bar([0, 1], [e3["mean_c"], p4["mean_c"]], color=[OK["green"], OK["orange"]], width=0.66)
    ax1.set_ylim(0.66, 0.70); ax1.set_xticks([0, 1]); ax1.set_xticklabels(["Distrib.", "Mean"], fontsize=6.6)
    ax1.tick_params(labelsize=6.3); ax1.set_xlabel("mean C", fontsize=7.6)
    for i, v in enumerate([e3["mean_c"], p4["mean_c"]]):
        ax1.text(i, v + 0.0004, f"{v:.3f}", ha="center", fontsize=6.1)
    ax2.bar([0, 1], [e3["unc_c_pooled"], e3["unc_age_floor"]], color=[OK["green"], "#999"], width=0.66)
    ax2.set_ylim(0.50, 0.53); ax2.set_xticks([0, 1]); ax2.set_xticklabels(["Model", "Age floor"], fontsize=6.6)
    ax2.tick_params(labelsize=6.3); ax2.set_xlabel("uncensored C", fontsize=7.6)
    for i, v in enumerate([e3["unc_c_pooled"], e3["unc_age_floor"]]):
        ax2.text(i, v + 0.0004, f"{v:.3f}", ha="center", fontsize=6.1)
    ax3.bar([0, 1], [fav_c, fav_r], color=[OK["green"], OK["orange"]], width=0.66)
    ax3.set_ylim(0, max(fav_c, fav_r) * 1.28); ax3.set_xticks([0, 1])
    ax3.set_xticklabels(["Distrib.", "Mean"], fontsize=6.6)
    ax3.tick_params(labelsize=6.3); ax3.set_xlabel("Head-to-head", fontsize=7.6)
    ax3.set_ylabel("Diseases won", fontsize=7.0, labelpad=1)
    for i, v in enumerate([fav_c, fav_r]):
        ax3.text(i, v + max(fav_c, fav_r) * 0.02, f"{v}", ha="center", fontsize=6.3, fontweight="bold")

    save(fig, "fig7_honest_eval")


# =========================================================================== #
# FIG (screening): prevalent-disease screening from the wrist embedding
# =========================================================================== #
def fig_screening_prevalent(bundle=None):
    """Wrist-only prevalent-disease screening, single panel: a phecode map of per-disease
    wrist-embedding prevalent-screening AUROC across all 632 screened outcomes, coloured by disease
    category and ranked within category (same category-ranked phecode-map style as the overview figure), with six leaders ringed. The three-arm
    comparison on the labelled outcomes (embedding vs demographics vs their combination) and
    Parkinson's ROC are reported in the body text and the Extended Data companion, ed_screening()."""
    B = F.load_screening_bundle(bundle)

    # ---- per-disease AUROC across all 632 screened outcomes (phecode map) ----
    sd = B["sd"].copy()
    pm = F.phecode_map()
    sd = sd.merge(pm, left_on="phecode", right_index=True, how="left")
    sd["cat"] = sd["PhecodeCategory"].fillna(sd["category"]).fillna("Other")
    catmean = sd.groupby("cat").sd_auroc.mean().sort_values()          # categories low -> high mean
    cat_order = list(catmean.index)
    sd["catrank"] = sd["cat"].map({c: i for i, c in enumerate(cat_order)})
    sd = sd.sort_values(["catrank", "sd_auroc"]).reset_index(drop=True)
    sd["x"] = np.arange(len(sd)); Nm = len(sd)

    fig = plt.figure(figsize=(F.W2, 2.45 + 0.38), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[2.45, 0.38])

    # phecode map: wrist-embedding screening AUROC by disease category (same category-ranked style as the overview figure) --------
    axm = fig.add_subplot(gs[0])
    for cat in cat_order:
        sub = sd[sd.cat == cat]
        axm.scatter(sub.x, sub.sd_auroc, s=9, color=F.cat_color(cat), edgecolor="none", alpha=0.85)
    axm.axhline(0.5, color="#999", ls=":", lw=0.9)
    axm.text(Nm + 4, 0.5, "chance", fontsize=6.3, color="#888", va="center", ha="left")
    # ring + leader-line key diseases (movement/neuro/respiratory/mood leaders + age-driven contrast)
    mspec = [("NS_324.11", "Parkinson's disease", 0.93),
             ("EM_236.1",  "Obesity",             0.87),
             ("NS_326.1",  "Multiple sclerosis",  0.81),
             ("RE_474",    "COPD",                0.75),
             ("NS_333.1",  "Sleep apnea",         0.69),
             ("MB_286.2",  "Major depression",    0.63)]
    for ph, lab, ly in mspec:
        r = sd[sd.phecode == ph]
        if not len(r):
            continue
        r = r.iloc[0]; hx, hc = int(r.x), float(r.sd_auroc)
        lo, hi = float(r.sd_lo), float(r.sd_hi)                  # 95% CI (per_disease_sd.csv)
        axm.errorbar(hx, hc, yerr=[[hc - lo], [hi - hc]], fmt="none", ecolor="#111",
                     elinewidth=0.9, capsize=2.0, capthick=0.9, zorder=5)
        axm.scatter([hx], [hc], s=40, facecolor="none", edgecolor="#111", linewidth=1.1, zorder=6)
        axm.annotate(f"{lab} ({hc:.2f})", xy=(hx, hc), xytext=(Nm + 28, ly),
                     fontsize=6.3, va="center", ha="left", color="#111",
                     arrowprops=dict(arrowstyle="-", lw=0.55, color="#b0b0b0",
                                     shrinkA=3.5, shrinkB=4, relpos=(0, 0.5)))
    axm.set_xticks([]); axm.set_xlim(-3, Nm + 190); axm.set_ylim(0.28, 0.965)
    axm.set_xlabel("632 prevalent-screening outcomes, grouped by disease category and ranked by AUROC within each",
                   fontsize=7.6)
    axm.set_ylabel("Prevalent-screening AUROC")

    # disease-category colour key (own row, below the map)
    axleg = fig.add_subplot(gs[1]); axleg.axis("off")
    cat_handles = [Line2D([0], [0], marker="o", ls="", mfc=F.cat_color(c), mec="none", ms=4.5,
                          label=F.cat_label(c)) for c in reversed(cat_order)]
    axleg.legend(handles=cat_handles, loc="center", ncol=7, fontsize=6.0, frameon=True,
                 facecolor="white", edgecolor="#bbbbbb", framealpha=1.0, handletextpad=0.3,
                 columnspacing=1.0, labelspacing=0.6, borderaxespad=0.1)

    # [Three-arm forest on the labelled outcomes removed per review; the embedding-vs-demographics
    #  comparison is reported in the body text and the Extended Data companion, ed_screening().]

    save(fig, "fig_screening")


def main():
    # Only the figures used in the manuscript are built by default. Superseded /
    # development builders (fig1_overview, fig2_phewas, fig3_forest_breadth,
    # fig5_interpretability, fig7_honest_eval) are kept defined above for
    # recoverability but are not in the default run.
    builders = [fig4_wrist_vs_clinic, fig6_saturation, fig_screening_prevalent]
    for b in builders:
        try:
            b()
        except Exception as e:
            print(f"  !! {b.__name__} FAILED: {e}")
            traceback.print_exc()
    print("main figures done")


if __name__ == "__main__":
    main()
