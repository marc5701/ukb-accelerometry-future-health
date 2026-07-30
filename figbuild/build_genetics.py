"""Genetics x actigraphy figures (reuses figlib for style and loaders).

Main figure (complementary modalities):
  fig_genetics  -> (A) orthogonality scatter, (B) exemplar wrist/PRS/fusion bars.
Extended Data (separate figures, appended at end of extended_data.tex):
  ed_genetics_forest  -> full 26-disease delta-C forest with 95% CI + permuted-PRS null.
  ed_genetics_robust  -> (a) ancestry sensitivity, (b) PRS-set / Enhanced-subgroup.
  ed_genetics_matched -> matched PRS: (a) global concat flat plus within-concat
                          cardiometabolic gains, (b) matched +0.042 vs panel-as-features -0.040.

Reads the crossfit genetics outputs read-only.
"""
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import figlib as F
from figlib import OK, CAT_COLORS, cat_color, C_WRIST, C_EMBEDDING, W2, W15, save

UKB_ROOT = os.environ.get("UKB_ROOT", "data")
CF = Path(f"{UKB_ROOT}/crossfit")
C_GEN = OK["purple"]      # genetics
C_FUSE = OK["green"]      # fusion
C_NULLp = OK["grey"]

# Disease-category colours come from the SINGLE unified figlib palette so genetics figures
# match the rest of the paper (colourblind-safe, same category == same colour everywhere).
ccolor = F.cat_color

# Sensible display order for the category legend (only those present in the data are shown).
CAT_ORDER = ["Cardiovascular", "Neurological", "Endocrine/Metab", "Respiratory",
             "Gastrointestinal", "Sense organs", "Dermatological", "Musculoskeletal",
             "Neoplasms", "Mental", "Genitourinary", "Blood/Immune", "Other"]

# phecode prefix -> disease category (matches figlib CAT_COLORS keys)
PREFIX_CAT = {
    "CV": "Cardiovascular", "NS": "Neurological", "EM": "Endocrine/Metab",
    "RE": "Respiratory", "GI": "Gastrointestinal", "SO": "Sense organs",
    "DE": "Dermatological", "MS": "Musculoskeletal", "CA": "Neoplasms",
    "MB": "Mental", "GU": "Genitourinary", "BI": "Blood/Immune",
}
LABEL = {
    "GI_525.1": "Coeliac disease", "EM_202.1": "Type 1 diabetes", "SO_375.11": "Glaucoma (open-angle)",
    "CV_402": "Hypertension", "GI_522.11": "Crohn's disease", "SO_375.1": "Glaucoma",
    "CV_416.212": "Atrial fibrillation", "CV_416.211": "Atrial fibrillation", "CV_416.2": "Atrial fibrillation",
    "DE_664.4": "Psoriasis", "CV_404.2": "Coronary atherosclerosis", "CV_404.11": "Myocardial infarction",
    "RE_475": "Asthma", "EM_202.2": "Type 2 diabetes", "CV_404.1": "Ischemic heart disease",
    "CV_440.3": "Pulmonary embolism", "GI_522.12": "Ulcerative colitis", "CV_401.1": "Hypertension",
    "CV_404": "Ischemic heart disease", "CA_103": "Melanoma", "MS_705.1": "Rheumatoid arthritis",
    "SO_374.5": "Macular degeneration", "NS_324.11": "Parkinson's disease", "NS_328.1": "Dementia",
    "CV_431.11": "Ischaemic stroke", "NS_328.11": "Alzheimer's disease",
}
# Forest-only labels that DISTINGUISH hierarchical phecode subtypes. The LABEL map above
# deliberately collapses them to one repeated name so the fig_genetics scatter (which uses
# drop_duplicates("label")) shows one point per disease; the forest, where every phecode is
# its own row, needs distinct names. Strings are the authoritative phecodeX 2.0 vocabulary
# (PheWAS/PhecodeX), British spelling to match the manuscript body. Used only in the forest.
FOREST_LABEL = {
    "CV_416.2": "Atrial fibrillation", "CV_416.211": "Atrial fibrillation (paroxysmal)",
    "CV_416.212": "Atrial fibrillation (persistent)",
    "CV_401.1": "Essential hypertension", "CV_402": "Hypertension (elevated BP)",
    "CV_404": "Ischaemic heart disease", "CV_404.1": "Myocardial infarction",
    "CV_404.11": "Acute myocardial infarction", "CV_404.2": "Coronary atherosclerosis",
}
# exemplar diseases for panel B (cardiometabolic/autoimmune wins + Parkinson's wrist-only)
EXEMPLARS = ["GI_525.1", "CV_416.2", "EM_202.2", "CV_404.2", "SO_375.1", "NS_324.11"]


def _per():
    d = pd.read_csv(CF / "perdisease_eur" / "perdisease_ci.csv")
    d["dcat"] = [PREFIX_CAT.get(str(p).split("_")[0], "Other") for p in d.phecode]
    d["label"] = [LABEL.get(p, p) for p in d.phecode]
    d["sig"] = d.p_pos >= 0.975
    return d


def _summ(name):
    p = CF / name / "summary.json"
    return json.load(open(p)) if p.exists() else None


# --------------------------------------------------------------------------- MAIN
def fig_genetics_main():
    # one point per disease: drop duplicate phecodes that share a disease label (e.g. three
    # atrial-fibrillation codes) so the centre is a readable scatter, not a blob of identical dots
    d = (_per().sort_values("n_ev", ascending=False)
         .drop_duplicates("label", keep="first").set_index("phecode"))
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(F.W2, 3.7), layout="constrained",
                                   gridspec_kw=dict(width_ratios=[1.12, 1.0]))

    # (A) orthogonality scatter: wrist-C (x) vs PRS-only-C (y) -----------------
    LO, HI = 0.38, 0.95
    axA.set_aspect("equal")                       # square box -> y=x renders at a true 45 deg
    axA.plot([LO, HI], [LO, HI], color="#b8b8b8", lw=1.0, zorder=1)   # equal-prediction reference
    # dominance regions: below the diagonal the wrist predicts better, above it genetics does.
    # faint modality-tinted half-planes (well below the points) make each disease's dominant
    # axis readable at a glance from which shaded half it falls in.
    axA.fill_between([LO, HI], LO, [LO, HI], color=C_WRIST, alpha=0.085, lw=0, zorder=0)
    axA.fill_between([LO, HI], [LO, HI], HI, color=C_GEN, alpha=0.085, lw=0, zorder=0)
    # labelled exemplars get a dark ring so the leader target is unambiguous in the cluster
    labeled = {"GI_525.1", "EM_202.1", "NS_324.11", "NS_328.11"}
    for ph, r in d.iterrows():
        hot = ph in labeled
        axA.scatter(r.c_embed, r.c_prs_only, s=14 + 2.0 * np.sqrt(r.n_ev),   # gentle area-by-events
                    color=ccolor(r.dcat), edgecolor=("#1a1a1a" if hot else "white"),
                    linewidth=(0.9 if hot else 0.5), alpha=0.9, zorder=(4 if hot else 3))
    axA.set_xlim(LO, HI); axA.set_ylim(LO, HI)
    axA.set_xlabel("Wrist embedding C-index (genetics-free)")
    axA.set_ylabel("PRS-only C-index (wrist-free)")
    F.panel(axA, "a")

    # name the two regions in their modality colour so the dominance map reads instantly:
    # purple above the diagonal (genetics wins), blue below it (wrist wins).
    axA.text(0.405, 0.935, "genetics dominates", ha="left", va="top",
             fontsize=7.0, style="italic", color=C_GEN, zorder=2)
    axA.text(0.94, 0.405, "wrist dominates", ha="right", va="bottom",
             fontsize=7.0, style="italic", color=C_WRIST, zorder=2)

    # label only the spatially-clear extremes that anchor the two regions (the cardiometabolic
    # cluster is shown disease-by-disease in panel b). Two genetics-dominant anchors above the
    # diagonal, two wrist-dominant neurodegenerative anchors below it.
    lab_pos = {
        "GI_525.1":  (0.560, 0.910, "center", "Coeliac disease"),      # autoimmune, genetics-dominant
        "EM_202.1":  (0.40, 0.560, "left", "Type 1 diabetes"),         # far-left, genetics-dominant
        "NS_328.11": (0.815, 0.715, "center", "Alzheimer's\ndisease"), # neuro, wrist-dominant
        "NS_324.11": (0.905, 0.475, "center", "Parkinson's\ndisease"), # neuro, far-right, wrist-dominant
    }
    for ph, (tx, ty, ha, txt) in lab_pos.items():
        r = d.loc[ph]
        axA.annotate(txt, (r.c_embed, r.c_prs_only), xytext=(tx, ty), ha=ha, va="center",
                     fontsize=6.2, color="#222222", linespacing=0.95,
                     arrowprops=dict(arrowstyle="-", lw=0.5, color="#aaaaaa", shrinkA=1, shrinkB=4))

    cats = [c for c in CAT_ORDER if c in set(d.dcat)]
    handles = [Line2D([0], [0], marker="o", ls="", mec="white", mew=0.5, mfc=F.cat_color(c),
                      ms=5, label=F.cat_label(c)) for c in cats]
    # category legend below panel a (reserved cleanly by constrained_layout, never covers data)
    axA.legend(handles=handles, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.16),
               ncol=5, frameon=False, handletextpad=0.2, columnspacing=0.9, labelspacing=0.3)

    # (B) exemplar grouped bars: wrist / PRS / fusion --------------------------
    ex = d.loc[EXEMPLARS].reset_index()
    x = np.arange(len(ex)); w = 0.26
    axB.bar(x - w, ex.c_embed, w, color=C_WRIST, label="Wrist only")
    axB.bar(x, ex.c_prs_only, w, color=C_GEN, label="PRS only")
    axB.bar(x + w, ex.c_fusion, w, color=C_FUSE, label="Wrist + PRS (fusion)")
    axB.axhline(0.5, color="#999999", lw=0.7, ls=":")
    axB.set_xticks(x)
    axB.set_xticklabels([LABEL[p] for p in EXEMPLARS], rotation=32, ha="right", fontsize=6.6)
    axB.set_ylabel("C-index"); axB.set_ylim(0.4, 1.06)
    F.panel(axB, "b")
    axB.legend(fontsize=5.8, loc="upper left", frameon=False, handlelength=1.1,
               handletextpad=0.4, labelspacing=0.3, borderaxespad=0.4)
    # fusion-gain cue: where combining genuinely wins, print the gain over the better single
    # modality above the fusion bar, so the complementarity is quantitative and obvious at a glance.
    for i, p in enumerate(EXEMPLARS):
        r = ex.iloc[i]
        g = r.c_fusion - max(r.c_embed, r.c_prs_only)
        if g >= 0.01:
            axB.text(i + w, r.c_fusion + 0.012, f"+{g:.3f}", ha="center", va="bottom",
                     fontsize=5.4, color=C_FUSE, fontweight="bold")
    # Parkinson's: wrist ~= fusion, PRS low -> short cue above the bars, no leader arrow
    pidx = EXEMPLARS.index("NS_324.11")
    ptop = max(ex.iloc[pidx].c_embed, ex.iloc[pidx].c_fusion)
    axB.text(pidx, ptop + 0.03, "genetics adds\nnothing", ha="center", va="bottom",
             fontsize=5.6, style="italic", color="#666666", linespacing=0.95)

    save(fig, "fig_genetics")


# --------------------------------------------------------------------------- ED: forest
def ed_genetics_forest():
    d = _per().sort_values("delta").reset_index(drop=True)
    d["label"] = [FOREST_LABEL.get(p, l) for p, l in zip(d.phecode, d.label)]  # distinct subtype names
    d["ci_excl0"] = d.delta_lo > 0          # 95% CI excludes zero
    y = np.arange(len(d))
    mean = d.delta.mean()
    from matplotlib.patches import Patch
    nd = (d.c_null - d.c_embed).to_numpy()
    nlo, nhi = np.percentile(nd, [10, 90])
    fig, ax = plt.subplots(figsize=(F.W2, 6.8), layout="constrained")

    # permuted-PRS null as a single clean shaded band (10-90% across outcomes), behind everything
    ax.axvspan(nlo, nhi, color="#d9d9d9", alpha=0.7, lw=0, zorder=0)
    ax.axvline(0, color="#333333", lw=0.9, zorder=1)
    ax.axvline(mean, color=C_WRIST, lw=1.1, ls=":", zorder=1)

    # one clean series: CI bar + significance-coloured point per disease (sorted by effect)
    ax.errorbar(d.delta, y, xerr=[d.delta - d.delta_lo, d.delta_hi - d.delta], fmt="none",
                ecolor="#b8b8b8", elinewidth=1.3, capsize=2, zorder=3)
    colors = [C_FUSE if e else C_NULLp for e in d.ci_excl0]
    ax.scatter(d.delta, y, c=colors, s=34, edgecolor="k", linewidth=0.3, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{l}  (n={int(n)})" for l, n in zip(d.label, d.n_ev)], fontsize=6.8)
    ax.set_xlabel("Δ C-index: wrist embedding + matched PRS  minus  wrist embedding")
    ax.set_ylim(-1, len(d) + 0.5)
    # mean label in the open upper-left corner, clear of both verticals
    ax.text(0.02, 0.985, f"dotted line = mean (+{mean:.3f})", transform=ax.transAxes,
            fontsize=7.5, color=C_WRIST, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))

    legend = [
        Line2D([0], [0], marker="o", ls="", mfc=C_FUSE, mec="k", mew=0.3, ms=6,
               label="Fusion gain, 95% CI excludes 0"),
        Line2D([0], [0], marker="o", ls="", mfc=C_NULLp, mec="k", mew=0.3, ms=6,
               label="Fusion gain, not significant"),
        Patch(facecolor="#d9d9d9", edgecolor="none", label="Permuted-PRS null (10-90%)"),
    ]
    # legend INSIDE the axes, lower-right open region (bottom rows are near zero, leaving
    # the right side clear) so it never covers any disease row
    ax.legend(handles=legend, fontsize=7, loc="lower right", frameon=True, framealpha=0.85,
              edgecolor="none", handletextpad=0.4, labelspacing=0.4)
    save(fig, "ed_genetics_forest")


# --------------------------------------------------------------------------- ED: robustness
def ed_genetics_robust():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(F.W2, 3.1), layout="constrained")
    # (a) ancestry sensitivity: matched delta over joint, EUR vs all
    eur = _summ("std_matched_eur"); allc = _summ("std_matched_all")
    vals = [eur["matched_only_delta_prs_over_joint_mean"], allc["matched_only_delta_prs_over_joint_mean"]]
    axA.bar(["European\n(primary)", "All ancestries\n(sensitivity)"], vals,
            color=[C_FUSE, OK["skyblue"]], width=0.6)
    for i, v in enumerate(vals):
        axA.text(i, v + 0.001, f"+{v:.3f}", ha="center", fontsize=7.5)
    axA.axhline(0, color="#333333", lw=0.7)
    axA.set_ylabel("Matched mean Δ C-index"); axA.set_ylim(0, max(vals) * 1.25)
    F.panel(axA, "a")
    # (b) PRS-set within the Enhanced subgroup (same 19,647 people)
    st = _summ("std_matched_enhsub"); en = _summ("enh_matched_enhsub"); be = _summ("best_matched_enhsub")
    labs, vv = ["Standard", "Enhanced", "Best-available"], [
        st["matched_only_delta_prs_over_joint_mean"], en["matched_only_delta_prs_over_joint_mean"],
        be["matched_only_delta_prs_over_joint_mean"]]
    axB.bar(labs, vv, color=[C_WRIST, C_GEN, C_EMBEDDING], width=0.6)
    for i, v in enumerate(vv):
        axB.text(i, v + 0.0007, f"+{v:.3f}", ha="center", fontsize=7.5)
    axB.axhline(0, color="#333333", lw=0.7)
    axB.set_ylabel("Matched mean Δ C-index"); axB.set_ylim(0, max(vv) * 1.3)
    F.panel(axB, "b")
    save(fig, "ed_genetics_robust")


# --------------------------------------------------------------------------- ED: matched
def ed_genetics_matched():
    cc = json.load(open(CF / "concat_benchmark.json"))
    panel = _summ("std_panel_all")
    matched = _summ("std_matched_eur")["matched_only_delta_prs_over_joint_mean"]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(F.W2, 3.1), layout="constrained",
                                   gridspec_kw=dict(width_ratios=[1.25, 1.0]))
    # (a) global concat flat, but matched cardiometabolic diseases improve within it
    md = cc["matched_disease_deltas_within_concat"]
    order = ["type2_diabetes", "atrial_fibrillation", "heart_failure", "dementia",
             "alzheimers", "parkinsons"]
    names = {"type2_diabetes": "Type 2 diabetes", "atrial_fibrillation": "Atrial fibrillation",
             "heart_failure": "Heart failure", "dementia": "Dementia", "alzheimers": "Alzheimer's",
             "parkinsons": "Parkinson's"}
    vals = [md[k] for k in order]
    axA.barh(range(len(order)), vals, color=[C_FUSE if v > 0.005 else C_NULLp for v in vals])
    axA.set_yticks(range(len(order))); axA.set_yticklabels([names[k] for k in order], fontsize=7)
    axA.invert_yaxis(); axA.axvline(0, color="#333", lw=0.7)
    axA.set_xlabel("Δ C-index within concat model (vs wrist alone)")
    F.panel(axA, "a")
    # (b) matched vs panel-as-features
    pv = panel["delta_prs_over_joint"]["mean"]
    # 95% phecode-panel bootstrap CIs (B=1000; ed_bar_cis.py): matched n=31, full n=332
    _bar_cis = Path(__file__).resolve().parent / "ed_bar_cis.json"
    _gci = json.load(open(_bar_cis))["figures"]["genetics_matched"]
    _bvals = [matched, pv]; _bk = ["matched", "full"]
    assert all(abs(_bvals[i] - _gci[k]["point"]) < 5e-4 for i, k in enumerate(_bk)), "genetics_matched CI/bar mismatch"
    blo = [_bvals[i] - _gci[k]["ci_lo"] for i, k in enumerate(_bk)]
    bhi = [_gci[k]["ci_hi"] - _bvals[i] for i, k in enumerate(_bk)]
    axB.bar(["Matched\n(one PRS/disease)", "Full panel\n(all 36 PRS/disease)"], _bvals,
            color=[C_FUSE, OK["vermillion"]], width=0.6, zorder=2)
    axB.errorbar([0, 1], _bvals, yerr=[blo, bhi], fmt="none", ecolor="#222", elinewidth=1.0,
                 capsize=3, capthick=1.0, zorder=3)
    for i, k in enumerate(_bk):
        yy = _gci[k]["ci_hi"] + 0.004 if _bvals[i] > 0 else _gci[k]["ci_lo"] - 0.004
        axB.text(i, yy, f"{_bvals[i]:+.3f}", ha="center",
                 va="bottom" if _bvals[i] > 0 else "top", fontsize=7.5)
    axB.axhline(0, color="#333", lw=0.7)
    axB.set_ylim(_gci["full"]["ci_lo"] - 0.014, _gci["matched"]["ci_hi"] + 0.016)   # headroom for CI caps + labels
    axB.set_ylabel("Mean Δ C-index"); F.panel(axB, "b")
    save(fig, "ed_genetics_matched")


if __name__ == "__main__":
    print("building genetics figures...")
    fig_genetics_main()
    ed_genetics_forest()
    ed_genetics_robust()
    ed_genetics_matched()
    print("done.")
