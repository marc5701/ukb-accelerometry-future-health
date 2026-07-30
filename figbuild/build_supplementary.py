"""Supplementary confusion, screening-genetics, poor-fit, and PRS-nonlinearity figures.
Reads the per-analysis CSVs and renders Extended Data figures via figlib. Each figure is
guarded so one failure does not block the rest.
"""
import os
import json
import re
import textwrap
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import figlib as F
from figlib import OK, save, cat_color, cat_label, panel

UKB_ROOT = os.environ.get("UKB_ROOT", str(Path(__file__).resolve().parent.parent / "data"))

CONF = f"{UKB_ROOT}/confusion"
SCR = f"{UKB_ROOT}/screening_genetics"
POOR = f"{UKB_ROOT}/poor_fit"
PRSN = f"{UKB_ROOT}/prs_nonlinear"
CT = f"{UKB_ROOT}/confusion_targets"
FP = f"{UKB_ROOT}/confusion_fp"
DEFR = f"{UKB_ROOT}/false_positives_burden_adjusted"   # burden-adjusted false-positive analysis

WATCH = {"NS_324.11": "Parkinson's", "NS_328.1": "Dementia", "EM_202.2": "Type 2 diabetes",
         "CV_416.2": "Atrial fibrillation", "RE_474": "COPD", "MB_286.2": "Depression"}


# --------------------------------------------------------------------------- #
# Confusion / misdiagnosis structure
# --------------------------------------------------------------------------- #
def ed_confusion():
    Cc = np.load(f"{CONF}/confusion_category_matrix.npy")   # age/sex-adjusted (not PC1-removed)
    cats = json.load(open(f"{CONF}/confusion_categories.json"))
    disp = [cat_label(c) for c in cats]

    n = len(cats)
    # Single-panel figure: only the confusion matrix (the own-vs-other lift scatter was dropped).
    fig, axh = plt.subplots(1, 1, figsize=(5.2, 4.6), constrained_layout=True)
    # panel a: age/sex-adjusted confusion matrix. Each cell = the average risk PERCENTILE (0-100,
    # 50 = median participant) at which the model ranks row-category patients for column-category
    # diseases. Numbers shown; diagonal boxed = same-category confusion.
    V = (Cc + 0.5) * 100.0
    # Centre the diverging scale on the matrix mean (not 50): every cell is above the cohort
    # median because participants with any disease rank high across the board, so a 50-centred
    # scale wastes the blue half and washes everything to red. Centring on the mean with robust
    # limits makes the real between-category gradient visible; the printed number is the exact
    # percentile, so absolute values are never hidden by the colour choice.
    ctr = float(np.nanmean(V))
    half = float(np.nanpercentile(np.abs(V - ctr), 93))
    im = axh.imshow(V, cmap="RdBu_r", norm=TwoSlopeNorm(ctr, ctr - half, ctr + half), aspect="auto")
    for r in range(n):
        for c in range(n):
            v = V[r, c]
            if not np.isfinite(v):
                continue
            tc = "white" if abs(v - ctr) > 0.6 * half else "black"
            axh.text(c, r, f"{int(round(v))}", ha="center", va="center", fontsize=4.6, color=tc)
    axh.set_xticks(range(n)); axh.set_xticklabels(disp, rotation=45, ha="right",
                                                  rotation_mode="anchor", fontsize=5.8)
    axh.set_yticks(range(n)); axh.set_yticklabels(disp, fontsize=5.8)
    axh.set_xlabel("Model-flagged category", fontsize=7.5)
    axh.set_ylabel("True disease category", fontsize=7.5)
    for a in range(n):
        axh.add_patch(plt.Rectangle((a - 0.5, a - 0.5), 1, 1, fill=False, ec="k", lw=1.0))
    cb = fig.colorbar(im, ax=axh, fraction=0.045, pad=0.03)
    cb.set_label("Average risk percentile\n(colour centred on matrix mean)", fontsize=6.0)
    cb.ax.tick_params(labelsize=5)
    save(fig, "ed_confusion")


# --------------------------------------------------------------------------- #
# High-specificity screening x genetics
# --------------------------------------------------------------------------- #
def ed_screening_genetics():
    arms = pd.read_csv(f"{SCR}/screening_arms.csv")
    ds = pd.read_csv(f"{SCR}/screening_dsens.csv")
    fig, (axpd, axf) = plt.subplots(1, 2, figsize=(F.W2, 3.2), constrained_layout=True)

    # panel a: PD sensitivity by operating point, key arms
    pd_rows = arms[(arms.phecode == "NS_324.11") & (arms.framing == "incident")].set_index("arm")
    specs = [0.90, 0.95, 0.99]
    armlist = [("actig", "wrist", OK["blue"]), ("actig_prs", "wrist+PRS", OK["green"]),
               ("prs", "PRS only", OK["purple"]), ("clinical", "clinical", OK["vermillion"])]
    x = np.arange(len(specs)); w = 0.2
    for i, (a, lab, col) in enumerate(armlist):
        present = a in pd_rows.index
        vals = np.array([pd_rows.loc[a, f"sens@{s}"] if present else np.nan for s in specs], float)
        lo = np.array([pd_rows.loc[a, f"sens_lo@{s}"] if present else np.nan for s in specs], float)
        hi = np.array([pd_rows.loc[a, f"sens_hi@{s}"] if present else np.nan for s in specs], float)
        yerr = np.vstack([np.clip(vals - lo, 0, None), np.clip(hi - vals, 0, None)])
        xs = x + (i - 1.5) * w
        axpd.bar(xs, vals, w, label=lab, color=col, zorder=2)
        # 95% subject-bootstrap CIs (only 17 incident PD cases -> intervals are wide by construction)
        axpd.errorbar(xs, vals, yerr=yerr, fmt="none", ecolor="#333", elinewidth=0.8,
                      capsize=1.8, capthick=0.8, zorder=3)
    axpd.set_xticks(x); axpd.set_xticklabels([f"{int(s*100)}%\nspec" for s in specs], fontsize=6.5)
    axpd.set_ylabel("sensitivity (incident PD, 6 yr)", fontsize=7)
    axpd.set_ylim(0, 1.03); axpd.legend(fontsize=5.6, frameon=False, loc="upper right", ncol=2)
    panel(axpd, "a")

    # panel b: per-disease Delta-sens@95% from +PRS over wrist (forest)
    dd = ds[(ds.base == "wrist") & (ds.spec == 0.95) & (ds.framing == "incident") & ds.dsens.notna()].copy()
    dd = dd.sort_values("dsens")
    y = np.arange(len(dd))
    sig = (dd.lo > 0).to_numpy()
    axf.errorbar(dd.dsens, y, xerr=[dd.dsens - dd.lo, dd.hi - dd.dsens], fmt="none",
                 ecolor="#bbb", lw=0.8, capsize=0)
    axf.scatter(dd.dsens, y, s=10, c=np.where(sig, OK["green"], OK["grey"]), zorder=3)
    axf.axvline(0, color="k", lw=0.8, ls="--")
    pm = F.phecode_map()["PhecodeString"]
    lbls = [_clean_name(pm.get(p, t.replace("_", " "))) for p, t in zip(dd.phecode, dd.trait)]
    axf.set_yticks(y); axf.set_yticklabels(lbls, fontsize=5.0)
    axf.set_ylim(-0.8, len(dd) - 0.2)
    axf.set_xlabel("Δ sensitivity at 95% spec (+PRS over wrist)", fontsize=7)
    leg = [Line2D([0], [0], marker="o", linestyle="none", markersize=4, markeredgecolor="none",
                  markerfacecolor=OK["green"], label="95% CI excludes 0"),
           Line2D([0], [0], marker="o", linestyle="none", markersize=4, markeredgecolor="none",
                  markerfacecolor=OK["grey"], label="CI includes 0")]
    axf.legend(handles=leg, fontsize=5.2, frameon=False, loc="lower right",
               handletextpad=0.3, borderpad=0.2, labelspacing=0.3,
               title=f"{int(sig.sum())} of {len(dd)} outcomes significant", title_fontsize=5.8)
    panel(axf, "b")
    save(fig, "ed_screening_genetics")


# --------------------------------------------------------------------------- #
# Poorly-fit individuals vs PRS extremes
# --------------------------------------------------------------------------- #
def ed_poor_fit():
    dec = pd.read_csv(f"{POOR}/poor_fit_decile_curve.csv")
    per = pd.read_csv(f"{POOR}/poor_fit_perdisease.csv")
    fig, (axc, axf) = plt.subplots(1, 2, figsize=(F.W2, 3.0), constrained_layout=True)

    # panel a: pooled miss-rate vs PRS decile (mean +/- SE across diseases)
    g = dec.groupby("prs_decile")["miss_rate"]
    mean = g.mean(); se = g.std() / np.sqrt(g.count())
    axc.axhline(0.5, color="k", lw=0.8, ls="--")
    axc.errorbar(mean.index, mean.values, yerr=se.values, fmt="o", color=OK["blue"],
                 ms=3, lw=1.0, capsize=2)
    zc = np.polyfit(mean.index.to_numpy(float), mean.values, 1)      # near-flat trend
    axc.plot(mean.index, np.polyval(zc, mean.index), color=OK["vermillion"], lw=1.2,
             label=f"slope {zc[0]:+.4f}/decile")
    axc.legend(fontsize=5.6, frameon=False, loc="lower right")
    axc.set_xlabel("matched-PRS decile (of true cases)", fontsize=7)
    axc.set_ylabel("miss rate (case in bottom-half risk)", fontsize=7)
    axc.set_ylim(0.3, 0.7); axc.set_xticks(range(0, 10, 2))
    panel(axc, "a")

    # panel b: PRS(missed)-PRS(caught) per disease (forest, centered on 0)
    dd = per[per.prs_missed_minus_caught.notna()].sort_values("prs_missed_minus_caught")
    y = np.arange(len(dd))
    axf.errorbar(dd.prs_missed_minus_caught, y,
                 xerr=[dd.prs_missed_minus_caught - dd.missed_caught_lo,
                       dd.missed_caught_hi - dd.prs_missed_minus_caught],
                 fmt="none", ecolor="#bbb", lw=0.8)
    axf.scatter(dd.prs_missed_minus_caught, y, s=9, c=OK["grey"], zorder=3)
    axf.axvline(0, color="k", lw=0.8, ls="--")
    pm = F.phecode_map()["PhecodeString"]
    lbls = [_clean_name(pm.get(p, t.replace("_", " "))) for p, t in zip(dd.phecode, dd.trait)]
    axf.set_yticks(y); axf.set_yticklabels(lbls, fontsize=5.0)
    axf.set_xlabel("PRS(missed) − PRS(caught), SD", fontsize=7)
    axf.text(0.97, 0.04, "0/24 CI>0", transform=axf.transAxes, ha="right", va="bottom", fontsize=6.4)
    panel(axf, "b")
    save(fig, "ed_poorfit")


# --------------------------------------------------------------------------- #
# PRS non-linearity
# --------------------------------------------------------------------------- #
def ed_prs_nonlinear():
    dec = pd.read_csv(f"{PRSN}/prs_decile_curve.csv")
    per = pd.read_csv(f"{PRSN}/prs_nonlinear_perdisease.csv")
    fig, (axd, axp) = plt.subplots(1, 2, figsize=(F.W2, 3.0), constrained_layout=True)

    # panel a: decile log-HR curve for a few well-powered exemplars
    exemplars = per.sort_values("n_ev", ascending=False).head(4)
    for _, r in exemplars.iterrows():
        d = dec[dec.phecode == r.phecode].sort_values("decile")
        axd.plot(d.decile, d.logHR, "o-", ms=2.5, lw=1.0,
                 label=f"{r.trait.replace('_',' ')} (n={int(r.n_ev)})")
    axd.axhline(0, color="k", lw=0.8, ls="--")
    axd.set_xlabel("matched-PRS decile", fontsize=7)
    axd.set_ylabel("log hazard ratio vs middle decile", fontsize=7)
    axd.legend(fontsize=5.2, frameon=False, loc="upper left")
    panel(axd, "a")

    # panel b: RCS-vs-linear ΔC (fusion) per disease + LR p (none significant)
    dd = per.sort_values("dC_fusion")
    y = np.arange(len(dd))
    axp.scatter(dd.dC_fusion, y, s=10, c=OK["grey"], zorder=3)
    axp.axvline(0, color="k", lw=0.8, ls="--")
    axp.set_yticks([]); axp.set_ylabel(f"matched diseases (n={len(dd)})", fontsize=7)
    axp.set_xlabel("ΔC-index, RCS − linear (fusion)", fontsize=7)
    panel(axp, "b")
    save(fig, "ed_prs_nonlinear")


# Faithful short display names for a couple of over-long phecode strings, so every y-label
# wraps to at most two lines and never needs "..." truncation.
_NAME_FIX = {
    "Complications and ill-defined descriptions of heart disease": "Ill-defined heart disease",
    "Nonspecific abnormal results of function study of liver": "Abnormal liver function study",
    "Elevated blood pressure reading without diagnosis of hypertension": "Elevated blood pressure reading",
}


def _clean_name(s):
    """Readable disease label: drop a trailing bracketed synonym or phecode asterisk and
    shorten a couple of over-long strings, so labels wrap cleanly instead of truncating."""
    s = str(s).strip()
    if s in _NAME_FIX:
        return _NAME_FIX[s]
    s = re.sub(r"\s*[\[(][^\])]*[\])]\s*$", "", s)   # drop a trailing [synonym] or (abbrev)
    s = s.rstrip("*").strip()
    return _NAME_FIX.get(s, s)


def _wrap_label(s, width=28):
    """Full disease name wrapped to at most two lines (no ellipsis)."""
    return "\n".join(textwrap.wrap(_clean_name(s), width=width) or [_clean_name(s)])


def _short(s, n=23):
    s = str(s)
    return s if len(s) <= n else s[:n - 1] + "…"


# --------------------------------------------------------------------------- #
# Per-disease false-positive / misdiagnosis profiles (10 diseases)
# --------------------------------------------------------------------------- #
def ed_confusion_targets():
    # False-positive / misdiagnosis view: for each disease, among the participants the model FLAGS
    # for it (95% specificity) but who do NOT have it, what do they actually have, corrected for
    # prevalence (enrichment vs cohort base rate). Panel title carries the precision context
    # (PPV and false positives per 100 flagged).
    df = pd.read_csv(f"{FP}/confusion_fp.csv")
    summ = pd.read_csv(f"{FP}/confusion_fp_summary.csv").set_index("target")
    targets = list(dict.fromkeys(df.target))          # preserve order
    fig, axes = plt.subplots(5, 2, figsize=(F.W2, 7.4), constrained_layout=True)
    axes = axes.ravel()
    letters = "abcdefghij"
    for i, (tg, ax) in enumerate(zip(targets, axes)):
        sub = df[df.target == tg].sort_values("lo")           # ascending -> most robust (highest lower CI) at top
        y = np.arange(len(sub))
        cols = [cat_color(c) for c in sub.actual_category]
        ax.barh(y, sub.enrichment, xerr=[sub.enrichment - sub.lo, sub.hi - sub.enrichment],
                color=cols, height=0.72, error_kw=dict(lw=0.6, ecolor="#777"))
        ax.axvline(1.0, color="k", lw=0.7, ls="--")           # enrichment = 1 (cohort base rate)
        xmax = max(2.0, float(sub.hi.max()) * 1.18)           # headroom for the per-bar % label
        for yi, (h, pc) in enumerate(zip(sub.hi.to_numpy(), sub.pct_of_fp.to_numpy())):
            ax.text(h + 0.012 * xmax, yi, f"{pc:.0f}%", ha="left", va="center",
                    fontsize=4.3, color="#333")               # share of false positives with the condition
        ax.set_yticks(y)
        ax.set_yticklabels([_wrap_label(nm) for nm in sub.actual_name], fontsize=4.9)
        ax.set_xlim(0, xmax)
        ax.set_ylim(-0.7, len(sub) - 0.3 + 1.5)               # white headroom for the 2-line title
        ax.tick_params(axis="x", labelsize=5)
        r = summ.loc[tg]
        # panel title sits in the white headroom above the bars (white bbox as insurance).
        # UK Biobank small-numbers policy: when a disease has fewer than five true cases at this
        # threshold, showing PPV and FP/100 alongside the flagged total would let a reader recover
        # the <5 true-positive count (flagged x PPV), so we suppress the precision for those panels.
        if int(r.n_tp) < 5:
            _title = f"{tg}\nflagged {int(r.n_flagged)}; precision not shown (fewer than five true cases)"
        else:
            _title = (f"{tg}\nflagged {int(r.n_flagged)}, PPV {100*r.ppv:.0f}%, "
                      f"~{round(r.fp_per_100)} FP/100")
        ax.text(0.98, 0.98, _title, transform=ax.transAxes, ha="right", va="top",
                fontsize=5.5, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", boxstyle="square,pad=0.2"))
        panel(ax, letters[i])
    fig.supxlabel("enrichment over cohort prevalence (fold)", fontsize=7)
    save(fig, "ed_confusion_targets")


# --------------------------------------------------------------------------- #
# Frailty-adjusted false-positive / misdiagnosis
# --------------------------------------------------------------------------- #
_DEFR_SHORT = {
    "Parkinson's disease (Primary)": "Parkinson's disease",
    "Chronic obstructive pulmonary disease [COPD]": "COPD",
    "Atrioventricular block, first degree": "AV block (1st degree)",
    "Altered mental status, unspecified": "Altered mental status",
    "Peripheral vascular disease NOS [includes PAD]": "Peripheral vascular disease",
    "Cerebral infarction [Ischemic stroke]": "Ischemic stroke",
    "Current tobacco use and nicotine dependence": "Tobacco use",
    "Phlebitis and thrombophlebitis": "Phlebitis",
    "Urinary incontinence and enuresis": "Urinary incontinence",
    "Acute lower respiratory infection": "Acute resp. infection",
    "Major depressive disorder": "Depression", "Anxiety and anxiety disorders": "Anxiety",
    "Abnormality of gait and mobility": "Gait abnormality",
    "Other cerebrovascular disease": "Cerebrovascular disease",
    "Abnormal findings on diagnostic imaging of the digestive tract": "Abnormal GI imaging",
    "Dizziness and giddiness": "Dizziness", "Repeated falls*": "Repeated falls",
    "Paroxysmal atrial fibrillation*": "Paroxysmal AF",
}


def _defr_wrap(s, n=22):
    s = _DEFR_SHORT.get(str(s), str(s))
    return s if len(s) <= n else s[:n - 1] + "…"


def ed_confusion_targets_burden_adjusted():
    # Frailty-adjusted misdiagnosis: for each disease, among the false positives at 95% specificity,
    # what do they actually have RELATIVE TO matched controls of the same prevalent-comorbidity burden
    # (E*), with the raw unadjusted enrichment shown as a grey tick. Panels k,l are the
    # pre-registered PD/AD prodrome forest.
    bars = pd.read_csv(f"{DEFR}/burden_adjusted_bars.csv")
    prod = pd.read_csv(f"{DEFR}/prodrome_check.csv")
    summ = {t["target"]: t for t in json.load(open(f"{DEFR}/summary.json"))["targets"]}
    pooled = bars[bars.framing == "pooled"]
    targets = list(dict.fromkeys(bars.target))
    fig, axes = plt.subplots(6, 2, figsize=(F.W2, 6.85), constrained_layout=True)
    axes = axes.ravel()
    letters = "abcdefghijkl"
    for i, tg in enumerate(targets):
        ax = axes[i]
        sub = pooled[pooled.target == tg].sort_values("e_star")
        y = np.arange(len(sub)); es = sub.e_star.to_numpy()
        ax.barh(y, es, xerr=[np.clip(es - sub.es_lo.to_numpy(), 0, None),
                             np.clip(sub.es_hi.to_numpy() - es, 0, None)],
                color=[cat_color(c) for c in sub.actual_category], height=0.70,
                error_kw=dict(lw=0.6, ecolor="#777"))
        ax.plot(sub.raw_enr.to_numpy(), y, marker="|", ls="none", ms=6, mec="#222", mew=0.9)
        ax.axvline(1.0, color="k", lw=0.7, ls="--")
        xmax = max(2.0, float(es.max()) * 1.28)
        for yi, (e, pc) in enumerate(zip(es, sub.pct_of_fp.to_numpy())):
            ax.text(min(e, xmax) + 0.015 * xmax, yi, f"{pc:.0f}%", ha="left", va="center",
                    fontsize=4.3, color="#333")
        ax.set_yticks(y); ax.set_yticklabels([_defr_wrap(nm) for nm in sub.actual_name], fontsize=4.9)
        ax.set_xlim(0, xmax); ax.set_ylim(-0.7, len(sub) - 0.3 + 1.4); ax.tick_params(axis="x", labelsize=5)
        r = summ[tg]
        ax.text(0.98, 0.98, f"{tg}\nflagged {int(r['n_flagged'])}, PPV {100*r['ppv']:.0f}%, "
                f"~{round(r['fp_per_100'])} FP/100", transform=ax.transAxes, ha="right", va="top",
                fontsize=5.4, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", boxstyle="square,pad=0.2"))
        panel(ax, letters[i])
    for j, tg in enumerate(["Parkinson's disease", "Alzheimer's disease"]):
        ax = axes[10 + j]
        sub = prod[(prod.target == tg) & (prod.framing == "pooled")].iloc[::-1].reset_index(drop=True)
        y = np.arange(len(sub)); pw = sub[~sub.underpowered]
        xmax = max(3.0, float((pw.es_hi.max() if len(pw) else sub.es_hi.max())) * 1.12)
        for yi, row in sub.iterrows():
            up = bool(row.underpowered); col = cat_color(row.actual_category)
            filled = bool(row.get("or_fdr_sig", False)) and not up   # survives age+sex+burden OR at FDR<0.10
            ax.errorbar(row.e_star, yi, xerr=[[max(row.e_star - row.es_lo, 0)],
                        [max(min(row.es_hi, xmax) - row.e_star, 0)]], fmt="o", ms=4.2, color=col,
                        mfc=(col if filled else "white"), mec=col, mew=1.0, ecolor="#888",
                        elinewidth=0.7, capsize=1.6)
            if up:
                ax.text(min(row.e_star + 0.04 * xmax, xmax * 0.92), yi, f"n={int(row.cnt_fp)}",
                        fontsize=4.1, va="center", ha="left", color="#999")
        ax.axvline(1.0, color="k", lw=0.7, ls="--")
        ax.set_yticks(y); ax.set_yticklabels([_defr_wrap(nm, 26) for nm in sub.actual_name], fontsize=4.9)
        ax.set_xlim(0, xmax); ax.set_ylim(-0.7, len(sub) - 0.3 + 1.4); ax.tick_params(axis="x", labelsize=5)
        ax.text(0.98, 0.98, f"{tg}\npre-registered prodrome", transform=ax.transAxes, ha="right",
                va="top", fontsize=5.4, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", boxstyle="square,pad=0.2"))
        panel(ax, letters[10 + j])
    fig.supxlabel("enrichment $E^{*}$ (fold)", fontsize=7)
    save(fig, "ed_confusion_targets_defrail")


if __name__ == "__main__":
    for fn in (ed_confusion, ed_confusion_targets, ed_confusion_targets_burden_adjusted,
               ed_screening_genetics, ed_poor_fit, ed_prs_nonlinear):
        try:
            fn()
        except Exception:
            print(f"[FAIL] {fn.__name__}"); traceback.print_exc()
