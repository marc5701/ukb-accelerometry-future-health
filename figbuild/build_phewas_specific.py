"""Extended Data PheWAS figure: disease-specific discrimination beyond the shared axis.

Phenome-wide companion to Fig 2 panel c, backing the claim that the shared actigraphic
axis carries general health and, on top of it, disease-specific signatures are encoded.

One uniform inclusion rule: every outcome with at least 10 incident test cases. For each
outcome it plots the gain in concordance from adding the disease-specific hazard on top of
the shared actigraphic axis (delta C = C_full - C_general, the "beyond the shared axis"
increment used throughout Results Section 2 and Fig 2c). Point size scales with incident-case
count so lower-power outcomes are visibly down-weighted without being hidden. A dashed line
marks delta C = 0.05; outcomes above it are drawn in organ-system colour, the rest muted.
A handful of central outcomes are labelled.

Reads only the frozen analysis output (Beyond_sharc_ge10); no re-computation.
Output: draft/figures/fig_phewas_specific.pdf  +  figbuild/qa/fig_phewas_specific.png
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import figlib as F
from figlib import OK, save, W2

UKB_ROOT = os.environ.get("UKB_ROOT", "data")
CONF = Path(f"{UKB_ROOT}/paper_experiments/output"
            "/Beyond_sharc_ge10/per_disease_residual_c.csv")

THRESH = 0.05
# narrative outcomes to label (leader lines). No exception logic, just labelled points.
LABELS = {
    "NS_324.11": "Parkinson’s", "NS_328.11": "Alzheimer’s", "NS_328.1": "Dementia",
    "CV_424": "Heart failure", "CV_416.2": "Atrial fibrillation",
    "RE_474": "COPD", "RE_468": "Pneumonia",
}
OFFSETS = {  # (dx pt, dy pt, ha) declutter tight clusters + lift labels off dense bubbles
    "NS_324.11": (12, 9, "left"),     # Parkinson's up-right
    "NS_328.11": (-12, 4, "right"),   # Alzheimer's up-left
    "NS_328.1":  (-3, 23, "right"),   # Dementia lifted up-left into the empty band, off its own whisker
    "CV_424":    (-11, 9, "right"),   # Heart failure up-left, short leader
    "CV_416.2":  (34, 20, "left"),    # Atrial fibrillation to the right, in the CV/neoplasms gap
    "RE_468":    (0, 20, "center"),   # Pneumonia straight up, higher
    "RE_474":    (22, 6, "left"),     # COPD to the right of its bubble
}


def _msize(n):
    # marker AREA proportional to case count (radius ~ sqrt(n)); low-power outcomes render small.
    return float(np.clip(0.10 * n + 3.0, 4.0, 46.0))


def build():
    d = pd.read_csv(CONF)
    assert (d.n_ev >= 10).all(), "expected all rows >=10 events"
    n_total = len(d)
    n_above = int((d.delta_incr > THRESH).sum())
    n_wp_above = int(((d.delta_incr > THRESH) & (d.n_ev >= 50)).sum())

    cat_mean = d.groupby("category")["delta_incr"].mean().sort_values()
    cat_order = list(cat_mean.index)

    xpos, rows, cat_spans = {}, [], []
    x, GAP = 0.0, 3.6
    for c in cat_order:
        sub = d[d.category == c].sort_values("delta_incr")
        x0 = x
        for _, r in sub.iterrows():
            xpos[r.phecode] = x
            rows.append((x, r.phecode, r.category, r.delta_incr, r.n_ev))
            x += 1.0
        cat_spans.append((c, x0 - 0.5, x - 0.5))
        x += GAP
    xmax = x - GAP

    fig = plt.figure(figsize=(W2, 4.05))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.12])
    ax = fig.add_subplot(gs[0])
    axleg = fig.add_subplot(gs[1]); axleg.axis("off")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.975, bottom=0.05, hspace=0.14)

    # alternating faint bands mark the organ-system groups (identity is given by the colour key below)
    for i, (c, a, b) in enumerate(cat_spans):
        if i % 2 == 0:
            ax.axvspan(a, b, color="#f5f5f5", zorder=0)

    ax.axhline(THRESH, ls="--", color="#555555", lw=0.9, zorder=1)
    ax.axhline(0.0, ls=":", color=OK["grey"], lw=0.8, zorder=1)
    ax.text(-1.5, THRESH + 0.004, "$\\Delta$C = 0.05", ha="left", va="bottom", fontsize=5.6,
            color="#555555")

    for xx, ph, cat, dc, nev in rows:
        if dc > THRESH:
            ax.scatter(xx, dc, s=_msize(nev), color=F.cat_color(cat), edgecolor="white",
                       linewidth=0.2, zorder=3, alpha=0.85)
        else:
            ax.scatter(xx, dc, s=_msize(nev) * 0.55, color="#cccccc", edgecolor="none", zorder=2)

    for _, r in d.iterrows():
        if r.phecode not in LABELS:
            continue
        xx, yy = xpos[r.phecode], r.delta_incr
        dx, dy, ha = OFFSETS.get(r.phecode, (0, 9, "center"))
        col = F.cat_color(r.category)
        # 95% CI on the labelled outcomes (ci_lo/ci_hi, same subject bootstrap as the point)
        ax.errorbar(xx, yy, yerr=[[yy - float(r.ci_lo)], [float(r.ci_hi) - yy]], fmt="none",
                    ecolor=col, elinewidth=0.8, capsize=1.6, capthick=0.8, alpha=0.9, zorder=5)
        # hollow ring around each labelled dot (matches the labelled-leader ring style used elsewhere) so the leader's target is clear
        ax.scatter([xx], [yy], s=_msize(r.n_ev) + 28, facecolor="none", edgecolor=col,
                   linewidth=0.9, zorder=6)
        ax.annotate(f"{LABELS[r.phecode]} +{r.delta_incr:.2f}", (xx, yy), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va="bottom", fontsize=5.6, color=col,
                    fontweight=("bold" if r.phecode == "NS_324.11" else "normal"),
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=col, shrinkA=0, shrinkB=4.5))

    ax.set_xlim(-2.0, xmax + 1.0)
    ax.set_ylim(-0.06, 0.40)   # headroom for the tall small-n CIs (Alzheimer's upper bound 0.36)
    ax.set_xticks([])
    ax.set_ylabel("Gain in concordance beyond the shared axis ($\\Delta$C)")
    ax.set_xlabel("Outcomes grouped and coloured by organ system", labelpad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.lines import Line2D
    key = [Line2D([0], [0], marker="o", color="none", markerfacecolor="#888",
                  markeredgecolor="white", markersize=np.sqrt(_msize(n)), label=f"{n} cases")
           for n in (25, 100, 400)]
    key.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="#cccccc",
                      markeredgecolor="none", markersize=3.5, label="$\\Delta$C $\\leq$ 0.05"))
    ax.legend(handles=key, loc="upper left", fontsize=5.6, frameon=False,
              handletextpad=0.5, labelspacing=0.55, borderaxespad=0.2)

    # disease-category colour key (boxed, own row below the plot; matches the boxed category-key style used elsewhere)
    cat_handles = [Line2D([0], [0], marker="o", ls="", mfc=F.cat_color(c), mec="none", ms=4.5,
                          label=F.cat_label(c)) for c in cat_order]
    axleg.legend(handles=cat_handles, loc="center", ncol=7, fontsize=6.0, frameon=True,
                 facecolor="white", edgecolor="#bbbbbb", framealpha=1.0, handletextpad=0.3,
                 columnspacing=1.0, labelspacing=0.6, borderaxespad=0.1)

    save(fig, "fig_phewas_specific")

    sd = d.copy()
    sd["above_0p05"] = sd.delta_incr > THRESH
    sd["labelled"] = sd.phecode.isin(LABELS)
    sd[["phecode", "label", "category", "n_ev", "delta_incr", "ci_lo", "ci_hi",
        "above_0p05", "labelled"]].to_csv(
        F.DRAFT_FIGS.parent / "fig_phewas_specific_source_data.csv", index=False)
    print(f"  {n_total} outcomes (>=10 cases); {n_above} with deltaC>0.05; "
          f"{n_wp_above} of the 101 well-powered (>=50) exceed 0.05")


if __name__ == "__main__":
    build()
    print("done: fig_phewas_specific (ge10)")
