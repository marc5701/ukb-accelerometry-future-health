"""Figure 1 bottom row (panels b, c, d) with 95% bootstrap CI whiskers, rendered as
individual panels (PNG + SVG) plus a stitched composite for QA.

Panels reuse the scatter/bar/point layout of build_paper_experiments.fig1_overview.
The additions are (i) numbered rings plus a top-left key instead of name leader lines,
and (ii) 95% CI whiskers from the bootstrap CI files:
  panel b: per_disease_c_ci.json  (whiskers on the 6 highlighted outcomes)
  panel c: per_category_c_ci.csv   (whiskers on the 14 category means)
  panel d: per_region_c_ci.csv     (whiskers on the 5 regions, both methods)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

import figlib as F                       # applies nature.mplstyle on import
from figlib import OK

OUT = F.QA / "phewas_bio"
OUT.mkdir(parents=True, exist_ok=True)

# 6 highlighted outcomes, numbered by descending concordance.
HL = [
    ("NS_324.11",     "Parkinson's",   1),
    ("CV_424",        "Heart failure", 2),
    ("NS_328.11",     "Alzheimer's",   3),
    ("EM_236.1",      "Obesity",       4),
    ("NS_328.1",      "Dementia",      5),
    ("time_to_death", "Mortality",     6),
]
ECOL = "#333"          # shared error-bar colour
CAP, ELW = 2, 0.8      # shared capsize / error-bar line width

REGION_ORDER = ["North England", "South England", "Central England", "Scotland", "Wales"]
REGION_SHORT = ["North", "South", "Central", "Scot.", "Wales"]


def _shared_df():
    df = F.load_per_disease().copy()
    catmean = df.groupby("PhecodeCategory").c.mean().sort_values()
    cat_order = list(catmean.index)
    df["catrank"] = df.PhecodeCategory.map({c: i for i, c in enumerate(cat_order)})
    df = df.sort_values(["catrank", "c"]).reset_index(drop=True)
    df["x"] = np.arange(len(df))
    return df, catmean, cat_order


def draw_panel_b(axb, axleg, df, cat_order, dci, xlabel_fs=7.6):
    """Manhattan scatter + numbered rings/key + 95% CI whiskers on the highlights."""
    N = len(df)
    for cat in cat_order:
        sub = df[df.PhecodeCategory == cat]
        axb.scatter(sub.x, sub.c, s=11, color=F.cat_color(cat), edgecolor="none", alpha=0.85)
    axb.axhline(0.5, color="#999", ls=":", lw=0.9)

    for ph, _short, num in HL:
        r = df[df.phecode == ph]
        if not len(r):
            continue
        hx, hc = int(r.iloc[0].x), float(r.iloc[0].c)
        rec = dci.get(ph)
        if rec is not None:
            lo, hi = float(rec["CI_lo"]), float(rec["CI_hi"])
            axb.errorbar([hx], [hc], yerr=[[hc - lo], [hi - hc]], fmt="none",
                         ecolor=ECOL, elinewidth=ELW, capsize=CAP, capthick=ELW, zorder=5)
        axb.scatter([hx], [hc], s=44, facecolor="none", edgecolor="#111", linewidth=1.2, zorder=6)
        axb.annotate(str(num), xy=(hx, hc), xytext=(6, 5), textcoords="offset points",
                     fontsize=7.2, fontweight="bold", color="#111", zorder=7)

    # top-left numbered key
    key = "\n".join(f"{num}  {short}" for _ph, short, num in HL)
    axb.text(0.015, 0.975, key, transform=axb.transAxes, va="top", ha="left",
             fontsize=6.4, linespacing=1.35, color="#111")

    axb.set_xticks([]); axb.set_xlim(-3, N + 3); axb.set_ylim(0.45, 0.995)
    axb.set_xlabel("388 incident outcomes, grouped by disease category\nand ranked by concordance within each",
                   fontsize=xlabel_fs)
    axb.set_ylabel("Concordance (6-year incident)")

    # disease-category colour key (horizontal legend below the scatter)
    axleg.axis("off")
    handles = [Line2D([0], [0], marker="o", ls="", mfc=F.cat_color(c), mec="none", ms=4.5,
                      label=F.cat_label(c)) for c in reversed(cat_order)]
    axleg.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                 ncol=4, fontsize=6.0, frameon=True, facecolor="white", edgecolor="#bbbbbb",
                 framealpha=1.0, handletextpad=0.3, columnspacing=1.0, labelspacing=0.5,
                 borderaxespad=0.0)


def draw_panel_c(axc, catmean, cci):
    """Category mean-C bars + horizontal 95% CI whiskers."""
    cats = list(catmean.index)
    vals = catmean.values
    axc.barh(range(len(cats)), vals, color=[F.cat_color(c) for c in cats],
             edgecolor="white", linewidth=0.3, zorder=2)
    los, his = [], []
    for i, c in enumerate(cats):
        v = float(catmean[c])
        if c in cci.index:
            lo, hi = float(cci.loc[c, "CI_lo"]), float(cci.loc[c, "CI_hi"])
        else:
            lo = hi = v
        los.append(lo); his.append(hi)
        axc.errorbar(v, i, xerr=[[max(0.0, v - lo)], [max(0.0, hi - v)]], fmt="none",
                     ecolor=ECOL, elinewidth=ELW, capsize=CAP, capthick=ELW, zorder=4)
    axc.set_yticks(range(len(cats)))
    axc.set_yticklabels([F.cat_label(c) for c in cats], fontsize=6.0)
    axc.axvline(0.5, color="#999", ls=":", lw=0.8, zorder=1)
    axc.set_xlim(0.5, max(0.76, max(his) + 0.01))
    axc.set_ylim(-0.7, len(cats) - 0.3)
    axc.set_xlabel("Mean concordance")


def draw_panel_d(axd, reg):
    """Per-region deployed vs leave-region-out + 95% CI whiskers + pooled reference lines."""
    dep = reg[(reg.method == "deployed") & (reg.region != "POOLED")].set_index("region").reindex(REGION_ORDER)
    lro = reg[(reg.method == "leave_region_out") & (reg.region != "POOLED")].set_index("region").reindex(REGION_ORDER)
    x = np.arange(len(REGION_ORDER))
    dodge = 0.09    # small horizontal offset so the two series' whiskers do not overlap

    def yerr(frame):
        lo = frame.mean_C.to_numpy() - frame.mean_C_lo.to_numpy()
        hi = frame.mean_C_hi.to_numpy() - frame.mean_C.to_numpy()
        return np.vstack([np.nan_to_num(lo, nan=0.0), np.nan_to_num(hi, nan=0.0)])

    axd.errorbar(x - dodge, dep.mean_C, yerr=yerr(dep), fmt="o", ms=5.2, color=OK["blue"],
                 ecolor=OK["blue"], elinewidth=ELW, capsize=CAP, capthick=ELW, zorder=5, label="Deployed")
    axd.errorbar(x + dodge, lro.mean_C, yerr=yerr(lro), fmt="s", ms=4.8, mfc="none", mec=OK["grey"],
                 mew=1.2, ecolor=OK["grey"], elinewidth=ELW, capsize=CAP, capthick=ELW, zorder=5,
                 label="Leave-region-out")

    dpool = float(reg[(reg.method == "deployed") & (reg.region == "POOLED")].mean_C.iloc[0])
    lpool = float(reg[(reg.method == "leave_region_out") & (reg.region == "POOLED")].mean_C.iloc[0])
    axd.axhline(dpool, color=OK["blue"], lw=0.9, ls="-", alpha=0.6, zorder=1)
    axd.axhline(lpool, color=OK["grey"], lw=0.9, ls="--", alpha=0.7, zorder=1)

    axd.set_xticks(x); axd.set_xticklabels(REGION_SHORT, fontsize=6.4)
    axd.set_xlim(-0.5, len(REGION_ORDER) - 0.5)
    lo_all = np.nanmin([dep.mean_C_lo.min(), lro.mean_C_lo.min()])
    hi_all = np.nanmax([dep.mean_C_hi.max(), lro.mean_C_hi.max()])
    axd.set_ylim(lo_all - 0.012, hi_all + 0.012)
    axd.set_ylabel("Mean concordance")
    n_wales = int(dep.loc["Wales", "n_subjects"]) if "n_subjects" in dep.columns else 193
    axd.text(len(REGION_ORDER) - 1, float(dep.mean_C_lo.iloc[-1]) - 0.006, f"n={n_wales}",
             fontsize=5.4, color="#888", ha="center", va="top")
    axd.legend(fontsize=5.8, loc="upper center", ncol=2, frameon=False,
               handletextpad=0.3, columnspacing=1.0)


def _save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=300, facecolor="white", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.png + {name}.svg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ci-dir", default=str(F.BENCH),
                    help="dir holding per_disease_c_ci.json / per_category_c_ci.csv / per_region_c_ci.csv")
    args = ap.parse_args()
    cd = args.ci_dir

    dci = json.load(open(os.path.join(cd, "per_disease_c_ci.json")))["diseases"]
    cci = pd.read_csv(os.path.join(cd, "per_category_c_ci.csv")).set_index("category")
    reg = pd.read_csv(os.path.join(cd, "per_region_c_ci.csv"))

    df, catmean, cat_order = _shared_df()

    # Panel figsizes chosen so the stitched strip's aspect is about 3.751 (b:c:d
    # widths roughly 1181:755:896 at a common height). A final exact resize then hits
    # 2832x755 with negligible (under 1%) scaling.
    # panel b (scatter + category legend below)
    figb = plt.figure(figsize=(5.5, 3.7))
    gsb = GridSpec(2, 1, height_ratios=[2.0, 0.46], hspace=0.32, figure=figb)
    draw_panel_b(figb.add_subplot(gsb[0, 0]), figb.add_subplot(gsb[1, 0]), df, cat_order, dci)
    _save(figb, "row_phewas")

    # panel c
    figc, axc = plt.subplots(figsize=(3.35, 3.7))
    draw_panel_c(axc, catmean, cci)
    _save(figc, "row_fig1c")

    # panel d
    figd, axd = plt.subplots(figsize=(4.0, 3.7))
    draw_panel_d(axd, reg)
    _save(figd, "row_fig1d")

    # Composite = the three self-contained panels stitched (no cross-panel label overlap),
    # then resized to exactly the original slot (2832 x 755). Panel figsizes above make the
    # strip's native aspect about 3.75, so this resize is nearly uniform and does not visibly
    # distort the markers.
    TARGET = (2832, 755)
    from PIL import Image
    ims = [Image.open(OUT / f"{n}.png") for n in ("row_phewas", "row_fig1c", "row_fig1d")]
    h = min(im.height for im in ims)
    ims = [im.resize((max(1, round(im.width * h / im.height)), h), Image.LANCZOS) for im in ims]
    strip = Image.new("RGB", (sum(im.width for im in ims), h), "white")
    xo = 0
    for im in ims:
        strip.paste(im, (xo, 0)); xo += im.width
    print(f"  stitched strip {strip.width}x{strip.height} (aspect {strip.width/strip.height:.3f}) "
          f"-> resize to {TARGET[0]}x{TARGET[1]}")
    strip.resize(TARGET, Image.LANCZOS).save(OUT / "row_preview.png")
    print(f"  wrote row_preview.png  ({TARGET[0]}x{TARGET[1]})")


if __name__ == "__main__":
    main()
