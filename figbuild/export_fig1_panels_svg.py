"""Export Figure 1 panels b, c, d as standalone vector SVGs. Reuses the exact code from
build_paper_experiments.py::fig1_overview() (panel b = gs[1,:], c = gs[3,0], d = gs[3,1]).
Outputs: figbuild/qa/fig1{b,c,d}.svg (+ .png QC).
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams["svg.fonttype"] = "none"          # keep text editable in the SVG
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import figlib as F                                     # applies nature.mplstyle on import
from figlib import OK

UKB_ROOT = os.environ.get("UKB_ROOT", str(Path(__file__).resolve().parent.parent / "data"))

PE = Path(f"{UKB_ROOT}/paper_experiments")
QA = str(Path(__file__).resolve().parent / "qa")

# shared data (as in fig1_overview)
df = F.load_per_disease().copy()
catmean = df.groupby("PhecodeCategory").c.mean().sort_values()
cat_order = list(catmean.index)
df["catrank"] = df.PhecodeCategory.map({c: i for i, c in enumerate(cat_order)})
df = df.sort_values(["catrank", "c"]).reset_index(drop=True)
df["x"] = np.arange(len(df)); N = len(df)


def _save(fig, name):
    fig.savefig(f"{QA}/{name}.svg", facecolor="white", bbox_inches="tight")
    fig.savefig(f"{QA}/{name}.png", dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---- panel b: phenome-wide breadth (scatter + category legend) ----
fig = plt.figure(figsize=(F.W2, 3.3))
from matplotlib.gridspec import GridSpec
gs = GridSpec(2, 1, height_ratios=[2.0, 0.34], hspace=0.32, figure=fig)
axb = fig.add_subplot(gs[0, 0])
for cat in cat_order:
    sub = df[df.PhecodeCategory == cat]
    axb.scatter(sub.x, sub.c, s=11, color=F.cat_color(cat), edgecolor="none", alpha=0.85)
axb.axhline(0.5, color="#999", ls=":", lw=0.9)
def pick(ph=None, name=None):
    r = df[df.phecode == ph] if ph else df[df.PhecodeString.str.contains(name, case=False, na=False)].sort_values("c")
    return r.iloc[-1] if len(r) else None
spec = [("NS_324.11", "Parkinson's disease", N + 22, 0.965), ("NS_328.11", "Alzheimer's disease", N + 22, 0.925),
        ("CV_424", "Heart failure", N + 22, 0.865), ("EM_236.1", "Obesity", N + 22, 0.825),
        ("NS_328.1", "Dementia", N + 22, 0.795), ("time_to_death", "All-cause mortality", N + 22, 0.765)]
for ph, lab, lx, ly in spec:
    r = pick(ph=ph) if ph else pick(name=lab)
    if r is None:
        continue
    hx, hc = int(r.x), float(r.c)
    axb.scatter([hx], [hc], s=44, facecolor="none", edgecolor="#111", linewidth=1.2, zorder=6)
    axb.annotate(f"{lab} ({hc:.2f})", xy=(hx, hc), xytext=(lx, ly), fontsize=6.3, va="center", ha="left",
                 color="#111", arrowprops=dict(arrowstyle="-", lw=0.55, color="#b0b0b0", shrinkA=3.5, shrinkB=4, relpos=(0, 0.5)))
axb.set_xticks([]); axb.set_xlim(-3, N + 150); axb.set_ylim(0.45, 0.995)
axb.set_xlabel("388 incident outcomes, grouped by disease category and ranked by concordance within each", fontsize=7.6)
axb.set_ylabel("Concordance (6-year incident)")
axleg = fig.add_subplot(gs[1, 0]); axleg.axis("off")
cat_handles = [Line2D([0], [0], marker="o", ls="", mfc=F.cat_color(c), mec="none", ms=4.5, label=F.cat_label(c))
               for c in reversed(cat_order)]
axleg.legend(handles=cat_handles, loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=7, fontsize=6.0,
             frameon=True, facecolor="white", edgecolor="#bbbbbb", framealpha=1.0, handletextpad=0.3,
             columnspacing=1.0, labelspacing=0.6, borderaxespad=0.1)
_save(fig, "fig1b")

# ---- panel c: mean concordance by category ----
figc = plt.figure(figsize=(3.9, 2.7))
axc = figc.add_subplot(111)
cm = catmean
axc.barh(range(len(cm)), cm.values, color=[F.cat_color(c) for c in cm.index], edgecolor="white", linewidth=0.3)
axc.set_yticks(range(len(cm))); axc.set_yticklabels([F.cat_label(c) for c in cm.index], fontsize=6.0)
axc.axvline(0.5, color="#999", ls=":", lw=0.8)
axc.set_xlim(0.5, 0.76); axc.set_xlabel("Mean concordance by category")
_save(figc, "fig1c")

# ---- panel d: transportability across assessment-centre regions ----
figd = plt.figure(figsize=(3.3, 2.7))
axd = figd.add_subplot(111)
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
axd.legend(fontsize=5.6, loc="upper center", ncol=2, frameon=False, handletextpad=0.3, columnspacing=1.0)
_save(figd, "fig1d")
