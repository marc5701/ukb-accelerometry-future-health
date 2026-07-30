"""Interpretability figures in the Okabe-Ito colour scheme with a clean layout.

Consumes the on-disk analysis artifacts (ablation npz, surrogate CSV, modality npz,
REMOVE_AND_RETRAIN npz) and renders the figures into the output directory. Uses constrained_layout
and generous sizing, and drops the bottom source strip because it collided with the
x-labels. Significance is shown by bold cell text rather than an asterisk.
"""

from __future__ import annotations

import os
import re

from ukb_disease.paths import UKB_ROOT

os.environ.setdefault("MPLCONFIGDIR", os.environ.get("MPLCONFIGDIR", str(UKB_ROOT / ".mpl_tmp")))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OK = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
      "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "grey": "#999999"}
OUT = f"{UKB_ROOT}/figures_out/interpretability"
WATCH = {"NS_324.11": "Parkinson's", "NS_328.11": "Alzheimer's", "NS_328.1": "Dementia",
         "time_to_death": "Mortality", "NS_333.1": "Sleep apnea", "EM_202.2": "Type 2 DM",
         "EM_236.1": "Obesity", "CV_416.2": "Atrial fib", "RE_474": "COPD",
         "MB_286.2": "Depression", "CV_424": "Heart failure", "NS_350.5": "Falls",
         "NS_325": "Gait abnorm."}


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[fig] wrote {p}")


#
def fig_surrogate_gap(merged_csv: str):
    df = pd.read_csv(merged_csv)
    fin = df.dropna(subset=["gap"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.35, 1]})

    # Panel 1: scatter with selected-disease list names via a leader-line callout column (no overlap)
    ax = axes[0]
    ax.scatter(fin.C_surrogate, fin.C_deep, s=10, alpha=0.28, color=OK["grey"],
               label="all 390 phecodes", linewidths=0)
    diag = [0.45, 0.95]
    ax.plot(diag, diag, "--", color="k", lw=0.9, label="deep = surrogate")
    wl = []
    for code, name in WATCH.items():
        r = df[df.phecode == code]
        if len(r) and np.isfinite(r.gap.values[0]):
            wl.append((name, float(r.C_surrogate.values[0]), float(r.C_deep.values[0])))
    ax.scatter([w[1] for w in wl], [w[2] for w in wl], s=70, color=OK["red"],
               edgecolor="k", linewidth=0.6, zorder=6, label="selected-disease list disease")
    # right-side label gutter; names in a sorted, evenly-spaced column with leader lines
    ax.set_xlim(0.45, 1.20); ax.set_ylim(0.45, 0.95)
    wl.sort(key=lambda t: -t[2])                      # by deep C, top to bottom
    ycol = np.linspace(0.93, 0.48, len(wl))
    xlab = 0.99
    for (name, xs, yd), yl in zip(wl, ycol):
        ax.plot([xs, xlab - 0.005], [yd, yl], color=OK["grey"], lw=0.6, alpha=0.8, zorder=4)
        ax.text(xlab, yl, name, fontsize=8.5, va="center", ha="left")
    ax.axvline(0.95, color="k", lw=0.6, alpha=0.25)
    ax.set_xticks(np.arange(0.5, 0.96, 0.1))
    ax.set_xlabel("C-index, transparent surrogate (named features)", fontsize=10)
    ax.set_ylabel("C-index, deep AcceleRest embedding", fontsize=10)
    ax.set_title(f"Deep model vs interpretable surrogate\nmean gap +{fin.gap.mean():.3f}, "
                 f"surrogate recovers {100*fin.C_surrogate.mean()/fin.C_deep.mean():.0f}%", fontsize=11)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
    ax.grid(alpha=0.15)

    # Panel 2: horizontal bars of selected-disease list gaps (sorted; labels on y-axis, no overlap)
    ax = axes[1]
    rows = [(WATCH[c], df[df.phecode == c].gap.values[0])
            for c in WATCH if len(df[df.phecode == c]) and np.isfinite(df[df.phecode == c].gap.values[0])]
    rows.sort(key=lambda t: t[1])
    names = [r[0] for r in rows]; gaps = [r[1] for r in rows]
    cols = [OK["red"] if g > 0 else OK["sky"] for g in gaps]
    ax.barh(range(len(names)), gaps, color=cols, edgecolor="k", linewidth=0.4)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=10)
    ax.axvline(0, color="k", lw=0.8)
    for i, g in enumerate(gaps):
        ax.text(g + (0.002 if g >= 0 else -0.002), i, f"{g:+.3f}",
                va="center", ha="left" if g >= 0 else "right", fontsize=8)
    ax.set_xlabel("gap = C_deep − C_surrogate  (signal the embedding finds beyond named features)", fontsize=9.5)
    ax.set_title("Where the embedding beats hand-engineered features", fontsize=11)
    ax.set_xlim(min(gaps) - 0.03, max(gaps) + 0.04)
    ax.grid(axis="x", alpha=0.15)
    _save(fig, "ws1_surrogate_gap.png")


#
def fig_modality_heatmap(dc_table: dict, diseases: list, fname: str, title: str,
                         decimals: int = 2, show_sig: bool = True):
    cells = list(dc_table["cell_names"])
    ph = list(dc_table["phecodes"])
    pidx = [ph.index(d) for d in diseases if d in ph]
    names = [WATCH.get(d, d) for d in diseases if d in ph]
    M = dc_table["dc_mean"][:, pidx]
    sig = dc_table.get("significant")
    sig = (sig[:, pidx] if (show_sig and sig is not None) else np.zeros_like(M, dtype=bool))
    nrow, ncol = len(cells), len(names)
    fig, ax = plt.subplots(figsize=(0.92 * ncol + 3.4, 0.66 * nrow + 2.4),
                           constrained_layout=True)
    vmax = float(np.nanpercentile(np.abs(M), 98))
    vmax = vmax if vmax > 0 else 0.01
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(ncol)); ax.set_xticklabels(names, rotation=38, ha="right", fontsize=9.5)
    ax.set_yticks(range(nrow)); ax.set_yticklabels(cells, fontsize=9.5)
    ax.set_xticks(np.arange(-.5, ncol, 1), minor=True)
    ax.set_yticks(np.arange(-.5, nrow, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    fs = 8.5 if ncol <= 10 else 7.5
    for i in range(nrow):
        for j in range(ncol):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.{decimals}f}", ha="center", va="center",
                        fontsize=fs, fontweight="bold" if sig[i, j] else "normal",
                        color="white" if abs(M[i, j]) > vmax * 0.55 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("ΔC  (drop in C-index when cell removed)", fontsize=9.5)
    sub = "\n(bold = BH q<0.05)" if show_sig else "\n(values small, cells are redundant; see colour)"
    ax.set_title(title + sub, fontsize=11.5)
    _save(fig, fname)


#
def _short(s: str) -> str:
    """Compact, human-readable label (no mid-word truncation)."""
    m = re.match(r"respeventbouts_(R\d)_?(daytime|nighttime|morning_window|evening_window)?_n_bouts", s)
    if m:
        win = {"daytime": " day", "nighttime": " night", "morning_window": " morn",
               "evening_window": " eve", None: ""}[m.group(2)]
        return f"resp-bouts {m.group(1)}{win}"
    mn = re.match(r"naps_(N\d)_any_nap_with_high_respevent_burden", s)
    if mn:
        return f"nap {mn.group(1)} resp-burden"
    table = {
        "sleep_p_wake_mean": "wake fraction", "sleep_p_deep_mean": "deep fraction",
        "n_stage_transitions_total_smoothed": "stage transitions",
        "ra_M10": "act. M10 (most active 10h)", "ra_L5": "act. L5 (least active 5h)",
        "ra_RA": "rest-activity amplitude", "ra_M10_onset_hour": "M10 onset hour",
        "ra_L5_onset_hour": "L5 onset hour",
    }
    return table.get(s, s.replace("_", " ")[:26])


def fig_modality_subspace(modality_npz: str):
    z = dict(np.load(modality_npz, allow_pickle=True))
    rn = [str(s) for s in z["resp_targets"]]
    an = [str(s) for s in z["act_targets"]]
    nbar = len(rn) + len(an)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, max(5.2, 0.34 * nbar)),
                             constrained_layout=True, gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    yr = np.arange(len(rn)); ya = np.arange(len(rn), nbar)
    ax.barh(yr, z["r2_resp"], color=OK["blue"], label="respiration target", edgecolor="k", linewidth=0.3)
    ax.barh(ya, z["r2_act"], color=OK["orange"], label="activity target", edgecolor="k", linewidth=0.3)
    ax.set_yticks(range(nbar))
    ax.set_yticklabels([_short(s) for s in rn + an], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("probe R²  (how well the concept is decodable from the AR embedding)", fontsize=10)
    ax.set_title("Concept decodability from the embedding", fontsize=11.5)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(0, 1.0); ax.grid(axis="x", alpha=0.15)

    ax = axes[1]
    ang = z["principal_angles_deg"]
    ax.bar(range(len(ang)), ang, color=OK["green"], edgecolor="k", linewidth=0.4)
    ax.axhline(90, ls="--", color="k", lw=0.8)
    ax.text(len(ang) - 0.5, 91, "90° = orthogonal (fully separable)", ha="right", fontsize=8)
    ax.set_xlabel("principal-angle index", fontsize=10)
    ax.set_ylabel("angle between resp & activity subspaces (deg)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title(f"Respiration vs activity cell_correlation\nshared variance = {float(z['shared_variance']):.2f} "
                 f"({'inseparable' if z['shared_variance']>=0.5 else 'partially entangled' if z['shared_variance']>=0.1 else 'clean'})",
                 fontsize=11.5)
    ax.grid(axis="y", alpha=0.15)
    _save(fig, "modality_subspace.png")
