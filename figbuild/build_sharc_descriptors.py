"""Extended Data figure: SHARC (Shared Actigraphic Risk Component) vs known health descriptors.

Single-panel grouped forest plot of the raw Spearman correlation (with 95% bootstrap CI)
between the SHARC score and each descriptor, grouped Demographics / Activity /
Sleep (actigraphy-derived) / Outcomes (reference). Numbers come from
ukb_disease.paper_additions.sharc.sharc_descriptor_assoc (assoc_table.csv).

Descriptive text (lifestyle-not-available, n per row, CI definition, the BMI-adjusted
robustness column) lives in the LaTeX caption, NOT in the artwork (Nature rule).

Run:  cd figbuild && python3 build_sharc_descriptors.py
"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figlib as F
from figlib import OK, save, W1, W2, CONST

ASSOC = (F.UKB / "paper_additions" / "sharc" / "sharc_descriptors" / "assoc_table_descriptor_set.csv")

GROUP_ORDER = ["Controls / anchors", "Movement / activity",
               "Rest-activity rhythm & timing", "Sleep & breathing", "Outcome validation"]
GROUP_COLOR = {
    "Controls / anchors":            OK["grey"],
    "Movement / activity":           OK["blue"],
    "Rest-activity rhythm & timing": OK["orange"],
    "Sleep & breathing":             OK["green"],
    "Outcome validation":            OK["vermillion"],
}
# faint background band per group (alternate) to reinforce grouping
GROUP_BAND = {
    "Controls / anchors": "#f4f4f4", "Movement / activity": "white",
    "Rest-activity rhythm & timing": "#f4f4f4", "Sleep & breathing": "white",
    "Outcome validation": "#f4f4f4",
}
# Display rename applied at read time so the figure grouping does not depend on the CSV group
# key. The descriptor CSV groups the amplitude/phase metrics under "Circadian rhythm & timing".
# The figure shows "Rest-activity rhythm & timing" because these are nonparametric rest-activity
# proxies (M10/L5/RA and their onsets), not endogenous circadian phase. Kept here (not in the
# CSV) so a CSV re-curation cannot silently revert it.
GROUP_RENAME = {"Circadian rhythm & timing": "Rest-activity rhythm & timing"}
# concise display labels (keep the label column narrow); order = figure row order
LABEL = {
    # Controls / anchors
    "bmi": "Body-mass index", "age": "Age", "sex": "Sex (M vs F)",
    # Movement / activity
    "enmo_p95": "95th-percentile acceleration",
    "enmo_mean": "Mean acceleration (ENMO)",
    "ukb_accel": "UK Biobank overall acceleration",
    "enmo_M10": "M10 (most-active 10 h)", "enmo_L5": "L5 (least-active 5 h)",
    "enmo_frac_sed": "Sedentary fraction", "enmo_frac_light": "Light-activity fraction",
    "enmo_frac_mod": "Moderate-activity fraction", "enmo_frac_vig": "Vigorous-activity fraction",
    "enmo_frac_mvpa": "MVPA fraction",
    # Rest-activity rhythm & timing
    "enmo_RA": "Relative amplitude", "sri": "Sleep Regularity Index",
    "m10_onset_h": "M10 onset time", "l5_onset_h": "L5 onset time",
    # Sleep & breathing
    "ahi_analog": "Apnoea–hypopnoea events/hour (wrist)", "tst_min": "Total sleep time",
    "rem_frac_mean": "REM %",
    "frag_mean": "Sleep fragmentation", "eff_mean": "Sleep efficiency",
    # Outcome validation
    "prevalent_burden_wp": "Prevalent disease burden", "burden": "Incident disease burden",
    "died6": "Six-year mortality",
}


def fig_sharc_descriptors():
    t = pd.read_csv(ASSOC)
    t["group"] = t["group"].replace(GROUP_RENAME)   # display rename (see GROUP_RENAME note)
    t = t.set_index("key")

    # ----- build vertical layout: header row then its data rows, gap between groups
    ypos, ylab, bold_idx = [], [], []
    data = []          # (y, key, group)
    band = []          # (y0, y1, color)
    y = 0.0
    for g in GROUP_ORDER:
        keys = [k for k in LABEL if (k in t.index and t.loc[k, "group"] == g)]
        y0 = y - 0.5
        # header
        ypos.append(y); ylab.append(g); bold_idx.append(len(ypos) - 1)
        y += 1.0
        for k in keys:
            ypos.append(y); ylab.append(LABEL[k])
            data.append((y, k, g))
            y += 1.0
        band.append((y0, y - 0.5, GROUP_BAND[g]))
        y += 0.5       # gap between groups

    ytop, ybot = -0.8, y - 0.6
    fig, ax = plt.subplots(figsize=(W2, 0.20 * len(ypos) + 0.6))
    fig.subplots_adjust(left=0.28, right=0.74, top=0.94, bottom=0.085)

    # group background bands
    for y0, y1, c in band:
        if c != "white":
            ax.axhspan(y0, y1, color=c, zorder=0, lw=0)

    ax.axvline(0, ls=":", color="#666666", lw=0.9, zorder=1)

    # forest: CI whisker + point, coloured by group
    for (yy, k, g) in data:
        rho, lo, hi = t.loc[k, "rho"], t.loc[k, "ci_lo"], t.loc[k, "ci_hi"]
        ax.errorbar([rho], [yy], xerr=[[rho - lo], [hi - rho]], fmt="none",
                    ecolor="#b5b5b5", elinewidth=1.3, capsize=2.2, zorder=3)
        ax.scatter([rho], [yy], s=30, color=GROUP_COLOR[g], edgecolor="k",
                   linewidth=0.3, zorder=4)
        # right-hand value column: r (95% CI)
        ax.text(1.012, yy, f"{rho:+.2f} ({lo:+.2f}, {hi:+.2f})",
                transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=6.4, color="#222222", clip_on=False)

    # value-column header
    ax.text(1.012, ytop + 0.15, "r (95% CI)", transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=6.6, fontweight="bold", clip_on=False)

    ax.set_yticks(ypos)
    ax.set_yticklabels(ylab, fontsize=7.0)
    for i, lab in enumerate(ax.get_yticklabels()):
        if i in bold_idx:
            lab.set_fontweight("bold"); lab.set_fontsize(7.4)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(ybot, ytop)          # inverted (top row first)
    ax.set_xlim(-0.62, 0.62)
    nlo, nhi = int(t["n"].min()), int(t["n"].max())
    ax.set_xlabel(f"Spearman correlation with SHARC score (n = {nlo:,}–{nhi:,} test participants)")

    save(fig, "fig_sarc_descriptors")

    # ----- source data CSV (Nature requirement)
    sd = t.reset_index()[["group", "label", "key", "n", "rho", "ci_lo", "ci_hi",
                          "rho_bmiadj", "ci_lo_bmiadj", "ci_hi_bmiadj"]].copy()
    sd["label"] = sd["key"].map(LABEL)   # display labels are the single source of truth
    sd = sd.rename(columns={
        "rho": "spearman_r", "ci_lo": "ci95_lo", "ci_hi": "ci95_hi",
        "rho_bmiadj": "spearman_r_bmi_adjusted", "ci_lo_bmiadj": "ci95_lo_bmi_adjusted",
        "ci_hi_bmiadj": "ci95_hi_bmi_adjusted"})
    out = F._PAPER_ROOT / "draft" / "sharc_descriptors_source_data.csv"
    sd.to_csv(out, index=False)
    print(f"  wrote source data -> draft/{out.name}")


if __name__ == "__main__":
    fig_sharc_descriptors()
    print("done: fig_sarc_descriptors")
