"""Figures for the cross-entropy screening result.

Reads `screening_results_<variant>.csv` (long: phecode x arm, with name/category,
metric panel, CIs) and writes:
  fig1_spotlight_bars.png    grouped AUROC bars (wrist / demo / combined) with CIs
                             for PD/AD/dementia plus notable wrist-win diseases
  fig2_combined_increment.png histograms of (combined - wrist) and (combined - demo)
  fig3_category_bars.png     mean AUROC by ICD category (wrist / demo / combined)

PNGs default to the package screening/figures/ directory so they sync with the code.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ukb_disease.baseline.config import resolve_oak_path

FIGURES_DIR = Path(__file__).resolve().parent / "figures"

WRIST, DEMO, COMB = "logistic_emb", "demo", "bce_demo"
SPOTLIGHT = ["NS_324.11", "NS_328.11", "NS_328.1", "MB_286.2", "RE_474", "NS_325", "RE_474.1"]
COLORS = {"wrist": "#2c7fb8", "demo": "#969696", "combined": "#d95f0e"}


def _pivot(df, metric):
    return df.pivot_table(index="phecode", columns="arm", values=metric)


def fig_spotlight(df, out):
    rows = []
    for ph in SPOTLIGHT:
        sub = df[df.phecode == ph]
        if not len(sub):
            continue
        nm = str(sub["name"].iloc[0])[:20] or ph
        npos = int(sub["n_test_pos"].iloc[0])
        d = {"label": f"{nm}\n(n={npos})"}
        for arm, key in [(WRIST, "wrist"), (DEMO, "demo"), (COMB, "combined")]:
            r = sub[sub.arm == arm]
            if len(r):
                d[key] = float(r["auroc"].iloc[0])
                d[f"{key}_lo"] = float(r["auroc_lo"].iloc[0])
                d[f"{key}_hi"] = float(r["auroc_hi"].iloc[0])
        rows.append(d)
    P = pd.DataFrame(rows)
    x = np.arange(len(P)); w = 0.26
    fig, ax = plt.subplots(figsize=(max(8, 1.5 * len(P)), 5))
    for i, key in enumerate(["wrist", "demo", "combined"]):
        vals = P[key].to_numpy()
        lo = P.get(f"{key}_lo", pd.Series(np.nan, index=P.index)).to_numpy()
        hi = P.get(f"{key}_hi", pd.Series(np.nan, index=P.index)).to_numpy()
        yerr = np.vstack([vals - lo, hi - vals])
        ax.bar(x + (i - 1) * w, vals, w, yerr=yerr, capsize=3,
               label={"wrist": "wrist embedding", "demo": "age+sex+BMI",
                      "combined": "embedding + age+sex+BMI"}[key], color=COLORS[key])
    ax.axhline(0.5, ls="--", color="grey", alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(P["label"], fontsize=8)
    ax.set_ylabel("Screening AUROC (95% CI)"); ax.set_ylim(0.4, 1.0)
    ax.set_title("Current-disease screening: wrist vs demographics vs combined")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_increment(df, out):
    pa = _pivot(df, "auroc")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (a, b, title) in zip(axes, [
        (COMB, WRIST, "combined minus wrist-only\n(does demographics help the wrist?)"),
        (COMB, DEMO, "combined minus demographics\n(does the wrist help demographics?)")]):
        if not {a, b}.issubset(pa.columns):
            continue
        d = (pa[a] - pa[b]).dropna()
        ax.hist(d, bins=40, color=COLORS["combined"], alpha=0.8)
        ax.axvline(0, color="k", lw=1)
        ax.axvline(float(d.mean()), color="red", ls="--",
                   label=f"mean {d.mean():+.3f}\nwin {100*(d>0).mean():.0f}%")
        ax.set_xlabel("AUROC difference"); ax.set_ylabel("diseases")
        ax.set_title(title, fontsize=10); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_category(df, out, min_n=8):
    g = (df[df.arm.isin([WRIST, DEMO, COMB])]
         .pivot_table(index=["phecode", "category"], columns="arm", values="auroc")
         .reset_index())
    cat = g.groupby("category").agg(n=("phecode", "size"), wrist=(WRIST, "mean"),
                                    demo=(DEMO, "mean"), combined=(COMB, "mean"))
    cat = cat[cat["n"] >= min_n].sort_values("combined", ascending=False)
    x = np.arange(len(cat)); w = 0.26
    fig, ax = plt.subplots(figsize=(max(9, 0.7 * len(cat)), 5))
    for i, (key, col) in enumerate([("wrist", "wrist"), ("demo", "demo"), ("combined", "combined")]):
        ax.bar(x + (i - 1) * w, cat[col].to_numpy(), w,
               label={"wrist": "wrist", "demo": "age+sex+BMI", "combined": "combined"}[key],
               color=COLORS[key])
    ax.axhline(0.5, ls="--", color="grey", alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(cat.index, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Mean screening AUROC"); ax.set_ylim(0.45, 0.8)
    ax.set_title("Mean AUROC by disease category")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--variant", default="primary")
    ap.add_argument("--out-dir", default=str(FIGURES_DIR))
    args = ap.parse_args()
    df = pd.read_csv(os.path.join(resolve_oak_path(args.in_dir), f"screening_results_{args.variant}.csv"))
    os.makedirs(args.out_dir, exist_ok=True)
    fig_spotlight(df, os.path.join(args.out_dir, f"fig1_spotlight_bars_{args.variant}.png"))
    fig_increment(df, os.path.join(args.out_dir, f"fig2_combined_increment_{args.variant}.png"))
    fig_category(df, os.path.join(args.out_dir, f"fig3_category_bars_{args.variant}.png"))
    print(f"[figs] wrote 3 figures for {args.variant} to {args.out_dir}")


if __name__ == "__main__":
    main()
