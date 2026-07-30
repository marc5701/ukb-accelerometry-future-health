"""95% bootstrap CIs for the three Extended Data bar panels that lacked uncertainty.

All are the paper's canonical disease/phecode-panel bootstrap (resample the panel rows with
replacement, B=1000, seed 42, paired within each figure), matching the per-disease forest panel
and the genetics-forest CIs. Reproduces the paper's stored point values and (where they exist) stored CIs.

Panels:
  residual (b)          two bars: mean per-disease dC_wrist and dC_null, over the 331
                        non-mortality diseases the builder plots.
  agepretrain (b)       three bars: mean per-disease C for embedding (main), sequence, mean-pooling.
  genetics_matched (b)  two bars: mean per-disease (c_joint_prs - c_joint) over the 31
                        matched phecodes, and over the 332 full-panel phecodes.

Writes: figbuild/ed_bar_cis.json
"""
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd

UKB_ROOT = os.environ.get("UKB_ROOT", str(Path(__file__).resolve().parent.parent / "data"))

B, SEED = 1000, 42
OUT = Path(__file__).resolve().parent / "ed_bar_cis.json"


def boot_ci(cols, n_rows):
    """Paired disease/phecode-panel bootstrap: one resampled index set applied to every column.
    `cols` = dict name->1D array (length n_rows). Returns name->(point, lo, hi)."""
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n_rows, size=(B, n_rows))
    out = {}
    for name, a in cols.items():
        a = np.asarray(a, float)
        bs = a[idx].mean(1)
        out[name] = {"point": round(float(a.mean()), 5),
                     "ci_lo": round(float(np.percentile(bs, 2.5)), 5),
                     "ci_hi": round(float(np.percentile(bs, 97.5)), 5)}
    return out


def main():
    res = {"method": f"disease/phecode-panel bootstrap, B={B}, seed={SEED}, paired within figure", "figures": {}}

    # residual panel b (exclude mortality, as the builder does)
    e1 = pd.read_csv(f"{UKB_ROOT}/investigations/residual_partial/output/exp1_per_disease.csv")
    e1 = e1[e1.phecode != "time_to_death"].dropna(subset=["dC_wrist", "dC_null"])
    res["figures"]["residual"] = {"n": int(len(e1)),
        **boot_ci({"wrist": e1.dC_wrist.values, "null": e1.dC_null.values}, len(e1))}

    # agepretrain panel b (drive bar heights from the CSV too)
    s6 = pd.read_csv(f"{UKB_ROOT}/benchmark/output/disjoint_split_v1/fig6_seq_vs_mean.csv").dropna(
        subset=["embedding_c", "seq_c", "mean_c"])
    res["figures"]["agepretrain"] = {"n": int(len(s6)),
        **boot_ci({"main": s6.embedding_c.values, "sequence": s6.seq_c.values, "mean": s6.mean_c.values}, len(s6))}

    # genetics_matched panel b
    gm = pd.read_csv(f"{UKB_ROOT}/crossfit/std_matched_eur/per_phecode.csv")
    gm = gm[gm.n_prs_cols > 0].dropna(subset=["c_joint_prs", "c_joint"])
    d_matched = (gm.c_joint_prs - gm.c_joint).values
    gf = pd.read_csv(f"{UKB_ROOT}/crossfit/std_panel_all/per_phecode.csv").dropna(subset=["c_joint_prs", "c_joint"])
    d_full = (gf.c_joint_prs - gf.c_joint).values
    res["figures"]["genetics_matched"] = {"n_matched": int(len(d_matched)), "n_full": int(len(d_full)),
        "matched": boot_ci({"d": d_matched}, len(d_matched))["d"],
        "full": boot_ci({"d": d_full}, len(d_full))["d"]}

    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
