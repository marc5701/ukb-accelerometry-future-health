"""95% confidence intervals for the four clinical-baseline mean-concordance bars.

Disease-panel bootstrap (resample the 388 outcomes, B=1000, seed 42) on the mean six-year
concordance of each arm, the same resampling used for the win-rate CIs and the text
deltas. Paired across arms (one resampled index set applied to all four), so the wrist-only
vs wrist+demographics contrast can be read as a paired difference.

Arms (all on the 388-outcome panel, 5,253 test participants):
  wrist only              nodemo_c        (age_mechanism/per_disease_age_contribution.csv)
  wrist + age/sex/BMI     deep-joint C    (embedding_clinical_embedding_disjoint_split, load_seed_avg)
  age/sex/BMI only        c_clin          (summary_baseline/output/full/per_disease_summary.csv)
  activity summary        c_summary       (summary_baseline/output/full/per_disease_summary.csv)

Run:  python3 -m ukb_disease.investigations.clinical_baseline.clinical_baseline_bar_cis
Writes: investigations/clinical_baseline/output/clinical_baseline_bar_cis.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from ukb_disease.paper_additions.io_utils import (
    load_seed_avg, evaluable_diseases, per_disease_c, cindex, RUNS_ROOT)

R = Path(__file__).resolve().parents[2]   # the ukb_disease package directory
OUT = R / "investigations/clinical_baseline/output/clinical_baseline_bar_cis.json"
B, SEED = 1000, 42   # B=1000 matches the win-rate and text-delta bootstraps


def main():
    deep = load_seed_avg("embedding_clinical_embedding_disjoint_split", root=RUNS_ROOT)   # wrist + age+sex+BMI
    idx_d = evaluable_diseases(deep)
    deep_df = pd.DataFrame({"phecode": np.asarray(deep["phecodes"])[idx_d],
                            "c_deep": per_disease_c(deep, idx_d)})

    am = pd.read_csv(R / "investigations/age_mechanism/output/per_disease_age_contribution.csv")
    su = pd.read_csv(R / "investigations/summary_baseline/output/full/per_disease_summary.csv")
    m = (am[["phecode", "nodemo_c"]]
         .merge(deep_df, on="phecode", how="inner")
         .merge(su[["phecode", "c_clin", "c_summary"]], on="phecode", how="inner")).dropna()

    arms = {"wrist_only": "nodemo_c", "wrist_demobmi": "c_deep",
            "clinical": "c_clin", "activity_summary": "c_summary"}
    n = len(m)
    idx = np.random.default_rng(SEED).integers(0, n, size=(B, n))
    out = {"method": f"disease-panel bootstrap, B={B}, seed={SEED}", "n_outcomes": int(n), "arms": {}}
    for a, c in arms.items():
        pt = float(m[c].mean())
        bs = m[c].values[idx].mean(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out["arms"][a] = {"point": round(pt, 5), "ci_lo": round(float(lo), 5), "ci_hi": round(float(hi), 5)}
    # paired wrist+demographics - wrist (the +0.005 claim)
    d = m["c_deep"].values - m["nodemo_c"].values
    bd = d[idx].mean(1)
    out["paired_demobmi_minus_wrist"] = {"point": round(float(d.mean()), 5),
                                         "ci_lo": round(float(np.percentile(bd, 2.5)), 5),
                                         "ci_hi": round(float(np.percentile(bd, 97.5)), 5)}
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
