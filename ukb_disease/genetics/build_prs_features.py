"""Turn the raw PRS cohort extract into a subject feature parquet for the fusion cross-fit.

Input : prs_cohort.parquet  (eid + 87 PRS field cols + 40 PCs + covars + 26200 flag), from
        extract_prs_cohort.py.
Output: prs_features.parquet  (eid + trait-named PRS columns for the three tracks + PCs +
        covariates + ancestry flags), consumed by cox_crossfit_prs.py.

Track columns (one per disease/trait, named by trait):
  prs_std_<trait>   Standard PRS                       (97.6% coverage)
  prs_enh_<trait>   Enhanced PRS                       (~19% coverage = PRS testing subgroup)
  prs_best_<trait>  best-available per SUBJECT = coalesce(enh, std)  (Track C)
For Standard-only traits (Crohn's, ulcerative colitis) prs_enh/prs_best fall back to std.

Ancestry:
  eur_genetic   1 if genetic ethnic grouping 22006 == 1 (in-sample white British)  [PRIMARY]
  eur_selfreport 1 if self-reported ethnic background 21000 in the British/Irish/White set
PCs kept as pc1..pc40 for within-European adjustment.

PRS are NOT z-scored here (the cross-fit harness z-scores per train-fold to avoid leakage);
we only rename and coalesce so distributions are preserved.
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from ukb_disease.paths import UKB_ROOT

HERE = os.path.dirname(__file__)
FIELD_MAP = os.path.join(HERE, "prs_field_map.csv")

# Self-reported ethnic-background codes (field 21000) counted as European for the sensitivity
# contrast: 1 White, 1001 British, 1002 Irish, 1003 Any other white background.
EUR_SELFREPORT_CODES = {1, 1001, 1002, 1003}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-parquet", default=f"{UKB_ROOT}/cohort/prs_cohort.parquet")
    ap.add_argument("--out", default=f"{UKB_ROOT}/cohort/prs_features.parquet")
    args = ap.parse_args()

    d = pd.read_parquet(args.in_parquet)
    fm = pd.read_csv(FIELD_MAP)
    fm = fm[fm.kind.isin(["Standard", "Enhanced"])].copy()

    out = pd.DataFrame({"eid": d["eid"].astype("int64")})

    # per-trait std / enh / best columns
    traits = sorted(fm.trait_short.unique())
    n_both = n_std_only = 0
    for tr in traits:
        rows = fm[fm.trait_short == tr]
        std_fid = rows[rows.kind == "Standard"]["field_id"]
        enh_fid = rows[rows.kind == "Enhanced"]["field_id"]
        std_col = f"{std_fid.iloc[0]}-0.0" if len(std_fid) else None
        enh_col = f"{enh_fid.iloc[0]}-0.0" if len(enh_fid) else None
        std = d[std_col] if std_col in d.columns else None
        enh = d[enh_col] if enh_col in d.columns else None
        if std is not None:
            out[f"prs_std_{tr}"] = pd.to_numeric(std, errors="coerce")
        if enh is not None:
            out[f"prs_enh_{tr}"] = pd.to_numeric(enh, errors="coerce")
        # best = coalesce(enh, std) per subject
        if enh is not None and std is not None:
            out[f"prs_best_{tr}"] = pd.to_numeric(enh, errors="coerce").fillna(
                pd.to_numeric(std, errors="coerce"))
            n_both += 1
        elif std is not None:
            out[f"prs_best_{tr}"] = pd.to_numeric(std, errors="coerce")
            n_std_only += 1
        elif enh is not None:
            out[f"prs_best_{tr}"] = pd.to_numeric(enh, errors="coerce")

    # PCs
    for i in range(1, 41):
        c = f"22009-0.{i}"
        if c in d.columns:
            out[f"pc{i}"] = pd.to_numeric(d[c], errors="coerce")

    # covariates / ancestry
    out["sex"] = pd.to_numeric(d["31-0.0"], errors="coerce")
    out["array"] = pd.to_numeric(d["22000-0.0"], errors="coerce")
    out["centre"] = pd.to_numeric(d["54-0.0"], errors="coerce")
    out["prs_testing_subgroup"] = (pd.to_numeric(d["26200-0.0"], errors="coerce") == 1).astype(int)
    out["eur_genetic"] = (pd.to_numeric(d["22006-0.0"], errors="coerce") == 1).astype(int)
    sr = pd.to_numeric(d["21000-0.0"], errors="coerce")
    out["eur_selfreport"] = sr.isin(EUR_SELFREPORT_CODES).astype(int)

    out.to_parquet(args.out, index=False)

    std_traits = [c for c in out.columns if c.startswith("prs_std_")]
    enh_traits = [c for c in out.columns if c.startswith("prs_enh_")]
    best_traits = [c for c in out.columns if c.startswith("prs_best_")]
    summary = {
        "n_subjects": int(len(out)),
        "n_std_traits": len(std_traits), "n_enh_traits": len(enh_traits),
        "n_best_traits": len(best_traits), "n_both": n_both, "n_std_only": n_std_only,
        "eur_genetic_n": int(out["eur_genetic"].sum()),
        "eur_selfreport_n": int(out["eur_selfreport"].sum()),
        "prs_testing_subgroup_n": int(out["prs_testing_subgroup"].sum()),
        "enh_mean_coverage": float(out[enh_traits].notna().mean().mean()),
        "std_mean_coverage": float(out[std_traits].notna().mean().mean()),
    }
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}  ({out.shape[0]} x {out.shape[1]})")


if __name__ == "__main__":
    main()
