"""Out-of-fold cross-fit of PRS added to the frozen-embedding hazard (the fusion test).

Additive sibling of baseline/cox_crossfit_features.py. Reuses the same
leakage-safe within-fold-C-then-average machinery (imported, not duplicated),
extended with (1) per-disease matched PRS columns (each phecode uses its own
disease PRS), (2) a full-panel mode (all PRS as features, same for every
phecode), (3) optional subject subsetting (European-ancestry or Enhanced
subgroup), and (4) a phecode-bootstrap CI on the aggregate delta mean-C.

Arms compared (all per-phecode, fixed evaluable panel):
  hazard_only      [hazard]                     embedding-only baseline
  joint            [hazard, age, sex]           plus demographics
  joint_prs        [hazard, age, sex, <PRS>]    fusion (PRS added)
  prs_only         [<PRS>(+age,sex)]            standalone genetics (orthogonal denom)

<PRS> is the matched disease column (matched mode) or all track columns (panel mode).

Usage (in-container, foreground, light):
  python3 -m ukb_disease.genetics.cox_crossfit_prs --track std --mode matched \
      --subset eur_genetic --out crossfit/std_matched_eur
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from ukb_disease.baseline.cox_crossfit_features import crossfit_phecode
from ukb_disease.benchmark.concordance_benchmark import DEFAULT_RUNS, harrell_counts, load_model
from ukb_disease.baseline.cox_crossfit_features import load_age_covars
from ukb_disease.baseline.config import resolve_oak_path
from ukb_disease.paths import UKB_ROOT

HERE = os.path.dirname(__file__)

FEATURES = f"{UKB_ROOT}/cohort/prs_features.parquet"
CROSSWALK_CURATED = os.path.join(HERE, "prs_phecode_crosswalk.json")
CROSSWALK_CANDIDATE = os.path.join(HERE, "prs_phecode_crosswalk_candidate.json")

TRACK_PREFIX = {"std": "prs_std_", "enh": "prs_enh_", "best": "prs_best_"}


def build_phecode_to_trait(panel_names: list[str]) -> dict[str, str]:
    """phecode -> PRS trait. Prefer the CURATED hand-verified map (direct phecode->trait,
    drops cancer/mismatched codes); fall back to inverting the keyword candidate map."""
    if os.path.exists(CROSSWALK_CURATED):
        d = json.load(open(CROSSWALK_CURATED))
        return {k: v for k, v in d.items() if not k.startswith("_")}
    xwalk = json.load(open(CROSSWALK_CANDIDATE))
    p2t = {}
    for trait, phs in xwalk.items():
        for p in phs:
            p2t.setdefault(p, trait)
    return p2t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["std", "enh", "best"], default="std")
    ap.add_argument("--mode", choices=["matched", "panel"], default="matched")
    ap.add_argument("--subset", choices=["all", "eur_genetic", "eur_selfreport",
                                         "enh_subgroup"], default="all")
    ap.add_argument("--deep-prefix", default="embedding_disjoint_split")
    ap.add_argument("--runs-dir", default=DEFAULT_RUNS)
    ap.add_argument("--age-dir", default="age_sex_disjoint_split/sequence_model_age")
    ap.add_argument("--features", default=FEATURES)
    ap.add_argument("--min-events", type=int, default=10)
    ap.add_argument("--penalizer-c0", type=float, default=10.0)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = resolve_oak_path(args.out)
    os.makedirs(out_dir, exist_ok=True)

    runs_dir = resolve_oak_path(args.runs_dir)
    mean_h, _by_seed, base, seeds = load_model(runs_dir, args.deep_prefix)
    T, E, M, eids = base["T"], base["E"], base["M"], base["eids"]
    N, P = T.shape
    cfg = json.load(open(f"{runs_dir}/{args.deep_prefix}_seed{seeds[0]}/config.json"))
    names = cfg["phecodes"]

    # age/sex covars aligned to eids
    age_dir = args.age_dir
    if not os.path.isabs(age_dir):
        age_dir = os.path.join(os.path.dirname(runs_dir), age_dir)
    ac = load_age_covars(age_dir, eids)
    age = ac["age"].to_numpy(float)
    sex = ac["sex"].to_numpy(float)

    # PRS features aligned to eids
    feat = pd.read_parquet(args.features).set_index("eid")
    feat = feat.reindex(index=eids)
    prefix = TRACK_PREFIX[args.track]
    prs_cols_all = [c for c in feat.columns if c.startswith(prefix)]

    # subject subset mask (applied on top of each phecode eval mask)
    if args.subset == "all":
        keep_subj = np.ones(N, bool)
    elif args.subset == "enh_subgroup":
        keep_subj = (feat["prs_testing_subgroup"].fillna(0).to_numpy() == 1)
    else:
        keep_subj = (feat[args.subset].fillna(0).to_numpy() == 1)
    print(f"[prs] track={args.track} mode={args.mode} subset={args.subset} "
          f"keep_subj={int(keep_subj.sum())}/{N}  PRS_cols={len(prs_cols_all)}")

    p2t = build_phecode_to_trait(names)

    # standardize all candidate PRS cols once (global z; fold-z is re-applied inside crossfit)
    Z = feat[prs_cols_all].apply(pd.to_numeric, errors="coerce")
    Z = (Z - Z.mean()) / Z.std(ddof=0)
    Z = Z.fillna(0.0).to_numpy(float)
    zcol = {c: i for i, c in enumerate(prs_cols_all)}
    # matched-complexity NULL control: subject-permuted PRS (same columns, shuffled rows). If the
    # real-PRS gain is just "one more covariate helps", Znull reproduces it; if PRS carries signal,
    # Znull's delta collapses to ~0.
    rng0 = np.random.default_rng(12345)
    Znull = Z[rng0.permutation(N), :]

    rows = []
    for j in range(P):
        ph = names[j]
        m = M[:, j] & keep_subj
        n_ev = int(E[m, j].sum())
        if int(m.sum()) < args.min_events or n_ev < args.min_events:
            continue
        T_j = T[m, j].astype(float)
        E_j = E[m, j].astype(float)
        h = mean_h[m, j]

        # choose PRS columns for this phecode
        if args.mode == "panel":
            cols = prs_cols_all
        else:
            trait = p2t.get(ph)
            col = f"{prefix}{trait}" if trait else None
            cols = [col] if (col in zcol) else []
        idx = [zcol[c] for c in cols]
        Zsub = Z[np.ix_(m, idx)] if cols else np.zeros((int(m.sum()), 0))
        Zsub_null = Znull[np.ix_(m, idx)] if cols else np.zeros((int(m.sum()), 0))

        def Cof(X):
            r = crossfit_phecode(X, T_j, E_j, args.penalizer_c0)
            return r["c"]

        c_haz, comp = harrell_counts(h, T_j, E_j)
        c_haz = c_haz / comp if comp > 0 else np.nan
        X_joint = np.column_stack([h, age[m], sex[m]])
        c_joint = Cof(X_joint)
        if Zsub.shape[1]:
            c_jprs = Cof(np.column_stack([X_joint, Zsub]))
            c_jprs_null = Cof(np.column_stack([X_joint, Zsub_null]))
            c_prs = Cof(np.column_stack([Zsub, age[m], sex[m]]))
        else:
            c_jprs = c_joint
            c_jprs_null = c_joint
            c_prs = np.nan
        rows.append({"phecode": ph, "trait": p2t.get(ph), "n_ev": n_ev, "n_eval": int(m.sum()),
                     "n_prs_cols": Zsub.shape[1], "c_hazard": c_haz, "c_joint": c_joint,
                     "c_joint_prs": c_jprs, "c_joint_prs_null": c_jprs_null, "c_prs_only": c_prs})

    df = pd.DataFrame(rows)
    df.to_csv(f"{out_dir}/per_phecode.csv", index=False)

    # aggregate + phecode-bootstrap CI on Delta(joint_prs - joint) and (joint_prs - hazard)
    def boot_delta(a, b):
        d = (df[a] - df[b]).dropna().to_numpy()
        if len(d) == 0:
            return (np.nan, np.nan, np.nan, np.nan)
        rng = np.random.default_rng(0)
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(args.n_boot)]
        return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), \
            float((np.array(bs) > 0).mean())

    summary = {"track": args.track, "mode": args.mode, "subset": args.subset,
               "n_phecodes": int(len(df)),
               "n_matched": int((df["n_prs_cols"] > 0).sum()),
               "mean_c_hazard": float(df["c_hazard"].mean()),
               "mean_c_joint": float(df["c_joint"].mean()),
               "mean_c_joint_prs": float(df["c_joint_prs"].mean())}
    for label, a, b in [("delta_prs_over_joint", "c_joint_prs", "c_joint"),
                        ("delta_null_over_joint", "c_joint_prs_null", "c_joint"),
                        ("delta_prs_over_null", "c_joint_prs", "c_joint_prs_null"),
                        ("delta_prs_over_hazard", "c_joint_prs", "c_hazard"),
                        ("delta_demo_over_hazard", "c_joint", "c_hazard")]:
        mean, lo, hi, ppos = boot_delta(a, b)
        summary[label] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "p_pos": ppos}
    # matched-only aggregate (where PRS actually entered)
    mdf = df[df["n_prs_cols"] > 0]
    if len(mdf):
        summary["matched_only_n"] = int(len(mdf))
        summary["matched_only_delta_prs_over_joint_mean"] = float(
            (mdf["c_joint_prs"] - mdf["c_joint"]).mean())
        summary["matched_only_delta_null_over_joint_mean"] = float(
            (mdf["c_joint_prs_null"] - mdf["c_joint"]).mean())
        summary["matched_only_delta_prs_over_null_mean"] = float(
            (mdf["c_joint_prs"] - mdf["c_joint_prs_null"]).mean())

    json.dump(summary, open(f"{out_dir}/summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[prs] wrote {out_dir}/summary.json + per_phecode.csv")


if __name__ == "__main__":
    main()
