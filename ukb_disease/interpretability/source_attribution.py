#!/usr/bin/env python3
"""Per-disease source-attribution sweep for the "what predicts each disease" figure.

For every disease we want one comparable per-source contribution, the extra help on top of the
other sources (marginal Delta-C), across 8 sources:

  day_activity      HA_day   (human-activity model, daytime)      } reused from the wrist
  day_sleepbreath   AR_day   (AcceleRest sleep+breathing, daytime) } removal attribution already
  night_activity    HA_night (human-activity model, nighttime)    } computed in
  night_sleepbreath AR_night (AcceleRest sleep+breathing, night)  } day_night_profiles.csv
  genetics          matched disease PRS, reused from perdisease_ci.csv (delta = c_fusion-c_embed)
  age, sex, bmi     computed here: per-disease drop-one Delta-C of each demographic on top of
                    [wrist hazard + the other two demographics], with a permuted-covariate
                    null for significance.

The wrist hazard is embedding_disjoint_split (trained with age+sex), so age/sex correctly come out near 0
(the wrist embedding already encodes them), the intended "demographics add nothing" message.

All three pieces are marginal Delta-C in concordance units, so they share one white-to-red scale
in the figure. Reuses the cross-fit engine (cox_crossfit_features.crossfit_phecode) so the demo
numbers reconcile with the rest of the analysis. CPU-only, ~10 min.

Run:
  PYTHONDONTWRITEBYTECODE=1 python3 \
    -m ukb_disease.interpretability.source_attribution --out <output dir>
Smoke: add --phecode-limit 8
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from ukb_disease.baseline.config import resolve_oak_path
from ukb_disease.baseline.cox_crossfit_features import crossfit_phecode
from ukb_disease.benchmark.concordance_benchmark import DEFAULT_RUNS, load_model
from ukb_disease.paths import UKB_ROOT

# Reused inputs (already-computed per-disease contributions)
DAY_NIGHT = f"{UKB_ROOT}/day_night_profiles.csv"                 # AR/HA x day/night removal dC
GENETICS = f"{UKB_ROOT}/crossfit/perdisease_eur/perdisease_ci.csv"  # matched-PRS fusion delta
# Demographics raw values (same sources residual_partial.load_demo uses)
DEM = f"{UKB_ROOT}/metadata/acc_dem.csv"     # age_at_first_run, sex
BMI = f"{UKB_ROOT}/metadata/BMI_yes.csv"     # 21001-0.0

# Canonical source order for the figure
SOURCES = ["day_activity", "day_sleepbreath", "night_activity", "night_sleepbreath",
           "genetics", "age", "sex", "bmi"]
# day_night cell to source name
CELL2SRC = {"HA_day": "day_activity", "AR_day": "day_sleepbreath",
            "HA_night": "night_activity", "AR_night": "night_sleepbreath"}


def load_demo(eids: np.ndarray):
    """Raw (age, sex, bmi) aligned to eids, NaN filled with column mean (inlined from residual_partial)."""
    dem = pd.read_csv(resolve_oak_path(DEM)).set_index("eid")
    bmi = (pd.read_csv(resolve_oak_path(BMI), usecols=["eid", "21001-0.0"])
           .rename(columns={"21001-0.0": "bmi"}).groupby("eid").mean())
    age = dem.reindex(eids)["age_at_first_run"].to_numpy(float)
    sex = dem.reindex(eids)["sex"].to_numpy(float)
    bm = bmi.reindex(eids)["bmi"].to_numpy(float)
    for a in (age, sex, bm):
        a[~np.isfinite(a)] = np.nanmean(a)
    return {"age": age, "sex": sex, "bmi": bm}


def compute_demo_deltas(args) -> pd.DataFrame:
    """Per-disease drop-one Delta-C of age/sex/bmi on top of [wrist hazard + other two demographics],
    plus a permuted-covariate null. Returns long-form rows (phecode, source, dC, dC_null, n_event)."""
    runs_dir = resolve_oak_path(args.runs_dir)
    mean_h, _by_seed, base, seeds = load_model(runs_dir, args.deep_prefix)
    T, E, M, eids = base["T"], base["E"], base["M"], base["eids"]
    N, P = T.shape
    cfg = json.load(open(f"{runs_dir}/{args.deep_prefix}_seed{seeds[0]}/config.json"))
    names = cfg["phecodes"]
    demo = load_demo(eids)
    print(f"[src-attr] deep={args.deep_prefix} seeds={seeds} N={N} P={P}", flush=True)

    ev_per = (E * M).sum(0)
    panel = [j for j in range(P) if ev_per[j] >= args.min_events]
    if args.phecode_limit:
        panel = panel[:args.phecode_limit]
    print(f"[src-attr] demo drop-one over {len(panel)} phecodes (>= {args.min_events} events)", flush=True)

    rng = np.random.default_rng(0)
    covs = ["age", "sex", "bmi"]
    rows = []
    t0 = time.time()
    for ji, j in enumerate(panel):
        m = M[:, j]
        T_j = T[m, j].astype(float)
        E_j = E[m, j].astype(float)
        h = mean_h[m, j]
        cov_m = {c: demo[c][m] for c in covs}

        def C(extra_cols):
            X = np.column_stack([h] + extra_cols)  # col 0 = hazard = fallback
            return crossfit_phecode(X, T_j, E_j, args.penalizer_c0)["c"]

        c_full = C([cov_m[c] for c in covs])
        for cov in covs:
            others = [cov_m[c] for c in covs if c != cov]
            c_drop = C(others)                                   # base: hazard + other two demos
            dC = c_full - c_drop                                 # real extra help from `cov`
            # null: same model but `cov` permuted among the eval subjects (matched-complexity)
            cols_null = []
            for c in covs:
                v = cov_m[c]
                cols_null.append(rng.permutation(v) if c == cov else v)
            c_null = C(cols_null)
            dC_null = c_null - c_drop
            rows.append({"phecode": names[j], "source": cov, "dC": dC, "dC_null": dC_null,
                         "n_event": int(E_j.sum())})
        if (ji + 1) % 50 == 0:
            print(f"[src-attr]   {ji+1}/{len(panel)}  ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def assemble(demo_df: pd.DataFrame) -> pd.DataFrame:
    """Combine actigraphy (day_night_profiles), genetics (perdisease_ci), and demo into one tidy
    frame: phecode, source, dC, significant, available, n_event."""
    out = []

    # actig 4 cells (reuse wrist REMOVE_AND_RETRAIN removal dC; has its own significance flag)
    dn = pd.read_csv(resolve_oak_path(DAY_NIGHT))
    sig_dn = dn["significant"].astype(str).str.lower().isin(["true", "1", "1.0"])
    for r, sig in zip(dn.itertuples(index=False), sig_dn):
        if r.cell in CELL2SRC:
            out.append({"phecode": r.phecode, "source": CELL2SRC[r.cell], "dC": float(r.dc_mean),
                        "significant": bool(sig), "available": True,
                        "n_event": int(getattr(r, "n_event", 0) or 0)})

    # genetics (reuse matched-PRS fusion delta; CI excluding 0 counts as significant)
    gen = pd.read_csv(resolve_oak_path(GENETICS))
    # collapse duplicate phecodes (some appear at multiple resolutions), keep the largest n_eval row
    gen = gen.sort_values("n_eval").drop_duplicates("phecode", keep="last")
    for r in gen.itertuples(index=False):
        out.append({"phecode": r.phecode, "source": "genetics", "dC": float(r.delta),
                    "significant": bool(float(r.delta_lo) > 0), "available": True,
                    "n_event": int(getattr(r, "n_ev", 0) or 0)})

    # demographics (computed here; pooled-null threshold per source for significance)
    for cov in ["age", "sex", "bmi"]:
        sub = demo_df[demo_df.source == cov]
        thr = float(np.nanpercentile(sub["dC_null"], 95)) if len(sub) else 0.0
        for r in sub.itertuples(index=False):
            out.append({"phecode": r.phecode, "source": cov, "dC": float(r.dC),
                        "significant": bool(r.dC > thr and r.dC > 0), "available": True,
                        "n_event": int(r.n_event)})

    df = pd.DataFrame(out)
    # mark genetics as unavailable (grey) for diseases with no matched PRS
    gen_have = set(df[df.source == "genetics"].phecode)
    all_ph = set(df.phecode)
    miss = [{"phecode": p, "source": "genetics", "dC": np.nan, "significant": False,
             "available": False, "n_event": 0} for p in (all_ph - gen_have)]
    df = pd.concat([df, pd.DataFrame(miss)], ignore_index=True)
    df["source"] = pd.Categorical(df["source"], categories=SOURCES, ordered=True)
    return df.sort_values(["phecode", "source"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deep-prefix", default="embedding_disjoint_split")
    ap.add_argument("--runs-dir", default=DEFAULT_RUNS)
    ap.add_argument("--min-events", type=int, default=10)
    ap.add_argument("--penalizer-c0", type=float, default=10.0)
    ap.add_argument("--phecode-limit", type=int, default=0, help="smoke: only first N phecodes")
    ap.add_argument("--out", default=f"{UKB_ROOT}/source_attribution/output")
    args = ap.parse_args()

    out_dir = resolve_oak_path(args.out)
    os.makedirs(out_dir, exist_ok=True)

    demo_df = compute_demo_deltas(args)
    demo_df.to_csv(f"{out_dir}/demo_deltas.csv", index=False)

    tidy = assemble(demo_df)
    tidy.to_csv(f"{out_dir}/perdisease_sources.csv", index=False)

    # quick sanity print
    print("\n[src-attr] ===== per-source summary (mean dC, %significant, n diseases) =====")
    for s in SOURCES:
        sub = tidy[(tidy.source == s) & tidy.available]
        if len(sub):
            print(f"  {s:18s} mean dC {sub.dC.mean():+.4f}   "
                  f"sig {100*sub.significant.mean():4.0f}%   n={len(sub)}")
    print(f"\n[src-attr] wrote {out_dir}/perdisease_sources.csv  (+ demo_deltas.csv)")


if __name__ == "__main__":
    main()
