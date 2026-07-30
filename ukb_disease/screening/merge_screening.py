"""Merge sharded screening outputs into the final tables and summary.

Reads `shard_<variant>_*.csv` (long format: one row per phecode x arm), produces:
  screening_results_<variant>.csv : merged long table (deduped, with names+categories)
  screening_auroc_<variant>.csv   : wide AUROC per disease x arm (with names+categories)
  spotlight_<variant>.csv         : PD / AD / dementia rows (all metrics)
  summary_<variant>.json          : mean AUROC per arm, wrist-vs-demo, BCE-vs-Cox,
                                    combined-arm increments, stack coverage, selected_diseases
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from ukb_disease.paths import UKB_ROOT
from ukb_disease.baseline.config import PHECODE_MAP_CSV, resolve_oak_path

SPOTLIGHT = {"NS_324.11": "Parkinson's", "NS_328.11": "Alzheimer's", "NS_328.1": "dementia"}
ARMS = ["logistic_emb", "mlp_emb", "demo", "cox", "bce_demo", "cox_demo"]
SELECTED_DISEASES_CSV = f"{UKB_ROOT}/benchmark/selected_diseases_frozen.csv"


def _name_maps() -> tuple[dict, dict]:
    """phecode -> (PhecodeString, PhecodeCategory) from the phecode map CSV."""
    try:
        m = pd.read_csv(resolve_oak_path(PHECODE_MAP_CSV)).drop_duplicates("Phecode")
        nm = m.set_index("Phecode")["PhecodeString"].to_dict()
        cat = m.set_index("Phecode")["PhecodeCategory"].to_dict()
        return nm, cat
    except Exception as e:  # noqa: BLE001
        print(f"[merge] name map unavailable: {e}")
        return {}, {}


def _attach(df: pd.DataFrame, nm: dict, cat: dict) -> pd.DataFrame:
    out = df.copy()
    out.insert(1, "name", out["phecode"].map(lambda p: nm.get(p, "")))
    out.insert(2, "category", out["phecode"].map(lambda p: cat.get(p, "")))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--variant", default="primary")
    args = ap.parse_args()
    d = resolve_oak_path(args.in_dir)
    nm, cat = _name_maps()

    shards = sorted(glob.glob(os.path.join(d, f"shard_{args.variant}_*.csv")))
    if not shards:
        raise SystemExit(f"no shards found in {d} for variant {args.variant}")
    df = pd.concat([pd.read_csv(s) for s in shards], ignore_index=True)
    df = df.drop_duplicates(subset=["phecode", "arm"], keep="last")
    df = _attach(df, nm, cat)
    df.to_csv(os.path.join(d, f"screening_results_{args.variant}.csv"), index=False)
    arms_present = sorted(df["arm"].unique())
    print(f"[merge] {len(shards)} shards -> {len(df)} rows, "
          f"{df['phecode'].nunique()} diseases, arms={arms_present}")

    # wide AUROC table (names + counts + per-arm AUROC)
    wide = df.pivot_table(index="phecode", columns="arm", values="auroc")
    wide.columns = [f"auroc_{c}" for c in wide.columns]
    counts = df.groupby("phecode")[["n_test_pos", "n_test_ctrl", "n_train_pos",
                                    "n_val_pos", "prevalence_pop"]].first()
    wide = counts.join(wide).reset_index()
    wide = _attach(wide, nm, cat).sort_values("n_test_pos", ascending=False)
    wide.to_csv(os.path.join(d, f"screening_auroc_{args.variant}.csv"), index=False)

    # spotlight (full metrics)
    spot = df[df["phecode"].isin(SPOTLIGHT)].copy()
    spot.to_csv(os.path.join(d, f"spotlight_{args.variant}.csv"), index=False)
    print("\n[merge] ===== SPOTLIGHT (AUROC [95% CI], n_test_pos) =====")
    for ph, name in SPOTLIGHT.items():
        sub = spot[spot["phecode"] == ph]
        if not len(sub):
            print(f"  {name:12s} ({ph}): not in panel")
            continue
        npos = int(sub["n_test_pos"].iloc[0])
        cells = []
        for arm in ARMS:
            r = sub[sub["arm"] == arm]
            if len(r):
                r = r.iloc[0]
                cells.append(f"{arm}={r['auroc']:.3f}[{r['auroc_lo']:.3f},{r['auroc_hi']:.3f}]")
        print(f"  {name:12s} ({ph}, n_pos={npos}): " + "  ".join(cells))

    # summary
    def arm_mean(arm, metric="auroc"):
        s = df[(df["arm"] == arm)][metric].dropna()
        return float(s.mean()) if len(s) else float("nan")

    piv = df.pivot_table(index="phecode", columns="arm", values="auroc")

    def increment(a, b):
        """mean/winrate of arm a over arm b on diseases where both exist."""
        if not {a, b}.issubset(piv.columns):
            return None
        dd = (piv[a] - piv[b]).dropna()
        if not len(dd):
            return None
        return {"mean_delta_auroc": float(dd.mean()),
                "winrate": float((dd > 0).mean()),
                "n_better": int((dd > 0).sum()), "n_total": int(len(dd))}

    summ = {
        "variant": args.variant,
        "n_diseases": int(df["phecode"].nunique()),
        "arms": arms_present,
        "mean_auroc": {a: arm_mean(a) for a in ARMS if a in arms_present},
        "mean_auprc": {a: arm_mean(a, "auprc") for a in ARMS if a in arms_present},
        "mean_sens_at_95spec": {a: arm_mean(a, "sens_at_spec") for a in ARMS if a in arms_present},
        # headline comparisons
        "wrist_vs_demo": increment("logistic_emb", "demo"),
        "bce_vs_cox": increment("logistic_emb", "cox"),
        # combined (embedding + age/sex/BMI) value
        "bcedemo_over_emb": increment("bce_demo", "logistic_emb"),  # does demo help the wrist?
        "bcedemo_over_demo": increment("bce_demo", "demo"),         # does the wrist help demo?
        "coxdemo_over_cox": increment("cox_demo", "cox"),
        "coxdemo_over_demo": increment("cox_demo", "demo"),
        # how often does the combined screen win vs the better single component?
        "stack_coverage_bce_demo": int(piv["bce_demo"].notna().sum()) if "bce_demo" in piv else 0,
        "stack_coverage_cox_demo": int(piv["cox_demo"].notna().sum()) if "cox_demo" in piv else 0,
    }
    if {"bce_demo", "logistic_emb", "demo"}.issubset(piv.columns):
        best_single = piv[["logistic_emb", "demo"]].max(axis=1)
        dd = (piv["bce_demo"] - best_single).dropna()
        summ["bcedemo_over_best_single"] = {
            "mean_delta_auroc": float(dd.mean()),
            "winrate": float((dd > 0).mean()), "n_total": int(len(dd))}

    if os.path.exists(SELECTED_DISEASES_CSV):
        try:
            wl = pd.read_csv(SELECTED_DISEASES_CSV)
            wl_codes = set(wl.get("phecode", pd.Series(dtype=str)).astype(str))
            summ["selected_diseases_coverage"] = int(df[df["phecode"].isin(wl_codes)]["phecode"].nunique())
        except Exception:
            pass

    with open(os.path.join(d, f"summary_{args.variant}.json"), "w") as f:
        json.dump(summ, f, indent=2, default=float)
    print("\n[merge] ===== SUMMARY =====")
    print(json.dumps(summ, indent=2, default=float))
    print(f"\n[merge] wrote screening_results/auroc/spotlight/summary_{args.variant}.* to {d}")


if __name__ == "__main__":
    main()
