"""Per-disease Harrell-C export, the per-phecode breakdown the headline mean-C hides.

Two uses:
  (1) Composition check. Confirm a mean-C shift reflects panel composition rather
      than regression: run on two model prefixes (e.g. embedding_disjoint_split vs
      embedding_fixed_split), diff the per-disease C on the SHARED phecodes (should be
      roughly stable), and show that the mean shift is driven by phecodes only
      present in the larger panel.
  (2) Disease monitoring list. Per-disease C for a fixed set of monitored phecodes
      (PD/AD/dementia plus the wrist-plausible set), keyed by phecode id.

Imports the primitives from concordance_benchmark so it stays consistent with that
harness, and never perturbs its aggregate JSON output.
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np

from ukb_disease.benchmark.concordance_benchmark import (
    load_model, harrell_counts, disease_columns,
    MEANC_MIN_EVAL, DEFAULT_RUNS,
)
from ukb_disease.benchmark.selected_diseases import SELECTED_DISEASES, DISJOINT_SPLIT_NEW


def phecode_names(runs_dir: str, prefix: str, P: int) -> list[str]:
    """Phecode column names (config.json), same key-probing as disease_columns."""
    seed_dirs = sorted(glob.glob(os.path.join(runs_dir, f"{prefix}_seed*")))
    cfgp = os.path.join(seed_dirs[0], "config.json")
    if os.path.exists(cfgp):
        cfg = json.load(open(cfgp))
        for k in ("phecodes", "phecode_cols", "outcomes", "columns"):
            v = cfg.get(k)
            if isinstance(v, list) and len(v) == P:
                return [str(x) for x in v]
    return [f"col{j}" for j in range(P)]


def per_disease_c(runs_dir: str, prefix: str) -> dict:
    """{phecode_name: {c, n_eval, n_event, is_mortality}} over the per_disease_eval floor
    (>=MEANC_MIN_EVAL evaluable subjects, >=1 event), hazards averaged across seeds."""
    mean_h, _by_seed, base, seeds = load_model(runs_dir, prefix)
    T, E, M = base["T"], base["E"], base["M"]
    _N, P = T.shape
    names = phecode_names(runs_dir, prefix, P)
    _all, _panel, death = disease_columns(runs_dir, prefix, P)
    out = {}
    for j in range(P):
        m = M[:, j]
        n_eval, n_event = int(m.sum()), int(E[m, j].sum())
        if n_eval < MEANC_MIN_EVAL or n_event < 1:
            continue
        conc, comp = harrell_counts(mean_h[m, j], T[m, j], E[m, j])
        if comp <= 0:
            continue
        out[names[j]] = dict(c=round(conc / comp, 5), n_eval=n_eval,
                             n_event=n_event, is_mortality=bool(j in death))
    return dict(seeds=seeds, n_eval_diseases=len(out),
                mean_c=round(float(np.mean([v["c"] for v in out.values()])), 5),
                median_c=round(float(np.median([v["c"] for v in out.values()])), 5),
                per_disease=out)


def composition_diff(a: dict, b: dict) -> dict:
    """Shared vs A-only vs B-only phecodes; mean-C on each subset (a=larger panel)."""
    pa, pb = a["per_disease"], b["per_disease"]
    shared = sorted(set(pa) & set(pb))
    a_only = sorted(set(pa) - set(pb))
    b_only = sorted(set(pb) - set(pa))
    def mc(keys, src): return round(float(np.mean([src[k]["c"] for k in keys])), 5) if keys else None
    deltas = {k: round(pa[k]["c"] - pb[k]["c"], 5) for k in shared}
    return dict(
        n_shared=len(shared), n_a_only=len(a_only), n_b_only=len(b_only),
        shared_meanC_a=mc(shared, pa), shared_meanC_b=mc(shared, pb),
        shared_meanC_delta=(round(mc(shared, pa) - mc(shared, pb), 5) if shared else None),
        a_only_meanC=mc(a_only, pa), b_only_meanC=mc(b_only, pb),
        shared_delta_mean=round(float(np.mean(list(deltas.values()))), 5) if deltas else None,
        shared_delta_abs_median=round(float(np.median(np.abs(list(deltas.values())))), 5) if deltas else None,
        a_only_phecodes=a_only,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", required=True,
                    help="run-dir prefixes (e.g. embedding_disjoint_split embedding_fixed_split)")
    ap.add_argument("--runs-dir", default=DEFAULT_RUNS)
    ap.add_argument("--out", default="./per_disease_c.json")
    ap.add_argument("--watch", nargs="*", default=None,
                    help="phecode ids to spotlight (default: selected_diseases.SELECTED_DISEASES)")
    ap.add_argument("--no-watch", action="store_true", help="disable the selected-disease list section")
    ap.add_argument("--compose", nargs=2, default=None, metavar=("A", "B"),
                    help="composition diff of two prefixes (A=larger/disjoint_split panel)")
    args = ap.parse_args()

    watch = None if args.no_watch else (args.watch if args.watch else list(SELECTED_DISEASES))

    result = {}
    for prefix in args.models:
        r = per_disease_c(args.runs_dir, prefix)
        result[prefix] = r
        print(f"{prefix}: {r['n_eval_diseases']} evaluable diseases  "
              f"mean C={r['mean_c']:.4f}  median C={r['median_c']:.4f}  seeds={r['seeds']}")
        if watch:
            wsec = {}
            print(f"    --- selected-disease list ({len(watch)} diseases) ---")
            for w in watch:
                v = r["per_disease"].get(w)
                label = SELECTED_DISEASES.get(w, "")
                tag = "  <new>" if w in DISJOINT_SPLIT_NEW else ""
                wsec[w] = dict(label=label, in_panel=bool(v),
                               is_new=bool(w in DISJOINT_SPLIT_NEW),
                               **({"c": v["c"], "n_eval": v["n_eval"], "n_event": v["n_event"]}
                                  if v else {}))
                print(f"      {w:14s} {label[:38]:38s}" + (
                      f"  C={v['c']:.4f}  n_eval={v['n_eval']:5d}  n_event={v['n_event']:4d}{tag}"
                      if v else f"  NOT IN PANEL (below floor){tag}"))
            r["selected_diseases"] = wsec

    if args.compose:
        A, B = args.compose
        if A in result and B in result:
            comp = composition_diff(result[A], result[B])
            result["_composition"] = {"A": A, "B": B, **comp}
            print(f"\n=== COMPOSITION {A} vs {B} ===")
            print(f"  shared={comp['n_shared']}  {A}-only={comp['n_a_only']}  {B}-only={comp['n_b_only']}")
            print(f"  shared mean-C: {A}={comp['shared_meanC_a']}  {B}={comp['shared_meanC_b']}  "
                  f"delta={comp['shared_meanC_delta']} (per-disease mean delta {comp['shared_delta_mean']}, "
                  f"abs-median {comp['shared_delta_abs_median']})")
            print(f"  {A}-only diseases mean-C: {comp['a_only_meanC']}  "
                  f"(these dilute the headline; {B}-only mean-C {comp['b_only_meanC']})")

    json.dump(result, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
