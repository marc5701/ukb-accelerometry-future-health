"""
Concordance benchmark for UKB disease-prediction models.

Model comparison in this project always looks at THREE quantities, because a single
averaged C-index hides what actually matters:

  1. MEAN C-INDEX -> "How good is the model overall?" (the headline, comparable to
     the field). Per-disease Harrell C, averaged equally across phecodes INCLUDING
     mortality, with per_disease_eval's floor (>=10 evaluable subjects, >=1 event), so it
     reconciles with baseline/compile_cox_report.py. Hazards: higher = more risk
     = earlier event (same orientation per_disease_eval encodes by passing -hazards).

  2. UNCENSORED C -> "What KIND of signal is it: real onset timing, or just sorting
     future-cases from non-cases?" Concordance computed only among subjects who
     actually had the disease. Reported POOLED across diseases (per-disease is too
     noisy), next to an age-only floor over the IDENTICAL pairs: if the model is no
     better than age, the 'timing' is really just the labels.

  3. PAIRED HEAD-TO-HEAD -> "Of two near-tied models, which one actually wins?" (the
     tie-breaker / model-selection referee). Compare two models only on the
     comparable patient-pairs where they DISAGREE; the roughly 90%+ they agree on
     (the shared age axis) cancels, leaving the real difference with much lower
     variance. Reported with per-seed reproducibility (paired by seed NUMBER) and an
     across-disease sign test, plus an age-only guardrail (does each model beat a
     dumb age model?).

Run (small runs are quick; large sweeps go to a cluster):
  python3 -m ukb_disease.benchmark.concordance_benchmark \
      --candidates embedding_fixed_split --reference sequence_model_fixed_split \
      --out ./benchmark_out

A model is read from one or more seed run-dirs <prefix>_seed<S> under --runs-dir, each
holding test_hazards/test_event_times/test_is_event/test_eval_mask/test_eids .npy.
Hazard arrays are averaged across seeds for the headline; per-seed (matched by seed
number) is used for the paired reproducibility check. Non-finite hazards are excluded,
not counted as losses. Nothing is written outside --out; inputs are read-only.

Tie-in-time handling is strict textbook Harrell (an event vs a strictly-later subject,
plus an event vs a censored subject at the same time), which equals lifelines on
distinct event times and differs only negligibly on tied event times (panel-mean
effect ~0.000 on the fixed split). View 1 reconciles with compile_cox_report; the
custom paired/uncensored views are intentionally their own statistics.
"""
from __future__ import annotations
import argparse, glob, json, os, re
from math import sqrt, erf
import numpy as np
import pandas as pd

from ukb_disease.paths import UKB_ROOT

DEFAULT_RUNS = f"{UKB_ROOT}/runs_cox"
DEFAULT_DEM = f"{UKB_ROOT}/metadata/acc_dem.csv"

# Monitoring-only per-selected-disease list. Does NOT affect model selection (that is the
# paired head-to-head below). Reported additively per model via the tiered watch view
# plus emerging scan (curate_selected_diseases.py, lazy-imported in main()).
from ukb_disease.benchmark.selected_diseases import load_frozen

# inclusion floors
MEANC_MIN_EVAL = 10     # >=10 evaluable (masked) subjects, matches per_disease_eval.min_eval
UNC_MIN_EVENTS = 8      # uncensored-C needs case-case pairs; require a few cases
PAIR_MIN_EVENTS = 5     # paired head-to-head per-disease floor


# ----------------------------------------------------------------------------- loading
def discover_seeds(runs_dir: str, prefix: str) -> list[str]:
    pat = re.compile(rf"^{re.escape(prefix)}_seed\d+$")
    return sorted(d for d in glob.glob(os.path.join(runs_dir, f"{prefix}_seed*"))
                  if pat.match(os.path.basename(d)))


def load_model(runs_dir: str, prefix: str):
    """Return (mean_hazards[N,P], {seed:hazards}, shared {eids,T,E,M}, seeds)."""
    seed_dirs = discover_seeds(runs_dir, prefix)
    if not seed_dirs:
        raise FileNotFoundError(f"no run dirs for prefix '{prefix}' under {runs_dir}")
    by_seed, base, seeds = {}, None, []
    for d in seed_dirs:
        s = int(re.search(r"seed(\d+)$", d).group(1))
        seeds.append(s)
        h = np.load(f"{d}/test_hazards.npy")
        if np.isnan(h).any():
            print(f"  WARNING: {os.path.basename(d)} has {int(np.isnan(h).sum())} "
                  f"NaN hazards (excluded from comparisons, not counted as losses)")
        by_seed[s] = h
        cur = dict(eids=np.load(f"{d}/test_eids.npy"),
                   T=np.load(f"{d}/test_event_times.npy"),
                   E=np.load(f"{d}/test_is_event.npy"),
                   M=np.load(f"{d}/test_eval_mask.npy"))
        if base is None:
            base = cur
        else:
            assert np.array_equal(cur["eids"], base["eids"]), f"eids mismatch in {d}"
            assert np.array_equal(cur["M"], base["M"]), f"eval_mask mismatch in {d}"
            assert np.array_equal(cur["E"], base["E"]), f"is_event mismatch in {d}"
    mean_h = np.stack([by_seed[s] for s in seeds]).mean(0)
    return mean_h, by_seed, base, seeds


def disease_columns(runs_dir: str, prefix: str, P: int):
    """Return (all_cols incl mortality, panel_cols excl mortality, death_idx)."""
    cfgp = os.path.join(discover_seeds(runs_dir, prefix)[0], "config.json")
    phe = None
    if os.path.exists(cfgp):
        cfg = json.load(open(cfgp))
        for k in ("phecodes", "phecode_cols", "outcomes", "columns"):
            if isinstance(cfg.get(k), list) and len(cfg[k]) == P:
                phe = cfg[k]; break
    if phe is not None:
        death = [i for i, n in enumerate(phe)
                 if "death" in str(n).lower() or "mort" in str(n).lower()]
    else:
        death = [0]  # known: column 0 is time_to_death in fixed-split runs
    return list(range(P)), [j for j in range(P) if j not in death], death


def load_age(dem_path: str, eids: np.ndarray) -> np.ndarray:
    dem = pd.read_csv(dem_path, usecols=["eid", "age_at_first_run"])
    amap = dict(zip(dem.eid.values, dem.age_at_first_run.values))
    return np.array([amap.get(int(x), np.nan) for x in eids], dtype=float)


# --------------------------------------------------------------- per-disease primitives
def harrell_counts(h, t, e):
    """(concordant+0.5*ties, comparable) Harrell-C counts. Higher h = more risk =
    earlier event. Comparable partner of event a: anyone observed strictly later, OR a
    censored subject at the same time (matches lifelines). Non-finite hazards excluded."""
    fin = np.isfinite(h)
    ev = np.where((e == 1) & fin)[0]
    conc = comp = 0.0
    for a in ev:
        b = ((t > t[a]) | ((t == t[a]) & (e == 0))) & fin
        if not b.any():
            continue
        comp += int(b.sum())
        d = h[a] - h[b]
        conc += float(np.sum(d > 0)) + 0.5 * float(np.sum(d == 0))
    return conc, comp


def uncensored_counts(h, t, e):
    """(concordant+0.5*ties, comparable) among EVENT-ONLY subjects (case-vs-case).
    Earlier-onset case should rank riskier. Strict in time (tied onsets unorderable)."""
    fin = np.isfinite(h)
    idx = np.where((e == 1) & fin)[0]
    if idx.size < 2:
        return 0.0, 0.0
    tt, hh = t[idx], h[idx]
    conc = comp = 0.0
    for i in range(idx.size):
        later = tt > tt[i]
        if not later.any():
            continue
        comp += int(later.sum())
        d = hh[i] - hh[later]
        conc += float(np.sum(d > 0)) + 0.5 * float(np.sum(d == 0))
    return conc, comp


def paired_disagreements(hX, hY, t, e):
    """On comparable pairs, count where exactly one of X/Y ranks the earlier-event
    subject as riskier. Returns (X_wins, Y_wins, comparable). Both hazards must be
    finite for a pair to count."""
    fin = np.isfinite(hX) & np.isfinite(hY)
    ev = np.where((e == 1) & fin)[0]
    xw = yw = comp = 0
    for a in ev:
        b = ((t > t[a]) | ((t == t[a]) & (e == 0))) & fin
        if not b.any():
            continue
        comp += int(b.sum())
        xc = hX[a] > hX[b]
        yc = hY[a] > hY[b]
        xw += int(np.sum(xc & ~yc))
        yw += int(np.sum(yc & ~xc))
    return xw, yw, comp


# ------------------------------------------------------------------------ view builders
def single_model_views(h, T, E, M, age, all_cols, panel_cols, death):
    """View 1 (mean-C, incl mortality + reported separately) and View 2 (uncensored-C
    + age floor over identical pairs)."""
    c_per, c_panel, mort_c = [], [], float("nan")
    for j in all_cols:
        m = M[:, j]
        if int(m.sum()) < MEANC_MIN_EVAL or int(E[m, j].sum()) < 1:
            continue
        conc, comp = harrell_counts(h[m, j], T[m, j], E[m, j])
        if comp <= 0:
            continue
        c = conc / comp
        c_per.append(c)
        if j in death:
            mort_c = c
        else:
            c_panel.append(c)
    c_per, c_panel = np.array(c_per), np.array(c_panel)

    unc_per, unc_c, unc_n, age_c, age_n = [], 0.0, 0.0, 0.0, 0.0
    for j in panel_cols:
        m = M[:, j]
        fin = np.isfinite(age[m])              # pair model & age over identical subjects
        t, e = T[m, j][fin], E[m, j][fin]
        hj, aj = h[m, j][fin], age[m][fin]
        if int(e.sum()) < UNC_MIN_EVENTS:
            continue
        c, n = uncensored_counts(hj, t, e)
        if n <= 0:
            continue
        unc_per.append(c / n); unc_c += c; unc_n += n
        ca, na = uncensored_counts(aj, t, e)
        age_c += ca; age_n += na
    unc_per = np.array(unc_per)
    return dict(
        mean_c=float(c_per.mean()) if c_per.size else float("nan"),
        mean_c_excl_mort=float(c_panel.mean()) if c_panel.size else float("nan"),
        mortality_c=float(mort_c), median_c=float(np.median(c_per)) if c_per.size else float("nan"),
        n_panel=int(c_per.size),
        unc_c_pooled=float(unc_c / unc_n) if unc_n else float("nan"),
        unc_c_disease_mean=float(unc_per.mean()) if unc_per.size else float("nan"),
        unc_frac_above_half=float(np.mean(unc_per > 0.5)) if unc_per.size else float("nan"),
        unc_age_floor=float(age_c / age_n) if age_n else float("nan"),
        n_unc=int(unc_per.size),
    )


def paired_view(hX_mean, hY_mean, X_by_seed, Y_by_seed, T, E, M, age, panel_cols):
    """View 3: paired head-to-head of X (candidate) vs Y (reference)."""
    def run(hX, hY):
        xw = yw = comp = favx = favy = 0
        for j in panel_cols:
            m = M[:, j]
            if int(m.sum()) < 2 or int(E[m, j].sum()) < PAIR_MIN_EVENTS:
                continue
            a, b, c = paired_disagreements(hX[m, j], hY[m, j], T[m, j], E[m, j])
            xw += a; yw += b; comp += c
            if a > b: favx += 1
            elif b > a: favy += 1
        return xw, yw, comp, favx, favy

    xw, yw, comp, favx, favy = run(hX_mean, hY_mean)
    disagree = xw + yw
    winrate = 100 * xw / disagree if disagree else float("nan")
    nfav = favx + favy
    if nfav:
        z = (abs(favx - nfav / 2) - 0.5) / sqrt(nfav / 4)   # continuity-corrected
        z = max(z, 0.0)
        p = 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))
    else:
        p = float("nan")

    # per-seed reproducibility: pair by matching SEED NUMBER, not position
    shared = sorted(set(X_by_seed) & set(Y_by_seed))
    mism = (set(X_by_seed) != set(Y_by_seed))
    if mism:
        print(f"  WARNING: candidate seeds {sorted(X_by_seed)} != reference "
              f"{sorted(Y_by_seed)}; per-seed check uses intersection {shared}")
    seed_rates = []
    for s in shared:
        a, b, _, _, _ = run(X_by_seed[s], Y_by_seed[s])
        if a + b:
            seed_rates.append(100 * a / (a + b))

    def vs_age(h):
        xw = aw = 0
        for j in panel_cols:
            m = M[:, j]
            if int(m.sum()) < 2 or int(E[m, j].sum()) < PAIR_MIN_EVENTS:
                continue
            a, b, _ = paired_disagreements(h[m, j], age[m], T[m, j], E[m, j])
            xw += a; aw += b
        return 100 * xw / (xw + aw) if (xw + aw) else float("nan")

    return dict(
        winrate_candidate=winrate, comparable=int(comp), disagree=int(disagree),
        disagree_pct=100 * disagree / comp if comp else float("nan"),
        diseases_favor_candidate=favx, diseases_favor_reference=favy, sign_test_p=p,
        seed_winrates=[round(r, 2) for r in seed_rates], n_matched_seeds=len(shared),
        seed_set_mismatch=bool(mism),
        seed_min=min(seed_rates) if seed_rates else float("nan"),
        seed_max=max(seed_rates) if seed_rates else float("nan"),
        candidate_vs_age=vs_age(hX_mean), reference_vs_age=vs_age(hY_mean),
    )


# ------------------------------------------------------------------ selected-disease list (monitoring)
def _phecode_names(runs_dir: str, prefix: str, P: int) -> list[str]:
    cfgp = os.path.join(discover_seeds(runs_dir, prefix)[0], "config.json")
    if os.path.exists(cfgp):
        cfg = json.load(open(cfgp))
        for k in ("phecodes", "phecode_cols", "outcomes", "columns"):
            v = cfg.get(k)
            if isinstance(v, list) and len(v) == P:
                return [str(x) for x in v]
    return [f"col{j}" for j in range(P)]


# The tiered watch view (importance + detectability + surprise), CI-aware flags, and the
# emerging-signal scan live in curate_selected_diseases.py (tiered_watch_view / run_emerging_scan
# / print_tiered_watch / print_emerging), lazy-imported in main() to avoid a circular
# import. They reuse the harness primitives above so the monitoring numbers stay
# consistent.


# --------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Concordance benchmark for UKB disease-prediction models.")
    ap.add_argument("--candidates", nargs="+", required=True,
                    help="run-dir prefixes to evaluate (e.g. embedding_fixed_split)")
    ap.add_argument("--reference", required=True,
                    help="run-dir prefix used as the paired-comparison anchor")
    ap.add_argument("--runs-dir", default=DEFAULT_RUNS)
    ap.add_argument("--dem", default=DEFAULT_DEM)
    ap.add_argument("--out", default="./benchmark_out")
    ap.add_argument("--selected_diseases-frozen", default=None,
                    help="frozen selected_diseases JSON (default: benchmark/selected_diseases_frozen.json if present)")
    ap.add_argument("--watch-csv", default=None,
                    help="raw phecode CSV for any-positive/emerging recompute (default: config)")
    ap.add_argument("--no-emerging", action="store_true",
                    help="skip the per-run emerging-signal panel scan")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Tiered selected-disease list plus emerging scan (monitoring only, never affects the 3 views).
    # Lazy import avoids a circular import with curate_selected_diseases.
    from ukb_disease.benchmark.curate_selected_diseases import (
        tiered_watch_view, run_emerging_scan, print_tiered_watch, print_emerging, load_sex,
    )
    from ukb_disease.baseline.config import MORTALITY_PHECODES_CSV
    frozen = load_frozen(args.selected_diseases_frozen)
    watch_csv = args.watch_csv or MORTALITY_PHECODES_CSV
    model_hazards = {}

    ref_mean, ref_by_seed, base, _ = load_model(args.runs_dir, args.reference)
    T, E, M, eids = base["T"], base["E"], base["M"], base["eids"]
    N, P = T.shape
    all_cols, panel_cols, death = disease_columns(args.runs_dir, args.reference, P)
    age = load_age(args.dem, eids)
    watch_sex = load_sex(args.dem, eids)

    print(f"\n{'='*72}\nCONCORDANCE BENCHMARK   N={N} subjects | {len(panel_cols)} disease "
          f"phecodes (+mortality) considered | age for {int(np.isfinite(age).sum())}/{N}"
          f"\n{'='*72}")

    out = {"reference": args.reference, "n_subjects": int(N),
           "n_phecodes_considered": len(panel_cols), "models": {},
           "paired_vs_reference": {}}

    names = _phecode_names(args.runs_dir, args.reference, P)

    ref_v = single_model_views(ref_mean, T, E, M, age, all_cols, panel_cols, death)
    ref_v["selected_diseases"] = tiered_watch_view(ref_mean, T, E, M, age, watch_sex, names, eids, frozen, watch_csv)
    out["models"][args.reference] = ref_v
    model_hazards[args.reference] = ref_mean
    _print_single(args.reference + "  [REFERENCE]", ref_v)
    print_tiered_watch(args.reference, ref_v["selected_diseases"])

    for cand in args.candidates:
        c_mean, c_by_seed, cbase, _ = load_model(args.runs_dir, cand)
        assert np.array_equal(cbase["eids"], eids), f"{cand} eids differ from reference"
        v = single_model_views(c_mean, T, E, M, age, all_cols, panel_cols, death)
        v["selected_diseases"] = tiered_watch_view(c_mean, T, E, M, age, watch_sex, names, eids, frozen, watch_csv)
        out["models"][cand] = v
        model_hazards[cand] = c_mean
        _print_single(cand, v)
        print_tiered_watch(cand, v["selected_diseases"])
        if cand == args.reference:
            continue
        pv = paired_view(c_mean, ref_mean, c_by_seed, ref_by_seed, T, E, M, age, panel_cols)
        out["paired_vs_reference"][cand] = pv
        _print_paired(cand, args.reference, pv)

    if frozen is not None and not args.no_emerging:
        out["emerging"] = run_emerging_scan(model_hazards, T, E, M, age, watch_sex, names, eids,
                                            frozen, watch_csv)
        print_emerging(out["emerging"])

    jpath = os.path.join(args.out, "concordance_benchmark.json")
    json.dump(out, open(jpath, "w"), indent=2)
    print(f"\nSaved -> {jpath}\n")


def _print_single(name, v):
    print(f"\n--- {name} ---")
    print(f"  [1] MEAN C-INDEX (headline, incl mortality): {v['mean_c']:.4f}  "
          f"(panel-only {v['mean_c_excl_mort']:.4f}, mortality {v['mortality_c']:.4f}, "
          f"median {v['median_c']:.4f}, n={v['n_panel']})")
    print(f"  [2] UNCENSORED C (timing, pooled): {v['unc_c_pooled']:.4f}   "
          f"age-only floor: {v['unc_age_floor']:.4f}   "
          f"frac diseases>0.5: {v['unc_frac_above_half']:.2f}  (n={v['n_unc']})")
    if not np.isnan(v["unc_c_pooled"]):
        gap = v["unc_c_pooled"] - v["unc_age_floor"]
        verdict = ("genuine onset-timing beyond age" if gap > 0.02 else
                   "timing ~ no better than age/labels" if gap < 0.01 else "borderline")
        print(f"      -> {verdict} (model - age = {gap:+.3f})")


def _print_paired(cand, ref, pv):
    print(f"\n  [3] PAIRED HEAD-TO-HEAD: {cand}  vs  {ref}")
    print(f"      models disagree on {pv['disagree_pct']:.1f}% of comparable pairs")
    print(f"      >>> {cand} wins {pv['winrate_candidate']:.1f}% of disagreements "
          f"(50% = tie)")
    print(f"      reproducibility across {pv['n_matched_seeds']} seeds: "
          f"{pv['seed_min']:.1f}%-{pv['seed_max']:.1f}% {pv['seed_winrates']}"
          + ("  [SEED-SET MISMATCH]" if pv["seed_set_mismatch"] else ""))
    print(f"      breadth: {pv['diseases_favor_candidate']} diseases favor {cand} vs "
          f"{pv['diseases_favor_reference']} (sign-test p={pv['sign_test_p']:.1e})")
    print(f"      guardrail vs dumb age model: {cand} {pv['candidate_vs_age']:.1f}% | "
          f"{ref} {pv['reference_vs_age']:.1f}%  (both >>50% = real signal beyond age)")


if __name__ == "__main__":
    main()
