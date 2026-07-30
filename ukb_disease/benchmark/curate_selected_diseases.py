"""Curate the data-driven detectability and surprise axes of the selected-disease list.

One-time scan that turns the importance-only selected-disease list into a three-axis curated
list, freezing the result into `selected_diseases_frozen.json`. It keeps two reasons-to-watch
separate and never lets one gate the other:

  1. Importance: clinical or public-health priority (the hand-curated `SELECTED_DISEASES`,
     watched regardless of detectability). Loaded, not computed here.
  2. Detectability: the model predicts the disease with statistical confidence.
     Per-disease incident Harrell-C, with a subject-cluster bootstrap CI, must clear
     all of {CI_lo > 0.5, BH-FDR q_chance < 0.05, point C >= 0.65, n_event >= 20} in
     both models (cross-model replication). Low-prior-plausibility diseases can qualify.
  3. Surprise: a detectability member that is (a) low-prior plausibility category and
     (b) beyond-age (the wrist beats an age-only model: paired bootstrap delta CI_lo > 0
     and BH-FDR q_beyond < 0.05, in both models). Uncapped.

Plausibility is a pre-registered tag (selected_diseases.is_plausible), used only to mark
surprises. It does not filter detectability.

Events are incident-primary: the on-disk test `eval_mask` already excludes prevalent
and 7-day-zone subjects (baseline/outcomes_processing.build_survival_arrays), so the
incident bar is computed directly off the artifacts and matches per_disease_c.json. The
any-positive C (prevalent re-included as immediate events) is reconstructed from the raw
phecode CSV and reported alongside for context. It never gates anything.

Reuses the shared primitives (harrell_counts, load_model, load_age, bh_fdr,
percentile_ci, load_phecode_meta) so it cannot drift from the main harness.

Run (sharded curation, merge after):
  python3 -m ukb_disease.benchmark.curate_selected_diseases --mode shard \
      --models embedding_disjoint_split sequence_model_disjoint_split --task-id $T --n-tasks 32 --out DATA/selected_diseases
  python3 -m ukb_disease.benchmark.curate_selected_diseases --mode merge \
      --models embedding_disjoint_split sequence_model_disjoint_split --out DATA/selected_diseases \
      --frozen-out benchmark/selected_diseases_frozen.json

Smoke test (a few diseases, no sharding):
  python3 -m ukb_disease.benchmark.curate_selected_diseases --mode shard \
      --diseases NS_324.11 CV_416.2 time_to_death --n-boot 200 --out /tmp/wl_smoke --task-id 0 --n-tasks 1
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ukb_disease.benchmark.concordance_benchmark import (
    DEFAULT_DEM,
    DEFAULT_RUNS,
    disease_columns,
    harrell_counts,
    load_age,
    load_model,
)
from ukb_disease.benchmark.per_disease_c import phecode_names
from ukb_disease.benchmark.selected_diseases import SELECTED_DISEASES, DISJOINT_SPLIT_NEW, is_plausible
from ukb_disease.baseline._per_phecode_utils import bh_fdr, load_phecode_meta, percentile_ci
from ukb_disease.baseline.config import (
    MORTALITY_PHECODES_CSV,
    PHECODE_MAP_CSV,
    resolve_oak_path,
)
from ukb_disease.paths import UKB_ROOT

# Locked selection thresholds for the selected-disease list.
SEVEN_DAY = 7.0 / 365.25
B_CURATE = 2000          # bootstrap resamples for curation
C_FLOOR = 0.65           # point-C effect floor (detectability)
N_EVENT_FLOOR = 20       # min incident events for a stable per-disease C
FDR_ALPHA = 0.05         # BH-FDR level
PLAUSIBLE_CAP = 20       # cap on plausible detectability members (surprises uncapped)
CI_LO_PCT, CI_HI_PCT = 2.5, 97.5

DEFAULT_MODELS = ("embedding_disjoint_split", "sequence_model_disjoint_split")
DEFAULT_OUT = f"{UKB_ROOT}/selected_diseases"


# Statistics.
def weighted_c(h, t, e, w):
    """Count-weighted Harrell C on a single risk set; w = bootstrap multiplicity (>=0).
    Vectorized (event x member) grid, identical to harrell_counts when w is all-ones.
    Inputs must already be restricted to finite-hazard rows. Returns NaN if no comparable
    pair. Higher h = more risk = earlier event."""
    ev = (e == 1) & (w > 0)
    if not ev.any():
        return np.nan
    pos = w > 0
    h_e, t_e, w_e = h[ev], t[ev], w[ev]
    h_b, t_b, e_b, w_b = h[pos], t[pos], e[pos], w[pos]
    comp_mask = (t_b[None, :] > t_e[:, None]) | (
        (t_b[None, :] == t_e[:, None]) & (e_b[None, :] == 0))
    wpair = (w_e[:, None] * w_b[None, :]) * comp_mask
    comp = float(wpair.sum())
    if comp <= 0:
        return np.nan
    d = h_e[:, None] - h_b[None, :]
    conc = float((wpair * ((d > 0) + 0.5 * (d == 0))).sum())
    return conc / comp


def paired_bootstrap(scores, t, e, n_boot, seed, model_keys):
    """Subject-cluster bootstrap on a common risk set (1 row/subject, so cluster=subject).
    `scores` = dict name->risk-score array (finite, aligned). One resample per replicate,
    shared across every score, so every pairwise delta (model vs age, model vs demographics)
    and the cross-model mean are all paired (low-variance, valid CIs). `model_keys` lists
    which score keys are models (averaged into the reference '__mean__' draw).

    Returns {name: boot_c[B] for each score} plus '__mean__' = cross-model-mean boot[B]."""
    n = len(t)
    rng = np.random.default_rng(seed)
    out = {k: np.full(n_boot, np.nan) for k in scores}
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        for k, s in scores.items():
            out[k][b] = weighted_c(s, t, e, w)
    out["__mean__"] = np.nanmean(np.stack([out[k] for k in model_keys]), axis=0)
    return out


def one_sided_p(boot, threshold, *, leq):
    """One-sided bootstrap p-value with the +1/(B+1) correction. `leq=True` tests the
    fraction of replicates at/below `threshold` (H0: stat <= threshold)."""
    v = np.asarray(boot, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    hits = int(np.sum(v <= threshold)) if leq else int(np.sum(v >= threshold))
    return (1.0 + hits) / (v.size + 1.0)


# Data loading.
def load_models(runs_dir, prefixes):
    """Return ({prefix: mean_hazards[N,P]}, shared base{eids,T,E,M}, names, panel_cols,
    death_idx). Asserts all models share eids/T/E/M (so the risk set is identical)."""
    runs_dir = resolve_oak_path(runs_dir)
    hazards, base = {}, None
    for pfx in prefixes:
        mean_h, _by_seed, b, _seeds = load_model(runs_dir, pfx)
        hazards[pfx] = mean_h
        if base is None:
            base = b
        else:
            assert np.array_equal(b["eids"], base["eids"]), f"{pfx} eids differ"
            assert np.array_equal(b["E"], base["E"]), f"{pfx} is_event differ"
            assert np.array_equal(b["M"], base["M"]), f"{pfx} eval_mask differ"
    _N, P = base["T"].shape
    names = phecode_names(runs_dir, prefixes[0], P)
    _all, panel_cols, death = disease_columns(runs_dir, prefixes[0], P)
    return hazards, base, names, panel_cols, death


def load_anypos_raw(csv_path, eids, cols):
    """Raw years matrix (N, len(cols)) for the requested phecode columns, reindexed to
    `eids`. NaN = never-diagnosed. Used to re-include prevalent cases for the any-positive
    variant (build_survival_arrays train rule)."""
    csv_path = resolve_oak_path(csv_path)
    usecols = ["eid"] + [c for c in cols if c != "eid"]
    df = pd.read_csv(csv_path, usecols=lambda c: c in set(usecols))
    df = df.drop_duplicates("eid").set_index("eid")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"phecode columns absent from {csv_path}: {missing[:5]} ...")
    return df.reindex(index=np.asarray(eids, dtype=np.int64))[cols].to_numpy(dtype=np.float64)


def load_sex(dem_path, eids):
    """Sex per subject (0/1) from the demographics CSV, NaN if missing. Used to build the
    age+sex demographic floor for the surprise gate. The wrist encodes sex strongly
    (AUROC around 0.99), so an age-only floor would over-credit sex-imbalanced diseases
    as 'surprises'."""
    dem = pd.read_csv(resolve_oak_path(dem_path), usecols=["eid", "sex"])
    smap = dict(zip(dem.eid.values, dem.sex.values))
    return np.array([smap.get(int(x), np.nan) for x in eids], dtype=float)


def _demo_score(age, sex, e):
    """Age+sex demographic risk score = logistic(event ~ age + sex) linear predictor on the
    at-risk set. The in-sample fit makes this a deliberately strong, conservative floor, so
    a wrist signal that still beats it is demographics-independent. Higher = more risk.
    Falls back to raw age if sex is unusable or a class is absent. Only ranks matter, so the
    scale is irrelevant."""
    age = np.asarray(age, float)
    sex = np.asarray(sex, float)
    y = (np.asarray(e) == 1).astype(int)
    ok = np.isfinite(age) & np.isfinite(sex)
    if y.sum() < 1 or y.sum() == y.size or ok.sum() < y.size or np.unique(sex[ok]).size < 2:
        return age
    try:
        clf = LogisticRegression(max_iter=1000)
        clf.fit(np.column_stack([age, sex]), y)
        return clf.decision_function(np.column_stack([age, sex]))
    except Exception:
        return age


def anypos_arrays(raw_col, max_follow_up):
    """(t, e) for the any-positive definition of one phecode column (all subjects at-risk).
    incident (raw>7d) = event at raw; effective-prevalent (raw<=7d incl <0) = immediate
    event at ~0; NaN = censored at max_follow_up. Mirrors the train/val rule."""
    nan = np.isnan(raw_col)
    e = np.zeros_like(raw_col)
    t = np.full_like(raw_col, max_follow_up)
    incident = (~nan) & (raw_col > SEVEN_DAY)
    e[incident] = 1.0
    t[incident] = raw_col[incident]
    effprev = (~nan) & (raw_col <= SEVEN_DAY)
    e[effprev] = 1.0
    t[effprev] = 0.0001
    return t, e


# Scoring.
def score_disease(j, name, hazards, base, age, sex, raw_anypos, max_follow_up,
                  prefixes, do_bootstrap, n_boot, seed):
    """Score ONE phecode column across all models. Returns the per-disease record.

    Incident point C uses harrell_counts on the eval_mask (matches per_disease_c.json).
    The bootstrap (only when do_bootstrap) runs on the common finite-hazard risk set
    shared by all models, so per-model CI, cross-model mean CI, and the model vs age delta
    CI are all paired."""
    T, E, M = base["T"], base["E"], base["M"]
    m = M[:, j]
    n_eval = int(m.sum())
    n_event = int(E[m, j].sum())  # oracle convention (matches per_disease_c.json)

    rec = {"phecode": name, "j": int(j), "n_eval": n_eval, "n_event": n_event,
           "per_model": {}, "bootstrapped": False}

    # incident point C per model (oracle)
    for pfx in prefixes:
        conc, comp = harrell_counts(hazards[pfx][m, j], T[m, j], E[m, j])
        c_inc = round(conc / comp, 6) if comp > 0 else None
        rec["per_model"][pfx] = {"c_inc": c_inc}

    # any-positive point C per model (all subjects at-risk, finite hazard)
    t_ap, e_ap = anypos_arrays(raw_anypos, max_follow_up)
    n_event_anypos = int(e_ap.sum())
    rec["n_event_anypos"] = n_event_anypos
    for pfx in prefixes:
        fin = np.isfinite(hazards[pfx][:, j])
        c_ap = weighted_c(hazards[pfx][fin, j], t_ap[fin], e_ap[fin],
                          np.ones(int(fin.sum())))
        rec["per_model"][pfx]["c_anypos"] = round(c_ap, 6) if np.isfinite(c_ap) else None

    if not do_bootstrap:
        return rec

    # common finite-hazard risk set (so a single resample aligns across models + age)
    fin = m.copy()
    for pfx in prefixes:
        fin &= np.isfinite(hazards[pfx][:, j])
    idx = np.where(fin)[0]
    if idx.size < 2 or int(E[idx, j].sum()) < 1:
        return rec

    h_models = {pfx: hazards[pfx][idx, j] for pfx in prefixes}
    t_idx, e_idx, age_idx, sex_idx = T[idx, j], E[idx, j], age[idx], sex[idx]
    demo_lp = _demo_score(age_idx, sex_idx, e_idx)          # age+sex floor
    scores = {**h_models, "age": age_idx, "demo": demo_lp}
    bs = paired_bootstrap(scores, t_idx, e_idx, n_boot, seed, model_keys=prefixes)

    rec["bootstrapped"] = True
    rec["n_event_used"] = int(e_idx.sum())  # finite-hazard events actually used in C
    ones = np.ones(idx.size)
    ref_lo, ref_hi = (float(x) for x in percentile_ci(bs["__mean__"], CI_LO_PCT, CI_HI_PCT))
    c_inc_vals = [rec["per_model"][p]["c_inc"] for p in prefixes
                  if rec["per_model"][p]["c_inc"] is not None]
    c_age_point = weighted_c(age_idx, t_idx, e_idx, ones)
    c_demo_point = weighted_c(demo_lp, t_idx, e_idx, ones)
    rec["ref"] = {
        "c_inc": round(float(np.mean(c_inc_vals)), 6) if c_inc_vals else None,
        "ci_lo": round(ref_lo, 6), "ci_hi": round(ref_hi, 6),
        "c_age": round(float(c_age_point), 6) if np.isfinite(c_age_point) else None,
        "c_demo": round(float(c_demo_point), 6) if np.isfinite(c_demo_point) else None,
    }
    for pfx in prefixes:
        bc = bs[pfx]
        lo, hi = (float(x) for x in percentile_ci(bc, CI_LO_PCT, CI_HI_PCT))
        d_age = bc - bs["age"]
        da_lo, da_hi = (float(x) for x in percentile_ci(d_age, CI_LO_PCT, CI_HI_PCT))
        d_demo = bc - bs["demo"]
        dd_lo, dd_hi = (float(x) for x in percentile_ci(d_demo, CI_LO_PCT, CI_HI_PCT))
        pm = rec["per_model"][pfx]
        pm.update(
            ci_lo=round(lo, 6), ci_hi=round(hi, 6),
            c_age=round(float(c_age_point), 6) if np.isfinite(c_age_point) else None,
            c_demo=round(float(c_demo_point), 6) if np.isfinite(c_demo_point) else None,
            delta_age=round(float(np.nanmean(d_age)), 6),
            delta_age_lo=round(da_lo, 6), delta_age_hi=round(da_hi, 6),
            delta_demo=round(float(np.nanmean(d_demo)), 6),
            delta_demo_lo=round(dd_lo, 6), delta_demo_hi=round(dd_hi, 6),
            p_chance=one_sided_p(bc, 0.5, leq=True),
            p_beyond_age=one_sided_p(d_age, 0.0, leq=True),
            p_beyond_demo=one_sided_p(d_demo, 0.0, leq=True),
        )
    return rec


def should_bootstrap(j, name, base, prefixes):
    """Bootstrap iff a disease could matter for curation: n_event >= floor in EITHER model
    (here labels are shared so it's one value), OR it's an importance member (so its
    reference snapshot is recorded even when below the event floor, e.g. Alzheimer's)."""
    n_event = int(base["E"][base["M"][:, j], j].sum())
    return (n_event >= N_EVENT_FLOOR) or (name in SELECTED_DISEASES)


# Shard / merge.
def run_shard(args):
    prefixes = list(args.models)
    hazards, base, names, panel_cols, death = load_models(args.runs_dir, prefixes)
    age = load_age(resolve_oak_path(args.dem), base["eids"])
    sex = load_sex(args.dem, base["eids"])
    max_follow_up = float(base["T"].max())

    # which columns this shard handles
    if args.diseases:
        want = set(args.diseases)
        cols = [j for j, n in enumerate(names) if n in want]
    else:
        # round-robin over ALL columns (panel + mortality) so importance mortality is covered
        cols = list(range(len(names)))[args.task_id::args.n_tasks]

    need = [j for j in cols if (names[j] in SELECTED_DISEASES) or (j in panel_cols)]
    raw = load_anypos_raw(args.csv, base["eids"], [names[j] for j in need])
    raw_by_j = {j: raw[:, k] for k, j in enumerate(need)}

    out = {"shard": args.task_id, "n_tasks": args.n_tasks, "models": prefixes,
           "n_event_floor": N_EVENT_FLOOR, "n_boot": args.n_boot,
           "max_follow_up": max_follow_up, "diseases": {}}
    for j in need:
        name = names[j]
        do_bs = should_bootstrap(j, name, base, prefixes)
        # per-disease seed shared across models so resamples align (reproducible)
        seed = args.seed + j
        rec = score_disease(j, name, hazards, base, age, sex, raw_by_j[j], max_follow_up,
                            prefixes, do_bs, args.n_boot, seed)
        rec["is_mortality"] = bool(j in death)
        rec["in_importance"] = bool(name in SELECTED_DISEASES)
        out["diseases"][name] = rec

    out_dir = resolve_oak_path(args.out)
    os.makedirs(os.path.join(out_dir, "shards"), exist_ok=True)
    p = os.path.join(out_dir, "shards", f"shard_{args.task_id:03d}.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"[shard {args.task_id}/{args.n_tasks}] scored {len(out['diseases'])} diseases "
          f"({sum(d['bootstrapped'] for d in out['diseases'].values())} bootstrapped) to {p}")


def _passes(pm, n_event):
    """Per-model detectability predicate (incident): CI excludes 0.5, FDR-significant,
    effect floor, event floor. Requires the bootstrap fields + q_chance (added in merge)."""
    return (pm.get("ci_lo") is not None and pm["ci_lo"] > 0.5
            and pm.get("q_chance") is not None and pm["q_chance"] < FDR_ALPHA
            and pm.get("c_inc") is not None and pm["c_inc"] >= C_FLOOR
            and n_event >= N_EVENT_FLOOR)


def _beyond_age(pm):
    """Wrist beats an age-only model (reported for transparency)."""
    return (pm.get("delta_age_lo") is not None and pm["delta_age_lo"] > 0
            and pm.get("q_beyond_age") is not None and pm["q_beyond_age"] < FDR_ALPHA)


def _beyond_demo(pm):
    """Wrist beats an age+sex demographic model, the surprise gate. Removes detections
    that merely exploit the wrist's sex/age encoding rather than a disease signal."""
    return (pm.get("delta_demo_lo") is not None and pm["delta_demo_lo"] > 0
            and pm.get("q_beyond_demo") is not None and pm["q_beyond_demo"] < FDR_ALPHA)


def merge_shards(args):
    prefixes = list(args.models)
    out_dir = resolve_oak_path(args.out)
    shard_files = sorted(glob.glob(os.path.join(out_dir, "shards", "shard_*.json")))
    if not shard_files:
        raise FileNotFoundError(f"no shards under {out_dir}/shards")
    diseases = {}
    meta = None
    for f in shard_files:
        s = json.load(open(f))
        meta = meta or s
        diseases.update(s["diseases"])
    print(f"[merge] {len(shard_files)} shards, {len(diseases)} diseases")

    # phecode descriptions / categories
    phe_meta = load_phecode_meta(resolve_oak_path(PHECODE_MAP_CSV))

    # BH-FDR per model over the scan family (bootstrapped, n_event >= floor).
    for pfx in prefixes:
        fam = [(name, d) for name, d in diseases.items()
               if d.get("bootstrapped") and d["n_event"] >= N_EVENT_FLOOR
               and not d.get("is_mortality")]
        for key in ("p_chance", "p_beyond_age", "p_beyond_demo"):
            pv = np.array([d["per_model"][pfx].get(key, np.nan) for _, d in fam], dtype=float)
            q, _ = bh_fdr(pv, FDR_ALPHA)
            qkey = key.replace("p_", "q_")
            for (name, d), qv in zip(fam, q):
                d["per_model"][pfx][qkey] = (float(qv) if np.isfinite(qv) else None)

    # Per-disease detectability / beyond-age (cross-model replication).
    for name, d in diseases.items():
        desc = phe_meta.get(name, {}).get("name", SELECTED_DISEASES.get(name, name))
        cat = phe_meta.get(name, {}).get("category", "Other")
        n_event = d["n_event"]
        det = (not d.get("is_mortality")) and all(
            _passes(d["per_model"][p], n_event) for p in prefixes)
        bage = all(_beyond_age(d["per_model"][p]) for p in prefixes)
        bdemo = all(_beyond_demo(d["per_model"][p]) for p in prefixes)
        plaus = is_plausible(name, desc)
        # Surprise gate uses beyond age+sex (bdemo), not age alone. This strips sex-confounded
        # "surprises" (the wrist encodes sex), keeping only demographics-independent signal.
        d.update(label=SELECTED_DISEASES.get(name, desc), description=desc, category=cat,
                 plausibility=("plausible" if plaus else "low-prior"),
                 is_plausible=bool(plaus), is_new=bool(name in DISJOINT_SPLIT_NEW),
                 detectable=bool(det), beyond_age=bool(bage), beyond_demo=bool(bdemo),
                 surprise=bool(det and (not plaus) and bdemo))

    # Tier assembly (importance frozen; plausible-detectable capped; surprise uncapped).
    importance = [n for n in SELECTED_DISEASES]
    det_all = [n for n, d in diseases.items() if d["detectable"]]
    surprise = sorted(
        [n for n in det_all if diseases[n]["surprise"]],
        key=lambda n: -min(diseases[n]["per_model"][p].get("delta_demo_lo", -9) for p in prefixes))
    plaus_det = sorted(
        [n for n in det_all if diseases[n]["is_plausible"]],
        key=lambda n: -min(diseases[n]["per_model"][p].get("ci_lo", 0) for p in prefixes))
    # Detectability tier = the expected wrist-detectable diseases (top plausible, capped)
    # plus the surprises (low-prior detectable that beat age+sex, uncapped). Low-prior
    # diseases detectable only via demographics (not beyond-demo) are neither important,
    # plausible, nor surprising, so they are dropped from the frozen list (they still
    # surface in the per-run emerging scan if they ever clear the surprise bar).
    detectability = list(dict.fromkeys(plaus_det[:PLAUSIBLE_CAP] + surprise))

    # axis tags + primary display tier
    for name, d in diseases.items():
        d["axes"] = {
            "importance": name in SELECTED_DISEASES,
            "detectability": name in detectability,
            "surprise": name in surprise,
        }
        d["tier"] = ("importance" if d["axes"]["importance"]
                     else "surprise" if d["axes"]["surprise"]
                     else "detectability" if d["axes"]["detectability"]
                     else None)

    # Write frozen artifact.
    frozen = {
        "schema_version": "selected_diseases_v2",
        "frozen": True,
        "curated_from": {
            "models": prefixes, "split": "disjoint_split_v1",
            "runs_dir": resolve_oak_path(args.runs_dir),
            "n_subjects": _n_subjects_from_runs(args.runs_dir, prefixes[0]),
        },
        "selection_params": {
            "B": meta["n_boot"], "ci_pct": [CI_LO_PCT, CI_HI_PCT],
            "c_floor": C_FLOOR, "n_event_floor": N_EVENT_FLOOR, "fdr_alpha": FDR_ALPHA,
            "plausible_cap": PLAUSIBLE_CAP, "event_def": "incident-primary",
            "surprise_gate": "beyond age+SEX (demographic logistic floor): paired delta_demo "
                             "CI_lo > 0 AND BH q_beyond_demo < FDR, both models",
            "beyond_age_rule": "paired delta_age CI_lo > 0 AND BH q_beyond_age < FDR (reported)",
            "replication": "BOTH models clear the bar",
        },
        "tiers": {"importance": importance, "detectability": detectability, "surprise": surprise},
        "diseases": {},
    }
    keep = set(importance) | set(detectability) | set(surprise)
    for name in sorted(keep):
        d = diseases.get(name)
        if d is None:
            frozen["diseases"][name] = {"label": SELECTED_DISEASES.get(name, name),
                                        "in_panel": False, "note": "not scored (below floor / absent)"}
            continue
        ref = d.get("ref", {})
        frozen["diseases"][name] = {
            "label": d.get("label", name), "description": d.get("description", name),
            "category": d.get("category", "Other"),
            "axes": d["axes"], "tier": d["tier"],
            "is_new": d.get("is_new", False), "plausibility": d.get("plausibility"),
            "is_plausible": d.get("is_plausible"), "is_mortality": d.get("is_mortality", False),
            "detectable": d.get("detectable"), "beyond_age": d.get("beyond_age"),
            "beyond_demo": d.get("beyond_demo"),
            "reference": {
                "C_incident": ref.get("c_inc"), "CI_lo": ref.get("ci_lo"),
                "CI_hi": ref.get("ci_hi"), "C_age": ref.get("c_age"), "C_demo": ref.get("c_demo"),
                "n_event_incident": d["n_event"], "n_event_anypos": d.get("n_event_anypos"),
            },
            "per_model": {p: _public_pm(d["per_model"][p]) for p in prefixes},
        }

    frozen_local = args.frozen_out
    oak_json = resolve_oak_path(os.path.join(out_dir, "selected_diseases_frozen.json"))
    oak_csv = resolve_oak_path(os.path.join(out_dir, "selected_diseases_frozen.csv"))
    # The canonical copy may be a read-only mount, so guard the write.
    try:
        json.dump(frozen, open(oak_json, "w"), indent=2)
    except OSError as exc:
        print(f"[merge] WARN: could not write OAK copy {oak_json} ({exc}); relying on --frozen-out")
        oak_csv = None
    if frozen_local:
        os.makedirs(os.path.dirname(os.path.abspath(frozen_local)), exist_ok=True)
        json.dump(frozen, open(frozen_local, "w"), indent=2)
    _write_csv(frozen, prefixes, oak_csv,
               frozen_local.replace(".json", ".csv") if frozen_local else None)

    print(f"[merge] importance={len(importance)} detectability={len(detectability)} "
          f"surprise={len(surprise)} | total unique={len(keep)}")
    print(f"[merge] surprises: {surprise}")
    print(f"[merge] frozen to {os.path.join(out_dir, 'selected_diseases_frozen.json')}"
          + (f" + {frozen_local}" if frozen_local else ""))
    return frozen


def _public_pm(pm):
    keys = ("c_inc", "c_anypos", "ci_lo", "ci_hi", "c_age", "c_demo",
            "delta_age", "delta_age_lo", "delta_age_hi", "p_beyond_age", "q_beyond_age",
            "delta_demo", "delta_demo_lo", "delta_demo_hi", "p_beyond_demo", "q_beyond_demo",
            "p_chance", "q_chance")
    return {k: pm[k] for k in keys if k in pm}


def _n_subjects_from_runs(runs_dir, prefix):
    runs_dir = resolve_oak_path(runs_dir)
    sd = sorted(glob.glob(os.path.join(runs_dir, f"{prefix}_seed*")))
    if not sd:
        return None
    return int(np.load(os.path.join(sd[0], "test_eids.npy")).shape[0])


def _write_csv(frozen, prefixes, oak_path, local_path):
    rows = []
    for name, d in frozen["diseases"].items():
        if "reference" not in d:
            continue
        ref, ax = d["reference"], d["axes"]
        row = {
            "phecode": name, "label": d.get("label"), "category": d.get("category"),
            "tier": d.get("tier"), "axis_importance": ax["importance"],
            "axis_detectability": ax["detectability"], "axis_surprise": ax["surprise"],
            "is_new": d.get("is_new"), "plausibility": d.get("plausibility"),
            "detectable": d.get("detectable"), "beyond_age": d.get("beyond_age"),
            "beyond_demo": d.get("beyond_demo"),
            "C_incident": ref.get("C_incident"), "CI_lo": ref.get("CI_lo"),
            "CI_hi": ref.get("CI_hi"), "C_age": ref.get("C_age"), "C_demo": ref.get("C_demo"),
            "n_event_incident": ref.get("n_event_incident"),
            "n_event_anypos": ref.get("n_event_anypos"),
        }
        for p in prefixes:
            pm = d.get("per_model", {}).get(p, {})
            row[f"{p}_C_inc"] = pm.get("c_inc")
            row[f"{p}_CI_lo"] = pm.get("ci_lo")
            row[f"{p}_q_chance"] = pm.get("q_chance")
        rows.append(row)
    df = pd.DataFrame(rows)
    if oak_path:
        try:
            df.to_csv(oak_path, index=False)
        except OSError as exc:
            print(f"[merge] WARN: could not write OAK csv {oak_path} ({exc})")
    if local_path:
        df.to_csv(local_path, index=False)


# Per-run reporting (called by concordance_benchmark).
# These reuse the curation primitives so the monitoring numbers cannot drift from the
# frozen reference. concordance_benchmark imports them lazily (inside main) to avoid a
# circular import. Monitoring only; they never affect model selection.
WATCH_NBOOT = 400    # per-run CIs over the ~40 frozen diseases (cheap; flags only need
                     # a coarse CI, the frozen reference CIs are the B=2000 curation ones)


def _current_disease(h_col, age, sex, t_col, e_col, m, raw_col, max_follow_up, n_boot, seed):
    """Recompute a single disease for the model under eval. Returns current incident C
    (+bootstrap CI), age-only and age+sex C, beyond_age/beyond_demo flags, any-positive C."""
    conc, comp = harrell_counts(h_col[m], t_col[m], e_col[m])
    c_inc = (conc / comp) if comp > 0 else None
    fin = m & np.isfinite(h_col)
    idx = np.where(fin)[0]
    out = {"c_inc": (round(c_inc, 6) if c_inc is not None else None),
           "n_event": int(e_col[m].sum())}
    # any-positive (all subjects at-risk, finite hazard)
    finall = np.isfinite(h_col)
    t_ap, e_ap = anypos_arrays(raw_col, max_follow_up)
    c_ap = weighted_c(h_col[finall], t_ap[finall], e_ap[finall], np.ones(int(finall.sum())))
    out["c_anypos"] = round(float(c_ap), 6) if np.isfinite(c_ap) else None
    out["n_event_anypos"] = int(e_ap.sum())
    if idx.size >= 2 and int(e_col[idx].sum()) >= 1:
        demo_lp = _demo_score(age[idx], sex[idx], e_col[idx])
        bs = paired_bootstrap({"m": h_col[idx], "age": age[idx], "demo": demo_lp},
                              t_col[idx], e_col[idx], n_boot, seed, model_keys=["m"])
        lo, hi = (float(x) for x in percentile_ci(bs["m"], CI_LO_PCT, CI_HI_PCT))
        d_age_lo = float(percentile_ci(bs["m"] - bs["age"], CI_LO_PCT, CI_HI_PCT)[0])
        d_demo_lo = float(percentile_ci(bs["m"] - bs["demo"], CI_LO_PCT, CI_HI_PCT)[0])
        c_age = weighted_c(age[idx], t_col[idx], e_col[idx], np.ones(idx.size))
        c_demo = weighted_c(demo_lp, t_col[idx], e_col[idx], np.ones(idx.size))
        out.update(ci_lo=round(lo, 6), ci_hi=round(hi, 6),
                   c_age=round(float(c_age), 6) if np.isfinite(c_age) else None,
                   c_demo=round(float(c_demo), 6) if np.isfinite(c_demo) else None,
                   beyond_age=bool(d_age_lo > 0), beyond_demo=bool(d_demo_lo > 0))
    return out


def tiered_watch_view(hazards, T, E, M, age, sex, names, eids, frozen, csv_path,
                      n_boot=WATCH_NBOOT, seed=20260618):
    """Per-disease monitoring for the model under eval against the FROZEN reference.
    Returns {params, summary, diseases:{code:{axes,tier,current,reference,delta,flags}}}.
    Degrades to the importance-only set if `frozen` is None (pre-first-curation)."""
    col = {n: j for j, n in enumerate(names)}
    if frozen is None:  # graceful fallback: importance-only point C
        diseases = {}
        for code, label in SELECTED_DISEASES.items():
            j = col.get(code)
            rec = {"label": label, "in_panel": False,
                   "axes": {"importance": True, "detectability": False, "surprise": False},
                   "tier": "importance", "flags": []}
            if j is not None and int(M[:, j].sum()) >= 10 and int(E[M[:, j], j].sum()) >= 1:
                conc, comp = harrell_counts(hazards[M[:, j], j], T[M[:, j], j], E[M[:, j], j])
                if comp > 0:
                    rec.update(in_panel=True, current={"c_inc": round(conc / comp, 4),
                               "n_event": int(E[M[:, j], j].sum())})
            diseases[code] = rec
        return {"params": {"frozen": False}, "summary": {"n_total": len(diseases)},
                "diseases": diseases}

    fcodes = list(frozen["diseases"])
    need = [c for c in fcodes if c in col]
    raw = load_anypos_raw(csv_path, eids, need) if need else np.zeros((len(eids), 0))
    raw_by_code = {c: raw[:, k] for k, c in enumerate(need)}
    max_follow_up = float(T.max())

    diseases, flag_counts = {}, {}
    for code in fcodes:
        fz = frozen["diseases"][code]
        ref = fz.get("reference", {})
        rec = {"label": fz.get("label", code), "category": fz.get("category"),
               "axes": fz.get("axes", {}), "tier": fz.get("tier"),
               "is_new": fz.get("is_new", False), "plausibility": fz.get("plausibility"),
               "reference": ref, "in_panel": False, "flags": []}
        j = col.get(code)
        if j is not None:
            m = M[:, j]
            seed_c = seed + j
            cur = _current_disease(hazards[:, j], age, sex, T[:, j], E[:, j], m,
                                   raw_by_code[code], max_follow_up, n_boot, seed_c)
            rec["in_panel"] = cur["c_inc"] is not None
            rec["current"] = cur
            if cur["c_inc"] is not None and ref.get("C_incident") is not None:
                rec["delta_C_incident"] = round(cur["c_inc"] - ref["C_incident"], 6)
            # CI-aware flags vs frozen reference.
            flags = []
            if (cur.get("c_inc") is not None and ref.get("CI_lo") is not None
                    and cur["c_inc"] < ref["CI_lo"]):
                flags.append("REGRESSION")
            if cur.get("ci_lo") is not None and cur["ci_lo"] <= 0.5:
                flags.append("SIGNAL-LOST(CI<=0.5)")
            if (fz.get("axes", {}).get("surprise") and cur.get("beyond_demo") is False):
                flags.append("SIGNAL-LOST(beyond-demo off)")
            rec["flags"] = flags
            for fl in flags:
                flag_counts[fl] = flag_counts.get(fl, 0) + 1
        diseases[code] = rec

    n_in = sum(d["in_panel"] for d in diseases.values())
    return {"params": {"frozen": True, "n_boot": n_boot,
                       "curated_from": frozen.get("curated_from")},
            "summary": {"n_total": len(diseases), "n_in_panel": n_in,
                        "flag_counts": flag_counts},
            "diseases": diseases}


def run_emerging_scan(model_hazards, T, E, M, age, sex, names, eids, frozen, csv_path,
                      n_boot=WATCH_NBOOT, seed=20260619):
    """Per-run scan of the full panel for off-list diseases that newly clear the
    detectability bar (point C >= floor AND bootstrap CI_lo > 0.5 in all models, n_event
    >= floor). Candidate additions only; never auto-added (the list stays frozen).
    Uses the cross-model-replication bar minus the panel-wide FDR step, since this is a
    per-run alert rather than a freezing decision. Re-run curate_selected_diseases to formally add."""
    col = {n: j for j, n in enumerate(names)}
    on_list = set()
    if frozen:
        for axis in ("importance", "detectability", "surprise"):
            on_list |= set(frozen["tiers"].get(axis, []))
    models = list(model_hazards)
    max_follow_up = float(T.max())

    # cheap point screen over all panel diseases: require point C >= floor AND point
    # beyond-demographics (C_model > C_demo) in ALL models BEFORE the expensive bootstrap.
    # This prunes the broad demographics-explained signal so only candidate SURPRISES are
    # bootstrapped (keeps the per-run scan fast).
    shortlist = []
    for code, j in col.items():
        if code in on_list or code == "time_to_death":
            continue
        m = M[:, j]
        if int(E[m, j].sum()) < N_EVENT_FLOOR:
            continue
        cs = [(lambda cc: cc[0] / cc[1] if cc[1] > 0 else 0.0)(
                  harrell_counts(model_hazards[mh][m, j], T[m, j], E[m, j])) for mh in models]
        if not all(c >= C_FLOOR for c in cs):
            continue
        fin = m.copy()
        for mh in models:
            fin &= np.isfinite(model_hazards[mh][:, j])
        idx = np.where(fin)[0]
        if idx.size < 2 or int(E[idx, j].sum()) < 1:
            continue
        demo_lp = _demo_score(age[idx], sex[idx], E[idx, j])
        c_demo_pt = weighted_c(demo_lp, T[idx, j], E[idx, j], np.ones(idx.size))
        beyond_pt = all(
            weighted_c(model_hazards[mh][idx, j], T[idx, j], E[idx, j], np.ones(idx.size))
            > c_demo_pt for mh in models)
        if beyond_pt:
            shortlist.append((code, j, cs))

    phe_meta = load_phecode_meta(resolve_oak_path(PHECODE_MAP_CSV))
    raw = load_anypos_raw(csv_path, eids, [c for c, _, _ in shortlist]) if shortlist \
        else np.zeros((len(eids), 0))
    raw_by_code = {c: raw[:, k] for k, (c, _, _) in enumerate(shortlist)}

    candidates = []
    for code, j, cs in shortlist:
        per_model, ci_ok = {}, True
        beyond = True
        for k, mh in enumerate(models):
            cur = _current_disease(model_hazards[mh][:, j], age, sex, T[:, j], E[:, j], M[:, j],
                                   raw_by_code[code], max_follow_up, n_boot, seed + j + k)
            per_model[mh] = {"C": cur["c_inc"], "CI_lo": cur.get("ci_lo"),
                             "CI_hi": cur.get("ci_hi")}
            ci_ok &= (cur.get("ci_lo") is not None and cur["ci_lo"] > 0.5)
            beyond &= bool(cur.get("beyond_demo"))
        # Emerging = off-list diseases that clear the surprise-worthy bar (beyond chance AND
        # beyond age+sex, both models). Demographics-explained detections are NOT surfaced.
        if not (ci_ok and beyond):
            continue
        desc = phe_meta.get(code, {}).get("name", code)
        plaus = is_plausible(code, desc)
        candidates.append({
            "phecode": code, "label": desc,
            "category": phe_meta.get(code, {}).get("category", "Other"),
            "n_event_incident": int(E[M[:, j], j].sum()),
            "is_plausible": bool(plaus), "beyond_age": bool(beyond),
            "would_be_surprise": bool(beyond and not plaus),
            "per_model": per_model,
        })
    candidates.sort(key=lambda c: -min(c["per_model"][m]["C"] for m in models))
    return {"params": {"bar": "point C>=%.2f & CI_lo>0.5 in all models, n_event>=%d (no panel FDR)"
                       % (C_FLOOR, N_EVENT_FLOOR), "models": models},
            "scanned": len(col), "n_shortlist": len(shortlist),
            "n_candidates": len(candidates), "candidates": candidates}


def print_tiered_watch(model_name, wsec):
    s = wsec["summary"]
    fc = s.get("flag_counts", {})
    fstr = ("   FLAGS: " + ", ".join(f"{v} {k}" for k, v in fc.items())) if fc else ""
    print(f"\n  [SELECTED DISEASES] {model_name}  ({s.get('n_in_panel', 0)}/{s['n_total']} in panel)"
          f"{fstr}")
    if not wsec["params"].get("frozen"):
        print("      (importance-only fallback: run curate_selected_diseases to enable tiers)")
    order = {"importance": 0, "surprise": 1, "detectability": 2, None: 3}
    rows = sorted(wsec["diseases"].items(), key=lambda kv: (order.get(kv[1].get("tier"), 3),
                  -(kv[1].get("current", {}) or {}).get("c_inc", 0)))
    cur_tier = None
    for code, d in rows:
        if d.get("tier") != cur_tier:
            cur_tier = d.get("tier")
            labels = {"importance": "IMPORTANCE (frozen, watched regardless of detectability)",
                      "surprise": "SURPRISE (low-prior category AND beyond age+sex, uncapped)",
                      "detectability": "DETECTABILITY (cross-model replicated, FDR, beyond-age)"}
            print(f"    [ {labels.get(cur_tier, cur_tier)} ]")
        cur = d.get("current") or {}
        ref = d.get("reference") or {}
        tags = []
        if d.get("is_new"):
            tags.append("new")
        if d.get("plausibility") == "low-prior":
            tags.append("lowP")
        tagstr = (" [" + ",".join(tags) + "]") if tags else ""
        if cur.get("c_inc") is None:
            body = "below floor / not in panel"
        else:
            ci = f"[{cur.get('ci_lo','?')},{cur.get('ci_hi','?')}]"
            dref = d.get("delta_C_incident")
            dstr = f"{dref:+.3f}" if dref is not None else "  n/a"
            body = (f"C={cur['c_inc']:.3f} {ci:18s} Δref={dstr} n_ev={cur.get('n_event',0):4d}")
        fl = ("  " + " ".join(d["flags"])) if d.get("flags") else ""
        print(f"      {code:14s} {d['label'][:30]:30s} {body}{tagstr}{fl}")


def print_emerging(esec):
    print(f"\n  [EMERGING SIGNALS] {esec['n_candidates']} off-list disease(s) clear the bar "
          f"this run (candidate additions; list stays FROZEN)")
    for c in esec["candidates"]:
        cs = " / ".join(f"{c['per_model'][m]['C']:.3f}" for m in esec["params"]["models"])
        sflag = "  SURPRISE-candidate" if c["would_be_surprise"] else ""
        print(f"      {c['phecode']:14s} {c['label'][:30]:30s} C({cs}) "
              f"n_ev={c['n_event_incident']:4d}{sflag}")
    if esec["n_candidates"]:
        print("      review with curate_selected_diseases before any append (frozen policy).")


# CLI.
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["shard", "merge"], required=True)
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    ap.add_argument("--runs-dir", default=DEFAULT_RUNS)
    ap.add_argument("--dem", default=DEFAULT_DEM)
    ap.add_argument("--csv", default=MORTALITY_PHECODES_CSV)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--frozen-out", default=None,
                    help="also write a checked-in copy here (e.g. benchmark/selected_diseases_frozen.json)")
    ap.add_argument("--n-boot", type=int, default=B_CURATE)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--diseases", nargs="*", default=None,
                    help="explicit phecode ids (smoke test); overrides round-robin sharding")
    args = ap.parse_args()
    if args.mode == "shard":
        run_shard(args)
    else:
        merge_shards(args)


if __name__ == "__main__":
    main()
