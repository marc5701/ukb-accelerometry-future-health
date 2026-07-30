"""Cross-fit evaluation of engineered covariates added to the per-phecode Cox combiner.

In-sample C is optimistically biased and the bias grows with the number of covariates per
event, so adding new covariates and re-fitting in-sample would manufacture a fake win on the
thin phecodes (PD 25 events, AD 15 events). This module evaluates any covariate bundle by
cross-fitting the combiner instead.

Design notes:
  * Within-fold Harrell C, then comparable-pair-weighted average. We never pool raw out-of-fold
    linear predictors across folds: per-fold centring/scaling of a penalised-Cox linear
    predictor corrupts cross-fold pairs and biases the augmented arm down (fake tie). We
    accumulate harrell_counts() within each held-out fold and sum conc/comp across folds.
  * Event-stratified folds, adaptive K by event count (>=50 -> K=10, >=20 -> 5, >=10 -> 3).
  * Fixed ridge penalty ~ C0/n_events, l1_ratio=0, identical for both arms (no tuning on OOF).
  * Train-fold z-scoring of covariates (mean/std from the train folds only; applied to the
    held-out fold) so the ridge penalty is on a comparable scale and there is no leakage.
  * Per-fold convergence fallback: if the penalised Cox fails on a fold's train partition, that
    fold's held-out linear predictor falls back to the deep hazard column (never silently drop
    a fold).

Survival arrays come straight from the deep run dir (test_event_times/is_event/eval_mask), the
same (T, E, M) the canonical concordance_benchmark uses.

Example:
  python3 -m ukb_disease.baseline.cox_crossfit_features \
      --deep-prefix embedding_disjoint_split --age-dir age_sex_disjoint_split/sequence_model_age \
      --arms hazard_only insample_joint crossfit_joint crossfit_no_ar_emb \
      --out crossfit_step0
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError

from ukb_disease.baseline.config import resolve_oak_path
from ukb_disease.benchmark.concordance_benchmark import DEFAULT_RUNS, harrell_counts, load_model

# Covariate bundles.
BASE_COVARS = ["hazard", "age", "sex", "ar_emb"]  # the current joint-Cox covariates

# Named feature bundles (column names must exist in the subject-feature parquet). Step 0 uses
# only the 'baseline'/'no_ar_emb' arms which need no parquet.
FEATURE_SETS: dict[str, list[str]] = {
    "baseline":   [],
    "resp":       ["resp_frac_mean", "resp_frac_std", "resp_rate_mean", "resp_rate_std", "ahi_analog"],
    "regularity": ["sri", "onset_h_std", "midpoint_h_std", "social_jetlag_h"],
    "rem":        ["rem_frac_mean", "rem_frac_std", "rem_entry_mean", "deep_frac_mean", "deep_frac_std"],
    "sleepqual":  ["eff_mean", "eff_std", "awakenings_mean", "longest_sleep_mean", "frag_mean", "frag_std"],
    "agegap_ext": ["ar_emb_embedding", "ar_emb_sleep"],
    "null_rar":   ["enmo_RA", "enmo_mean", "enmo_M10", "enmo_L5"],
}
# 'all_sleep' = union of the wrist-derived sleep families (defined after the literal so it can
# reference the others). Used as the combined-panel ceiling arm.
FEATURE_SETS["all_sleep"] = sorted(set(
    FEATURE_SETS["resp"] + FEATURE_SETS["regularity"]
    + FEATURE_SETS["rem"] + FEATURE_SETS["sleepqual"]))

# Dense-physiology proxy bundles. Columns live in dense_baseline_panel_nm/all.parquet (built by
# build_dense_baseline_parquet.py). Append-only, so prior crossfit runs reproduce identically.
FEATURE_SETS["dense_hrv"] = ["hrv_rmssd_mean", "hrv_rmssd_std", "hrv_rmssd_median", "hrv_frac_lowmotion"]
FEATURE_SETS["dense_spectral"] = ["spectral_3_8hz_ratio_lowmotion", "spectral_07_3hz_ratio_lowmotion",
                                  "spectral_trem_card_ratio_lowmotion", "spectral_3_8hz_ratio_highmotion"]
FEATURE_SETS["dense_enmodist"] = ["enmo_l1", "enmo_l2", "enmo_l3", "enmo_frac_zero",
                                  "enmo_p5", "enmo_p95", "enmo_skew", "enmo_ig", "enmo_frac_mvpa"]
FEATURE_SETS["dense_circ"] = ["enmo_IS", "enmo_IV", "enmo_RA", "enmo_cosinor_amp", "enmo_mesor"]
FEATURE_SETS["dense_frag"] = ["frag_mean", "frag_std"]
FEATURE_SETS["dense_physio"] = sorted(set(
    FEATURE_SETS["dense_hrv"] + FEATURE_SETS["dense_spectral"] + FEATURE_SETS["dense_enmodist"]
    + FEATURE_SETS["dense_circ"] + FEATURE_SETS["dense_frag"]))

# Nightbeat HR-from-accelerometer panel. Columns live in hr_nightbeat_panel_nm/all.parquet
# (compute_nightbeat_features.py --combine). The `null_hr` bundle is the matched-complexity
# control (the same 7 cols, subject-permuted, suffixed `_null`). Append-only.
FEATURE_SETS["nightbeat_hr"] = ["hr_rest", "hr_sleep_mean", "hr_within_night_sd", "hr_night_var",
                                "hr_dip", "hr_coverage", "n_usable_nights"]
FEATURE_SETS["null_hr"] = [c + "_null" for c in FEATURE_SETS["nightbeat_hr"]]

# Cleaned v2 HR panel. Drops the artifact-prone 5th-pct trough (hr_rest, inverted demographics)
# and adds HRV (SDNN/RMSSD) plus a robust resting quantile (hr_resting_q). Columns live in
# hr_nightbeat_panel_v2_nm/all.parquet. Append-only (v1 bundles above unchanged). SDNN kept
# (corr 0.89 vs ECG HRV on validation data); RMSSD dropped (corr 0.21, BCG peak jitter).
FEATURE_SETS["nightbeat_hr_v2"] = ["hr_sleep_mean", "hr_resting_q", "sdnn", "hr_dip",
                                   "hr_within_night_sd", "hr_night_var"]
FEATURE_SETS["nightbeat_hr_min"] = ["hr_sleep_mean"]          # minimal-complexity arm
FEATURE_SETS["null_hr_v2"] = [c + "_null" for c in FEATURE_SETS["nightbeat_hr_v2"]]

# SleepFM-v2 frozen-encoder day-pooled embedding (per-subject wear-masked mean of
# CARDIAC/EMG/RESP 128d, then label-free PCA to 24 components). Columns live in the sleepfm
# feature parquet (extraction/build_sleepfm_features.py). null_sleepfm = the same 24 cols
# subject-permuted (matched-complexity control). Append-only.
FEATURE_SETS["sleepfm_emb"] = [f"sleepfm_pc{j}" for j in range(24)]
FEATURE_SETS["null_sleepfm"] = [c + "_null" for c in FEATURE_SETS["sleepfm_emb"]]

# Higher within-recording L-moments (l4,l5,l6) of the AR+HA patches, PCA to 24. Tests whether
# the patch distribution carries disease signal beyond the l3 the main model uses. Columns from
# extraction/build_himoment_features.py.
FEATURE_SETS["himom"] = [f"himom_pc{j}" for j in range(24)]
FEATURE_SETS["null_himom"] = [c + "_null" for c in FEATURE_SETS["himom"]]

# UK Biobank distributed accelerometer summary features. The 100 official Doherty-pipeline
# activity summaries: overall accel avg + SD (90012/90013), day-of-week averages (90019-90025),
# hour-of-day averages (90027-90050), and the acceleration-intensity distribution fraction<=X mg
# (90092-90158). Columns are the raw UKB field ids ("<id>-0.0") stored in
# ukb_summary_activity.parquet (built by extract_ukb_summary_activity.py). This bundle lets
# pooled_crossfit.py test the pooled-OOF increment of adding the summaries to the frozen hazard
# (the unbiased analog of the clinical train_rolling_window_cox --add-bmi +0.0006), replacing the
# estimation-floor-biased per-disease emb_summary combiner in summary_baseline.py. Append-only.
_UKB_SUMMARY_IDS = ([90012, 90013] + list(range(90019, 90026))
                    + list(range(90027, 90051)) + list(range(90092, 90159)))
FEATURE_SETS["ukb_summary"] = [f"{i}-0.0" for i in _UKB_SUMMARY_IDS]           # 100 fields
FEATURE_SETS["ukb_summary_compact"] = ["90012-0.0", "90013-0.0"]              # 2 canonical scalars


# Fold machinery.
def adaptive_k(n_ev: int) -> int:
    if n_ev >= 50:
        return 10
    if n_ev >= 20:
        return 5
    if n_ev >= 10:
        return 3
    return 0


def event_stratified_folds(E_j: np.ndarray, K: int) -> np.ndarray:
    """Round-robin fold assignment within the event / censored strata (deterministic by
    position; the master frame is sorted by eid upstream so this is stable). Guarantees each
    fold gets floor/ceil of n_events/K events whenever n_events >= K."""
    fold = np.empty(len(E_j), dtype=int)
    ev = np.where(E_j == 1)[0]
    cen = np.where(E_j != 1)[0]
    fold[ev] = np.arange(len(ev)) % K
    fold[cen] = np.arange(len(cen)) % K
    return fold


def _fit_lp(Xtr: np.ndarray, Ttr: np.ndarray, Etr: np.ndarray, Xte: np.ndarray,
            penalizer: float) -> np.ndarray:
    """Train-fold z-score, fit ridge Cox, return the held-out linear predictor (higher=riskier).
    Raises on non-convergence (caller falls back)."""
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd[sd == 0] = 1.0
    Ztr = (Xtr - mu) / sd
    Zte = (Xte - mu) / sd
    cols = [f"x{i}" for i in range(Xtr.shape[1])]
    df = pd.DataFrame(Ztr, columns=cols)
    df["T"] = Ttr
    df["E"] = Etr
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=0.0)
    cph.fit(df, duration_col="T", event_col="E")
    beta = cph.params_.reindex(cols).to_numpy()
    return Zte @ beta


def crossfit_phecode(X: np.ndarray, T_j: np.ndarray, E_j: np.ndarray,
                     penalizer_c0: float, fallback_col: int = 0) -> dict:
    """Within-fold-C-then-average cross-fit for one phecode.
    X columns are [hazard, age, sex, ar_emb, *features]; col 0 (hazard) is the fallback LP."""
    n_ev = int(E_j.sum())
    K = adaptive_k(n_ev)
    if K == 0:
        return {"c": float("nan"), "K": 0, "n_ev": n_ev, "n_fail": 0}
    penalizer = max(0.1, penalizer_c0 / n_ev)
    fold = event_stratified_folds(E_j, K)
    conc = comp = 0.0
    n_fail = 0
    for f in range(K):
        te = fold == f
        tr = ~te
        if int(E_j[tr].sum()) < 1 or int(te.sum()) == 0:
            lp = X[te, fallback_col]
        else:
            try:
                lp = _fit_lp(X[tr], T_j[tr], E_j[tr], X[te], penalizer)
            except (ConvergenceError, ValueError, np.linalg.LinAlgError):
                n_fail += 1
                lp = X[te, fallback_col]
        c, m = harrell_counts(lp, T_j[te], E_j[te])
        conc += c
        comp += m
    return {"c": (conc / comp if comp > 0 else float("nan")), "K": K, "n_ev": n_ev,
            "n_fail": n_fail}


def insample_phecode(X: np.ndarray, T_j: np.ndarray, E_j: np.ndarray,
                     penalizer_c0: float, fallback_col: int = 0) -> float:
    n_ev = int(E_j.sum())
    penalizer = max(0.1, penalizer_c0 / n_ev)
    try:
        lp = _fit_lp(X, T_j, E_j, X, penalizer)
    except (ConvergenceError, ValueError, np.linalg.LinAlgError):
        lp = X[:, fallback_col]
    c, m = harrell_counts(lp, T_j, E_j)
    return c / m if m > 0 else float("nan")


# Data assembly.
def load_age_covars(age_dir: str, base_eids: np.ndarray) -> pd.DataFrame:
    """Load ar_emb/age/sex from a run_age_sex dir, collapse per-recording rows to one per subject
    (mean), and align to base_eids order. Returns a frame indexed like base_eids."""
    age_dir = resolve_oak_path(age_dir)
    df = pd.DataFrame({
        "eid": np.load(f"{age_dir}/test_eids.npy"),
        "ar_emb": np.load(f"{age_dir}/test_ar_emb.npy"),
        "age": np.load(f"{age_dir}/test_age.npy"),
        "sex": np.load(f"{age_dir}/test_sex.npy").astype(np.float64),
    }).groupby("eid", as_index=True).mean(numeric_only=True)
    out = df.reindex(index=base_eids)
    miss = int(out["ar_emb"].isna().sum())
    if miss:
        print(f"[crossfit] WARNING: {miss}/{len(base_eids)} subjects missing age covars "
              f"(imputed to column mean)")
        out = out.fillna(out.mean(numeric_only=True))
    return out


def load_features(feat_parquet: str | None, feat_cols: list[str],
                  base_eids: np.ndarray) -> np.ndarray | None:
    """Load named feature columns from a subject parquet, align to base_eids, z-score, NaN->0.
    Returns (N, n_feat) or None if no feat_cols requested."""
    if not feat_cols:
        return None
    if feat_parquet is None:
        raise SystemExit(f"[crossfit] feature set needs {feat_cols} but --feat-parquet not given")
    fp = pd.read_parquet(resolve_oak_path(feat_parquet)).set_index("eid")
    missing = [c for c in feat_cols if c not in fp.columns]
    if missing:
        raise SystemExit(f"[crossfit] parquet missing columns {missing}")
    sub = fp.reindex(index=base_eids)[feat_cols]
    mu = sub.mean(0)
    sd = sub.std(0).replace(0.0, 1.0)
    z = ((sub - mu) / sd).fillna(0.0)
    return z.to_numpy(dtype=np.float64)


def build_arm_matrix(arm: str, h_col: np.ndarray, age: np.ndarray, sex: np.ndarray,
                     ar_emb: np.ndarray, feats: np.ndarray | None, m: np.ndarray) -> np.ndarray:
    """Assemble the (n_eval, n_cov) covariate matrix for one phecode + arm; col 0 is the hazard.
    Arms: 'crossfit_joint'/'insample_joint' = [haz,age,sex,ar_emb]; 'crossfit_no_ar_emb' =
    [haz,age,sex]; '<featureset>' = [haz,age,sex,ar_emb,*feats]."""
    cols = [h_col[m], age[m], sex[m]]
    if arm != "crossfit_no_ar_emb":
        cols.append(ar_emb[m])
    if feats is not None and arm not in ("crossfit_joint", "insample_joint",
                                         "crossfit_no_ar_emb", "hazard_only"):
        cols.extend(feats[m, k] for k in range(feats.shape[1]))
    return np.column_stack(cols)


# Main.
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deep-prefix", default="embedding_disjoint_split",
                    help="run-dir prefix under --runs-dir (seed dirs averaged)")
    ap.add_argument("--runs-dir", default=DEFAULT_RUNS)
    ap.add_argument("--age-dir", default="age_sex_disjoint_split/sequence_model_age",
                    help="run_age_sex dir with test_ar_emb/age/sex/eids.npy (relative to results root or absolute)")
    ap.add_argument("--feat-parquet", default=None)
    ap.add_argument("--feature-set", default=None,
                    help="named bundle from FEATURE_SETS; its columns are added to the joint arm")
    ap.add_argument("--arms", nargs="+",
                    default=["hazard_only", "insample_joint", "crossfit_joint", "crossfit_no_ar_emb"])
    ap.add_argument("--min-events", type=int, default=10)
    ap.add_argument("--penalizer-c0", type=float, default=10.0,
                    help="ridge penalty = max(0.1, c0 / n_events)")
    ap.add_argument("--loose-panel-check", action="store_true",
                    help="also report hazard-only mean-C on the loose >=1-event panel to reconcile 0.6880")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = resolve_oak_path(args.out)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    runs_dir = resolve_oak_path(args.runs_dir)
    mean_h, _by_seed, base, seeds = load_model(runs_dir, args.deep_prefix)
    T, E, M, eids = base["T"], base["E"], base["M"], base["eids"]
    N, P = T.shape
    cfg = json.load(open(f"{runs_dir}/{args.deep_prefix}_seed{seeds[0]}/config.json"))
    names = cfg["phecodes"]
    print(f"[crossfit] deep={args.deep_prefix} seeds={seeds} N={N} P={P}")

    # Age dir given relative to the results root, for convenience.
    age_dir = args.age_dir
    if not os.path.isabs(age_dir):
        age_dir = os.path.join(os.path.dirname(resolve_oak_path(runs_dir)), age_dir)
    ac = load_age_covars(age_dir, eids)
    age = ac["age"].to_numpy(dtype=np.float64)
    sex = ac["sex"].to_numpy(dtype=np.float64)
    ar_emb = ac["ar_emb"].to_numpy(dtype=np.float64)

    feat_cols = FEATURE_SETS.get(args.feature_set, []) if args.feature_set else []
    feats = load_features(args.feat_parquet, feat_cols, eids)

    # Evaluable panel: events-under-mask >= min_events, fixed for all arms.
    ev_per = (E * M).sum(0)
    panel = [j for j in range(P) if ev_per[j] >= args.min_events]
    print(f"[crossfit] panel (>= {args.min_events} events) = {len(panel)} phecodes "
          f"(incl mortality={'time_to_death' in [names[j] for j in panel]})")

    # Optional sanity reconciliation with the headline C on the loose panel.
    if args.loose_panel_check:
        cs = []
        for j in range(P):
            m = M[:, j]
            if int(m.sum()) >= 10 and int(E[m, j].sum()) >= 1:
                c, comp = harrell_counts(mean_h[m, j], T[m, j], E[m, j])
                if comp > 0:
                    cs.append(c / comp)
        print(f"[crossfit] RECONCILE hazard-only mean-C on loose panel "
              f"(n={len(cs)}): {np.mean(cs):.4f}  (expect Embedding base ~0.6880)")

    # Per-phecode, per-arm C.
    rows = []
    for ji, j in enumerate(panel):
        m = M[:, j]
        T_j = T[m, j].astype(np.float64)
        E_j = E[m, j].astype(np.float64)
        rec = {"phecode": names[j], "j": j, "n_ev": int(E_j.sum()), "n_eval": int(m.sum())}
        for arm in args.arms:
            X = build_arm_matrix(arm, mean_h[:, j], age, sex, ar_emb, feats, m)
            if arm == "hazard_only":
                c, comp = harrell_counts(mean_h[m, j], T_j, E_j)
                rec[f"c_{arm}"] = (c / comp if comp > 0 else float("nan"))
            elif arm == "insample_joint":
                rec[f"c_{arm}"] = insample_phecode(X, T_j, E_j, args.penalizer_c0)
            else:  # crossfit_* and named feature arms
                r = crossfit_phecode(X, T_j, E_j, args.penalizer_c0)
                rec[f"c_{arm}"] = r["c"]
                rec[f"K_{arm}"] = r["K"]
                rec[f"nfail_{arm}"] = r["n_fail"]
        rows.append(rec)
        if (ji + 1) % 50 == 0:
            print(f"[crossfit] {ji+1}/{len(panel)} phecodes  ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(f"{out_dir}/per_phecode.csv", index=False)

    summary = {"deep_prefix": args.deep_prefix, "seeds": seeds, "n_panel": len(panel),
               "min_events": args.min_events, "penalizer_c0": args.penalizer_c0,
               "feature_set": args.feature_set, "feat_cols": feat_cols}
    for arm in args.arms:
        col = f"c_{arm}"
        summary[f"mean_c_{arm}"] = float(df[col].mean())
        summary[f"median_c_{arm}"] = float(df[col].median())
    # headline deltas
    if "c_crossfit_joint" in df and "c_insample_joint" in df:
        summary["optimism_gap"] = summary["mean_c_insample_joint"] - summary["mean_c_crossfit_joint"]
    if "c_crossfit_joint" in df and "c_crossfit_no_ar_emb" in df:
        summary["ar_emb_crossfit_delta"] = summary["mean_c_crossfit_joint"] - summary["mean_c_crossfit_no_ar_emb"]
    if "c_hazard_only" in df and "c_crossfit_joint" in df:
        summary["joint_over_hazard_crossfit"] = summary["mean_c_crossfit_joint"] - summary["mean_c_hazard_only"]
    summary["elapsed_sec"] = round(time.time() - t0, 1)

    with open(f"{out_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[crossfit] ===== SUMMARY =====")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:28s} {v:+.4f}" if "delta" in k or "gap" in k or "over" in k
                  else f"  {k:28s} {v:.4f}")
        else:
            print(f"  {k:28s} {v}")
    print(f"\n[crossfit] wrote {out_dir}/summary.json + per_phecode.csv  ({summary['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
