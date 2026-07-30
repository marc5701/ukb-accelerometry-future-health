"""Per-disease C-index with bootstrap CIs for the matched-PRS fusion (forest figure).

For each matched disease we build out-of-fold linear predictors (5-fold, event-stratified) for
three arms, then subject-bootstrap the frozen OOF LPs to get CIs on each arm's C and on the
fusion-minus-embedding delta.
  embed   : [hazard] (embedding-only, the deployed wrist model)
  fusion  : [hazard, age, sex, PRS] (embedding plus genetics)
  prs_only: [PRS, age, sex] (standalone genetics, orthogonality)

Each fold fits a lightly-ridged Cox on raw covariates (no per-fold z-scoring), so the held-out
LPs stay in common hazard units and pool across folds. Skipping standardization keeps the OOF
predictors poolable.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError

from ukb_disease.benchmark.concordance_benchmark import DEFAULT_RUNS, harrell_counts, load_model
from ukb_disease.baseline.cox_crossfit_features import load_age_covars
from ukb_disease.baseline.config import resolve_oak_path
from ukb_disease.genetics.cox_crossfit_prs import build_phecode_to_trait, TRACK_PREFIX, FEATURES


def event_folds(E, K, seed=0):
    rng = np.random.default_rng(seed)
    fold = np.empty(len(E), int)
    for cls in (1, 0):
        idx = np.where(E == cls)[0]
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % K
    return fold


def oof_lp(X, T, E, K=5, pen=1.0):
    """Pooled out-of-fold linear predictor (higher = riskier). Falls back to col 0 on failure."""
    lp = np.zeros(len(T))
    fold = event_folds(E, K)
    cols = [f"x{i}" for i in range(X.shape[1])]
    for f in range(K):
        te, tr = fold == f, fold != f
        if E[tr].sum() < 2 or te.sum() == 0:
            lp[te] = X[te, 0]
            continue
        df = pd.DataFrame(X[tr], columns=cols); df["T"] = T[tr]; df["E"] = E[tr]
        try:
            cph = CoxPHFitter(penalizer=pen, l1_ratio=0.0).fit(df, "T", "E")
            beta = cph.params_.reindex(cols).to_numpy()
            lp[te] = X[te] @ beta
        except (ConvergenceError, ValueError, np.linalg.LinAlgError):
            lp[te] = X[te, 0]
    return lp


def c_from(lp, T, E):
    c, comp = harrell_counts(lp, T, E)
    return c / comp if comp > 0 else np.nan


def boot_ci(lp_a, lp_b, T, E, n_boot=2000, seed=1):
    """Bootstrap subjects; return (cA, cB, dlo, dhi, p_pos) for C(a), C(b), delta=b-a."""
    rng = np.random.default_rng(seed)
    n = len(T)
    dA = c_from(lp_a, T, E); dB = c_from(lp_b, T, E)
    deltas = []
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        ca = c_from(lp_a[ix], T[ix], E[ix]); cb = c_from(lp_b[ix], T[ix], E[ix])
        if not (np.isnan(ca) or np.isnan(cb)):
            deltas.append(cb - ca)
    deltas = np.array(deltas)
    return dA, dB, float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)), \
        float((deltas > 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="std")
    ap.add_argument("--subset", default="eur_genetic")
    ap.add_argument("--deep-prefix", default="embedding_disjoint_split")
    ap.add_argument("--runs-dir", default=DEFAULT_RUNS)
    ap.add_argument("--age-dir", default="age_sex_disjoint_split/sequence_model_age")
    ap.add_argument("--features", default=FEATURES)
    ap.add_argument("--min-events", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out_dir = resolve_oak_path(args.out); os.makedirs(out_dir, exist_ok=True)

    runs_dir = resolve_oak_path(args.runs_dir)
    mean_h, _bs, base, seeds = load_model(runs_dir, args.deep_prefix)
    T, E, M, eids = base["T"], base["E"], base["M"], base["eids"]
    N, P = T.shape
    names = json.load(open(f"{runs_dir}/{args.deep_prefix}_seed{seeds[0]}/config.json"))["phecodes"]
    age_dir = args.age_dir
    if not os.path.isabs(age_dir):
        age_dir = os.path.join(os.path.dirname(runs_dir), age_dir)
    ac = load_age_covars(age_dir, eids); age = ac["age"].to_numpy(float); sex = ac["sex"].to_numpy(float)

    feat = pd.read_parquet(args.features).set_index("eid").reindex(index=eids)
    prefix = TRACK_PREFIX[args.track]
    keep_subj = np.ones(N, bool) if args.subset == "all" else \
        (feat[args.subset].fillna(0).to_numpy() == 1)
    p2t = build_phecode_to_trait(names)

    rows = []
    for j in range(P):
        ph = names[j]; trait = p2t.get(ph)
        col = f"{prefix}{trait}" if trait else None
        if col not in feat.columns:
            continue
        m = M[:, j] & keep_subj
        if int(m.sum()) < args.min_events or int(E[m, j].sum()) < args.min_events:
            continue
        Tj = T[m, j].astype(float); Ej = E[m, j].astype(float)
        prs = pd.to_numeric(feat[col], errors="coerce").to_numpy()[m]
        prs = np.nan_to_num((prs - np.nanmean(prs)) / (np.nanstd(prs) or 1.0))
        h = mean_h[m, j]; a = age[m]; s = sex[m]
        # Isolate the PRS contribution: [hazard, PRS] vs [hazard], no age/sex confound.
        lp_embed = h  # embedding hazard is itself the LP
        lp_fusion = oof_lp(np.column_stack([h, prs]), Tj, Ej)
        lp_prs = oof_lp(prs.reshape(-1, 1), Tj, Ej)
        # Simple-averaging late fusion (sanity baseline): mean of rank-standardized embedding
        # hazard and PRS-only LP, no learned weights.
        def rz(x):
            r = pd.Series(x).rank().to_numpy(); return (r - r.mean()) / (r.std() or 1.0)
        lp_avg = rz(h) + rz(lp_prs)
        c_avg = c_from(lp_avg, Tj, Ej)
        # null control: subject-permuted PRS added to the hazard (matched complexity)
        rngp = np.random.default_rng(7 + j)
        lp_null = oof_lp(np.column_stack([h, prs[rngp.permutation(len(prs))]]), Tj, Ej)
        c_e, c_f, dlo, dhi, ppos = boot_ci(lp_embed, lp_fusion, Tj, Ej, n_boot=args.n_boot)
        c_p = c_from(lp_prs, Tj, Ej)
        c_n = c_from(lp_null, Tj, Ej)
        rows.append({"phecode": ph, "trait": trait, "n_ev": int(Ej.sum()), "n_eval": int(m.sum()),
                     "c_embed": c_e, "c_fusion": c_f, "c_avg": c_avg, "c_prs_only": c_p, "c_null": c_n,
                     "delta": c_f - c_e, "delta_lo": dlo, "delta_hi": dhi, "p_pos": ppos})
    df = pd.DataFrame(rows).sort_values("delta", ascending=False)
    df.to_csv(f"{out_dir}/perdisease_ci.csv", index=False)
    print(df.round(3).to_string(index=False))
    print(f"\nmatched n={len(df)} mean delta={df.delta.mean():.4f} "
          f"improved={int((df.delta>0).sum())}/{len(df)} "
          f"sig(p_pos>=0.975)={int((df.p_pos>=0.975).sum())}")
    print(f"wrote {out_dir}/perdisease_ci.csv")


if __name__ == "__main__":
    main()
