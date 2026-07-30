"""Per-disease cross-entropy screener: train on incident, test on prevalent.

Five arms, all scored on the same per-disease prevalent test set:
  logistic_emb : L2 logistic on the pooled embedding (primary screener)
  mlp_emb      : shallow MLP (BCE) on the embedding (robustness/parity)
  demo         : logistic on [age, sex, BMI] (the confound baseline to beat)
  emb_demo     : logistic on [embedding || age, sex, BMI]
  cox          : penalized Cox (PCA + lifelines) on the embedding (BCE vs Cox)

Output: long-format CSV, one row per (phecode, arm) with the screening metric
panel plus bootstrap CIs. Sharded by disease for parallel execution.
"""

from __future__ import annotations

import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from ukb_disease.paths import UKB_ROOT
from ukb_disease.baseline.config import (
    ACC_DEM_CSV,
    SEVEN_DAY_YEARS,
    resolve_oak_path,
)
from ukb_disease.baseline.outcomes_processing import load_phecodes
from ukb_disease.screening.screening_metrics import metrics_with_ci
from ukb_disease.screening.screening_split import (
    audit_panel,
    disease_masks,
    disease_masks_same_dist_cv,
)

BMI_CSV = f"{UKB_ROOT}/metadata/BMI_yes.csv"
ARMS = ["logistic_emb", "mlp_emb", "demo", "cox", "bce_demo", "cox_demo"]


# ----------------------------------------------------------------------------- demographics
def load_demographics(eids: np.ndarray) -> np.ndarray:
    """Return (n, 3) raw [age, sex, bmi] aligned to eids (NaN where missing)."""
    dem = pd.read_csv(resolve_oak_path(ACC_DEM_CSV)).set_index("eid")
    bmi = (
        pd.read_csv(resolve_oak_path(BMI_CSV), usecols=["eid", "21001-0.0"])
        .rename(columns={"21001-0.0": "bmi"})
        .groupby("eid")
        .mean()
    )
    age = dem.reindex(eids)["age_at_first_run"].to_numpy(float)
    sex = dem.reindex(eids)["sex"].to_numpy(float)
    bm = bmi.reindex(eids)["bmi"].to_numpy(float)
    return np.column_stack([age, sex, bm])


def _standardize(train: np.ndarray, *others: np.ndarray):
    """Mean/std standardize using TRAIN stats; NaN imputed to the train mean."""
    mu = np.nanmean(train, axis=0)
    sd = np.nanstd(train, axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)

    def tf(x):
        x = np.where(np.isfinite(x), x, mu)
        return ((x - mu) / sd).astype(np.float32)

    return (tf(train), *[tf(o) for o in others])


# ----------------------------------------------------------------------------- arms
def _fit_logistic_clf(Xtr, ytr, C=1.0):
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(
        C=C, penalty="l2", solver="lbfgs", max_iter=2000, class_weight="balanced"
    )
    clf.fit(Xtr, ytr)
    return clf


def _fit_logistic(Xtr, ytr, Xte, C=1.0):
    return _fit_logistic_clf(Xtr, ytr, C=C).predict_proba(Xte)[:, 1]


def _logistic_with_val(Xtr, ytr, Xva, yva, Xte, c_grid):
    """Pick C by validation AUROC (incident task), refit on train.

    Returns (test_prob, val_logit, test_logit, best_c). The logits (decision
    function = log-odds) feed the val-fit demographics stack; they are
    out-of-sample on val (model trained on train), so the combiner is honest.
    """
    from sklearn.metrics import roc_auc_score

    best_c, best_auc = c_grid[0], -1.0
    if len(np.unique(yva)) >= 2:
        for C in c_grid:
            try:
                pv = _fit_logistic(Xtr, ytr, Xva, C=C)
                auc = roc_auc_score(yva, pv)
            except Exception:
                auc = -1.0
            if auc > best_auc:
                best_auc, best_c = auc, C
    clf = _fit_logistic_clf(Xtr, ytr, C=best_c)
    return (clf.predict_proba(Xte)[:, 1], clf.decision_function(Xva),
            clf.decision_function(Xte), best_c)


def _fit_mlp(Xtr, ytr, Xte):
    from sklearn.neural_network import MLPClassifier

    clf = MLPClassifier(
        hidden_layer_sizes=(128,),
        activation="relu",
        alpha=1e-3,
        batch_size=256,
        early_stopping=True,
        n_iter_no_change=8,
        max_iter=120,
        random_state=0,
    )
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def _fit_cox(Xtr_emb, t_tr, e_tr, Xva_emb, Xte_emb, n_comp=50, penalizer=0.1):
    """PCA(train) + penalized lifelines Cox on incident time-to-event.

    Returns (test_loghazard, val_loghazard). Log-partial-hazard (the linear
    predictor) is the natural feature for the demographics stack and is
    monotone in the partial hazard, so the cox-arm AUROC is unchanged.
    """
    from lifelines import CoxPHFitter
    from sklearn.decomposition import PCA

    k = int(min(n_comp, Xtr_emb.shape[1], max(2, Xtr_emb.shape[0] - 1)))
    pca = PCA(n_components=k, random_state=0).fit(Xtr_emb)
    Ztr, Zva, Zte = pca.transform(Xtr_emb), pca.transform(Xva_emb), pca.transform(Xte_emb)
    cols = [f"f{i}" for i in range(k)]
    dtr = pd.DataFrame(Ztr, columns=cols)
    dtr["duration"] = t_tr
    dtr["event"] = e_tr
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=0.0)
    cph.fit(dtr, duration_col="duration", event_col="event")
    lh_te = cph.predict_log_partial_hazard(pd.DataFrame(Zte, columns=cols)).to_numpy().ravel()
    lh_va = cph.predict_log_partial_hazard(pd.DataFrame(Zva, columns=cols)).to_numpy().ravel()
    return lh_te, lh_va


def _stack_valfit(score_va, demo_va, y_va, score_te, demo_te, min_val_pos=10):
    """Honest score-level combiner: logistic on [score, age, sex, bmi] fit on VAL.

    `score_*` is the embedding model's (out-of-sample on val) log-odds / log-hazard;
    `demo_*` is the train-standardized [age, sex, bmi]. Mirrors cox_valfit._fit_apply
    (fit on val, apply to test). Returns the test probability, or None if val has
    < min_val_pos positives (combiner unreliable -> arm = NaN, like cox_valfit min_events).
    """
    y_va = np.asarray(y_va).astype(int)
    if int(y_va.sum()) < min_val_pos or len(np.unique(y_va)) < 2:
        return None
    Xv = np.column_stack([score_va, demo_va]).astype(np.float64)
    Xt = np.column_stack([score_te, demo_te]).astype(np.float64)
    mu = np.nanmean(Xv, 0)
    sd = np.nanstd(Xv, 0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Xv = np.nan_to_num((Xv - mu) / sd)
    Xt = np.nan_to_num((Xt - mu) / sd)
    return _fit_logistic_clf(Xv, y_va, C=1.0).predict_proba(Xt)[:, 1]


# ----------------------------------------------------------------------------- per disease
def screen_one(
    col, t, bucket, Xemb, demo, *, nearterm_years, n_total, n_boot, c_grid,
    do_mlp, do_cox, pca_comp, max_train_controls,
):
    m = disease_masks(t, bucket, nearterm_years=nearterm_years)
    # Train rows as indices so we can subsample the (huge, imbalanced) control
    # pool for FITTING only; test is never touched. AUROC/CIs are computed on
    # the full prevalent test set. A balanced logistic with 30k controls
    # reproduces the all-controls boundary at a fraction of the MLP/Cox cost.
    rng = np.random.default_rng(0)
    tr_pos = np.where(m["train_pos"])[0]
    tr_ctrl = np.where(m["train_ctrl"])[0]
    n_train_ctrl_full = len(tr_ctrl)
    if max_train_controls and len(tr_ctrl) > max_train_controls:
        tr_ctrl = rng.choice(tr_ctrl, size=max_train_controls, replace=False)
    tr_idx = np.concatenate([tr_pos, tr_ctrl])
    y_tr = np.concatenate([np.ones(len(tr_pos), int), np.zeros(len(tr_ctrl), int)])

    va = m["val_pos"] | m["val_ctrl"]
    te = m["test_pos"] | m["test_ctrl"]
    y_va = m["val_pos"][va].astype(int)
    y_te = m["test_pos"][te].astype(int)
    n_test_pos = int(m["test_pos"].sum())
    prevalence = n_test_pos / n_total

    base = {
        "phecode": col,
        "n_test_pos": n_test_pos,
        "n_test_ctrl": int(m["test_ctrl"].sum()),
        "n_train_pos": int(len(tr_pos)),
        "n_train_ctrl": int(n_train_ctrl_full),
        "n_train_ctrl_used": int(len(tr_ctrl)),
        "n_val_pos": int(m["val_pos"].sum()),
        "prevalence_pop": prevalence,
    }

    # standardized feature blocks (train-fit stats)
    Xe_tr, Xe_va, Xe_te = _standardize(Xemb[tr_idx], Xemb[va], Xemb[te])
    Dd_tr, Dd_va, Dd_te = _standardize(demo[tr_idx], demo[va], demo[te])

    scores = {}
    # logistic on embedding (primary); val-selected C, keep val/test logits for the stack
    prob_te, emb_logit_va, emb_logit_te, _ = _logistic_with_val(
        Xe_tr, y_tr, Xe_va, y_va, Xe_te, c_grid)
    scores["logistic_emb"] = prob_te
    # demographics-only (age, sex, BMI)
    scores["demo"] = _fit_logistic(Dd_tr, y_tr, Dd_te, C=1.0)
    # embedding(BCE) + demographics: val-fit stack on [emb log-odds, age, sex, bmi]
    s = _stack_valfit(emb_logit_va, Dd_va, y_va, emb_logit_te, Dd_te)
    if s is not None:
        scores["bce_demo"] = s
    # MLP on embedding
    if do_mlp and len(np.unique(y_tr)) >= 2:
        try:
            scores["mlp_emb"] = _fit_mlp(Xe_tr, y_tr, Xe_te)
        except Exception as ex:
            print(f"  [warn] mlp failed {col}: {ex}", flush=True)
    # Cox reference (+ Cox + demographics stack)
    if do_cox:
        try:
            t_inc = t[tr_idx].astype(float)
            e_tr = y_tr.astype(float)
            max_fu = float(np.nanmax(t_inc[e_tr > 0])) + 1.0 if (e_tr > 0).any() else 10.0
            dur = np.where(e_tr > 0, np.clip(t_inc, SEVEN_DAY_YEARS, None), max_fu)
            cox_lh_te, cox_lh_va = _fit_cox(Xe_tr, dur, e_tr, Xe_va, Xe_te, n_comp=pca_comp)
            scores["cox"] = cox_lh_te
            sc = _stack_valfit(cox_lh_va, Dd_va, y_va, cox_lh_te, Dd_te)
            if sc is not None:
                scores["cox_demo"] = sc
        except Exception as ex:
            print(f"  [warn] cox failed {col}: {ex}", flush=True)

    rows = []
    for arm in ARMS:
        if arm not in scores:
            continue
        mt = metrics_with_ci(y_te, scores[arm], prevalence, n_boot=n_boot)
        rows.append({**base, "arm": arm, **mt})
    return rows


# --------------------------------------------------------------- same-distribution CV
def screen_one_same_dist_cv(
    col, t, cv_assignment, Xemb, demo, *, n_total, n_boot, c_grid, do_mlp,
    n_folds, val_frac, cv_fixed_c, cv_min_val_pos, cv_eval_controls, max_train_controls,
):
    """Same-distribution K-fold CV detector: train AND test on prevalent cases.

    The transfer screener (`screen_one`) trains on incident and tests on prevalent, so
    it pays a transfer tax. Here each prevalent case (plus a capped control sample) is
    scored OUT-OF-FOLD by a model trained on the other folds, then metrics are computed
    once on the pooled out-of-fold predictions. Headline arms are WRIST-EMBEDDING-ONLY
    (`logistic_emb`, `mlp_emb`); `demo` (age/sex/BMI) is kept as a reference baseline,
    never the model. The Cox arm is dropped: prevalent cases have no incident onset time.

    AUROC, sens@95spec and PPV/NPV (at population prevalence) are directly comparable to
    the transfer numbers; AUPRC depends on the eval-set positive:negative ratio (the
    control sample is capped), so it is reported but not cross-mode comparable.
    """
    from sklearn.metrics import roc_auc_score

    notna = ~np.isnan(t)
    prevalent = notna & (t <= SEVEN_DAY_YEARS)
    n_test_pos = int(prevalent.sum())
    n_ctrl_total = int((~notna).sum())
    prevalence = n_test_pos / n_total
    arms = ["logistic_emb", "demo"] + (["mlp_emb"] if do_mlp else [])

    oof = {a: [] for a in arms}            # list of (y_fold, pred_fold)
    per_fold_auroc = {a: [] for a in arms}
    tr_pos_counts, val_pos_counts = [], []
    eval_ctrl_per_fold = max(1, int(cv_eval_controls // n_folds))

    for k in range(n_folds):
        fm = disease_masks_same_dist_cv(t, cv_assignment, k)
        te_pos = np.where(fm["test_pos"])[0]
        te_ctrl = np.where(fm["test_ctrl"])[0]
        tp_pos = np.where(fm["trainpool_pos"])[0]
        tp_ctrl = np.where(fm["trainpool_ctrl"])[0]
        if len(tp_pos) < 2 or len(te_pos) == 0:
            continue  # cannot fit a positive class, or nothing to score this fold

        rng = np.random.default_rng(1000 + k)
        # cap eval controls; AUROC/sens unchanged in expectation, bounds bootstrap cost
        if len(te_ctrl) > eval_ctrl_per_fold:
            te_ctrl = rng.choice(te_ctrl, size=eval_ctrl_per_fold, replace=False)

        # stratified validation carve from the train pool (for logistic C selection)
        n_vp = int(round(len(tp_pos) * val_frac))
        n_vc = int(round(len(tp_ctrl) * val_frac))
        vp = rng.choice(tp_pos, size=n_vp, replace=False) if 0 < n_vp < len(tp_pos) else np.empty(0, int)
        vc = rng.choice(tp_ctrl, size=n_vc, replace=False) if 0 < n_vc < len(tp_ctrl) else np.empty(0, int)
        tr_pos = np.setdiff1d(tp_pos, vp)
        tr_ctrl = np.setdiff1d(tp_ctrl, vc)
        if max_train_controls and len(tr_ctrl) > max_train_controls:
            tr_ctrl = rng.choice(tr_ctrl, size=max_train_controls, replace=False)

        tr_idx = np.concatenate([tr_pos, tr_ctrl])
        y_tr = np.concatenate([np.ones(len(tr_pos), int), np.zeros(len(tr_ctrl), int)])
        te_idx = np.concatenate([te_pos, te_ctrl])
        y_te = np.concatenate([np.ones(len(te_pos), int), np.zeros(len(te_ctrl), int)])
        if len(np.unique(y_tr)) < 2:
            continue
        tr_pos_counts.append(len(tr_pos))
        val_pos_counts.append(len(vp))

        Dd_tr, Dd_te = _standardize(demo[tr_idx], demo[te_idx])
        fold_pred = {"demo": _fit_logistic(Dd_tr, y_tr, Dd_te, C=1.0)}

        # logistic on embedding (wrist-only primary); pick C on val when it has positives
        use_val = len(vp) >= cv_min_val_pos and len(vc) > 0
        if use_val:
            val_idx = np.concatenate([vp, vc])
            y_va = np.concatenate([np.ones(len(vp), int), np.zeros(len(vc), int)])
            Xe_tr, Xe_va, Xe_te = _standardize(Xemb[tr_idx], Xemb[val_idx], Xemb[te_idx])
            fold_pred["logistic_emb"] = _logistic_with_val(Xe_tr, y_tr, Xe_va, y_va, Xe_te, c_grid)[0]
        else:
            Xe_tr, Xe_te = _standardize(Xemb[tr_idx], Xemb[te_idx])
            fold_pred["logistic_emb"] = _fit_logistic(Xe_tr, y_tr, Xe_te, C=cv_fixed_c)

        if do_mlp:
            try:
                fold_pred["mlp_emb"] = _fit_mlp(Xe_tr, y_tr, Xe_te)
            except Exception as ex:  # noqa: BLE001
                print(f"  [warn] mlp failed {col} fold {k}: {ex}", flush=True)

        for a in arms:
            if a in fold_pred:
                oof[a].append((y_te, fold_pred[a]))
                if len(np.unique(y_te)) >= 2:
                    per_fold_auroc[a].append(float(roc_auc_score(y_te, fold_pred[a])))

    base = {
        "phecode": col,
        "n_test_pos": n_test_pos,
        "n_test_ctrl": min(cv_eval_controls, n_ctrl_total),
        "n_train_pos": int(round(np.mean(tr_pos_counts))) if tr_pos_counts else 0,
        "n_train_ctrl": n_ctrl_total,
        "n_train_ctrl_used": int(max_train_controls),
        "n_val_pos": int(round(np.mean(val_pos_counts))) if val_pos_counts else 0,
        "prevalence_pop": prevalence,
    }

    rows = []
    for a in arms:
        if not oof[a]:
            continue
        ys = np.concatenate([p[0] for p in oof[a]])
        ss = np.concatenate([p[1] for p in oof[a]])
        mt = metrics_with_ci(ys, ss, prevalence, n_boot=n_boot)
        fa = per_fold_auroc[a]
        rows.append({
            **base, "arm": a, **mt,
            "n_test_ctrl": int((ys == 0).sum()),
            "auroc_cv_mean": float(np.mean(fa)) if fa else float("nan"),
            "auroc_cv_std": float(np.std(fa)) if fa else float("nan"),
            "n_folds_used": len(fa),
        })
    return rows


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pooled-cache", required=True, help="npz from pool_embeddings (eids,X,bucket)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--variant", choices=["primary", "nearterm"], default="primary")
    ap.add_argument("--nearterm-years", type=float, default=2.5)
    ap.add_argument("--mode", choices=["prevalent_v1", "same_dist_cv"], default="prevalent_v1",
                    help="prevalent_v1 = transfer (train incident, test prevalent); "
                         "same_dist_cv = train AND test on prevalent via K-fold CV")
    ap.add_argument("--n-folds", type=int, default=5, help="folds for same_dist_cv")
    ap.add_argument("--cv-val-frac", type=float, default=0.15,
                    help="validation fraction carved from each CV train pool (C selection)")
    ap.add_argument("--cv-fixed-c", type=float, default=0.1,
                    help="logistic C used when a fold's val is too thin for selection")
    ap.add_argument("--cv-min-val-pos", type=int, default=8,
                    help="min val positives to trust val-based C selection in same_dist_cv")
    ap.add_argument("--cv-eval-controls", type=int, default=8000,
                    help="cap on pooled OOF eval controls (bounds bootstrap cost; "
                         "AUROC precision is set by the ~positives, not the controls)")
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--min-test-pos", type=int, default=10)
    ap.add_argument("--min-train-pos", type=int, default=50)
    ap.add_argument("--pca-comp", type=int, default=50)
    ap.add_argument("--max-train-controls", type=int, default=30000,
                    help="subsample train controls for FITTING only (0 = use all)")
    ap.add_argument("--c-grid", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    ap.add_argument("--diseases", nargs="+", default=None, help="explicit phecode list (smoke)")
    ap.add_argument("--no-mlp", action="store_true")
    ap.add_argument("--no-cox", action="store_true")
    ap.add_argument("--write-audit", action="store_true", help="also write the panel audit CSV")
    args = ap.parse_args()

    out_dir = resolve_oak_path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    nearterm = args.nearterm_years if args.variant == "nearterm" else None

    # load pooled embeddings + buckets
    z = np.load(resolve_oak_path(args.pooled_cache), allow_pickle=True)
    eids = z["eids"].astype(np.int64)
    Xemb = z["X"].astype(np.float32)
    bucket = z["bucket"].astype(object)
    n_total = len(eids)
    print(f"[run] pooled cache: {Xemb.shape} subjects; "
          f"buckets {pd.Series(bucket).value_counts().to_dict()}", flush=True)

    cv_assignment = None
    if args.mode == "same_dist_cv":
        from ukb_disease.baseline.make_cv_folds import fold_of_eid
        cv_assignment = np.array([fold_of_eid(int(e), args.n_folds) for e in eids], dtype=np.int64)
        print(f"[run] same_dist_cv: {args.n_folds} folds, "
              f"counts {pd.Series(cv_assignment).value_counts().sort_index().to_dict()}", flush=True)

    # phecodes aligned to the embedding universe
    phe = load_phecodes().reindex(index=eids)
    demo = load_demographics(eids)
    print(f"[run] phecodes {phe.shape}; demo coverage "
          f"age={np.isfinite(demo[:,0]).mean():.3f} sex={np.isfinite(demo[:,1]).mean():.3f} "
          f"bmi={np.isfinite(demo[:,2]).mean():.3f}", flush=True)

    # panel audit + viable list (deterministic order by phecode name for stable sharding)
    audit = audit_panel(phe, bucket, args.min_test_pos, args.min_train_pos, nearterm)
    if args.write_audit and args.shard_idx == 0:
        ap_path = os.path.join(out_dir, f"panel_audit_{args.variant}.csv")
        audit.to_csv(ap_path, index=False)
        print(f"[run] wrote audit {ap_path} ({audit['viable'].sum()} viable / {len(audit)})", flush=True)

    if args.diseases:
        panel = [c for c in sorted(args.diseases) if c in phe.columns]
    else:
        panel = sorted(audit.loc[audit["viable"], "phecode"].tolist())
    shard = panel[args.shard_idx :: args.n_shards]
    print(f"[run] mode={args.mode} variant={args.variant} panel={len(panel)} "
          f"shard {args.shard_idx}/{args.n_shards} -> {len(shard)} diseases", flush=True)

    all_rows = []
    t0 = time.time()
    for i, col in enumerate(shard):
        t = phe[col].to_numpy(dtype=np.float64)
        try:
            if args.mode == "same_dist_cv":
                rows = screen_one_same_dist_cv(
                    col, t, cv_assignment, Xemb, demo,
                    n_total=n_total, n_boot=args.n_boot, c_grid=args.c_grid,
                    do_mlp=not args.no_mlp, n_folds=args.n_folds, val_frac=args.cv_val_frac,
                    cv_fixed_c=args.cv_fixed_c, cv_min_val_pos=args.cv_min_val_pos,
                    cv_eval_controls=args.cv_eval_controls,
                    max_train_controls=args.max_train_controls,
                )
            else:
                rows = screen_one(
                    col, t, bucket, Xemb, demo,
                    nearterm_years=nearterm, n_total=n_total, n_boot=args.n_boot,
                    c_grid=args.c_grid, do_mlp=not args.no_mlp, do_cox=not args.no_cox,
                    pca_comp=args.pca_comp, max_train_controls=args.max_train_controls,
                )
            all_rows.extend(rows)
        except Exception as ex:
            print(f"  [error] {col}: {type(ex).__name__}: {ex}", flush=True)
        if (i + 1) % 5 == 0 or i + 1 == len(shard):
            el = time.time() - t0
            print(f"[run] {i+1}/{len(shard)} diseases  ({el/ (i+1):.1f}s/disease)", flush=True)

    df = pd.DataFrame(all_rows)
    out_csv = os.path.join(out_dir, f"shard_{args.variant}_{args.shard_idx:03d}.csv")
    df.to_csv(out_csv, index=False)
    print(f"[run] wrote {out_csv}: {len(df)} rows ({df['phecode'].nunique() if len(df) else 0} diseases)", flush=True)


if __name__ == "__main__":
    main()
