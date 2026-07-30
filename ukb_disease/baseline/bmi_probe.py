"""CPU experiment runner for the BMI analysis.

One experiment fits a head on TRAIN embeddings, predicts VAL, computes the
headline r (with CI) and the four confounding controls, compares to the current
best via paired Δr (keep or kill), and appends a ledger row. TEST is never
touched here.

Usage:
    python3 -m ukb_disease.baseline.bmi_probe \
        --exp-id bmi_0000 --hypothesis "baseline ridge mean both" \
        --pooling mean --modality both --head ridge --transform raw
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np

from ukb_disease.baseline import bmi_lib as L
from ukb_disease.baseline import bmi_metrics as M


# ---------------------------------------------------------------------------
# target transforms (fit on train; inverse before every metric -> raw-scale r)
# ---------------------------------------------------------------------------
def _make_transform(kind, y_tr):
    if kind == "raw":
        return (lambda y: y), (lambda z: z)
    if kind == "log":
        return (lambda y: np.log(y)), (lambda z: np.exp(z))
    if kind == "rankgauss":
        from sklearn.preprocessing import QuantileTransformer
        qt = QuantileTransformer(output_distribution="normal",
                                 n_quantiles=min(1000, len(y_tr)), subsample=10**9,
                                 random_state=0)
        qt.fit(y_tr.reshape(-1, 1))
        return (lambda y: qt.transform(y.reshape(-1, 1)).ravel()), \
               (lambda z: qt.inverse_transform(z.reshape(-1, 1)).ravel())
    raise ValueError(kind)


def _make_head(kind, alpha, seed):
    if kind == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=alpha)
    if kind == "mlp":
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor(hidden_layer_sizes=(256, 64), alpha=alpha,
                            max_iter=300, early_stopping=True, random_state=seed)
    if kind == "mlp_tuned":
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor(hidden_layer_sizes=(512, 128), alpha=1e-4,
                            max_iter=500, early_stopping=True, n_iter_no_change=20,
                            learning_rate_init=1e-3, random_state=seed)
    if kind == "huber":
        from sklearn.linear_model import HuberRegressor
        return HuberRegressor(alpha=alpha, max_iter=500)
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# proxy predictions (for paired confounding tests)
# ---------------------------------------------------------------------------
def _demo_pred(bmi_tr, age_tr, sex_tr, age_ho, sex_ho):
    am, asd = float(np.mean(age_tr)), float(np.std(age_tr))
    beta = M._ols_fit(M.demo_design(age_tr, sex_tr, am, asd), bmi_tr)
    return M.demo_design(age_ho, sex_ho, am, asd) @ beta


def _proxy_pred(emb_tr, bmi_tr, age_tr, sex_tr, emb_ho, alpha=10.0):
    from sklearn.linear_model import Ridge, LogisticRegression
    age_dec = Ridge(alpha=alpha).fit(emb_tr, age_tr)
    sex_dec = LogisticRegression(max_iter=2000).fit(emb_tr, sex_tr.astype(int))
    a_tr, s_tr = age_dec.predict(emb_tr), sex_dec.predict_proba(emb_tr)[:, 1]
    a_ho, s_ho = age_dec.predict(emb_ho), sex_dec.predict_proba(emb_ho)[:, 1]
    am, asd = float(np.mean(a_tr)), float(np.std(a_tr))
    beta = M._ols_fit(M.demo_design(a_tr, s_tr, am, asd), bmi_tr)
    return M.demo_design(a_ho, s_ho, am, asd) @ beta


# ---------------------------------------------------------------------------
# best-model tracking
# ---------------------------------------------------------------------------
BEST_JSON = os.path.join(L.LOOP_DIR, "best.json")


def _load_best():
    if not os.path.exists(BEST_JSON):
        return None
    best = json.load(open(BEST_JSON))
    d = os.path.join(L.RUNS_DIR, best["exp_id"])
    pred = np.load(os.path.join(d, "val_pred.npy"))
    eids = np.load(os.path.join(d, "val_eids.npy"))
    return {"exp_id": best["exp_id"], "r": best["r_raw"], "eids": eids, "pred": pred}


def _align(eids_a, pred_a, eids_b, pred_b, y_b):
    """Intersect two val prediction sets by eid; return aligned (a, b, y)."""
    map_a = {int(e): p for e, p in zip(eids_a, pred_a)}
    keep = np.array([int(e) in map_a for e in eids_b])
    a = np.array([map_a[int(e)] for e in eids_b[keep]])
    return a, pred_b[keep], y_b[keep]


# ---------------------------------------------------------------------------
# one experiment
# ---------------------------------------------------------------------------
def run_experiment(cfg) -> dict:
    tr = L.load_dataset("train", cfg["pooling"], cfg["modality"])
    va = L.load_dataset("val", cfg["pooling"], cfg["modality"])
    return evaluate(tr, va, cfg)


def evaluate(tr, va, cfg, update_best: bool = True) -> dict:
    """Fit head on TRAIN, score VAL, log ledger row. tr/va from L.load_dataset."""
    from sklearn.preprocessing import StandardScaler

    t0 = time.time()
    scaler = StandardScaler().fit(tr["X"])
    Xtr, Xva = scaler.transform(tr["X"]), scaler.transform(va["X"])
    fwd, inv = _make_transform(cfg["transform"], tr["bmi"])
    head = _make_head(cfg["head"], cfg["alpha"], cfg["seed"])
    head.fit(Xtr, fwd(tr["bmi"]))
    yhat = inv(head.predict(Xva))

    bmi_va, age_va, sex_va = va["bmi"], va["age"], va["sex"]

    r_raw = M.boot_r_ci(yhat, bmi_va, seed=cfg["seed"])
    demo_pred = _demo_pred(tr["bmi"], tr["age"], tr["sex"], age_va, sex_va)
    proxy_pred = _proxy_pred(Xtr, tr["bmi"], tr["age"], tr["sex"], Xva)
    rdemo = M.pearson(demo_pred, bmi_va)
    rproxy = M.pearson(proxy_pred, bmi_va)
    rpart = M.r_partial(yhat, bmi_va, age_va, sex_va, seed=cfg["seed"])
    rwith = M.r_within(yhat, bmi_va, age_va, sex_va, seed=cfg["seed"])

    dr_vs_demo = M.paired_dr(demo_pred, yhat, bmi_va, seed=cfg["seed"])
    dr_vs_proxy = M.paired_dr(proxy_pred, yhat, bmi_va, seed=cfg["seed"])

    verdict = {
        "beats_demo": bool(dr_vs_demo["lo"] > 0 and (r_raw["point"] - rdemo) >= 0.05),
        "partial_ok": bool(rpart["point"] >= 0.5 * r_raw["point"] and rpart["lo"] > 0),
        "within_ok": bool(abs(rwith["point"] - rpart["point"]) <= 0.03),
        "beats_proxy_ceiling": bool(dr_vs_proxy["lo"] > 0),
    }
    verdict["genuine_signal"] = all(verdict.values())

    # keep/kill vs current best (paired Δr)
    best = _load_best()
    if best is None:
        decision, rule, vs_best = "keep", "first_experiment", None
    else:
        a, b, y = _align(best["eids"], best["pred"], va["eids"], yhat, bmi_va)
        dr = M.paired_dr(a, b, y, seed=cfg["seed"])
        vs_best = {"best_exp_id": best["exp_id"], **dr, "n_common": int(len(y))}
        if dr["lo"] > 0 and dr["point"] >= 0.005 and dr["lo"] >= 0.001:
            decision, rule = "keep", "dr_lo>0 & dr>=0.005 & dr_lo>=0.001"
        elif dr["hi"] < 0.005:
            decision, rule = "kill", "dr_hi<0.005"
        else:
            decision, rule = "iterate", "ci_straddles_bar"

    # persist val predictions + config
    exp_dir = os.path.join(L.RUNS_DIR, cfg["exp_id"])
    os.makedirs(exp_dir, exist_ok=True)
    np.save(os.path.join(exp_dir, "val_pred.npy"), yhat.astype(np.float32))
    np.save(os.path.join(exp_dir, "val_eids.npy"), va["eids"].astype(np.int64))

    cfg_for_hash = {k: cfg[k] for k in ("pooling", "modality", "head", "transform", "alpha", "seed")}
    config_hash = hashlib.sha256(json.dumps(cfg_for_hash, sort_keys=True).encode()).hexdigest()[:16]

    row = {
        "exp_id": cfg["exp_id"], "timestamp_unix": time.time(),
        "hypothesis": cfg["hypothesis"], "single_change": cfg.get("single_change", ""),
        "config_hash": config_hash, "config": cfg_for_hash, "seed": cfg["seed"],
        "n_train": tr["n"], "n_val": va["n"],
        "val": {
            "r_raw": r_raw, "r_demo": rdemo, "r_proxy_ceiling": rproxy,
            "r_partial": rpart, "r_within": rwith,
        },
        "confound_verdict": verdict,
        "vs_best": vs_best, "decision": decision, "decision_rule_fired": rule,
        "wall_sec": round(time.time() - t0, 1),
    }
    with open(L.LEDGER, "a") as f:
        f.write(json.dumps(row) + "\n")
    json.dump(row, open(os.path.join(exp_dir, "metrics.json"), "w"), indent=2)

    if decision == "keep" and update_best:
        json.dump({"exp_id": cfg["exp_id"], "r_raw": r_raw["point"]},
                  open(BEST_JSON, "w"), indent=2)

    print(f"[{cfg['exp_id']}] r_raw={r_raw['point']:.4f} "
          f"[{r_raw['lo']:.4f},{r_raw['hi']:.4f}]  r_demo={rdemo:.4f}  "
          f"r_proxy={rproxy:.4f}  r_partial={rpart['point']:.4f}  "
          f"r_within={rwith['point']:.4f}  genuine={verdict['genuine_signal']}  "
          f"decision={decision}  ({row['wall_sec']}s)", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--hypothesis", required=True)
    ap.add_argument("--single-change", default="")
    ap.add_argument("--pooling", default="mean", choices=["mean", "l3"])
    ap.add_argument("--modality", default="both", choices=["ar", "ha", "both"])
    ap.add_argument("--head", default="ridge", choices=["ridge", "mlp", "mlp_tuned", "huber"])
    ap.add_argument("--transform", default="raw", choices=["raw", "log", "rankgauss"])
    ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seal-test", action="store_true", help="(re)write test_manifest.json first")
    args = ap.parse_args()
    if args.seal_test:
        man = L.seal_test()
        print(f"[seal] test sealed: N={man['n_test_subjects']} "
              f"bmi_mean={man['bmi_mean']:.2f} sd={man['bmi_sd']:.2f}", flush=True)
    run_experiment(vars(args))


if __name__ == "__main__":
    main()
