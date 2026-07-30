"""Characterise the shared disease-risk axis.

Per-disease predicted risks are strongly inter-correlated: one principal
component (PC1) explains most of the variance. This script quantifies that axis
and the residual group-specific structure, so we can state that the model
predicts one dominant disease-risk axis with partial, group-specific
specificity, rather than independent per-disease signatures.

Reuses the same helpers and data as disease_correlation.py, so PC1 is identical.
CPU-only, runs in seconds.

Outputs:
  pc1_subject.npz        : per-(demographics-complete)-subject PC1 score + incident-disease burden.
  loadings.csv           : per-disease PC1 loading + category + n_event.
  beyond_by_category.npz : S_beyond reordered by organ system + category block boundaries.
  variance_partition.csv : per-category shared / group-specific / idiosyncratic variance.
  summary.json           : headline scalars.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ukb_disease.paths import UKB_ROOT
from ukb_disease.baseline.outcomes_processing import load_covariates
from ukb_disease.paper_additions.io_utils import load_seed_avg
from ukb_disease.paper_additions.disease_correlation import (
    _phecode_meta, _residualize, MIN_EVENTS,
)

OUT = f"{UKB_ROOT}/paper_additions/shared_axis"
MIN_CAT = 3   # min diseases in a category to estimate a within-group residual axis


def _col_var(M: np.ndarray) -> np.ndarray:
    """Population variance of each column (subjects x diseases)."""
    return M.var(axis=0)


def main() -> None:
    Path(OUT).mkdir(parents=True, exist_ok=True)

    sa = load_seed_avg("embedding_disjoint_split")
    eids, H, events, mask, phe = (sa["eids"], sa["hazards"], sa["events"],
                                  sa["mask"], sa["phecodes"])
    H = H.astype(np.float64)

    cov = load_covariates().reindex(eids)
    age = cov["age"].to_numpy()
    sex = cov["sex"].to_numpy()
    ok = ~(np.isnan(age) | np.isnan(sex))

    # Well-powered diseases (same criterion as the correlation figure).
    n_event = np.array([int(events[mask[:, p], p].sum()) for p in range(len(phe))])
    W = np.where(n_event >= MIN_EVENTS)[0]
    phe_W = [phe[p] for p in W]
    meta = _phecode_meta()
    names = [meta["name"].get(p, p) for p in phe_W]
    cats = np.array([meta["cat"].get(p, "Other") for p in phe_W])
    n_dis = len(W)
    print(f"[axis] subjects={len(eids)} ok-demo={int(ok.sum())} well-powered diseases={n_dis}")

    # Demographic-residualized, standardized predicted risks (the PCA input).
    Hw_res = _residualize(H[ok][:, W], age[ok], sex[ok])
    Xc = Hw_res - Hw_res.mean(axis=0, keepdims=True)
    Xs = Xc / (Xc.std(axis=0, keepdims=True) + 1e-9)            # n_ok x n_dis, unit-variance cols
    n_ok = Xs.shape[0]

    U, svals, Vt = np.linalg.svd(Xs, full_matrices=False)
    var_explained = (svals ** 2) / np.sum(svals ** 2)
    loading = Vt[0].copy()
    if loading.sum() < 0:                                        # sign-align: positive = shared axis
        loading = -loading
    score1 = Xs @ loading                                        # per-subject position on the shared axis

    # The shared axis is a general disease-burden gradient.
    burden = (events[ok][:, W] > 0.5).sum(axis=1).astype(float)  # incident diseases per subject (of n_dis)
    rho_burden, p_burden = spearmanr(score1, burden)
    # mortality anchor (predicted all-cause-death hazard), if present in the panel
    rho_mort = float("nan")
    if "time_to_death" in phe:
        m = phe.index("time_to_death")
        rho_mort = float(spearmanr(score1, H[ok, m])[0])

    # Shared / beyond split (orthogonal projection onto the PC1 direction).
    Xs_shared = np.outer(score1, loading)                       # rank-1 PC1 reconstruction
    Xs_beyond = Xs - Xs_shared
    S_beyond = np.corrcoef(Xs_beyond, rowvar=False)
    v_tot = _col_var(Xs)
    v_shared = _col_var(Xs_shared)

    # Per-category variance partition. For each category g: shared = variance on
    # the global PC1 axis; group-specific = variance of the leading within-group
    # residual axis (top PC of Xs_beyond restricted to g); idiosyncratic = the
    # rest of the residual. Fractions normalise over (shared + residual), an
    # ANOVA-style partition.
    rows = []
    for c in pd.unique(cats):
        G = np.where(cats == c)[0]
        if len(G) < MIN_CAT:
            continue
        v_sh = float(v_shared[G].sum())
        Bg = Xs_beyond[:, G]
        sg = np.linalg.svd(Bg, full_matrices=False, compute_uv=False)
        v_beyond = float((sg ** 2).sum() / n_ok)
        v_group = float((sg[0] ** 2) / n_ok)                   # leading within-group residual axis
        v_idio = max(v_beyond - v_group, 0.0)
        denom = v_sh + v_beyond
        rows.append({
            "category": c,
            "n": int(len(G)),
            "shared_frac": v_sh / denom,
            "group_specific_frac": v_group / denom,
            "idiosyncratic_frac": v_idio / denom,
            "mean_loading": float(loading[G].mean()),
        })
    vp = pd.DataFrame(rows).sort_values("group_specific_frac", ascending=False).reset_index(drop=True)

    # S_beyond ordered by organ system (categories by descending mean loading).
    cat_mean_load = {c: loading[np.where(cats == c)[0]].mean() for c in pd.unique(cats)}
    cat_order = sorted(pd.unique(cats), key=lambda c: -cat_mean_load[c])
    perm = np.concatenate([np.where(cats == c)[0] for c in cat_order]).astype(int)
    bounds, lab_pos, k = [], [], 0
    for c in cat_order:
        m = int((cats == c).sum())
        bounds.append(k + m)
        lab_pos.append((c, k + m / 2.0))
        k += m

    # Save.
    np.savez(f"{OUT}/pc1_subject.npz", score1=score1, burden=burden)
    pd.DataFrame({
        "phecode": phe_W, "name": names, "category": cats,
        "n_event": n_event[W], "pc1_loading": loading,
    }).to_csv(f"{OUT}/loadings.csv", index=False)
    np.savez(f"{OUT}/beyond_by_category.npz",
             S_beyond_ordered=S_beyond[np.ix_(perm, perm)],
             bounds=np.array(bounds),
             cat_labels=np.array([c for c, _ in lab_pos], dtype=object),
             cat_label_pos=np.array([p for _, p in lab_pos]))
    vp.to_csv(f"{OUT}/variance_partition.csv", index=False)

    summary = {
        "n_subjects": int(len(eids)),
        "n_ok_demo": int(n_ok),
        "n_diseases": int(n_dis),
        "pc1_var_explained": float(var_explained[0]),
        "pc1_3_var_explained": float(var_explained[:3].sum()),
        "frac_loadings_positive": float((loading > 0).mean()),
        "burden_spearman": float(rho_burden),
        "burden_spearman_p": float(p_burden),
        "mortality_hazard_spearman": rho_mort,
        "mean_shared_frac": float(vp["shared_frac"].mean()),
        "top_group_specific": vp.loc[vp.index[:5], ["category", "group_specific_frac"]].to_dict("records"),
        "var_explained_head": [float(x) for x in var_explained[:10]],
    }
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(vp.to_string(index=False))
    print(f"[axis] wrote outputs to {OUT}")


if __name__ == "__main__":
    main()
