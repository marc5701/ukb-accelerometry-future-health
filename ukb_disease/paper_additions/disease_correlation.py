"""Which diseases share the same prediction signal, vs mere comorbidity.

Correlated predicted-risk scores across diseases can mean (a) the model reads a
shared actigraphy signal, or (b) the diseases simply co-occur in the same people
(comorbidity), or (c) a shared age/sex axis the embedding encodes. We disentangle
all three:

  - S        : Spearman correlation of predicted per-subject hazards across diseases.
  - S_resid  : same, after residualizing each disease's hazard on age + sex (removes
               the shared demographic axis the embedding encodes).
  - Co       : label co-occurrence (phi) of incident events across diseases (comorbidity).

We then ask how much of S_resid survives after accounting for Co (regress S_resid on
Co across disease pairs; the residual is "shared actigraphy signal beyond comorbidity")
and cluster S_resid to reveal disease groupings.

CPU-only, reads the seed-averaged Embedding hazards. Writes matrices + a summary to
output/disease_correlation/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

from ukb_disease.baseline.config import resolve_oak_path
from ukb_disease.baseline.outcomes_processing import load_covariates
from ukb_disease.paper_additions.io_utils import load_seed_avg
from ukb_disease.paths import UKB_ROOT

PHECODE_MAP = f"{UKB_ROOT}/metadata/Phecode_mapX_filtered.csv"
OUT = str(Path(__file__).resolve().parent / "output" / "disease_correlation")
MIN_EVENTS = 50   # well-powered diseases for a stable correlation/comorbidity estimate
N_CLUSTERS = 8


def _phecode_meta() -> dict:
    df = pd.read_csv(resolve_oak_path(PHECODE_MAP),
                     usecols=["Phecode", "PhecodeString", "PhecodeCategory"]).drop_duplicates("Phecode")
    name = dict(zip(df["Phecode"], df["PhecodeString"]))
    cat = dict(zip(df["Phecode"], df["PhecodeCategory"]))
    name["time_to_death"] = "All-cause mortality"
    cat["time_to_death"] = "Mortality"
    return {"name": name, "cat": cat}


def _spearman_cols(mat: np.ndarray) -> np.ndarray:
    """Spearman correlation matrix across columns of `mat` (subjects x diseases)."""
    ranks = np.apply_along_axis(rankdata, 0, mat)
    return np.corrcoef(ranks, rowvar=False)


def _residualize(H: np.ndarray, age: np.ndarray, sex: np.ndarray) -> np.ndarray:
    """Residualize each column of H (subjects x diseases) on [1, age_z, age_z^2, sex]."""
    az = (age - age.mean()) / (age.std() + 1e-9)
    X = np.column_stack([np.ones_like(az), az, az ** 2, sex.astype(float)])
    beta, *_ = np.linalg.lstsq(X, H, rcond=None)
    return H - X @ beta


def main() -> None:
    Path(OUT).mkdir(parents=True, exist_ok=True)
    sa = load_seed_avg("embedding_disjoint_split")
    eids, H, events, mask, phe = (sa["eids"], sa["hazards"], sa["events"],
                                  sa["mask"], sa["phecodes"])
    H = H.astype(np.float64)

    # Demographics aligned to test subjects.
    cov = load_covariates().reindex(eids)
    age = cov["age"].to_numpy()
    sex = cov["sex"].to_numpy()
    ok = ~(np.isnan(age) | np.isnan(sex))
    print(f"[corr] subjects={len(eids)} with demographics={int(ok.sum())}")

    # Well-powered diseases (stable correlation + comorbidity).
    n_event = np.array([int(events[mask[:, p], p].sum()) for p in range(len(phe))])
    W = np.where(n_event >= MIN_EVENTS)[0]
    print(f"[corr] well-powered diseases (>= {MIN_EVENTS} events): {len(W)} / {len(phe)}")
    phe_W = [phe[p] for p in W]

    # Predicted-score correlations (all subjects; the model predicts a hazard for everyone).
    Hw = H[:, W]
    S = _spearman_cols(Hw)
    Hw_res = _residualize(H[ok][:, W], age[ok], sex[ok])
    S_resid = _spearman_cols(Hw_res)

    # Comorbidity: phi (Pearson of binary incident-event indicators) across the test set.
    Ew = (events[:, W] > 0.5).astype(np.float64)
    # mask non-evaluable subjects per disease to NaN then pairwise-complete corr is heavy;
    # incident events are defined for all rows here (eval_mask only gates SCORING), so use all.
    Co = np.corrcoef(Ew, rowvar=False)

    meta = _phecode_meta()
    names = [meta["name"].get(p, p) for p in phe_W]
    cats = [meta["cat"].get(p, "Other") for p in phe_W]

    # Dominant shared axis: PCA of the demographic-residualized hazards.
    Xc = Hw_res - Hw_res.mean(axis=0, keepdims=True)
    # correlation-scale PCA (standardize each disease) so it is not dominated by scale
    Xs = Xc / (Xc.std(axis=0, keepdims=True) + 1e-9)
    _, svals, Vt = np.linalg.svd(Xs, full_matrices=False)
    var_explained = (svals ** 2) / np.sum(svals ** 2)
    # Project out PC1 (the shared axis) to expose SECONDARY disease-specific structure.
    score1 = Xs @ Vt[0]
    Xs_beyond = Xs - np.outer(score1, Vt[0])
    S_beyond = np.corrcoef(Xs_beyond, rowvar=False)

    # PC1 loading per disease (sign-aligned so positive = aligned with the shared axis).
    loading = Vt[0].copy()
    if np.sum(loading) < 0:
        loading = -loading
    ld_order = np.argsort(-loading)
    pc1_top = [(names[i], cats[i], float(loading[i])) for i in ld_order[:8]]
    pc1_bottom = [(names[i], cats[i], float(loading[i])) for i in ld_order[-8:]]

    # Cluster on the beyond-shared-axis structure (PC1 removed) so the
    # dominant common axis does not swamp the disease-specific grouping.
    d = 1.0 - S_beyond
    np.fill_diagonal(d, 0.0)
    d = np.clip((d + d.T) / 2.0, 0, 2)
    Z = linkage(squareform(d, checks=False), method="average")
    order = leaves_list(Z)
    clusters = fcluster(Z, t=N_CLUSTERS, criterion="maxclust")

    # Quantify: shared signal vs demographics vs comorbidity.
    iu = np.triu_indices(len(W), k=1)
    s_pairs, sr_pairs, co_pairs = S[iu], S_resid[iu], Co[iu]
    # regress S_resid on comorbidity across pairs: intercept = shared corr at zero comorbidity.
    A = np.column_stack([np.ones_like(co_pairs), co_pairs])
    coef, *_ = np.linalg.lstsq(A, sr_pairs, rcond=None)
    pred = A @ coef
    r2_co = 1.0 - float(np.sum((sr_pairs - pred) ** 2)) / float(np.sum((sr_pairs - sr_pairs.mean()) ** 2))
    low_co = co_pairs < 0.01  # disease pairs with essentially no comorbidity

    summary = {
        "n_subjects": int(len(eids)),
        "n_subjects_with_demo": int(ok.sum()),
        "min_events": MIN_EVENTS,
        "n_diseases_wellpowered": int(len(W)),
        "mean_score_corr_raw": float(s_pairs.mean()),
        "mean_score_corr_demo_residualized": float(sr_pairs.mean()),
        "demographic_share_of_corr": float(s_pairs.mean() - sr_pairs.mean()),
        "mean_comorbidity_phi": float(co_pairs.mean()),
        "S_resid_vs_comorbidity_R2": float(r2_co),
        "S_resid_intercept_at_zero_comorbidity": float(coef[0]),
        "S_resid_slope_per_comorbidity": float(coef[1]),
        "mean_S_resid_among_noncomorbid_pairs": float(sr_pairs[low_co].mean()),
        "frac_noncomorbid_pairs": float(low_co.mean()),
        "pc1_var_explained": float(var_explained[0]),
        "pc1_3_var_explained": float(var_explained[:3].sum()),
        "frac_pairs_resid_corr_gt_0.1": float((sr_pairs > 0.1).mean()),
        "n_clusters": int(N_CLUSTERS),
        "pc1_top_loading_diseases": pc1_top,
        "pc1_bottom_loading_diseases": pc1_bottom,
    }
    # Per-cluster within-group shared signal vs comorbidity.
    cl_rows = []
    for c in np.unique(clusters):
        members = np.where(clusters == c)[0]
        if len(members) < 2:
            continue
        sub = np.ix_(members, members)
        mu = np.triu_indices(len(members), k=1)
        cl_rows.append({
            "cluster": int(c),
            "n": int(len(members)),
            "mean_within_S_resid": float(S_resid[sub][mu].mean()),
            "mean_within_comorbidity": float(Co[sub][mu].mean()),
            "top_categories": pd.Series([cats[m] for m in members]).value_counts().head(3).to_dict(),
            "example_diseases": [names[m] for m in members[:6]],
        })
    summary["clusters"] = cl_rows

    # Save.
    np.save(f"{OUT}/score_corr.npy", S)
    np.save(f"{OUT}/score_corr_residualized.npy", S_resid)
    np.save(f"{OUT}/score_corr_beyond_axis.npy", S_beyond)
    np.save(f"{OUT}/var_explained.npy", var_explained)
    np.save(f"{OUT}/comorbidity_phi.npy", Co)
    np.save(f"{OUT}/cluster_order.npy", order)
    pd.DataFrame({
        "phecode": phe_W, "name": names, "category": cats,
        "n_event": n_event[W], "cluster": clusters, "pc1_loading": loading,
    }).to_csv(f"{OUT}/diseases.csv", index=False)
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps({k: v for k, v in summary.items() if k != "clusters"}, indent=2))
    print(f"[corr] wrote matrices + summary to {OUT}")


if __name__ == "__main__":
    main()
