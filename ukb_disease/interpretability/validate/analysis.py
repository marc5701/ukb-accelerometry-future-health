"""Turn ablation npz (from ladder/run_ablation.py) into per-(cell,disease) dC with
uncertainty and BH-FDR significance, the engine behind the heatmap and decision_rule.

Two uncertainty sources:
  * perm x seed spread: dC sampled across permutation seeds and model seeds
    (permutation noise plus init noise). Always available.
  * EID-clustered subject bootstrap: when ablated per-subject hazards were saved
    (--save-ablated-hazards), the canonical subject-sampling CI on dC. Primary for
    the Tier-2 heatmap.

dC = C_full - C_ablated  (positive means the cell carries signal).
"""

from __future__ import annotations

import glob
import os

import numpy as np

from ukb_disease.baseline.config import resolve_oak_path
from ukb_disease.baseline._per_phecode_utils import bh_fdr, percentile_ci
from ukb_disease.baseline.per_disease_eval import c_index_lifelines
from ukb_disease.paths import UKB_ROOT

ABL_ROOT = f"{UKB_ROOT}/interpretability/ablation"


def load_ablation_seeds(prefix: str, out_root: str = ABL_ROOT) -> list:
    d = resolve_oak_path(os.path.join(out_root, prefix))
    files = sorted(glob.glob(os.path.join(d, "ablation_seed*.npz")))
    if not files:
        raise FileNotFoundError(f"no ablation npz under {d}")
    return [dict(np.load(f, allow_pickle=True)) for f in files]


def stack_dc(npz_list: list, mode: str = "cond"):
    """Stack dC across model seeds by perm seeds.

    Returns (cell_names, phecodes, dc) with dc shape (n_cells, n_disease, n_samples).
    """
    cell_names = [str(c) for c in npz_list[0]["cell_names"]]
    phecodes = [str(p) for p in npz_list[0]["phecodes"]]
    dcs = []
    for z in npz_list:
        cfull = z["c_full"]                # (P,)
        arr = z[mode]                      # (n_cells, K, P)
        if arr.ndim != 3:
            continue
        dcs.append(cfull[None, None, :] - arr)   # (n_cells, K, P)
    dc = np.concatenate(dcs, axis=1)       # (n_cells, n_seeds*K, P)
    dc = np.transpose(dc, (0, 2, 1))       # (n_cells, P, n_samples)
    return cell_names, phecodes, dc


def summarize_spread(dc: np.ndarray):
    """mean, [lo,hi] 95% percentile, one-sided p(dC<=0) over the sample axis (=2)."""
    mean = np.nanmean(dc, axis=2)
    lo, hi = percentile_ci(dc, 2.5, 97.5, axis=2)
    with np.errstate(invalid="ignore"):
        p = np.nanmean((dc <= 0).astype(float), axis=2)
    return mean, lo, hi, p


def bh_per_cell(p: np.ndarray, alpha: float = 0.05):
    """BH-FDR across the disease axis, independently per cell. p shape (n_cells, P)."""
    q = np.full_like(p, np.nan)
    rej = np.zeros_like(p, dtype=bool)
    for ci in range(p.shape[0]):
        q[ci], rej[ci] = bh_fdr(p[ci], alpha=alpha)
    return q, rej


def bootstrap_dc(full_haz: np.ndarray, abl_haz: np.ndarray, et: np.ndarray,
                 ie: np.ndarray, em: np.ndarray, n_boot: int = 1000,
                 min_eval: int = 10, seed: int = 0):
    """EID-clustered (subject-level) paired bootstrap of per-disease dC.

    full_haz/abl_haz: (n_subj, P) per-subject hazards. Subjects are the cluster
    unit (rows already aggregated to subject). Returns (point, lo, hi, p_onesided).
    Subject-level clustering keeps the CI honest when one subject contributes
    several records.
    """
    rng = np.random.default_rng(seed)
    n, P = full_haz.shape
    point = np.full(P, np.nan)
    boots = np.full((n_boot, P), np.nan)
    for j in range(P):
        m = em[:, j].astype(bool)
        idx = np.where(m)[0]
        if len(idx) < min_eval or int((ie[idx, j] > 0).sum()) < 1:
            continue
        cf = c_index_lifelines(full_haz[idx, j], et[idx, j], ie[idx, j])
        ca = c_index_lifelines(abl_haz[idx, j], et[idx, j], ie[idx, j])
        point[j] = cf - ca
        for b in range(n_boot):
            rs = idx[rng.integers(0, len(idx), len(idx))]
            if int((ie[rs, j] > 0).sum()) < 1:
                continue
            boots[b, j] = (c_index_lifelines(full_haz[rs, j], et[rs, j], ie[rs, j])
                           - c_index_lifelines(abl_haz[rs, j], et[rs, j], ie[rs, j]))
    lo, hi = percentile_ci(boots, axis=0)
    with np.errstate(invalid="ignore"):
        p = np.nanmean((boots <= 0).astype(float), axis=0)
    return point, lo, hi, p


def dc_table(npz_list: list, mode: str = "cond", alpha: float = 0.05) -> dict:
    """Full per-(cell,disease) dC summary (spread-based) ready for the heatmap."""
    cell_names, phecodes, dc = stack_dc(npz_list, mode=mode)
    mean, lo, hi, p = summarize_spread(dc)
    q, rej = bh_per_cell(p, alpha=alpha)
    return {
        "cell_names": cell_names, "phecodes": phecodes,
        "dc_mean": mean, "dc_lo": lo, "dc_hi": hi,
        "p": p, "q": q, "significant": rej, "n_samples": dc.shape[2],
    }
