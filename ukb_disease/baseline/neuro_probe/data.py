"""Case/control builder for the unseen-neuro wrist-only probe.

Cases come from the held-out neuro cohort (`ukb_neuro_files.json`), the subjects
whose Parkinson's / Alzheimer's / dementia diagnoses the wrist foundation model
never saw in pretraining and that were filtered from the disease head. We attach
the per-disease time-to-diagnosis cell (years, relative to wear) and the cached
512-d wrist embedding.

Controls are drawn from the cached embedding pool, restricted to subjects with
no neuro diagnosis at all (every NS_324*/NS_328* column NaN) and not in the neuro
set, then age and sex matched to the case set (1:k greedy, caliper about 3y,
exact sex). The matching generalizes the 1:1 case/control matcher to k controls.

Wrist-only: age and sex live only in the matching step and the guardrail table.
They are never returned as probe features. `build_cohort` returns the 512-d
embedding matrix `X` plus the binary label `y`; the demographic columns travel
in a separate metadata frame used only for matching and reporting.

Disease columns (mortality_and_phecodes.csv, value = years to diagnosis,
NaN=never, <0=prevalent/before-wear, >0=incident):
  * PD       = NS_324.11
  * dementia = NS_328.1
  * Alzheimer= NS_328.11
Broader NS_324*/NS_328* columns define "any neuro diagnosis" for control
exclusion.

Timing filter (primary = prodromal/incident):
  * PD  : cell > 2.0  (prodromal, per Schalkamp et al. 2023)
  * AD  : cell >= 0    (incident)
  * dem : cell >= 0    (incident)
  * pooled "any-neurodegenerative" = union of the three incident sets
Prevalent (cell < 0) is kept as a separate secondary set.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ukb_disease.baseline.config import (
    ACC_DEM_CSV,
    DAY_MEAN_ROOT,
    MORTALITY_PHECODES_CSV,
    resolve_oak_path,
)
from ukb_disease.paths import UKB_ROOT

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
NEURO_FILES_JSON = f"{UKB_ROOT}/data_splits/ukb_split_full/ukb_neuro_files.json"

PD_PHECODE = "NS_324.11"
DEMENTIA_PHECODE = "NS_328.1"
ALZHEIMER_PHECODE = "NS_328.11"

NEURO_PREFIXES = ("NS_324", "NS_328")  # define "any neuro dx" for control exclusion

PD_PRODROMAL_BUFFER = 2.0  # years, splits prodromal from diagnosed PD

# disease -> (phecode column, primary timing rule)
#   ">2"  : cell > 2.0   (PD prodromal)
#   ">=0" : cell >= 0     (incident)
DISEASE_SPEC = {
    "pd": (PD_PHECODE, ">2"),
    "pd_diagnosed": (PD_PHECODE, "<=2"),  # diagnosed/prevalent (<=2y; motor signs present)
    "pd_any": (PD_PHECODE, "notna"),      # any PD ever (prodromal + diagnosed)
    "alzheimer": (ALZHEIMER_PHECODE, ">=0"),
    "dementia": (DEMENTIA_PHECODE, ">=0"),
}

EMB_DIM = 512  # concat(AR mean-over-valid-days 256, HA mean-over-valid-days 256)

SPLITS = ("train", "test")


# --------------------------------------------------------------------------- #
# Embedding pool (mask-aware subject-level 512-d vectors)
# --------------------------------------------------------------------------- #
@dataclass
class EmbeddingPool:
    """Subject-level wrist embeddings + demographics + neuro flags.

    Attributes
    ----------
    eids : (n,) int64, unique subject eids present in the cached pool.
    X    : (n, 512) float32, subject embedding = mean over the subject's
           recordings of concat(mask-aware mean-over-valid-days AR,
           mask-aware mean-over-valid-days HA).
    meta : DataFrame indexed by eid; cols [age, sex, has_age_sex, n_recordings].
           Demographics are matching/reporting only, never probe features.
    """

    eids: np.ndarray
    X: np.ndarray
    meta: pd.DataFrame

    def index_of(self, eid_array: np.ndarray) -> np.ndarray:
        """Row indices into X/eids for the given eids (must all be present)."""
        pos = {int(e): i for i, e in enumerate(self.eids)}
        return np.array([pos[int(e)] for e in eid_array], dtype=np.int64)


def _recording_embedding(ar_row: np.ndarray, ha_row: np.ndarray,
                         mask_row: np.ndarray) -> np.ndarray | None:
    """512-d recording embedding: mask-aware mean over valid days, AR then HA."""
    m = mask_row.astype(bool)
    if not m.any():
        return None
    ar_mean = ar_row[m].mean(axis=0)
    ha_mean = ha_row[m].mean(axis=0)
    return np.concatenate([ar_mean, ha_mean]).astype(np.float32)


def load_embedding_pool(means_root: str = DAY_MEAN_ROOT,
                        splits=SPLITS) -> EmbeddingPool:
    """Build subject-level 512-d embeddings from the stacked day-mean caches.

    Reads <means_root>/<split>/{ar.npy, ha.npy, valid_day_mask.npy,
    index.parquet}. A subject can have recordings across splits (e.g. a repeat
    visit lands in test while the first visit is in train); we pool ALL its
    recordings into one subject embedding by averaging the per-recording 512-d
    vectors.
    """
    rec_emb: dict[int, list[np.ndarray]] = {}
    for split in splits:
        d = resolve_oak_path(os.path.join(means_root, split))
        if not os.path.isdir(d):
            continue
        ar = np.load(os.path.join(d, "ar.npy"), mmap_mode="r")
        ha = np.load(os.path.join(d, "ha.npy"), mmap_mode="r")
        vm = np.load(os.path.join(d, "valid_day_mask.npy"))
        idx = pd.read_parquet(os.path.join(d, "index.parquet"))
        eid_arr = idx.eid.to_numpy()
        for i in range(len(idx)):
            emb = _recording_embedding(np.asarray(ar[i]), np.asarray(ha[i]), vm[i])
            if emb is None:
                continue
            rec_emb.setdefault(int(eid_arr[i]), []).append(emb)

    eids = np.array(sorted(rec_emb.keys()), dtype=np.int64)
    X = np.zeros((len(eids), EMB_DIM), dtype=np.float32)
    n_rec = np.zeros(len(eids), dtype=np.int32)
    for k, e in enumerate(eids):
        recs = rec_emb[int(e)]
        X[k] = np.mean(recs, axis=0)
        n_rec[k] = len(recs)

    # demographics (matching/reporting only)
    dem = pd.read_csv(resolve_oak_path(ACC_DEM_CSV),
                      usecols=["eid", "age_at_first_run", "sex"])
    dem = dem.rename(columns={"age_at_first_run": "age"}).set_index("eid")
    dd = dem.reindex(eids)
    meta = pd.DataFrame({
        "eid": eids,
        "age": dd["age"].to_numpy(),
        "sex": dd["sex"].to_numpy(),
        "n_recordings": n_rec,
    }).set_index("eid")
    meta["has_age_sex"] = meta.age.notna() & meta.sex.notna()
    return EmbeddingPool(eids=eids, X=X, meta=meta)


# --------------------------------------------------------------------------- #
# Neuro set + per-disease labels
# --------------------------------------------------------------------------- #
def load_neuro_eids() -> set[int]:
    """EIDs of the held-out neuro cohort (scrape eid = int before 1st '_')."""
    files = json.load(open(resolve_oak_path(NEURO_FILES_JSON)))
    return {int(os.path.basename(f).split("_")[0]) for f in files}


def load_phecode_table(extra_cols=()):
    """Read the disease + broad-neuro phecode columns, indexed by eid."""
    all_cols = pd.read_csv(resolve_oak_path(MORTALITY_PHECODES_CSV),
                           nrows=0).columns.tolist()
    neuro_cols = [c for c in all_cols
                  if any(c.startswith(p) for p in NEURO_PREFIXES)]
    disease_cols = [PD_PHECODE, DEMENTIA_PHECODE, ALZHEIMER_PHECODE]
    need = sorted(set(["eid"] + neuro_cols + disease_cols + list(extra_cols)))
    df = pd.read_csv(resolve_oak_path(MORTALITY_PHECODES_CSV),
                     usecols=need).set_index("eid")
    return df, neuro_cols


def _apply_timing(cell: pd.Series, rule: str) -> np.ndarray:
    """Boolean mask of cases under the primary timing rule."""
    c = cell.to_numpy()
    if rule == ">2":
        return c > PD_PRODROMAL_BUFFER
    if rule == ">=0":
        return c >= 0.0
    if rule == "<=2":          # diagnosed/prevalent PD (NaN excluded automatically)
        return c <= PD_PRODROMAL_BUFFER
    if rule == "notna":        # any PD ever
        return ~np.isnan(c)
    raise ValueError(rule)


# --------------------------------------------------------------------------- #
# Cohort construction
# --------------------------------------------------------------------------- #
@dataclass
class Cohort:
    disease: str
    case_eids: np.ndarray         # cases (in pool, has_age_sex)
    ctrl_eids: np.ndarray         # matched controls (in pool, has_age_sex)
    X: np.ndarray                 # (n_case+n_ctrl, 512), wrist embedding only
    y: np.ndarray                 # (n,) 1=case 0=ctrl
    eid: np.ndarray               # (n,) aligned to X/y
    age: np.ndarray               # (n,) reporting/guardrail only, not a feature
    sex: np.ndarray               # (n,) reporting/guardrail only, not a feature
    surv_time: np.ndarray         # (n,) years-to-event (cases) / horizon (ctrl)
    surv_event: np.ndarray        # (n,) bool, case had the dx
    n_case: int = 0
    n_ctrl: int = 0
    counts: dict = field(default_factory=dict)


def match_1tok(case_meta: pd.DataFrame, pool_meta: pd.DataFrame,
               k: int, rng: np.random.Generator, caliper: float = 3.0
               ) -> np.ndarray:
    """Greedy nearest-age, exact-sex, 1:k matching without replacement.

    Generalizes the 1:1 case/control matcher to k controls. Returns control eids
    (subject-disjoint from cases by construction: the pool is pre-filtered to
    exclude cases and neuro subjects upstream).
    """
    pool = pool_meta[pool_meta.has_age_sex].copy()
    case_meta = case_meta[case_meta.has_age_sex]
    # shuffle pool for random tie-breaking
    pool = pool.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
    age = pool.age.to_numpy()
    sex = pool.sex.to_numpy()
    eid = pool.index.to_numpy()
    used: set[int] = set()
    picks: list[int] = []
    for _, c in case_meta.iterrows():
        for _ in range(k):
            avail = np.array([e not in used for e in eid])
            same = avail & (sex == c.sex)
            if not same.any():
                break
            d = np.abs(age - c.age)
            d = np.where(same, d, np.inf)
            within = d <= caliper
            cand = within if within.any() else (d < np.inf)
            j = int(np.argmin(np.where(cand, d, np.inf)))
            used.add(int(eid[j]))
            picks.append(int(eid[j]))
    return np.array(picks, dtype=np.int64)


def build_cohort(disease: str, pool: EmbeddingPool, phecodes: pd.DataFrame,
                 neuro_eids: set[int], neuro_cols: list[str], *,
                 k: int = 3, caliper: float = 3.0, seed: int = 42,
                 prevalent: bool = False) -> Cohort:
    """Assemble a matched case/control cohort for one disease (or 'pooled').

    Cases = neuro-cohort subjects with the disease phecode, timing-filtered,
    that have a cached embedding. Controls = pool subjects with NO neuro dx,
    NOT in the neuro set, age+sex matched 1:k to the cases.
    """
    rng = np.random.default_rng(seed)
    pool_eid_set = set(int(e) for e in pool.eids)

    # --- case selection ---
    if disease == "pooled":
        # union of the three INCIDENT sets (PD>=0, AD>=0, dem>=0)
        case_mask = np.zeros(len(phecodes), dtype=bool)
        for col in (PD_PHECODE, ALZHEIMER_PHECODE, DEMENTIA_PHECODE):
            case_mask |= _apply_timing(phecodes[col], ">=0")
        # for survival anchoring use the earliest available positive cell.
        # Rows with no incident cell (all-NaN slice) stay NaN. They are not
        # cases, so they never reach the survival fields. Compute the per-row
        # min only over rows that have at least one incident cell to avoid the
        # benign All-NaN-slice RuntimeWarning.
        cells = phecodes[[PD_PHECODE, ALZHEIMER_PHECODE, DEMENTIA_PHECODE]].to_numpy()
        pos = np.where(cells >= 0, cells, np.nan)
        surv_cell = np.full(len(phecodes), np.nan)
        any_pos = ~np.isnan(pos).all(axis=1)
        surv_cell[any_pos] = np.nanmin(pos[any_pos], axis=1)
    else:
        col, rule = DISEASE_SPEC[disease]
        if prevalent:
            case_mask = phecodes[col].to_numpy() < 0.0
        else:
            case_mask = _apply_timing(phecodes[col], rule)
        surv_cell = phecodes[col].to_numpy()

    case_eids_all = set(phecodes.index[case_mask].astype(int))
    # cases must be in the held-out neuro cohort AND have embeddings
    case_eids = np.array(sorted(case_eids_all & neuro_eids & pool_eid_set),
                         dtype=np.int64)

    # --- control pool: no neuro dx at all, not in neuro set, in embedding pool ---
    has_any_neuro = phecodes[neuro_cols].notna().any(axis=1)
    neuro_dx_eids = set(phecodes.index[has_any_neuro].astype(int))
    ctrl_eligible = np.array(
        sorted((pool_eid_set - neuro_dx_eids - neuro_eids) - case_eids_all),
        dtype=np.int64)

    # restrict to has_age_sex on both sides (needed for matching)
    case_meta = pool.meta.loc[case_eids]
    case_meta = case_meta[case_meta.has_age_sex]
    case_eids = case_meta.index.to_numpy()

    pool_meta = pool.meta.loc[ctrl_eligible]
    ctrl_eids = match_1tok(case_meta, pool_meta, k=k, rng=rng, caliper=caliper)

    # --- assemble matrices (wrist embedding ONLY) ---
    all_eids = np.concatenate([case_eids, ctrl_eids])
    rows = pool.index_of(all_eids)
    X = pool.X[rows]                       # (n, 512), no age/sex
    y = np.concatenate([np.ones(len(case_eids), int),
                        np.zeros(len(ctrl_eids), int)])
    age = pool.meta.loc[all_eids, "age"].to_numpy()
    sex = pool.meta.loc[all_eids, "sex"].to_numpy()

    # survival fields (for time-dependent AUROC; controls censored at horizon)
    cell_by_eid = pd.Series(surv_cell, index=phecodes.index)
    case_cells = cell_by_eid.reindex(case_eids).to_numpy()
    HORIZON = float(np.nanmax(case_cells)) if len(case_cells) else 0.0
    surv_time = np.concatenate([
        np.where(case_cells > 0, case_cells, 1e-3),   # incident time
        np.full(len(ctrl_eids), HORIZON),
    ])
    surv_event = np.concatenate([
        np.ones(len(case_eids), bool),
        np.zeros(len(ctrl_eids), bool),
    ])

    counts = {
        "disease": disease,
        "prevalent": prevalent,
        "n_case_phecode_total": int(len(case_eids_all)),
        "n_case_in_neuro_cohort": int(len(case_eids_all & neuro_eids)),
        "n_case_with_embedding": int(len(case_eids)),
        "n_ctrl_eligible": int(len(ctrl_eligible)),
        "n_ctrl_matched": int(len(ctrl_eids)),
        "ratio_case_ctrl": (len(ctrl_eids) / max(len(case_eids), 1)),
        "k_requested": k,
    }
    return Cohort(
        disease=disease, case_eids=case_eids, ctrl_eids=ctrl_eids,
        X=X, y=y, eid=all_eids, age=age, sex=sex,
        surv_time=surv_time, surv_event=surv_event,
        n_case=int(len(case_eids)), n_ctrl=int(len(ctrl_eids)),
        counts=counts,
    )
