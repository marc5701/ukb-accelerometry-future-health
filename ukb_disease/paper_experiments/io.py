"""Thin loaders and shared constants for the paper experiments.

Reuses io_utils.load_seed_avg (seed-averaged canonical deep-model hazards/labels)
and the canonical label/demographic CSVs. Inputs are read-only, so analysis
outputs default to a writable staging directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ukb_disease.baseline.config import resolve_oak_path
from ukb_disease.paper_additions.io_utils import load_seed_avg
from ukb_disease.paths import UKB_ROOT

# --- canonical data paths --------------------------------------------------- #
RUNS_ROOT = f"{UKB_ROOT}/runs_cox"
DEEP_PREFIX = "embedding_disjoint_split"
DAY_MEAN_DISJOINT_SPLIT = f"{UKB_ROOT}/day_mean_disjoint_split"
MORTALITY_PHECODES_CSV = f"{UKB_ROOT}/metadata/mortality_and_phecodes.csv"
PHECODE_MAP_CSV = f"{UKB_ROOT}/metadata/Phecode_mapX_filtered.csv"
ACC_DEM_CSV = f"{UKB_ROOT}/metadata/acc_dem.csv"
BMI_CSV = f"{UKB_ROOT}/metadata/BMI_yes.csv"

# --- output layout ---------------------------------------------------------- #
# Writable staging directory for analysis outputs.
STAGING_BASE = f"{UKB_ROOT}/paper_experiments/output"
OAK_BASE = f"{UKB_ROOT}/paper_experiments"

HORIZON = 6.0  # canonical incident horizon (years)

# --- flagship outcomes ------------------------------------------------------ #
FLAGSHIPS = {
    "Parkinson's disease": "NS_324.11",
    "Alzheimer's disease": "NS_328.11",
    "All-cause dementia": "NS_328.1",
    "Type 2 diabetes": "EM_202.2",
    "Atrial fibrillation": "CV_416.2",
    "Heart failure": "CV_424",
    "All-cause mortality": "time_to_death",
}


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def out_dir(name: str, base: str = STAGING_BASE) -> str:
    """Staging output subdir for an analysis (e.g. 'Calibration')."""
    return ensure_dir(os.path.join(base, name))


def write_json(obj, path: str) -> None:
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not JSON serializable: {type(o)}")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)


# --- canonical model + labels ---------------------------------------------- #
def load_canonical() -> dict:
    """Seed-averaged canonical deep-model run.

    Returns dict: eids (N,), hazards (N,P) log-hazard LP, times (N,P),
    events (N,P), mask (N,P) bool, phecodes (list[P]), name2j (dict).
    """
    sa = load_seed_avg(DEEP_PREFIX, root=RUNS_ROOT)
    sa["events"] = sa["events"].astype(bool)
    sa["mask"] = sa["mask"].astype(bool)
    sa["name2j"] = {n: j for j, n in enumerate(sa["phecodes"])}
    return sa


def phecode_meta() -> dict:
    """phecode -> {'name': PhecodeString, 'cat': PhecodeCategory}."""
    m = pd.read_csv(resolve_oak_path(PHECODE_MAP_CSV))[
        ["Phecode", "PhecodeString", "PhecodeCategory"]].drop_duplicates("Phecode")
    name = dict(zip(m.Phecode, m.PhecodeString))
    cat = dict(zip(m.Phecode, m.PhecodeCategory))
    name["time_to_death"] = "All-cause mortality"
    cat["time_to_death"] = "Mortality"
    return {"name": name, "cat": cat}


def load_demographics(eids: np.ndarray):
    """Return (age, sex, bmi) aligned to eids; NaN -> column mean."""
    dem = pd.read_csv(resolve_oak_path(ACC_DEM_CSV)).set_index("eid")
    bmi = (pd.read_csv(resolve_oak_path(BMI_CSV), usecols=["eid", "21001-0.0"])
           .rename(columns={"21001-0.0": "bmi"}).groupby("eid").mean())
    age = dem.reindex(eids)["age_at_first_run"].to_numpy(float)
    sex = dem.reindex(eids)["sex"].to_numpy(float)
    bm = bmi.reindex(eids)["bmi"].to_numpy(float)
    cov = {}
    for nm, a in (("age", age), ("sex", sex), ("bmi", bm)):
        cov[nm] = float(np.isfinite(a).mean())
        a[~np.isfinite(a)] = np.nanmean(a)
    return age, sex, bm, cov


def disease_arrays(sa: dict, phecode: str, min_eval: int = 2):
    """Per-disease evaluable (lp, time, event, eids) for one phecode."""
    j = sa["name2j"][phecode]
    m = sa["mask"][:, j]
    return (sa["hazards"][m, j].astype(float), sa["times"][m, j].astype(float),
            sa["events"][m, j].astype(float), sa["eids"][m], j, int(m.sum()))
