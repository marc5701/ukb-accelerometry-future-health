"""Per-disease attribution profiles (all evaluable diseases) with uncertainty.

For each disease emits the vector of per-cell dC (mean, 95% CI, BH-q, significant,
decision_rule decision), the "what does the model attend to for this disease" profile.
Relates to clinical priors descriptively (names the dominant cell) without letting
priors gate a surprising but confirmed cell. Wide-CI diseases (n_event<25 or CI
half-width>0.10) are flagged, not hidden.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ukb_disease.baseline.config import resolve_oak_path
from ukb_disease.paths import UKB_ROOT

OUT_DEFAULT = f"{UKB_ROOT}/interpretability/profiles"


def build_profiles(dc_table: dict, n_event: dict | None = None,
                   decision: np.ndarray | None = None) -> pd.DataFrame:
    """dc_table from validate.analysis.dc_table. Long-form per (disease, cell)."""
    cells = dc_table["cell_names"]
    ph = dc_table["phecodes"]
    rows = []
    for j, code in enumerate(ph):
        col_dc = dc_table["dc_mean"][:, j]
        if not np.isfinite(col_dc).any():
            continue
        dom = int(np.nanargmax(col_dc))
        ci_hw = np.nanmax((dc_table["dc_hi"][:, j] - dc_table["dc_lo"][:, j]) / 2.0)
        nev = (n_event or {}).get(code, np.nan)
        wide = (np.isfinite(nev) and nev < 25) or (np.isfinite(ci_hw) and ci_hw > 0.10)
        for i, cell in enumerate(cells):
            rows.append({
                "phecode": code, "cell": cell,
                "dc_mean": dc_table["dc_mean"][i, j],
                "dc_lo": dc_table["dc_lo"][i, j], "dc_hi": dc_table["dc_hi"][i, j],
                "q": dc_table["q"][i, j], "significant": bool(dc_table["significant"][i, j]),
                "decision": int(decision[i, j]) if decision is not None else np.nan,
                "dominant_cell": cells[dom], "n_event": nev, "wide_ci": wide,
            })
    return pd.DataFrame(rows)


def save_profiles(df: pd.DataFrame, out_root: str = OUT_DEFAULT, tag: str = "tier1") -> str:
    out_dir = resolve_oak_path(out_root)
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"attribution_profiles_{tag}.csv")
    df.to_csv(p, index=False)
    return p
