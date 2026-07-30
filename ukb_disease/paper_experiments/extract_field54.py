"""Extract UKB assessment-centre (field 54-0.0) per eid from the main dataset.

Heavy gzip read of the wide main dataset. Streams only eid + 54-0.0 via
csv.reader (quote-safe) and writes a small parquet for G6 transportability.
Field 54 is used only as a grouping variable (region), never as a predictive
feature.
"""
from __future__ import annotations

import csv
import gzip
import os

import pandas as pd

from ukb_disease.paths import UKB_ROOT
from ukb_disease.baseline.config import resolve_oak_path

SRC = f"{UKB_ROOT}/metadata/ukb_main_dataset.csv.gz"
OUT = f"{UKB_ROOT}/paper_experiments/Regional_transport"


def main():
    src = resolve_oak_path(SRC)
    out = resolve_oak_path(OUT)
    os.makedirs(out, exist_ok=True)
    print(f"[field54] reading {src}", flush=True)
    eids, centre = [], []
    with gzip.open(src, "rt", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        # header entries are quoted like "eid","54-0.0"; csv.reader strips quotes
        ie = header.index("eid")
        ic = header.index("54-0.0")
        for i, row in enumerate(r):
            eids.append(row[ie])
            centre.append(row[ic])
            if (i + 1) % 100000 == 0:
                print(f"[field54] {i+1} rows...", flush=True)
    df = pd.DataFrame({"eid": pd.to_numeric(pd.Series(eids), errors="coerce"),
                       "centre": pd.to_numeric(pd.Series(centre), errors="coerce")})
    df = df.dropna(subset=["eid"]).astype({"eid": "int64"})
    p = os.path.join(out, "assessment_centre.parquet")
    df.to_parquet(p, index=False)
    n_centres = df["centre"].nunique(dropna=True)
    print(f"[field54] wrote {len(df)} rows, {n_centres} distinct centres -> {p}", flush=True)
    print(df["centre"].value_counts(dropna=False).head(30).to_string(), flush=True)


if __name__ == "__main__":
    main()
