"""Extract the actigraphy cohort's PRS and covariate columns from the full UKB dataset.

The full UKB file is 4.3 GB gzipped, 502,180 subjects x 20,847 columns. We project
only the columns we need and keep only the rows whose eid is in the known
actigraphy cohort (acc_dem.csv, 103,627 subjects). Filtering by the known cohort
EID set is more robust than guessing the accelerometer field; the
accelerometer-field non-empty count is then an independent cross-check.

Columns carried (resolved from the header at run time, so naming changes are caught):
  * eid
  * all disease/trait PRS (5-digit field IDs 26200-26434; the 4-digit
    2624/2634/2644 questionnaire fields that share the 262/263/264 prefix are excluded)
  * 40 genetic principal components (22009-0.1 to 22009-0.40)
  * genetic ethnic grouping 22006-0.0 (in-sample white-British flag)
  * self-reported ethnic background 21000-0.0 (for the ancestry sensitivity contrast)
  * sex 31-0.0, genetic sex 22001-0.0
  * genotyping array 22000-0.0
  * assessment centre 54-0.0
  * age at recruitment 21022-0.0, birth year 34-0.0 (covariate cross-checks)
  * accelerometer fields 90001-0.0 / 90004-0.0 / 90010-0.0 (cohort cross-check only)

Output:
  <out>/prs_cohort.parquet   (canonical, downstream)
  <out>/prs_cohort.csv       (inspectable)
  <out>/extract_meta.json    (provenance and counts)

Usage (~minutes, big file):
  python3 -m ukb_disease.genetics.extract_prs_cohort \
      --out genetics/cohort

Smoke test (sample of the gzip, no full scan):
  python3 -m ukb_disease.genetics.extract_prs_cohort --smoke-nrows 50000 \
      --out smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time

import pandas as pd

from ukb_disease.paths import UKB_ROOT

FULL_UKB = f"{UKB_ROOT}/metadata/ukb_main_dataset.csv.gz"
ACC_DEM = f"{UKB_ROOT}/metadata/acc_dem.csv"

# instance-0 single-column covariates / ancestry / accel / PRS-partition fields we keep verbatim
SCALAR_COLS = [
    "31-0.0",      # sex (reported)
    "22001-0.0",   # genetic sex
    "22000-0.0",   # genotyping array
    "54-0.0",      # assessment centre (instance 0)
    "21022-0.0",   # age at recruitment
    "34-0.0",      # year of birth
    "22006-0.0",   # genetic ethnic grouping (in-sample white British == 1)
    "21000-0.0",   # self-reported ethnic background (instance 0)
    "26200-0.0",   # UKB PRS Release testing-subgroup flag (needed for the Enhanced-PRS leakage audit)
    "90001-0.0",   # accelerometer wear start (cohort cross-check)
    "90004-0.0",   # accelerometer data-quality field
    "90010-0.0",   # accelerometer field
]

# Townsend-style multiple-deprivation indices (England/Wales/Scotland), instance 0. Carried as a
# socioeconomic confound block, not polygenic scores. (Field IDs 26410-26434.)
DEPRIVATION_FIELDS = [26410, 26411, 26412, 26413, 26414, 26415, 26416, 26417, 26418, 26419,
                      26420, 26421, 26422, 26423, 26424, 26425, 26426, 26427, 26428, 26429,
                      26430, 26431, 26432, 26433, 26434]


def resolve_columns(header: list[str]) -> tuple[list[str], list[str], list[str], list[str]]:
    """From the raw header, return (keep_cols, prs_cols, pc_cols, dep_cols) using exact names.

    PRS = the 87 real trait scores only (field id in [26202, 26290]). The
    26410-26434 deprivation block and 26200/26201/26300+ metadata are not PRS and
    are handled separately (deprivation carried as a confound block)."""
    present = set(header)

    # PRS trait fields: 5-digit field id in [26202, 26290] (Standard + Enhanced "PRS for X").
    prs_cols = []
    for c in header:
        m = re.match(r"^(\d+)-\d+\.\d+$", c)
        if not m:
            continue
        fid = m.group(1)
        if len(fid) == 5 and 26202 <= int(fid) <= 26290:
            prs_cols.append(c)

    # 40 genetic PCs 22009-0.1 .. 22009-0.40
    pc_cols = [c for c in header if re.match(r"^22009-0\.\d+$", c)]
    pc_cols.sort(key=lambda c: int(c.split(".")[-1]))

    # deprivation indices, instance 0
    dep_cols = [f"{f}-0.0" for f in DEPRIVATION_FIELDS if f"{f}-0.0" in present]

    scalars = [c for c in SCALAR_COLS if c in present]
    missing_scalars = [c for c in SCALAR_COLS if c not in present]
    if missing_scalars:
        print(f"[extract] WARNING: scalar cols absent from header: {missing_scalars}")

    keep = ["eid"] + scalars + pc_cols + dep_cols + prs_cols
    # de-dup preserving order
    seen, keep_u = set(), []
    for c in keep:
        if c not in seen:
            seen.add(c)
            keep_u.append(c)
    return keep_u, prs_cols, pc_cols, dep_cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-ukb", default=FULL_UKB)
    ap.add_argument("--acc-dem", default=ACC_DEM)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke-nrows", type=int, default=0,
                    help="if >0, read only this many data rows (container smoke test, no cohort match expected)")
    ap.add_argument("--chunksize", type=int, default=50000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    # cohort EID set
    cohort = pd.read_csv(args.acc_dem, usecols=["eid"])
    cohort_eids = set(int(x) for x in cohort["eid"].tolist())
    print(f"[extract] cohort EIDs from acc_dem: {len(cohort_eids)}")

    # header to column selection
    with __import__("gzip").open(args.full_ukb, "rt") as fh:
        header = next(csv.reader(fh))
    keep, prs_cols, pc_cols, dep_cols = resolve_columns(header)
    print(f"[extract] keeping {len(keep)} cols  (PRS={len(prs_cols)}, PCs={len(pc_cols)}, "
          f"deprivation={len(dep_cols)}, "
          f"scalars={len(keep) - 1 - len(prs_cols) - len(pc_cols) - len(dep_cols)})")

    # stream + filter
    reader = pd.read_csv(args.full_ukb, usecols=keep, dtype={"eid": "Int64"},
                         chunksize=args.chunksize,
                         nrows=(args.smoke_nrows or None), low_memory=False)
    parts, n_seen, n_kept = [], 0, 0
    for i, chunk in enumerate(reader):
        n_seen += len(chunk)
        if args.smoke_nrows:
            sub = chunk  # smoke: keep all sampled rows to validate column logic
        else:
            sub = chunk[chunk["eid"].isin(cohort_eids)]
        if len(sub):
            parts.append(sub)
            n_kept += len(sub)
        if (i + 1) % 20 == 0:
            print(f"[extract] {n_seen} rows scanned, {n_kept} kept  ({time.time()-t0:.0f}s)")

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=keep)
    # reorder columns: ensure eid first
    out = out[[c for c in keep if c in out.columns]]
    out.to_parquet(f"{args.out}/prs_cohort.parquet", index=False)
    out.to_csv(f"{args.out}/prs_cohort.csv", index=False)

    matched = int(out["eid"].isin(cohort_eids).sum()) if not args.smoke_nrows else None
    meta = {
        "full_ukb": args.full_ukb,
        "acc_dem": args.acc_dem,
        "cohort_eids": len(cohort_eids),
        "rows_scanned": n_seen,
        "rows_kept": int(len(out)),
        "rows_matched_cohort": matched,
        "n_cols_kept": len(keep),
        "n_prs_cols": len(prs_cols),
        "n_pc_cols": len(pc_cols),
        "n_dep_cols": len(dep_cols),
        "prs_cols": prs_cols,
        "pc_cols": pc_cols,
        "dep_cols": dep_cols,
        "scalar_cols": [c for c in keep if c not in prs_cols and c not in pc_cols
                        and c not in dep_cols and c != "eid"],
        "smoke_nrows": args.smoke_nrows,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    with open(f"{args.out}/extract_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[extract] DONE rows_kept={len(out)} (expect ~103627 for full run) "
          f"cols={len(keep)} in {meta['elapsed_sec']}s")
    print(f"[extract] wrote {args.out}/prs_cohort.parquet + .csv + extract_meta.json")
    if not args.smoke_nrows and len(out) != len(cohort_eids):
        print(f"[extract] NOTE: rows_kept {len(out)} != cohort {len(cohort_eids)} "
              f"(some cohort EIDs may be absent from the genetics file; audit explains)")


if __name__ == "__main__":
    main()
