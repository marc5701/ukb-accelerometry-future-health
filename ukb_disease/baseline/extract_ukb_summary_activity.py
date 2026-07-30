#!/usr/bin/env python3
"""Extract the UK Biobank *distributed* accelerometer summary fields into a compact
per-subject parquet.

The official UKB accelerometer summary metrics (Doherty pipeline: field 90012 overall
acceleration average, 90013 SD, the day-of-week / hour-of-day acceleration averages
[Category "Acceleration averages"], and the acceleration-intensity distribution
"fraction <= X mg" [Category "Acceleration intensity distribution"]) live ONLY in
`ukb_main_dataset.csv.gz`. They are absent from the derived demographics CSVs
(acc_dem.csv / BMI_yes.csv only carry age/sex/BMI/mortality + file-availability flags).

This pulls the whole contiguous accelerometer block (UKB field IDs 90010-90195) plus
`eid`, quoting-safe (the gz has embedded-comma ICD free-text fields earlier in the row,
so a byte-level `cut -d,` would corrupt column positions, so we use the pandas C parser
which respects the RFC-4180 double-quoting). Field-level include/exclude (activity vs
data-quality vs wear-timing) is applied DOWNSTREAM in `summary_baseline.py`, so we store
the full block once and never re-read this 4.3 GB file.

Usage:
  python3 -m ukb_disease.baseline.extract_ukb_summary_activity            # full run
  python3 -m ukb_disease.baseline.extract_ukb_summary_activity 2000       # smoke (first N rows)
"""
from __future__ import annotations

import os
import re
import sys
import time

import numpy as np
import pandas as pd

from ukb_disease.paths import UKB_ROOT

SRC = f"{UKB_ROOT}/metadata/ukb_main_dataset.csv.gz"
OUT_DIR = f"{UKB_ROOT}/caches/ukb_summary_activity"
OUT = f"{OUT_DIR}/ukb_summary_activity.parquet"
LO, HI = 90010, 90195  # accelerometer summary block (inclusive), by UKB field id


def field_id(col: str):
    m = re.match(r"^(\d+)-\d+\.\d+$", str(col))
    return int(m.group(1)) if m else None


def main() -> None:
    nrows = int(sys.argv[1]) if len(sys.argv) > 1 else None
    t0 = time.time()

    header = pd.read_csv(SRC, nrows=0)  # parses the quoted header only (fast)
    cols = list(header.columns)
    accel = [c for c in cols if (fid := field_id(c)) is not None and LO <= fid <= HI]
    keep = ["eid"] + accel
    print(f"[extract] header has {len(cols)} cols; keeping eid + {len(accel)} "
          f"accelerometer cols (field ids {LO}-{HI})", flush=True)
    print(f"[extract] first/last accel cols: {accel[0]} .. {accel[-1]}", flush=True)

    parts, seen = [], 0
    reader = pd.read_csv(SRC, usecols=keep, chunksize=20000, low_memory=False)
    for i, chunk in enumerate(reader):
        parts.append(chunk)
        seen += len(chunk)
        print(f"[extract] chunk {i:3d}  rows={seen:>7d}  ({time.time()-t0:6.0f}s)", flush=True)
        if nrows and seen >= nrows:
            break
    df = pd.concat(parts, ignore_index=True)

    # eid -> int; every accelerometer column -> numeric (wear-start datetime 90010 and
    # any stray strings coerce to NaN; those fields are excluded from the activity set
    # downstream anyway). Numeric-only keeps the parquet clean and small.
    df["eid"] = pd.to_numeric(df["eid"], errors="coerce").astype("Int64")
    for c in accel:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["eid"]).drop_duplicates(subset=["eid"], keep="first")

    cov = df[accel].notna().mean().round(3)
    print(f"[extract] N_subjects={len(df)}  cols={len(accel)}", flush=True)
    print(f"[extract] coverage 90012 (overall accel avg): {cov.get('90012-0.0', float('nan'))}", flush=True)
    print(f"[extract] coverage 90013 (SD):                {cov.get('90013-0.0', float('nan'))}", flush=True)
    print(f"[extract] min/median/max non-null frac across accel cols: "
          f"{cov.min():.3f} / {cov.median():.3f} / {cov.max():.3f}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    out = OUT if not nrows else OUT.replace(".parquet", f".smoke{nrows}.parquet")
    df.to_parquet(out, index=False)
    print(f"[extract] wrote {out}  ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
