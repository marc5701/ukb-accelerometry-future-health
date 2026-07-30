"""Build a manifest parquet for Embedding extraction.

A manifest is a pandas DataFrame with columns:

    eid_run         str   '<eid>_<run>' stem
    source_h5       str   local path to the preprocessed H5 (UKB_PP_DIR/<stem>.h5)
    split_path      str   original split-list path for the recording (for traceability)
    split           str   'audit' | 'test' | 'train'
    cache_h5        str   per-recording cache target (CACHE_ROOT/<split>/<stem>.h5)
    calib_failed    bool  always False after filtering (excluded_h5 entries dropped)

Calibration-failed recordings never get a manifest row, so downstream code can
trust that every source_h5 is in the preprocessed_h5/ pool.

Audit split sampling is deterministic via numpy.random.default_rng(SEED), so
re-running yields the same subjects.

Usage::

    python3 -m ukb_disease.extraction.build_manifest --split test  --out manifests/manifest_test.parquet
    python3 -m ukb_disease.extraction.build_manifest --split train --out manifests/manifest_train.parquet
    python3 -m ukb_disease.extraction.build_manifest --split audit --n 150 --out manifests/manifest_audit.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ukb_disease.extraction import config as C


def _read_split_paths(split: str) -> list[str]:
    """Return the raw H5 paths listed for the named split."""
    if split == "train":
        with open(C.resolve_oak_path(C.LEGACY_TRAIN_JSON)) as f:
            return json.load(f)
    if split in ("test", "audit"):
        with open(C.resolve_oak_path(C.LEGACY_TEST_JSON)) as f:
            return json.load(f)
    raise ValueError(f"Unknown split: {split!r}")


def build_manifest(split: str, n: int | None = None, seed: int = C.SEED) -> pd.DataFrame:
    """Build a manifest for the given split.

    Args:
        split: 'train' | 'test' | 'audit'. Audit samples `n` from the test split.
        n: For 'audit' split only: number of subjects to sample.
        seed: Random seed for audit sampling (deterministic).
    """
    raw_paths = _read_split_paths(split)
    rows: list[dict] = []
    drop_excluded = 0
    drop_missing = 0
    for split_path in raw_paths:
        local_path = C.map_split_path_to_local(split_path)
        eid_run = C.eid_run_from_h5(split_path)
        if C.is_calib_failed(local_path):
            drop_excluded += 1
            continue
        if not C.path_exists_resolved(local_path):
            drop_missing += 1
            continue
        rows.append({
            "eid_run": eid_run,
            "source_h5": local_path,
            "split_path": split_path,
            "split": split if split != "audit" else "audit",
            "cache_h5": C.cache_path(split if split != "audit" else "audit", eid_run),
            "calib_failed": False,
        })

    df = pd.DataFrame(rows)
    print(
        f"[build_manifest] split={split}  total={len(raw_paths)}  "
        f"kept={len(df)}  dropped_calib_failed={drop_excluded}  "
        f"dropped_missing={drop_missing}"
    )

    if split == "audit":
        if n is None:
            n = 150
        if len(df) < n:
            raise RuntimeError(
                f"Asked for {n} audit subjects but only {len(df)} are available."
            )
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(df), size=n, replace=False)
        df = df.iloc[idx].sort_values("eid_run").reset_index(drop=True)
        print(f"[build_manifest] audit subsample: n={len(df)} seed={seed}")

    # Sanity: every source_h5 must exist (resolved through bidirectional path map)
    missing = df.loc[~df["source_h5"].apply(C.path_exists_resolved), "source_h5"].tolist()
    if missing:
        raise RuntimeError(
            f"{len(missing)} source_h5 paths do not exist after filtering. "
            f"First 3: {missing[:3]}"
        )

    return df


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--split", required=True, choices=["train", "test", "audit"])
    p.add_argument("--n", type=int, default=None,
                   help="For --split audit: number of subjects to sample (default 150).")
    p.add_argument("--seed", type=int, default=C.SEED)
    p.add_argument("--out", required=True, help="Output parquet path.")
    args = p.parse_args()

    df = build_manifest(args.split, n=args.n, seed=args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[build_manifest] wrote {len(df)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
