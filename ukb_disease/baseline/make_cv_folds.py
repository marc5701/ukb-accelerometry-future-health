"""Freeze the split artifacts: 5-fold CV assignment and locked test set.

Two immutable JSON manifests, written once:

1. ``cv_folds_disjoint_split.json``: every disjoint_split TRAIN-pool subject (eid in
   acc_dem) assigned to one of K folds by ``md5(int eid) % K``. Subject-keyed,
   so all recordings of a subject land in the same fold (leak-free CV). This is
   the ranking/model-selection pool.

2. ``LOCKED_TEST_EIDS.json``: the unique disjoint_split TEST subjects (in
   acc_dem), evaluated once at the end.

Determinism: salt-free md5 on the integer eid, identical across machines and
runs (mirrors ``_split_hash.hash_unit``). Verifies train/test subject
disjointness and that the folds partition the pool exactly. Subject identifiers
are de-identified to positional row indices before being written to disk.

Run (CPU):
    python3 -m ukb_disease.baseline.make_cv_folds
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ukb_disease.baseline.config import CANONICAL_DAY_MEAN_ROOT, resolve_oak_path
from ukb_disease.baseline.outcomes_processing import load_covariates

# Default home for the manifests (repo subtree).
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "splits" / "disjoint_split_v1"
FROZEN_DATE = "2026-06-18"


def fold_of_eid(eid: int, n_folds: int) -> int:
    """Deterministic fold in [0, n_folds) from the integer eid (md5, salt-free)."""
    if isinstance(eid, str) and "_" in eid:
        raise ValueError(f"expects a bare integer eid, got stem {eid!r}")
    h = int(hashlib.md5(str(int(eid)).encode()).hexdigest(), 16)
    return int(h % n_folds)


def _pool_eids(split: str, day_mean_root: str) -> np.ndarray:
    """Unique subject eids present in <split>/index.parquet AND in acc_dem."""
    idx_path = resolve_oak_path(os.path.join(day_mean_root, split, "index.parquet"))
    idx = pd.read_parquet(idx_path)
    eids = np.unique(idx["eid"].to_numpy(dtype=np.int64))
    cov = load_covariates()
    eids = eids[np.isin(eids, cov.index.to_numpy())]
    return np.sort(eids)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--day-means-root", default=CANONICAL_DAY_MEAN_ROOT)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--overwrite", action="store_true",
                    help="Required to overwrite an existing manifest (guards the freeze).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv_path = out_dir / "cv_folds_disjoint_split.json"
    test_path = out_dir / "LOCKED_TEST_EIDS.json"

    for p in (cv_path, test_path):
        if p.exists() and not args.overwrite:
            raise SystemExit(
                f"[make_cv_folds] {p} already exists; refusing to regenerate a frozen "
                f"manifest without --overwrite (protects against drift)."
            )

    # --- TRAIN pool -> folds
    train_eids = _pool_eids("train", args.day_mean_root)
    folds = [fold_of_eid(int(e), args.n_folds) for e in train_eids]
    folds = np.asarray(folds, dtype=np.int64)
    # De-identify: write positional row indices into train_eids, not raw eids.
    train_index = np.arange(len(train_eids), dtype=np.int64)
    fold_lists = [sorted(int(i) for i in train_index[folds == k]) for k in range(args.n_folds)]
    counts = [len(f) for f in fold_lists]
    # Partition sanity
    assert sum(counts) == len(train_eids), "fold counts do not sum to pool size"
    flat = np.concatenate([np.asarray(f) for f in fold_lists])
    assert len(np.unique(flat)) == len(train_eids), "fold partition not disjoint/complete"

    # --- TEST locked set
    test_eids = _pool_eids("test", args.day_mean_root)

    # --- Disjointness train pool vs test
    overlap = np.intersect1d(train_eids, test_eids)
    assert overlap.size == 0, f"LEAKAGE: {overlap.size} eids in BOTH train pool and test"

    cv_obj = {
        "split": "disjoint_split_v1",
        "pool": "train index and acc_dem (subject-level)",
        "n_folds": args.n_folds,
        "hash": "md5(int eid) % n_folds (salt-free, subject-keyed)",
        "n_subjects": int(len(train_eids)),
        "fold_counts": counts,
        "frozen_date": FROZEN_DATE,
        "id_type": "deidentified_row_index",
        "folds": fold_lists,
    }
    test_obj = {
        "split": "disjoint_split_v1 TEST, locked, evaluated once at the end",
        "definition": "unique subject in test/index.parquet and acc_dem (subject-level)",
        "n_subjects": int(len(test_eids)),
        "frozen_date": FROZEN_DATE,
        "id_type": "deidentified_row_index",
        "row_index": [int(i) for i in range(len(test_eids))],
    }

    with open(cv_path, "w") as f:
        json.dump(cv_obj, f, indent=2)
    with open(test_path, "w") as f:
        json.dump(test_obj, f, indent=2)

    print(f"[make_cv_folds] train pool subjects: {len(train_eids)}  fold counts: {counts}")
    print(f"[make_cv_folds] LOCKED test subjects: {len(test_eids)}")
    print(f"[make_cv_folds] train/test overlap: {overlap.size} (must be 0)")
    print(f"[make_cv_folds] wrote {cv_path}")
    print(f"[make_cv_folds] wrote {test_path}")


if __name__ == "__main__":
    main()
