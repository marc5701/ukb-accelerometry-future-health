"""Aggregate per-subject day-means .npz into one per-split padded tensor.

Reads `compute_day_mean.py` outputs at `<means_root>/<split>/<eid_run>.npz`
and packs them into padded tensors per split. All subjects are padded to
the maximum observed `n_days` across the split using a valid-day mask.

Output:
    <out_root>/<split>/ar.npy             (n_subjects, max_days, 256) float32
    <out_root>/<split>/ha.npy             (n_subjects, max_days, 256) float32
    <out_root>/<split>/valid_day_mask.npy (n_subjects, max_days)      bool
    <out_root>/<split>/index.parquet      cols [eid, eid_run, n_valid_days,
                                                first_day_code]

Padded positions in `ar`/`ha` are filled with zeros. Downstream code must
use `valid_day_mask` to ignore them. We pad with 0.0 (not NaN) because
the downstream model layers (LSTM, transformer, mean-pool with mask)
multiply by the bool mask, and NaN would propagate and poison every output.

Idempotent: rewrites outputs from scratch each invocation. Uses parallel
threaded reads (ThreadPoolExecutor) in a single pass for I/O efficiency on
NFS/Lustre.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ukb_disease.baseline.config import (
    AR_EMB_DIM,
    DAY_MEAN_ROOT,
    HA_EMB_DIM,
    resolve_oak_path,
)

EID_RUN_RE = re.compile(r"^(?P<eid>\d+)_(?P<run>\d+)\.npz$")
DEFAULT_THREADS = 32


def list_eid_runs(split_dir_resolved: str) -> list[str]:
    if not os.path.isdir(split_dir_resolved):
        return []
    out: list[str] = []
    for f in os.listdir(split_dir_resolved):
        m = EID_RUN_RE.match(f)
        if m:
            out.append(f[:-4])  # strip ".npz"
    return sorted(out)


def _read_one(args):
    """Worker: read one .npz, return (eid_run, ar, ha, day_codes)."""
    split_dir, er = args
    with np.load(os.path.join(split_dir, f"{er}.npz")) as z:
        return er, z["ar"], z["ha"], z["day_codes"]


def aggregate_split(split_dir_resolved: str, out_split_dir_resolved: str,
                    n_threads: int = DEFAULT_THREADS) -> None:
    eid_runs = list_eid_runs(split_dir_resolved)
    if not eid_runs:
        raise RuntimeError(f"No .npz found under {split_dir_resolved}")
    n = len(eid_runs)
    print(f"[aggregate_day_mean] {os.path.basename(split_dir_resolved)}: "
          f"{n} subjects to aggregate (threads={n_threads})")

    # Single-pass parallel read: keep all (ar, ha, day_codes) in memory then
    # pack at the end. Memory bound = ~1.6 GB for the full train pool.
    ar_per_subj: dict[str, np.ndarray] = {}
    ha_per_subj: dict[str, np.ndarray] = {}
    codes_per_subj: dict[str, np.ndarray] = {}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        for i, (er, ar, ha, dc) in enumerate(
            ex.map(_read_one, [(split_dir_resolved, er) for er in eid_runs])
        ):
            ar_per_subj[er] = ar.astype(np.float32, copy=False)
            ha_per_subj[er] = ha.astype(np.float32, copy=False)
            codes_per_subj[er] = dc
            if (i + 1) % 10000 == 0:
                rate = (i + 1) / max(time.time() - t0, 1e-3)
                eta = (n - (i + 1)) / max(rate, 1e-3)
                print(f"[aggregate_day_mean] read {i+1}/{n}  "
                      f"rate={rate:.1f}/s  eta={eta/60:.1f}min")
    print(f"[aggregate_day_mean] all reads done in {(time.time()-t0)/60:.1f}min")

    # Compute padding shape
    n_days_per_subj = np.asarray(
        [ar_per_subj[er].shape[0] for er in eid_runs], dtype=np.int32
    )
    max_days = int(n_days_per_subj.max())
    first_day_code = np.asarray(
        [int(codes_per_subj[er][0]) if n_days_per_subj[i] > 0 else 0
         for i, er in enumerate(eid_runs)],
        dtype=np.int64,
    )
    print(f"[aggregate_day_mean] max_days={max_days}  "
          f"min={int(n_days_per_subj.min())}  mean={float(n_days_per_subj.mean()):.2f}")

    # Build padded tensors. Infer the per-day feature dim from the data rather
    # than the AR_EMB_DIM/HA_EMB_DIM constants, so distributional caches
    # (concatenated [mean, std] = 512-d per modality) pack correctly. Resolves
    # to 256 for the canonical mean cache, staying backward-compatible.
    ar_dim = int(next(iter(ar_per_subj.values())).shape[1])
    ha_dim = int(next(iter(ha_per_subj.values())).shape[1])
    if ar_dim != AR_EMB_DIM or ha_dim != HA_EMB_DIM:
        print(f"[aggregate_day_mean] non-canonical feature dims: "
              f"ar_dim={ar_dim} ha_dim={ha_dim} (canonical {AR_EMB_DIM}/{HA_EMB_DIM})")
    ar_block = np.zeros((n, max_days, ar_dim), dtype=np.float32)
    ha_block = np.zeros((n, max_days, ha_dim), dtype=np.float32)
    valid_mask = np.zeros((n, max_days), dtype=bool)
    eids = np.zeros(n, dtype=np.int64)
    for i, er in enumerate(eid_runs):
        eids[i] = int(er.split("_")[0])
        nd = n_days_per_subj[i]
        ar_block[i, :nd] = ar_per_subj[er]
        ha_block[i, :nd] = ha_per_subj[er]
        valid_mask[i, :nd] = True

    # Free the per-subject dicts before writing
    del ar_per_subj, ha_per_subj, codes_per_subj
    import gc; gc.collect()

    os.makedirs(out_split_dir_resolved, exist_ok=True)

    # Atomic-style write via tmp + rename. np.save auto-appends ".npy" to
    # path-strings; pass a file handle to keep the exact name.
    def _save_atomic(path: str, arr: np.ndarray) -> None:
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            np.save(fh, arr)
        os.rename(tmp, path)

    _save_atomic(os.path.join(out_split_dir_resolved, "ar.npy"), ar_block)
    _save_atomic(os.path.join(out_split_dir_resolved, "ha.npy"), ha_block)
    _save_atomic(os.path.join(out_split_dir_resolved, "valid_day_mask.npy"), valid_mask)

    idx = pd.DataFrame({
        "eid": eids,
        "eid_run": eid_runs,
        "n_valid_days": n_days_per_subj,
        "first_day_code": first_day_code,
    })
    idx.to_parquet(os.path.join(out_split_dir_resolved, "index.parquet"), index=False)

    print(f"[aggregate_day_mean] wrote {out_split_dir_resolved}: "
          f"ar={ar_block.shape} ha={ha_block.shape} "
          f"valid_frac={float(valid_mask.mean()):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--means-root", default=DAY_MEAN_ROOT,
                    help="Root where compute_day_mean wrote per-subject .npz")
    ap.add_argument("--out-root", default=None,
                    help="If unset, writes into --means-root/<split>/")
    ap.add_argument("--splits", nargs="+", default=["audit", "test", "train"],
                    choices=["audit", "test", "train", "val", "neuro"])
    ap.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                    help=f"ThreadPool workers for parallel .npz reads (default {DEFAULT_THREADS})")
    args = ap.parse_args()

    out_root = args.out_root if args.out_root else args.means_root
    for split in args.splits:
        split_dir = resolve_oak_path(os.path.join(args.means_root, split))
        out_split_dir = resolve_oak_path(os.path.join(out_root, split))
        if not os.path.isdir(split_dir):
            print(f"[aggregate_day_mean] SKIP {split}: input dir does not exist ({split_dir})")
            continue
        aggregate_split(split_dir, out_split_dir, n_threads=args.threads)


if __name__ == "__main__":
    main()
