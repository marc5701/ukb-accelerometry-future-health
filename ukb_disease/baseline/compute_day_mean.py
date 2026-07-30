"""Compute per-day pooled mean embeddings (AR + HA) per subject.

For each subject, takes the patch-level embeddings, applies the noon-aligned
daily binning logic from `compute_means.py`, but instead of collapsing across
days, KEEPS the per-day axis. Only days that have sufficient valid patches in
BOTH modalities (AR and HA) are retained, so the resulting per-day arrays are
aligned across modalities.

Output (one .npz per subject):

    <out_root>/<split>/<eid_run>.npz
        ar:        (n_days, 256) float32, per-day NaN-aware AR mean
        ha:        (n_days, 256) float32, per-day NaN-aware HA mean
        day_codes: (n_days,) int64, noon-aligned ns-since-epoch day code

`n_days` is the count of days valid for BOTH modalities (intersection).
Subjects with zero such days are skipped (no .npz written).

SLURM-array compatible via `--n-tasks N --task-id K` strided assignment.
Output is idempotent: subjects whose target .npz already exists are
skipped (resume-friendly).

Mirrors `compute_means.py` for structure, idempotency, and resume.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ukb_disease.baseline.config import (
    AR_PATCH_SECONDS,
    CACHE_ROOT_V2,
    DAY_MEAN_ROOT,
    HA_PATCH_SECONDS,
    resolve_oak_path,
)

MODEL_FILE = {
    "ar": "accelerest_embeddings.npy",
    "ha": "activity_mae_embeddings.npy",
}
PATCH_SECONDS = {"ar": AR_PATCH_SECONDS, "ha": HA_PATCH_SECONDS}


def list_subjects(cache_split_dir_resolved: str) -> list[tuple[str, str]]:
    if not os.path.isdir(cache_split_dir_resolved):
        raise FileNotFoundError(f"Cache split dir does not exist: {cache_split_dir_resolved}")
    entries = sorted(
        d for d in os.listdir(cache_split_dir_resolved)
        if os.path.isdir(os.path.join(cache_split_dir_resolved, d))
    )
    return [(d, os.path.join(cache_split_dir_resolved, d)) for d in entries]


def per_modality_day_mean(
    emb_path: str,
    start_time: pd.Timestamp,
    patch_sec: float,
    min_day_valid_patches: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute per-day mean for one modality.

    Returns (day_codes, per_day_mean) or None if nothing usable.
        day_codes:     (n_valid_days,) int64, noon-aligned ns-since-epoch
        per_day_mean: (n_valid_days, 256) float32
    """
    emb = np.load(emb_path)
    valid = ~np.isnan(emb[:, 0])
    if not valid.any():
        return None

    offsets = pd.to_timedelta(np.arange(emb.shape[0]) * patch_sec, unit="s")
    ts = start_time + offsets
    day_floor = (ts - pd.Timedelta(hours=12)).floor("D")
    day_codes_all = day_floor.astype("int64").to_numpy()
    unique_days, day_idx = np.unique(day_codes_all, return_inverse=True)

    kept_codes: list[int] = []
    kept_means: list[np.ndarray] = []
    for d in range(len(unique_days)):
        mask = (day_idx == d) & valid
        n_valid = int(mask.sum())
        if n_valid < min_day_valid_patches:
            continue
        kept_codes.append(int(unique_days[d]))
        kept_means.append(emb[mask].astype(np.float64).mean(axis=0))

    if not kept_codes:
        return None
    return (
        np.asarray(kept_codes, dtype=np.int64),
        np.stack(kept_means).astype(np.float32),
    )


def per_subject_day_mean(
    subj_dir: str,
    min_day_valid_patches: int,
    min_subject_valid_frac: float,
) -> dict | None:
    """Compute per-day means for both modalities and align by day code.

    Returns:
        {"ar": (n,256), "ha": (n,256), "day_codes": (n,)} or None
    A day is kept only if it has sufficient valid patches in BOTH AR and HA.
    """
    with open(os.path.join(subj_dir, "meta.json")) as f:
        meta = json.load(f)
    start_time = pd.to_datetime(meta["start_time"])

    out: dict[str, np.ndarray] = {}
    for model in ("ar", "ha"):
        emb_path = os.path.join(subj_dir, MODEL_FILE[model])
        if not os.path.exists(emb_path):
            return None

        # Coarse subject-level validity short-circuit (matches compute_means.py).
        # Use mmap so the .npy bytes aren't pulled into RAM just for the
        # one-column NaN check.
        try:
            emb_mm = np.load(emb_path, mmap_mode="r")
            valid_frac = float((~np.isnan(emb_mm[:, 0])).mean()) if emb_mm.size else 0.0
            del emb_mm
        except Exception:
            return None
        if valid_frac < min_subject_valid_frac:
            return None

        res = per_modality_day_mean(
            emb_path,
            start_time=start_time,
            patch_sec=PATCH_SECONDS[model],
            min_day_valid_patches=min_day_valid_patches,
        )
        if res is None:
            return None
        codes, means = res
        out[f"{model}_codes"] = codes
        out[f"{model}_means"] = means

    # Intersect day codes across modalities
    common, ar_idx, ha_idx = np.intersect1d(
        out["ar_codes"], out["ha_codes"], return_indices=True
    )
    if common.size == 0:
        return None
    return {
        "day_codes": common,
        "ar": out["ar_means"][ar_idx].astype(np.float32),
        "ha": out["ha_means"][ha_idx].astype(np.float32),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-root", default=CACHE_ROOT_V2)
    ap.add_argument("--out-root", default=DAY_MEAN_ROOT)
    ap.add_argument("--split", required=True,
                    choices=["audit", "test", "train", "val", "neuro"])
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--min-day-valid-patches", type=int, default=100,
                    help="Skip days with fewer valid patches in EITHER modality")
    ap.add_argument("--min-subject-valid-frac", type=float, default=0.05,
                    help="Skip subjects below this valid-patch fraction")
    ap.add_argument("--max-subjects", type=int, default=None, help="Cap for smoke testing")
    ap.add_argument("--no-resume", action="store_true",
                    help="Recompute even if output already exists")
    args = ap.parse_args()

    cache_split_resolved = resolve_oak_path(os.path.join(args.cache_root, args.split))
    out_split_resolved = resolve_oak_path(os.path.join(args.out_root, args.split))
    os.makedirs(out_split_resolved, exist_ok=True)
    print(f"[compute_day_mean] cache: {cache_split_resolved}")
    print(f"[compute_day_mean] out:   {out_split_resolved}")

    subjects = list_subjects(cache_split_resolved)
    print(f"[compute_day_mean] total subjects in split: {len(subjects)}")

    if args.n_tasks > 1:
        subjects = [s for i, s in enumerate(subjects) if i % args.n_tasks == args.task_id]
        print(f"[compute_day_mean] task {args.task_id}/{args.n_tasks}: "
              f"my slice = {len(subjects)} subjects")
    if args.max_subjects:
        subjects = subjects[: args.max_subjects]
        print(f"[compute_day_mean] capped to {len(subjects)} subjects (--max-subjects)")

    n_done = n_skipped_existing = n_skipped_lowvalid = n_errors = 0
    t0 = time.time()
    for i, (eid_run, subj_dir) in enumerate(subjects):
        out_path = os.path.join(out_split_resolved, f"{eid_run}.npz")
        if not args.no_resume and os.path.exists(out_path):
            n_skipped_existing += 1
            continue
        try:
            res = per_subject_day_mean(
                subj_dir,
                min_day_valid_patches=args.min_day_valid_patches,
                min_subject_valid_frac=args.min_subject_valid_frac,
            )
            if res is None:
                n_skipped_lowvalid += 1
                continue
            # Atomic write: tmp + rename. np.savez auto-appends ".npz" to
            # path-strings, so pass a file handle to keep the exact name.
            tmp_path = out_path + ".tmp"
            with open(tmp_path, "wb") as fh:
                np.savez(fh, **res)
            os.rename(tmp_path, out_path)
            n_done += 1
        except Exception as e:
            n_errors += 1
            print(f"[compute_day_mean] ERROR on {eid_run}: {type(e).__name__}: {e}")
        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(subjects) - (i + 1)) / rate if rate else 0
            print(f"[compute_day_mean] progress {i+1}/{len(subjects)}  "
                  f"done={n_done} skip_exist={n_skipped_existing} "
                  f"skip_lowvalid={n_skipped_lowvalid} err={n_errors}  "
                  f"rate={rate:.2f}/s  eta={remaining/60:.1f}min")

    elapsed = time.time() - t0
    print(f"[compute_day_mean] DONE in {elapsed/60:.1f}min: "
          f"done={n_done} skip_exist={n_skipped_existing} "
          f"skip_lowvalid={n_skipped_lowvalid} err={n_errors}")


if __name__ == "__main__":
    main()
