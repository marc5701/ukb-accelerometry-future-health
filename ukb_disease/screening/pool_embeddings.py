"""Precompute subject-level pooled embeddings for the screener.

Reads the per-day L-moment cache (per-recording tensors of shape (n_rec, 8, 768)
for the two accelerometer encoders, plus an (n_rec, 8) valid-day mask). For each
recording we mean-pool over its valid days; for each subject we mean over its
recordings. Output is one 1536-d vector per subject
([L1||L2||L3]_enc1 || [L1||L2||L3]_enc2), saved as an .npz. This only re-pools
tensors that already exist for the evaluable subjects (train+val+test of the
split).
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

from ukb_disease.baseline.config import CANONICAL_DAY_LMOMENT_ROOT, resolve_oak_path
from ukb_disease.screening.screening_split import assign_buckets

CHUNK = 8192


def _pool_recordings(root: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (eids, rec_vecs) for one split: rec_vecs (n_rec, 1536) mean over valid days."""
    sdir = resolve_oak_path(f"{root}/{split}")
    idx = pd.read_parquet(f"{sdir}/index.parquet")
    ar = np.load(f"{sdir}/ar.npy", mmap_mode="r")
    ha = np.load(f"{sdir}/ha.npy", mmap_mode="r")
    vm = np.load(f"{sdir}/valid_day_mask.npy", mmap_mode="r")
    n_rec, n_days, feat = ar.shape
    out = np.zeros((n_rec, 2 * feat), dtype=np.float32)
    for s in range(0, n_rec, CHUNK):
        e = min(s + CHUNK, n_rec)
        m = vm[s:e].astype(np.float32)            # (c, days)
        cnt = m.sum(1, keepdims=True)             # (c, 1)
        cnt = np.where(cnt < 1.0, 1.0, cnt)
        a = (np.asarray(ar[s:e]) * m[:, :, None]).sum(1) / cnt   # (c, feat)
        h = (np.asarray(ha[s:e]) * m[:, :, None]).sum(1) / cnt
        out[s:e] = np.concatenate([a, h], axis=1)
    return idx["eid"].to_numpy(dtype=np.int64), out


def pool_subjects(root: str = CANONICAL_DAY_LMOMENT_ROOT) -> tuple[np.ndarray, np.ndarray]:
    """Pool all splits to one (n_subj, 1536) array; subject vec = mean over recordings."""
    all_eids, all_vecs = [], []
    for split in ("train", "val", "test"):
        t0 = time.time()
        eids, vecs = _pool_recordings(root, split)
        all_eids.append(eids)
        all_vecs.append(vecs)
        print(f"[pool] {split}: {len(eids)} recordings pooled in {time.time()-t0:.1f}s", flush=True)
    eids = np.concatenate(all_eids)
    vecs = np.concatenate(all_vecs, axis=0)
    # subject = mean over its recordings
    df = pd.DataFrame(vecs)
    df["eid"] = eids
    g = df.groupby("eid").mean()
    subj_eids = g.index.to_numpy(dtype=np.int64)
    subj_vecs = g.to_numpy(dtype=np.float32)
    print(f"[pool] {len(subj_eids)} subjects from {len(eids)} recordings; dim={subj_vecs.shape[1]}", flush=True)
    return subj_eids, subj_vecs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=CANONICAL_DAY_LMOMENT_ROOT,
                    help="per-day L-moment cache root")
    ap.add_argument("--out", required=True,
                    help="output .npz path (must be writable)")
    args = ap.parse_args()
    eids, vecs = pool_subjects(args.root)
    bucket = assign_buckets(eids)
    out = resolve_oak_path(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, eids=eids, X=vecs, bucket=bucket.astype("U8"))
    print(f"[pool] wrote {out}: X={vecs.shape} n_subjects={len(eids)}", flush=True)


if __name__ == "__main__":
    main()
