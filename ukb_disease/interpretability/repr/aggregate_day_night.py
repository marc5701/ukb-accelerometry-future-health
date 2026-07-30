"""Index-driven aggregation of day/night per-subject .npz into padded split tensors.

Unlike ``baseline/aggregate_day_mean.py`` (which lists and sorts whatever .npz
are present), this stacks rows in the exact order of a reference cache's
``index.parquet`` (the L3 disjoint_split day-distribution cache). This guarantees
the output tensors are row-for-row aligned with the reference cache and with its
on-disk ``test_hazards.npy``, so Tier-1 occlusion and OOD comparisons need no
re-indexing, and the stratified model trains and scores on identical membership.

Source .npz are located across one or more source roots (``embeddings_v2`` day/
night pools plus an ``embeddings_extras`` day/night pool); the reference index
lists every eid_run we must find (confirmed complete, no missing rows).

Output per split (mirrors aggregate_day_mean and adds cell_counts):
    <out_root>/<split>/ar.npy             (n, max_days, n_bins*3*256) float32
    <out_root>/<split>/ha.npy             (n, max_days, n_bins*3*256) float32
    <out_root>/<split>/valid_day_mask.npy (n, max_days)               bool
    <out_root>/<split>/cell_counts.npy    (n, max_days, 2*n_bins)     int32
    <out_root>/<split>/index.parquet      [eid, eid_run, n_valid_days, first_day_code]

Padded positions are 0.0 (model masks by valid_day_mask). Asserts the per-subject
day count matches the reference's n_valid_days (same 100-patch intersection rule),
warns on any mismatch rather than silently misaligning windows.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from ukb_disease.baseline.config import resolve_oak_path

DEFAULT_THREADS = 32


def build_path_map(source_roots: list[str]) -> dict[str, str]:
    """eid_run -> first source path holding <eid_run>.npz (first root wins)."""
    pmap: dict[str, str] = {}
    for root in source_roots:
        root = resolve_oak_path(root)
        if not os.path.isdir(root):
            print(f"[agg-dn] WARN source root missing: {root}")
            continue
        for f in os.listdir(root):
            if f.endswith(".npz"):
                er = f[:-4]
                pmap.setdefault(er, os.path.join(root, f))
    return pmap


def _read_one(args):
    er, path = args
    with np.load(path) as z:
        return er, z["ar"], z["ha"], z["day_codes"], z["cell_counts"]


def aggregate_split(
    split: str,
    ref_index_path: str,
    pmap: dict[str, str],
    out_split_dir: str,
    n_threads: int = DEFAULT_THREADS,
) -> None:
    ref = pd.read_parquet(resolve_oak_path(ref_index_path))
    eid_runs = ref["eid_run"].tolist()                     # ground-truth order
    ref_ndays = dict(zip(ref["eid_run"], ref["n_valid_days"]))
    n = len(eid_runs)
    missing = [er for er in eid_runs if er not in pmap]
    if missing:
        raise RuntimeError(
            f"[agg-dn] {split}: {len(missing)} eid_runs absent from sources "
            f"(would break alignment) e.g. {missing[:5]}")
    print(f"[agg-dn] {split}: {n} rows (threads={n_threads})")

    ar_d: dict[str, np.ndarray] = {}
    ha_d: dict[str, np.ndarray] = {}
    cc_d: dict[str, np.ndarray] = {}
    code_d: dict[str, np.ndarray] = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        for i, (er, ar, ha, dc, cc) in enumerate(
            ex.map(_read_one, [(er, pmap[er]) for er in eid_runs])
        ):
            ar_d[er] = ar.astype(np.float32, copy=False)
            ha_d[er] = ha.astype(np.float32, copy=False)
            cc_d[er] = cc.astype(np.int32, copy=False)
            code_d[er] = dc
            if (i + 1) % 10000 == 0:
                rate = (i + 1) / max(time.time() - t0, 1e-3)
                print(f"[agg-dn] read {i+1}/{n} rate={rate:.0f}/s "
                      f"eta={(n-(i+1))/max(rate,1e-3)/60:.1f}min")

    n_days = np.asarray([ar_d[er].shape[0] for er in eid_runs], dtype=np.int32)
    max_days = int(n_days.max())
    ar_dim = int(next(iter(ar_d.values())).shape[1])
    ha_dim = int(next(iter(ha_d.values())).shape[1])
    cc_dim = int(next(iter(cc_d.values())).shape[1])
    print(f"[agg-dn] {split}: max_days={max_days} ar_dim={ar_dim} ha_dim={ha_dim} cc_dim={cc_dim}")

    # Day-count parity check against the reference (same inclusion rule expected).
    n_mismatch = sum(1 for er in eid_runs if int(ref_ndays[er]) != int(ar_d[er].shape[0]))
    if n_mismatch:
        print(f"[agg-dn] {split}: WARN {n_mismatch}/{n} rows differ in n_valid_days "
              f"vs reference (day-inclusion drift, investigate before trusting "
              f"window-level alignment)")

    ar_block = np.zeros((n, max_days, ar_dim), dtype=np.float32)
    ha_block = np.zeros((n, max_days, ha_dim), dtype=np.float32)
    cc_block = np.zeros((n, max_days, cc_dim), dtype=np.int32)
    valid = np.zeros((n, max_days), dtype=bool)
    eids = np.zeros(n, dtype=np.int64)
    first_code = np.zeros(n, dtype=np.int64)
    for i, er in enumerate(eid_runs):
        nd = int(n_days[i])
        ar_block[i, :nd] = ar_d[er]
        ha_block[i, :nd] = ha_d[er]
        cc_block[i, :nd] = cc_d[er]
        valid[i, :nd] = True
        eids[i] = int(er.split("_")[0])
        first_code[i] = int(code_d[er][0]) if nd > 0 else 0

    del ar_d, ha_d, cc_d, code_d
    import gc; gc.collect()

    out_split_dir = resolve_oak_path(out_split_dir)
    os.makedirs(out_split_dir, exist_ok=True)

    def _save(path, arr):
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            np.save(fh, arr)
        os.rename(tmp, path)

    _save(os.path.join(out_split_dir, "ar.npy"), ar_block)
    _save(os.path.join(out_split_dir, "ha.npy"), ha_block)
    _save(os.path.join(out_split_dir, "valid_day_mask.npy"), valid)
    _save(os.path.join(out_split_dir, "cell_counts.npy"), cc_block)
    pd.DataFrame({
        "eid": eids, "eid_run": eid_runs,
        "n_valid_days": n_days, "first_day_code": first_code,
    }).to_parquet(os.path.join(out_split_dir, "index.parquet"), index=False)
    print(f"[agg-dn] {split}: wrote ar={ar_block.shape} valid_frac={valid.mean():.3f} "
          f"-> {out_split_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-cache", required=True,
                    help="Reference cache with per-split index.parquet (membership and order).")
    ap.add_argument("--source-roots", nargs="+", required=True,
                    help="Dirs holding day/night per-subject <eid_run>.npz.")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    args = ap.parse_args()

    pmap = build_path_map(args.source_roots)
    print(f"[agg-dn] path map: {len(pmap)} eid_runs across {len(args.source_roots)} roots")
    for split in args.splits:
        ref_idx = os.path.join(args.reference_cache, split, "index.parquet")
        aggregate_split(split, ref_idx, pmap,
                        os.path.join(args.out_root, split), n_threads=args.threads)


if __name__ == "__main__":
    main()
