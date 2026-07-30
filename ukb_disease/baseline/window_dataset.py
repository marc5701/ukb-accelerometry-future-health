"""Rolling-window dataset: rolling N-day windows over per-day mean embeddings.

Reads the aggregated per-split padded tensors produced by
`aggregate_day_mean.py` and yields N-day windows for the multilabel
CoxPH trainer.

Modes:
    Training (default `__getitem__`):
        One random valid N-day window per subject per epoch (data
        augmentation). Subjects with fewer than N consecutive valid
        days are dropped entirely (caller is expected to use the
        `.eids` attribute to know who survived).

    Inference (`forward_subject_aggregated`):
        Forward ALL valid N-day windows per subject through a model,
        average the per-window log-hazards into one subject log-hazard
        vector. This is the "average at inference" protocol.

Yields per `__getitem__`:
    (window, valid_mask, t, e, covariates), the same tuple shape as
    `SurvivalMeansDataset` but with an extra valid_mask informational tensor
    (in default mode always all-True).

The dataset DOES NOT pad windows: every yielded window is exactly N
consecutive valid days. The `valid_mask` is therefore always all-True
in the default sampling path; we expose it for compatibility with
models that natively support partial masks.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ukb_disease.baseline.config import (
    AR_EMB_DIM,
    DAY_MEAN_ROOT,
    HA_EMB_DIM,
    resolve_oak_path,
)

WINDOW_IN_DIM = AR_EMB_DIM + HA_EMB_DIM  # 512, un-stratified concat of AR and HA
WINDOW_IN_DIM_STRAT = 2 * (AR_EMB_DIM + HA_EMB_DIM)  # 1024, concat of AR_sleep, AR_wake, HA_sleep, HA_wake


def _valid_window_starts(valid_mask: np.ndarray, n: int) -> np.ndarray:
    """Indices `i` such that valid_mask[i:i+n].all() == True.

    Args:
        valid_mask: (max_days,) bool
        n:          window size
    Returns:
        (n_windows,) int32 array of start positions; empty if none.
    """
    d = len(valid_mask)
    if d < n:
        return np.zeros(0, dtype=np.int32)
    # Vectorized: sliding sum over n positions; valid when sum == n.
    csum = np.concatenate(([0], np.cumsum(valid_mask.astype(np.int32))))
    sums = csum[n:] - csum[:-n]  # (d - n + 1,)
    return np.where(sums == n)[0].astype(np.int32)


class WindowDataset(Dataset):
    """One random valid N-day window per subject per __getitem__.

    Args:
        split:        "audit" | "test" | "train"
        window_size:  N days per window (3, 5, or 7 expected)
        outcomes:     per-split dict from `build_survival_arrays`, keys
                      "event_times", "is_event", "eval_mask", "eids".
        covariates:   DataFrame indexed by eid, columns [age, sex].
        eid_filter:   optional iterable of eids to restrict to (intersected
                      with index, outcomes, and covariates).
        day_mean_root: override for the aggregated-tensor root.
        rng_seed:     base seed for per-subject window sampling.
        eager_load:   if True (default), load ar.npy/ha.npy into RAM
                      immediately. Train pool fits in ~1.6 GB.
    """

    def __init__(
        self,
        split: str,
        window_size: int,
        outcomes: dict,
        covariates: pd.DataFrame,
        eid_filter=None,
        day_mean_root: str = DAY_MEAN_ROOT,
        rng_seed: int = 0,
        eager_load: bool = True,
        stratified: bool = False,
        aux_age=None,
    ):
        super().__init__()
        self.split = split
        self.window_size = int(window_size)
        self.rng_seed = int(rng_seed)
        self.stratified = bool(stratified)
        self.in_dim = WINDOW_IN_DIM_STRAT if stratified else WINDOW_IN_DIM

        split_dir = resolve_oak_path(os.path.join(day_mean_root, split))
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(
                f"Day-means split dir does not exist: {split_dir}. "
                "Run precompute + aggregate first."
            )
        if stratified:
            ars_path = os.path.join(split_dir, "ar_sleep.npy")
            arw_path = os.path.join(split_dir, "ar_wake.npy")
            has_path = os.path.join(split_dir, "ha_sleep.npy")
            haw_path = os.path.join(split_dir, "ha_wake.npy")
        else:
            ar_path = os.path.join(split_dir, "ar.npy")
            ha_path = os.path.join(split_dir, "ha.npy")
        mask_path = os.path.join(split_dir, "valid_day_mask.npy")
        idx_path = os.path.join(split_dir, "index.parquet")

        # Load index first so we know how many subjects + what eid_runs.
        idx = pd.read_parquet(idx_path)
        self.eid_run_all = idx["eid_run"].to_numpy()
        self.eid_all = idx["eid"].to_numpy(dtype=np.int64)
        self.n_valid_days_all = idx["n_valid_days"].to_numpy(dtype=np.int32)

        loader = (lambda p: np.load(p).astype(np.float32, copy=False)) if eager_load else (lambda p: np.load(p, mmap_mode="r"))
        mask_all = (np.load(mask_path).astype(bool, copy=False)
                    if eager_load else np.load(mask_path, mmap_mode="r"))
        if stratified:
            ars_all = loader(ars_path)
            arw_all = loader(arw_path)
            has_all = loader(has_path)
            haw_all = loader(haw_path)
            ar_all = None  # legacy slots unused in stratified mode
            ha_all = None
        else:
            ar_all = loader(ar_path)
            ha_all = loader(ha_path)
            ars_all = arw_all = has_all = haw_all = None

        # Infer the true per-day feature dim from the loaded blocks rather than
        # trusting the WINDOW_IN_DIM constant. This lets distributional caches
        # (where ar/ha are [mean, std] = 512-d each, giving in_dim 1024) flow
        # through unchanged. For the canonical 256-d mean cache this resolves to
        # 512 (un-stratified) or 1024 (stratified), identical to before.
        if stratified:
            self.in_dim = int(ars_all.shape[-1] + arw_all.shape[-1]
                              + has_all.shape[-1] + haw_all.shape[-1])
        else:
            self.in_dim = int(ar_all.shape[-1] + ha_all.shape[-1])

        # Compute valid-window-start lists per subject. Drop subjects with 0 windows.
        starts_per_subj = [
            _valid_window_starts(mask_all[i], self.window_size)
            for i in range(len(self.eid_all))
        ]
        n_windows_per_subj = np.asarray([len(s) for s in starts_per_subj], dtype=np.int32)
        has_window = n_windows_per_subj > 0

        # Intersect with phecode-outcomes EIDs + covariate EIDs.
        outcomes_eids = np.asarray(outcomes["eids"], dtype=np.int64)
        cov_index = covariates.index.to_numpy()
        has_outcome = np.isin(self.eid_all, outcomes_eids)
        has_cov = np.isin(self.eid_all, cov_index)
        keep = has_window & has_outcome & has_cov

        if eid_filter is not None:
            filter_arr = np.asarray(list(eid_filter), dtype=np.int64)
            keep = keep & np.isin(self.eid_all, filter_arr)

        keep_idx = np.where(keep)[0]
        if len(keep_idx) == 0:
            raise ValueError(
                f"No subjects pass window/outcome/covariate filter at split={split} "
                f"window={window_size}."
            )

        # Slice down to the keep set; ar_all/ha_all (or stratified blocks) are
        # read in __getitem__ via the original indices to avoid duplicating a
        # big block in RAM.
        self._ar = ar_all
        self._ha = ha_all
        self._ar_sleep = ars_all
        self._ar_wake = arw_all
        self._ha_sleep = has_all
        self._ha_wake = haw_all
        self._mask = mask_all
        self.subj_keep_idx = keep_idx  # original-row index per surviving subject
        self.eids = self.eid_all[keep_idx]
        self.eid_runs = self.eid_run_all[keep_idx]
        self.starts_per_subj = [starts_per_subj[i] for i in keep_idx]
        self.n_windows_per_subj = n_windows_per_subj[keep_idx]

        # Resolve outcomes per surviving subject. outcomes["eids"] is unique sorted.
        eid_to_out = pd.Series(np.arange(len(outcomes_eids)), index=outcomes_eids)
        out_idx = eid_to_out.loc[self.eids].to_numpy()
        self.event_times = outcomes["event_times"][out_idx].astype(np.float32)
        self.is_event = outcomes["is_event"][out_idx].astype(np.float32)
        self.eval_mask = outcomes["eval_mask"][out_idx]

        self.covariates = covariates.reindex(self.eids).to_numpy(dtype=np.float32)
        self.n_covar = self.covariates.shape[1]
        self.n_phecodes = self.event_times.shape[1]

        assert not np.isnan(self.covariates).any(), (
            "NaN covariates: drop subjects with missing demographics upstream"
        )

        # Optional aux-age target: per-subject out-of-fold ar_emb, appended to the returned
        # covariate vector. The model slices it off as input via n_covar; the loss reads it
        # as the aux regression target. n_covar stays the real count so the head width is
        # unchanged. NaN for subjects without an out-of-fold ar_emb (the loss masks them).
        # None disables it (the default path).
        if aux_age is not None:
            self.aux_age = aux_age.reindex(self.eids).to_numpy(dtype=np.float32)
        else:
            self.aux_age = None

    def __len__(self) -> int:
        return len(self.eids)

    def _read_window(self, orig_i: int, s: int) -> np.ndarray:
        """Read one (window_size, in_dim) window from the appropriate blocks."""
        if self.stratified:
            ars = np.asarray(self._ar_sleep[orig_i, s : s + self.window_size])
            arw = np.asarray(self._ar_wake[orig_i, s : s + self.window_size])
            has = np.asarray(self._ha_sleep[orig_i, s : s + self.window_size])
            haw = np.asarray(self._ha_wake[orig_i, s : s + self.window_size])
            return np.concatenate([ars, arw, has, haw], axis=-1).astype(np.float32, copy=False)
        ar_w = np.asarray(self._ar[orig_i, s : s + self.window_size])
        ha_w = np.asarray(self._ha[orig_i, s : s + self.window_size])
        return np.concatenate([ar_w, ha_w], axis=-1).astype(np.float32, copy=False)

    def _sample_window(self, subj_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Pick one random valid window for subject `subj_idx`.

        Returns: (window (N, in_dim) float32, valid_mask (N,) bool)
        """
        starts = self.starts_per_subj[subj_idx]
        s = int(np.random.choice(starts))
        orig_i = self.subj_keep_idx[subj_idx]
        window = self._read_window(orig_i, s)
        return window, np.ones(self.window_size, dtype=bool)

    def __getitem__(self, idx: int):
        window, mask = self._sample_window(idx)
        cov = self.covariates[idx]
        if self.aux_age is not None:
            cov = np.concatenate([cov, self.aux_age[idx:idx + 1]]).astype(np.float32)
        return (
            torch.from_numpy(window),
            torch.from_numpy(mask),
            torch.from_numpy(self.event_times[idx]),
            torch.from_numpy(self.is_event[idx]),
            torch.from_numpy(cov),
        )

    def get_all_windows(self, idx: int, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """All valid N-day windows for subject `idx`.

        Args:
            stride: subsample valid window starts via `[::stride]`. Default 1
                keeps every overlapping window. For stride>1, ceil(n/stride)
                windows are returned per subject (always at least 1 since
                subjects with 0 windows were filtered in `__init__`).

        Returns:
            windows: (n_windows, N, in_dim) float32
            mask:    (n_windows, N) bool, all-True by construction
        """
        starts = self.starts_per_subj[idx]
        if stride > 1:
            starts = starts[::stride]
        orig_i = self.subj_keep_idx[idx]
        n_w = len(starts)
        windows = np.empty((n_w, self.window_size, self.in_dim), dtype=np.float32)
        for j, s in enumerate(starts):
            windows[j] = self._read_window(orig_i, s)
        return windows, np.ones((n_w, self.window_size), dtype=bool)


@torch.no_grad()
def forward_subject_windows(
    model: torch.nn.Module,
    dataset: WindowDataset,
    device: str = "cpu",
    batch_size: int = 128,
    out_dim: int | None = None,
) -> dict:
    """Forward all valid windows per subject; return per-window log-hazards (not
    averaged) plus each window's start-day index.

    Sibling of `forward_subject_aggregated` with identical packing and forward pass,
    but it skips the per-subject mean so a caller can regroup windows by day-type
    (weekend/weekday composition contrast). Window order per subject matches
    `dataset.starts_per_subj[i]` (the output of `_valid_window_starts`, i.e.
    ascending window start positions into that subject's valid-day axis). Because
    the day-type experiment does not mask any day, every real day is valid and the
    start index `s` maps directly to day positions `s, s+1, ..., s+window_size-1`
    in the subject's npz `day_codes` ordering.

    Returns:
        {
          "hazards":      (total_w, P) float32, one log-hazard vector per window,
          "subj_id":      (total_w,) int64, dataset row index of the owning subject,
          "start_day":    (total_w,) int32, window start position (day index),
          "offsets":      (N_subj + 1,) int64, CSR-style window ranges per subject,
          "n_windows":    (N_subj,) int32, window count per subject,
          "event_times":  (N_subj, P), "is_event": (N_subj, P),
          "eval_mask":    (N_subj, P), "eids": (N_subj,) int64,
        }
    """
    model.eval()
    n = len(dataset)
    P = dataset.n_phecodes if out_dim is None else int(out_dim)
    n_windows = np.asarray(dataset.n_windows_per_subj, dtype=np.int32)
    total_w = int(n_windows.sum())
    N = dataset.window_size
    in_dim = dataset.in_dim

    all_w = np.empty((total_w, N, in_dim), dtype=np.float32)
    all_m = np.ones((total_w, N), dtype=bool)
    all_c = np.empty((total_w, dataset.n_covar), dtype=np.float32)
    subj_id = np.empty(total_w, dtype=np.int64)
    start_day = np.empty(total_w, dtype=np.int32)
    offsets = np.zeros(n + 1, dtype=np.int64)
    cur = 0
    for i in range(n):
        windows, mask = dataset.get_all_windows(i, stride=1)
        w = windows.shape[0]
        all_w[cur:cur + w] = windows
        all_m[cur:cur + w] = mask
        all_c[cur:cur + w] = dataset.covariates[i]
        subj_id[cur:cur + w] = i
        start_day[cur:cur + w] = np.asarray(dataset.starts_per_subj[i], dtype=np.int32)
        cur += w
        offsets[i + 1] = cur

    out = np.empty((total_w, P), dtype=np.float32)
    for s in range(0, total_w, batch_size):
        e = min(s + batch_size, total_w)
        bw = torch.from_numpy(all_w[s:e]).to(device, non_blocking=True)
        bm = torch.from_numpy(all_m[s:e]).to(device, non_blocking=True)
        bc = torch.from_numpy(all_c[s:e]).to(device, non_blocking=True)
        out[s:e] = model(bw, bm, bc).detach().cpu().numpy()

    return {
        "hazards": out,
        "subj_id": subj_id,
        "start_day": start_day,
        "offsets": offsets,
        "n_windows": n_windows,
        "event_times": dataset.event_times,
        "is_event": dataset.is_event,
        "eval_mask": dataset.eval_mask,
        "eids": dataset.eids,
    }


def collate_fn(batch):
    """Stack a list of WindowDataset __getitem__ tuples into batch tensors."""
    windows = torch.stack([b[0] for b in batch])     # (B, N, 512)
    masks = torch.stack([b[1] for b in batch])       # (B, N)
    event_times = torch.stack([b[2] for b in batch]) # (B, P)
    is_event = torch.stack([b[3] for b in batch])    # (B, P)
    covariates = torch.stack([b[4] for b in batch])  # (B, n_cov)
    return windows, masks, event_times, is_event, covariates


@torch.no_grad()
def forward_subject_aggregated(
    model: torch.nn.Module,
    dataset: WindowDataset,
    device: str = "cpu",
    batch_size: int = 128,
    stride: int = 1,
    out_dim: int | None = None,
) -> dict:
    """Forward all valid windows per subject; average per-window log-hazards.

    Batched across subjects + windows for GPU throughput: pre-pack all
    valid windows from all subjects into one big tensor, forward in
    batches of `batch_size`, then aggregate per subject by mean.

    Args:
        stride: at inference, subsample per-subject window starts via
            `[::stride]` before averaging. Default 1 (every overlapping
            window). For stride>1, each subject keeps ceil(n_windows/stride)
            windows, always at least 1.
        out_dim: model output width per window. Defaults to dataset.n_phecodes
            (Cox single-hazard). For E1 discrete-time the model emits
            n_phecodes * n_bins logits, so pass that here; `hazards` is then
            (N_subj, out_dim) and the caller reshapes/converts to a risk score.

    Returns:
        {
          "hazards": (N_subj, P) float32, average log-hazard per subject
          "event_times": (N_subj, P),
          "is_event": (N_subj, P),
          "eval_mask": (N_subj, P),
          "eids": (N_subj,) int64,
          "n_windows": (N_subj,) int32, windows averaged per subject,
        }
    """
    model.eval()
    n = len(dataset)
    P = dataset.n_phecodes if out_dim is None else int(out_dim)
    if stride > 1:
        n_windows = np.asarray(
            [-(-int(nw) // stride) for nw in dataset.n_windows_per_subj],
            dtype=np.int32,
        )
    else:
        n_windows = np.asarray(dataset.n_windows_per_subj, dtype=np.int32)
    total_w = int(n_windows.sum())
    N = dataset.window_size
    in_dim = dataset.in_dim

    # Pack all valid windows per subject into contiguous arrays.
    all_w = np.empty((total_w, N, in_dim), dtype=np.float32)
    all_m = np.ones((total_w, N), dtype=bool)
    all_c = np.empty((total_w, dataset.n_covar), dtype=np.float32)
    subj_id = np.empty(total_w, dtype=np.int64)
    offsets = np.zeros(n + 1, dtype=np.int64)
    cur = 0
    for i in range(n):
        windows, mask = dataset.get_all_windows(i, stride=stride)
        w = windows.shape[0]
        all_w[cur:cur + w] = windows
        all_m[cur:cur + w] = mask
        all_c[cur:cur + w] = dataset.covariates[i]
        subj_id[cur:cur + w] = i
        cur += w
        offsets[i + 1] = cur

    # Big forward pass.
    out = np.empty((total_w, P), dtype=np.float32)
    for s in range(0, total_w, batch_size):
        e = min(s + batch_size, total_w)
        bw = torch.from_numpy(all_w[s:e]).to(device, non_blocking=True)
        bm = torch.from_numpy(all_m[s:e]).to(device, non_blocking=True)
        bc = torch.from_numpy(all_c[s:e]).to(device, non_blocking=True)
        out[s:e] = model(bw, bm, bc).detach().cpu().numpy()

    # Aggregate per subject (mean across that subject's windows).
    hazards = np.zeros((n, P), dtype=np.float32)
    for i in range(n):
        lo, hi = offsets[i], offsets[i + 1]
        if hi > lo:
            hazards[i] = out[lo:hi].mean(axis=0)

    return {
        "hazards": hazards,
        "event_times": dataset.event_times,
        "is_event": dataset.is_event,
        "eval_mask": dataset.eval_mask,
        "eids": dataset.eids,
        "n_windows": n_windows,
    }


__all__ = [
    "WindowDataset",
    "collate_fn",
    "forward_subject_aggregated",
    "forward_subject_windows",
]
