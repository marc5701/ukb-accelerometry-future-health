"""Flexible-length day sequence dataset.

Day-level variant of a flexible-length transformer over day sequences. The
patch-level version of this idea did not improve over the fixed-window baseline,
so only the cheaper day-level variant is kept here.

Unlike WindowDataset (which samples a fixed N-day window and averages all
windows at inference), this yields each subject's FULL valid-day sequence
(variable 1..d_max days) with a REAL padding mask, trained/evaluated as one
variable-length sequence through TransformerSeqHead (which already threads
src_key_padding_mask). The aggregate already packs each subject's valid days
contiguously from index 0 and provides valid_day_mask, so there is nothing to
window: we return the padded row directly.

Yields per __getitem__:  (window (d_max,in_dim), valid_mask (d_max,), t, e, cov)
matching collate_fn / the trainer's (windows, masks, t, e, c) contract.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ukb_disease.baseline.config import DAY_MEAN_ROOT, resolve_oak_path


class FlexibleWindowDataset(Dataset):
    def __init__(
        self,
        split: str,
        outcomes: dict,
        covariates: pd.DataFrame,
        eid_filter=None,
        day_mean_root: str = DAY_MEAN_ROOT,
        rng_seed: int = 0,
        min_valid_days: int = 1,
        **_ignore,
    ):
        super().__init__()
        self.split = split
        split_dir = resolve_oak_path(os.path.join(day_mean_root, split))
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Day-means split dir missing: {split_dir}")
        idx = pd.read_parquet(os.path.join(split_dir, "index.parquet"))
        self.eid_all = idx["eid"].to_numpy(dtype=np.int64)
        ar = np.load(os.path.join(split_dir, "ar.npy"))           # (N, D, ar_dim)
        ha = np.load(os.path.join(split_dir, "ha.npy"))           # (N, D, ha_dim)
        mask = np.load(os.path.join(split_dir, "valid_day_mask.npy")).astype(bool)
        self.d_max = int(ar.shape[1])
        self.in_dim = int(ar.shape[-1] + ha.shape[-1])
        self._win = np.concatenate([ar, ha], axis=-1).astype(np.float32)  # (N, D, in_dim)
        self._mask = mask

        n_valid_days = self._mask.sum(axis=1)
        has_days = n_valid_days >= int(min_valid_days)
        outcomes_eids = np.asarray(outcomes["eids"], dtype=np.int64)
        cov_index = covariates.index.to_numpy()
        keep = has_days & np.isin(self.eid_all, outcomes_eids) & np.isin(self.eid_all, cov_index)
        if eid_filter is not None:
            keep &= np.isin(self.eid_all, np.asarray(list(eid_filter), dtype=np.int64))
        keep_idx = np.where(keep)[0]
        if len(keep_idx) == 0:
            raise ValueError(f"No subjects pass filter at split={split} (flexible).")

        self._win = self._win[keep_idx]
        self._mask = self._mask[keep_idx]
        self.eids = self.eid_all[keep_idx]

        eid_to_out = pd.Series(np.arange(len(outcomes_eids)), index=outcomes_eids)
        out_idx = eid_to_out.loc[self.eids].to_numpy()
        self.event_times = outcomes["event_times"][out_idx].astype(np.float32)
        self.is_event = outcomes["is_event"][out_idx].astype(np.float32)
        self.eval_mask = outcomes["eval_mask"][out_idx]
        self.covariates = covariates.reindex(self.eids).to_numpy(dtype=np.float32)
        self.n_covar = self.covariates.shape[1]
        self.n_phecodes = self.event_times.shape[1]
        assert not np.isnan(self.covariates).any(), "NaN covariates"

    def __len__(self) -> int:
        return len(self.eids)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self._win[idx]),
            torch.from_numpy(self._mask[idx]),
            torch.from_numpy(self.event_times[idx]),
            torch.from_numpy(self.is_event[idx]),
            torch.from_numpy(self.covariates[idx]),
        )


@torch.no_grad()
def forward_subject_flex(model, dataset: FlexibleWindowDataset, device: str = "cpu",
                         batch_size: int = 64, out_dim: int | None = None) -> dict:
    """One forward per subject over the full padded day sequence (no averaging)."""
    model.eval()
    n = len(dataset)
    P = dataset.n_phecodes if out_dim is None else int(out_dim)
    hazards = np.zeros((n, P), dtype=np.float32)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        bw = torch.from_numpy(dataset._win[s:e]).to(device)
        bm = torch.from_numpy(dataset._mask[s:e]).to(device)
        bc = torch.from_numpy(dataset.covariates[s:e]).to(device)
        hazards[s:e] = model(bw, bm, bc).detach().cpu().numpy()
    return {
        "hazards": hazards,
        "event_times": dataset.event_times,
        "is_event": dataset.is_event,
        "eval_mask": dataset.eval_mask,
        "eids": dataset.eids,
        "n_windows": np.ones(n, dtype=np.int32),
    }


__all__ = ["FlexibleWindowDataset", "forward_subject_flex"]
