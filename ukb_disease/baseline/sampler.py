"""Positive-enriched sampling for survival training.

Surfaces disease-relevant gradient signal when most subjects are censored.
Used by baseline control (run_baseline.py with `--curriculum pos_enriched`)
and by the Cox trainer (train_rolling_window_cox.py).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Sampler, WeightedRandomSampler


def compute_positive_mask(
    is_event: np.ndarray,
    eval_mask: np.ndarray | None = None,
) -> np.ndarray:
    """A subject is "positive" if any of its evaluated labels is a positive event.

    Args:
        is_event: (N, P) float32, 1.0 where event occurred.
        eval_mask: (N, P) bool, True where the (subject, label) is included
            in the loss/eval. None means all-True.
    Returns:
        (N,) bool, True for positive subjects.
    """
    if eval_mask is None:
        eval_mask = np.ones_like(is_event, dtype=bool)
    valid_pos = (is_event > 0.5) & eval_mask
    return valid_pos.any(axis=1)


def make_positive_enriched_sampler(
    is_event: np.ndarray,
    eval_mask: np.ndarray | None = None,
    balance: float = 0.5,
    seed: int = 42,
) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler giving `balance` mass to positive subjects.

    Each draw is i.i.d.; per epoch we draw `len(weights)` samples. Replacement
    is True so positives can be drawn multiple times (the oversampling effect).

    If the train pool has no positive subjects, falls back to uniform sampling.
    """
    pos_mask = compute_positive_mask(is_event, eval_mask)
    n = len(pos_mask)
    n_pos = int(pos_mask.sum())
    n_neg = n - n_pos

    weights = np.full(n, 1.0 / max(n, 1), dtype=np.float64)
    if n_pos > 0 and n_neg > 0:
        weights[pos_mask] = balance / n_pos
        weights[~pos_mask] = (1.0 - balance) / n_neg

    gen = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.float64),
        num_samples=n,
        replacement=True,
        generator=gen,
    )


class PerDiseaseBalancedBatchSampler(Sampler):
    """Yields fixed-composition batches: n_pos_per_batch positives + remainder negatives.

    Designed for per-disease specialization training. The "positive" pool for
    one batch is drawn (without replacement within the batch) from subjects
    whose target phecode column has `is_event == 1 AND eval_mask == True` in
    the train arrays. The "negative" pool is everyone else. Across batches,
    draws are with replacement, so a given positive subject is expected to
    appear in many batches per epoch (~n_batches * n_pos_per_batch / n_pos).

    Each batch is yielded as a shuffled list of int indices. __len__ returns
    a fixed n_batches per epoch so the cosine_warmup scheduler can size its
    per-step budget from len(train_loader).

    Args:
        pos_idx:           (n_pos,) int64 train-pool indices of disease-positive subjects.
        neg_idx:           (n_neg,) int64 train-pool indices of remaining subjects.
        batch_size:        Total batch size (e.g., 64).
        n_pos_per_batch:   Number of positives per batch (e.g., 32 for 50/50).
        n_batches:         Batches yielded per epoch. None uses len(pos)+len(neg) // batch_size.
        seed:              Base RNG seed. Different seed gives a different batch sequence.

    Raises:
        ValueError: if either pool is smaller than its per-batch draw.
    """

    def __init__(
        self,
        pos_idx: np.ndarray,
        neg_idx: np.ndarray,
        batch_size: int,
        n_pos_per_batch: int,
        n_batches: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__(data_source=None)
        pos_idx = np.asarray(pos_idx, dtype=np.int64)
        neg_idx = np.asarray(neg_idx, dtype=np.int64)
        n_neg_per_batch = int(batch_size) - int(n_pos_per_batch)
        if n_pos_per_batch <= 0 or n_neg_per_batch <= 0:
            raise ValueError(
                f"n_pos_per_batch ({n_pos_per_batch}) and n_neg_per_batch "
                f"({n_neg_per_batch}) must both be > 0 (batch_size={batch_size})"
            )
        if len(pos_idx) < n_pos_per_batch:
            raise ValueError(
                f"PerDiseaseBalancedBatchSampler: positive pool size {len(pos_idx)} "
                f"< n_pos_per_batch {n_pos_per_batch}. Lower n_pos_per_batch or "
                f"widen the target's positive definition."
            )
        if len(neg_idx) < n_neg_per_batch:
            raise ValueError(
                f"PerDiseaseBalancedBatchSampler: negative pool size {len(neg_idx)} "
                f"< n_neg_per_batch {n_neg_per_batch}."
            )
        if n_batches is None:
            n_batches = max(1, (len(pos_idx) + len(neg_idx)) // int(batch_size))

        self.pos_idx = pos_idx
        self.neg_idx = neg_idx
        self.batch_size = int(batch_size)
        self.n_pos_per_batch = int(n_pos_per_batch)
        self.n_neg_per_batch = n_neg_per_batch
        self.n_batches = int(n_batches)
        self.base_seed = int(seed)
        self._epoch = 0

    def __iter__(self):
        # Fresh RNG per epoch gives a different batch sequence each pass, reproducible
        # for fixed (base_seed, epoch_index).
        rng = np.random.default_rng(self.base_seed + self._epoch * 1_000_003)
        self._epoch += 1
        for _ in range(self.n_batches):
            pos_batch = rng.choice(self.pos_idx, size=self.n_pos_per_batch, replace=False)
            neg_batch = rng.choice(self.neg_idx, size=self.n_neg_per_batch, replace=False)
            batch = np.concatenate([pos_batch, neg_batch])
            rng.shuffle(batch)
            yield batch.tolist()

    def __len__(self) -> int:
        return self.n_batches


def make_per_disease_balanced_sampler(
    is_event: np.ndarray,
    eval_mask: np.ndarray,
    target_col: int,
    batch_size: int = 64,
    n_pos_per_batch: int = 32,
    n_batches: int | None = None,
    seed: int = 42,
) -> PerDiseaseBalancedBatchSampler:
    """Build a PerDiseaseBalancedBatchSampler for one phecode column.

    Args:
        is_event:        (N, P) float, 1.0 where train arrays have a positive event.
        eval_mask:       (N, P) bool, True where (subject, phecode) is in loss/eval.
        target_col:      Column index of the target phecode in the dataset's
                         phecode ordering (resolved via `phecodes.index(...)`).
        batch_size:      Total batch size.
        n_pos_per_batch: Positives per batch.
        n_batches:       Batches per epoch (None uses (n_pos+n_neg)//batch_size).
        seed:            Base RNG seed.

    Returns:
        Configured sampler. Pass to `DataLoader(..., batch_sampler=...)`
        (NOT `sampler=`); do NOT also pass `batch_size`/`shuffle`/`sampler`.
    """
    pos_mask = (is_event[:, target_col] > 0.5) & eval_mask[:, target_col]
    pos_idx = np.where(pos_mask)[0]
    neg_idx = np.where(~pos_mask)[0]
    return PerDiseaseBalancedBatchSampler(
        pos_idx=pos_idx,
        neg_idx=neg_idx,
        batch_size=batch_size,
        n_pos_per_batch=n_pos_per_batch,
        n_batches=n_batches,
        seed=seed,
    )


__all__ = [
    "compute_positive_mask",
    "make_positive_enriched_sampler",
    "PerDiseaseBalancedBatchSampler",
    "make_per_disease_balanced_sampler",
]
