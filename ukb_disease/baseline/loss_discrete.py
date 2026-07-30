"""Discrete-time logistic-hazard loss (Nnet-survival / PC-Hazard) for the
multilabel many-phecode regime.

The Cox partial-likelihood loss builds the risk set from the minibatch, so at
batch=64 and roughly 0.30% prevalence most of the ~390 phecode columns receive
zero gradient in a typical batch. The discrete-time logistic hazard reformulates
each phecode as a sequence of per-time-bin Bernoulli outcomes with no risk set:
every phecode head gets a dense gradient from every subject in every batch, and
censored subjects contribute negatives up to their censor bin. This mirrors the
structural choice large many-disease survival models use (Delphi-2M, SurvivEHR).

Convention: hazard logit h[b, p, j] = logit P(event of phecode p in bin j |
survived to bin j). Loss = masked BCEWithLogits over each subject's followed
bins (bins up to and including its event/censor bin). The per-subject
per-phecode risk score used downstream is the cumulative hazard
H = sum_j softplus(h_j), which is monotone-increasing in risk (higher = more
risk), matching the Cox convention the C-index scorer expects (it passes
`-hazards`).

All functions are torch-only and batch-shaped so they slot into the training
loop with no dataloader change.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def build_time_bins(
    event_times: np.ndarray,
    is_event: np.ndarray,
    n_bins: int,
    seven_day_years: float = 7.0 / 365.25,
) -> np.ndarray:
    """KM-quantile (equal-events) interior cut-points from train incident events.

    Args:
        event_times: (N, P) float, train event/censor times in years.
        is_event:    (N, P) float, 1 for events.
        n_bins:      number of time bins T (so T-1 interior cut-points).
        seven_day_years: incident threshold; events at/below it are the
            effective-prevalent t=0.0001 cluster and are excluded from the
            quantile grid so the cut-points reflect genuine incident timing.

    Returns:
        cuts: (n_bins - 1,) float64 ascending interior cut-points. A time t is
        assigned to bin `np.searchsorted(cuts, t, side="right")`, clipped to
        [0, n_bins-1].
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    ev = (is_event > 0.5) & np.isfinite(event_times) & (event_times > seven_day_years)
    times = event_times[ev]
    if times.size < n_bins:
        # Degenerate (tiny smoke runs): fall back to an even grid over observed range.
        lo = float(np.nanmin(event_times[np.isfinite(event_times)])) if np.isfinite(event_times).any() else 0.0
        hi = float(np.nanmax(event_times[np.isfinite(event_times)])) if np.isfinite(event_times).any() else 1.0
        hi = max(hi, lo + 1e-3)
        return np.linspace(lo, hi, n_bins + 1)[1:-1].astype(np.float64)
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]  # interior quantiles
    cuts = np.quantile(times, qs).astype(np.float64)
    # De-duplicate ties so searchsorted bins are monotone and non-degenerate.
    cuts = np.unique(cuts)
    if len(cuts) < n_bins - 1:
        # Pad by nudging (rare) to keep T stable for the head width.
        pad = np.linspace(cuts[-1] + 1e-4, cuts[-1] + 1e-3, (n_bins - 1) - len(cuts))
        cuts = np.concatenate([cuts, pad])
    return cuts.astype(np.float64)


def assign_bins(event_times: torch.Tensor, cuts: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Bin index per (subject, phecode). Shape preserved.

    bin = searchsorted(cuts, t, right) clipped to [0, n_bins-1].
    """
    idx = torch.searchsorted(cuts, event_times.contiguous(), right=True)
    return idx.clamp_(0, n_bins - 1).long()


def discrete_hazard_loss(
    logits: torch.Tensor,        # (B, P, T)
    event_times: torch.Tensor,   # (B, P)
    is_event: torch.Tensor,      # (B, P)
    cuts: torch.Tensor,          # (T-1,)
    eval_mask: torch.Tensor | None = None,  # (B, P) bool, True = use
    pos_weight: torch.Tensor | None = None,  # (T,) or scalar
) -> torch.Tensor:
    """Masked per-bin BCE over each subject's followed bins.

    A subject (b) for phecode (p) with event/censor bin k contributes a binary
    target to bins 0..k inclusive: 0 for bins < k, and `is_event` at bin k.
    Bins > k are not followed (masked out).
    """
    B, P, T = logits.shape
    k = assign_bins(event_times, cuts, T)              # (B, P)
    arange = torch.arange(T, device=logits.device).view(1, 1, T)
    followed = arange <= k.unsqueeze(-1)               # (B, P, T) bool
    target = ((arange == k.unsqueeze(-1)) & (is_event.unsqueeze(-1) > 0.5)).float()
    cell_mask = followed
    if eval_mask is not None:
        cell_mask = cell_mask & eval_mask.unsqueeze(-1)
    cell_mask = cell_mask.float()
    bce = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=pos_weight
    )
    denom = cell_mask.sum().clamp(min=1.0)
    return (bce * cell_mask).sum() / denom


def logits_to_risk(logits: np.ndarray | torch.Tensor, n_bins: int) -> np.ndarray:
    """Convert per-bin hazard logits to a scalar per-(subject, phecode) risk score.

    Risk = cumulative hazard H = sum_j softplus(h_j) over all T bins. Monotone
    increasing in risk; higher = more risk (matches the Cox log-hazard sign the
    downstream C-index scorer expects).

    Accepts logits of shape (..., P*T) or (..., P, T) and returns (..., P).
    """
    x = torch.as_tensor(np.asarray(logits)) if not isinstance(logits, torch.Tensor) else logits
    if x.shape[-1] != n_bins:
        x = x.reshape(*x.shape[:-1], -1, n_bins)
    risk = F.softplus(x).sum(dim=-1)
    return risk.detach().cpu().numpy()


__all__ = [
    "build_time_bins",
    "assign_bins",
    "discrete_hazard_loss",
    "logits_to_risk",
]
