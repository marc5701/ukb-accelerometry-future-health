"""Kernel-free, mask-aware selective state-space / recurrent block.

The CUDA selective-scan kernel (`mamba_ssm`) is not always available on the
compute environment, and compiling it against a given cluster toolchain is a
portability risk. Instead this module implements a self-contained, pure-PyTorch
linear-time selective recurrent layer from the SSM/minimal-RNN family
(minGRU; Feng et al. 2024, "Were RNNs All We Needed?"), which:

  - is a selective gated linear recurrence  h_t = a_t * h_{t-1} + b_t  with
    input-dependent a_t = 1-sigmoid(z_t) and b_t = sigmoid(z_t) * g(h_tilde_t),
    the same first-order diagonal recurrence Mamba's S6 reduces to;
  - is solved in parallel by a numerically-stable, chunked log-space scan
    (logcumsumexp), so it is O(L) and has no python per-step loop and no custom
    kernel, running on any torch build;
  - is mask-aware: at nonwear/pad positions a_t=1, b_t=0, so masked tokens
    neither inject signal nor disturb the carried state (the SSM analogue of
    -inf attention masking);
  - is causal when `bidirectional=False`: the forward scan's per-position
    outputs are the causal prefix states, so per-patch deep supervision gets
    every prefix for free in one pass.

Block: LayerNorm, in_proj (with optional left-padded causal depthwise conv),
SiLU, selective recurrence, SiLU gate, out_proj, residual.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Positivity activation (keeps the candidate state > 0 so its log is real).
def g_act(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))


def log_g(x: torch.Tensor) -> torch.Tensor:
    """log(g_act(x)) computed stably."""
    return torch.where(x >= 0, torch.log(F.relu(x) + 0.5), -F.softplus(-x))


# The stable, chunked, log-space linear scan.
def _scan_chunk(log_coeffs: torch.Tensor, log_b: torch.Tensor,
                log_h0: torch.Tensor | None) -> torch.Tensor:
    """Solve h_t = a_t*h_{t-1} + b_t over one chunk, given log(a)=log_coeffs,
    log(b)=log_b, and optional log(h0) carry. Returns h (B, L, D) > 0.

    Uses the Heinsen/minGRU identity:
        h_t = exp(A_t) * sum_{i<=t} exp(log_v_i - A_i),   A_t = sum_{j<=t} log a_j
    evaluated with logcumsumexp so no intermediate over/underflows.
    """
    B, L, D = log_coeffs.shape
    if log_h0 is None:
        log_h0 = torch.full((B, 1, D), float("-inf"), device=log_coeffs.device,
                            dtype=log_coeffs.dtype)
    else:
        log_h0 = log_h0.unsqueeze(1)
    log_values = torch.cat([log_h0, log_b], dim=1)             # (B, L+1, D)
    a_star = F.pad(log_coeffs.cumsum(dim=1), (0, 0, 1, 0))      # (B, L+1, D)
    log_h = a_star + torch.logcumsumexp(log_values - a_star, dim=1)
    return log_h[:, 1:].exp()


def parallel_scan_log(log_coeffs: torch.Tensor, log_b: torch.Tensor,
                      chunk: int = 512) -> torch.Tensor:
    """Chunked stable scan for arbitrary L. Carries log(h_last) between chunks
    so float32 precision stays bounded even for 20k to 60k-token recordings."""
    B, L, D = log_coeffs.shape
    if L <= chunk:
        return _scan_chunk(log_coeffs, log_b, None)
    outs = []
    log_h0 = None
    for s in range(0, L, chunk):
        e = min(s + chunk, L)
        h = _scan_chunk(log_coeffs[:, s:e], log_b[:, s:e], log_h0)
        outs.append(h)
        log_h0 = torch.log(h[:, -1].clamp_min(1e-30))
    return torch.cat(outs, dim=1)


class MinGRU(nn.Module):
    """Selective gated linear recurrence with a parallel log-space scan.

    forward(x (B,L,dim), mask (B,L) bool True=valid) -> (B,L,dim).
    bidirectional=False gives causal (prefix states); True sums forward+backward.
    """

    def __init__(self, dim: int, bidirectional: bool = False):
        super().__init__()
        self.dim = int(dim)
        self.bidirectional = bool(bidirectional)
        self.to_z = nn.Linear(dim, dim)
        self.to_h = nn.Linear(dim, dim)

    def _dir_scan(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        z = self.to_z(x)
        h_tilde = self.to_h(x)
        log_coeffs = -F.softplus(z)          # log(1 - sigmoid(z)) = log(a_t)
        log_z = -F.softplus(-z)              # log(sigmoid(z))
        log_b = log_z + log_g(h_tilde)       # log(sigmoid(z) * g(h_tilde))
        if mask is not None:
            m = mask.unsqueeze(-1)
            log_coeffs = torch.where(m, log_coeffs, torch.zeros_like(log_coeffs))
            log_b = torch.where(m, log_b, torch.full_like(log_b, float("-inf")))
        return parallel_scan_log(log_coeffs, log_b)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self._dir_scan(x, mask)
        if self.bidirectional:
            xr = torch.flip(x, dims=[1])
            mr = torch.flip(mask, dims=[1]) if mask is not None else None
            hr = torch.flip(self._dir_scan(xr, mr), dims=[1])
            h = h + hr
        return h


class MambaBlock(nn.Module):
    """Pre-norm residual selective-recurrence block (kernel-free).

    forward(x (B,L,dim), mask (B,L)) -> (B,L,dim).
    `causal=True` uses a left-padded depthwise conv plus a unidirectional scan
    (for per-patch deep supervision). `causal=False` is bidirectional (for
    pooling-only readout).
    """

    def __init__(self, dim: int, expand: int = 2, conv_kernel: int = 4,
                 causal: bool = True, dropout: float = 0.0):
        super().__init__()
        self.causal = bool(causal)
        self.conv_kernel = int(conv_kernel)
        inner = int(dim * expand)
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, inner)
        self.gate_proj = nn.Linear(dim, inner)
        self.conv = nn.Conv1d(inner, inner, kernel_size=conv_kernel,
                              groups=inner, padding=conv_kernel - 1)
        self.ssm = MinGRU(inner, bidirectional=not self.causal)
        self.out_proj = nn.Linear(inner, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        res = x
        x = self.norm(x)
        u = self.in_proj(x)                                   # (B,L,inner)
        gate = self.gate_proj(x)
        uc = u.transpose(1, 2)                                # (B,inner,L)
        uc = self.conv(uc)[..., : u.shape[1]]                 # causal: left-pad then trim
        u = uc.transpose(1, 2)
        if mask is not None:
            u = u * mask.unsqueeze(-1)
        u = F.silu(u)
        h = self.ssm(u, mask)
        h = h * F.silu(gate)
        h = self.out_proj(h)
        return res + self.drop(h)


class MaskedMeanPool(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=1)
        m = mask.float().unsqueeze(-1)
        return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


class MaskedLastPool(nn.Module):
    """Pool the LAST valid position per row (the full-recording causal state)."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            return x[:, -1]
        L = x.shape[1]
        # index of last True per row = L-1 - argmax(reversed mask)
        last = L - 1 - torch.argmax(mask.flip(dims=[1]).float(), dim=1)
        return x[torch.arange(x.shape[0], device=x.device), last]


__all__ = [
    "g_act", "log_g", "parallel_scan_log",
    "MinGRU", "MambaBlock", "MaskedMeanPool", "MaskedLastPool",
]
