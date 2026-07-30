"""Set Transformer primitives (Lee et al. 2019, arXiv:1810.00825).

Learnable permutation-invariant pooling that generalizes the fixed L-moments day-pool:
k seed queries cross-attend over a set of token embeddings.

- ``PMA``  (Pooling by Multihead Attention): k seeds attend over the set -> (B, k, dim).
  Replaces the across-day CLS pool (the day axis is short, so no ISAB is needed).
- ``ISAB`` (Induced Set Attention Block): O(N*m) attention via m inducing points, for the
  long within-day patch sets (~2880 AR / ~8640 HA).
- ``SetEncoder``: [ISAB x n] -> PMA(k) -> mean over seeds, a per-set vector.

Pre-norm (norm_first) to match the sequence-head transformer style. All blocks accept a
``key_padding_mask`` (True = pad / invalid) so NaN-masked nonwear patches and padded days
never enter attention.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MAB(nn.Module):
    """Multihead Attention Block (pre-norm): query set X attends over key set Y."""

    def __init__(self, dim: int, num_heads: int = 4, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(ff_mult * dim, dim),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, Q, D) queries; y: (B, K, D) keys/values; mask True = pad on y.
        # Guard a fully-padded row: nn.MultiheadAttention returns NaN if every key is
        # masked. Unmask those rows (their query output is unused / re-masked upstream).
        if key_padding_mask is not None:
            allpad = key_padding_mask.all(dim=1)
            if bool(allpad.any()):
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[allpad, 0] = False
        a, _ = self.attn(self.norm_q(x), self.norm_kv(y), self.norm_kv(y),
                         key_padding_mask=key_padding_mask, need_weights=False)
        h = x + a
        return h + self.ff(self.norm_ff(h))


class PMA(nn.Module):
    """Pooling by Multihead Attention: k learnable seeds attend over the set."""

    def __init__(self, dim: int, num_heads: int = 4, num_seeds: int = 1,
                 ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_seeds = int(num_seeds)
        self.seeds = nn.Parameter(torch.zeros(1, self.num_seeds, dim))
        nn.init.normal_(self.seeds, std=0.02)
        self.mab = MAB(dim, num_heads, ff_mult, dropout)

    def forward(self, z: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # z: (B, K, D); returns (B, num_seeds, D).
        B = z.shape[0]
        s = self.seeds.expand(B, -1, -1)
        return self.mab(s, z, key_padding_mask=key_padding_mask)


class ISAB(nn.Module):
    """Induced Set Attention Block: O(N*m) self-attention via m inducing points."""

    def __init__(self, dim: int, num_heads: int = 4, num_inducing: int = 32,
                 ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.inducing = nn.Parameter(torch.zeros(1, int(num_inducing), dim))
        nn.init.normal_(self.inducing, std=0.02)
        self.mab1 = MAB(dim, num_heads, ff_mult, dropout)  # I attends over X -> H
        self.mab2 = MAB(dim, num_heads, ff_mult, dropout)  # X attends over H

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        B = x.shape[0]
        i = self.inducing.expand(B, -1, -1)
        h = self.mab1(i, x, key_padding_mask=key_padding_mask)  # (B, m, D), no mask on H
        return self.mab2(x, h)                                  # (B, N, D)


class SetEncoder(nn.Module):
    """[ISAB x n_isab] -> PMA(k) -> mean over seeds: a permutation-invariant set vector."""

    def __init__(self, dim: int, num_heads: int = 4, num_inducing: int = 32,
                 num_seeds: int = 4, n_isab: int = 1, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.isabs = nn.ModuleList(
            [ISAB(dim, num_heads, num_inducing, ff_mult, dropout) for _ in range(n_isab)])
        self.pma = PMA(dim, num_heads, num_seeds, ff_mult, dropout)

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        for isab in self.isabs:
            x = isab(x, key_padding_mask=key_padding_mask)
        pooled = self.pma(x, key_padding_mask=key_padding_mask)  # (B, k, D)
        return pooled.mean(dim=1)                                # (B, D)


if __name__ == "__main__":
    torch.manual_seed(0)
    B, N, D = 8, 7, 32
    z = torch.randn(B, N, D)
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[0, 3:] = True            # subject 0 has only 3 valid days
    mask[1, :] = True             # subject 1 fully padded (degenerate guard)

    pma = PMA(D, num_heads=4, num_seeds=4)
    out = pma(z, key_padding_mask=mask).mean(dim=1)
    assert out.shape == (B, D), out.shape
    assert torch.isfinite(out).all(), "PMA produced NaN/Inf (fully-padded guard failed)"

    # permutation invariance: shuffling the (unmasked) set order leaves the pool unchanged.
    pma.eval()
    with torch.no_grad():
        out = pma(z, key_padding_mask=mask).mean(dim=1)
        perm = torch.randperm(N)
        out_p = pma(z[:, perm, :], key_padding_mask=mask[:, perm]).mean(dim=1)
    # subject 2 is fully valid (no masking) -> its pool must be order-invariant.
    assert torch.allclose(out[2], out_p[2], atol=1e-4), "PMA not permutation-invariant"

    enc = SetEncoder(D, num_inducing=16, num_seeds=4, n_isab=1)
    big = torch.randn(B, 500, D)
    bmask = torch.zeros(B, 500, dtype=torch.bool); bmask[0, 100:] = True
    e = enc(big, key_padding_mask=bmask)
    assert e.shape == (B, D) and torch.isfinite(e).all()
    print("OK: PMA/ISAB/SetEncoder finite, masked, permutation-invariant.")
