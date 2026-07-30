"""Differentiable multilabel pairwise concordance loss.

A DeepHit-style ranking term added to the multilabel Cox leaf loss in the
training loop. We implement it here rather than reusing an external margin-ranking
loss because the version we had available has a base-class initialization bug.

Sign convention (must match the Cox head): ``haz`` is a per-label log-hazard where
a higher value means higher instantaneous risk, i.e. a shorter survival time. So
for a comparable ordered pair (i, j) with ``t_i < t_j`` and ``e_i == 1`` (subject i
has the earlier, observed event), subject i should be ranked at higher risk than j,
i.e. we want ``haz_i > haz_j`` and penalize the opposite. This is Harrell's
concordant-pair quantity, so the term directly optimizes the evaluation metric on
every comparable pair rather than the looser per-batch risk-set approximation the
Cox partial likelihood uses.

The term is fully vectorized over the batch with broadcast ``(B, B, P)`` pair
matrices (no Python pair loop), normalized per label by its comparable-pair count
so common diseases (huge pair sets) do not dominate rare ones and the loss scale
stays comparable to the Cox term. Memory is ``O(B^2 * P)``. At B=64, P~390 that is
about 6 MB per tensor, so the caller keeps the microbatch at 64 and grows the
effective batch via gradient accumulation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def multilabel_ranking_loss(
    haz: torch.Tensor,
    t: torch.Tensor,
    e: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    phi: str = "sigmoid",
    margin: float = 0.0,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Mean pairwise concordance penalty over comparable, valid, ordered pairs.

    Args:
        haz: ``(B, P)`` per-label log-hazards (higher = higher risk = shorter survival).
        t:   ``(B, P)`` event/censoring times.
        e:   ``(B, P)`` event indicator in {0, 1}.
        valid_mask: ``(B, P)`` bool of eval-valid rows per label (effective-prevalent rows
            excluded), or ``None`` during training (all rows valid). Mirrors the eval mask
            so the term never drifts from the scored quantity.
        phi: ``"sigmoid"`` or ``"softplus"`` (DeepHit smooth) or ``"margin"`` (hinge).
        margin: hinge margin ``m`` (only used when ``phi == "margin"``).
        sigma: sigmoid sharpness (only used when ``phi == "sigmoid"``).

    Returns:
        Scalar loss (mean over labels that had >=1 comparable pair). Zero if none.
    """
    if haz.dim() != 2:
        raise ValueError(f"multilabel_ranking_loss expects (B, P) haz, got {tuple(haz.shape)}")
    B, P = haz.shape
    if valid_mask is None:
        valid_mask = torch.ones_like(e, dtype=torch.bool)
    else:
        valid_mask = valid_mask.bool()

    # Broadcast pair matrices: dim0 = i, dim1 = j, per label p.
    ti = t.unsqueeze(1)              # (B, 1, P), [i, j, p] = t_i
    tj = t.unsqueeze(0)              # (1, B, P), [i, j, p] = t_j
    ei = e.unsqueeze(1) > 0.5        # (B, 1, P) -> event at i
    vi = valid_mask.unsqueeze(1)     # (B, 1, P)
    vj = valid_mask.unsqueeze(0)     # (1, B, P)

    comparable = (ti < tj) & ei & vi & vj            # (B, B, P) bool

    # diff[i, j, p] = haz_i - haz_j ; we want this positive (i is higher risk).
    diff = haz.unsqueeze(1) - haz.unsqueeze(0)       # (B, B, P)

    if phi == "sigmoid":
        pair_loss = torch.sigmoid(-diff / sigma)     # DeepHit-style smooth ranking
    elif phi == "softplus":
        pair_loss = F.softplus(-diff)
    elif phi == "margin":
        pair_loss = F.relu(margin - diff)            # hinge: 0 once diff >= margin
    else:
        raise ValueError(f"unknown phi '{phi}' (expected sigmoid|softplus|margin)")

    pair_loss = pair_loss * comparable.to(pair_loss.dtype)   # zero non-comparable pairs
    n_pairs = comparable.sum(dim=(0, 1)).clamp_min(1.0)      # (P,) per-label pair count
    per_label = pair_loss.sum(dim=(0, 1)) / n_pairs          # (P,) normalized per label
    has = comparable.any(dim=(0, 1))                         # (P,) labels with >=1 pair
    if bool(has.any()):
        return per_label[has].mean()
    return haz.new_zeros(())


if __name__ == "__main__":
    # Self-test: a perfectly concordant ranking has near-zero loss, an
    # anti-concordant ranking has a large loss, and the gradient pushes haz
    # toward concordance.
    torch.manual_seed(0)
    B, P = 32, 5
    t = torch.rand(B, P) * 10.0
    e = (torch.rand(B, P) > 0.4).float()

    # haz_concordant = -t (earlier time -> higher hazard) should give near-minimal loss.
    haz_good = -t.clone()
    haz_bad = t.clone()           # anti-concordant
    haz_rand = torch.randn(B, P)

    for name, h in [("concordant(-t)", haz_good), ("anti(+t)", haz_bad), ("random", haz_rand)]:
        for phi in ("sigmoid", "softplus", "margin"):
            v = multilabel_ranking_loss(h, t, e, phi=phi, margin=0.1, sigma=1.0)
            print(f"{name:16s} phi={phi:8s} loss={float(v):.4f}")
        print()

    # Gradient sanity: a learnable haz should reduce the loss and approach concordance.
    h = torch.randn(B, P, requires_grad=True)
    opt = torch.optim.Adam([h], lr=0.2)
    l0 = float(multilabel_ranking_loss(h, t, e, phi="sigmoid"))
    for _ in range(300):
        opt.zero_grad()
        loss = multilabel_ranking_loss(h, t, e, phi="sigmoid")
        loss.backward()
        opt.step()
    l1 = float(multilabel_ranking_loss(h, t, e, phi="sigmoid"))
    # correlation of learned haz with -t (concordance target), averaged over labels
    corr = float(torch.stack([
        torch.corrcoef(torch.stack([h[:, p].detach(), -t[:, p]]))[0, 1] for p in range(P)
    ]).mean())
    print(f"gradient descent: loss {l0:.4f} -> {l1:.4f} ; mean corr(haz, -t) = {corr:.3f}")
    assert l1 < l0, "ranking loss did not decrease under gradient descent"
    assert corr > 0.5, "learned hazard did not become concordant with event ordering"
    print("OK: ranking loss is concordance-aligned and differentiable.")
