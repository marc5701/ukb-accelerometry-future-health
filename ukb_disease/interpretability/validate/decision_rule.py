"""Decision_rule rule: combine ladder tiers into a per-(cell,disease) decision.

A (cell, disease) is CONFIRMED iff:
  1. sign agreement across all available tiers (Tier-1 occlusion, Tier-2
     conditional ablation, Tier-3 REMOVE_AND_RETRAIN);
  2. magnitude agreement between REMOVE_AND_RETRAIN and Tier-2-conditional within
     max(0.01, 0.5*|dC_remove_and_retrain|);
  3. REMOVE_AND_RETRAIN BH-FDR q < 0.05 (the arbiter; falls back to the strongest available
     tier's q when REMOVE_AND_RETRAIN is absent);
  4. global sanity checks pass (model-parameter randomization, enforced upstream);
  5. not an OOD artifact: |dC_tier1 - dC_remove_and_retrain| <= 0.02 (when both present).
Else: ENTANGLED (conditional collapses vs marginal, set upstream), UNFAITHFUL
(sign disagreement or fails sanity), or INCONCLUSIVE (ungated, wide-CI, or NaN).
"""

from __future__ import annotations

import numpy as np

OOD_TOL = 0.02


def _sign(x):
    return np.sign(np.where(np.isfinite(x), x, 0.0))


def bank(tier_means: dict, tier_q: dict, remove_and_retrain_key: str = "remove_and_retrain",
         cond_key: str = "tier2_cond", occ_key: str = "tier1",
         alpha: float = 0.05) -> dict:
    """tier_means/tier_q: {tier_name: (n_cells, P) arrays of dC mean / BH-q}.

    Missing tiers may be omitted. Returns decision codes per (cell, disease):
      2=confirmed, 1=suggestive(single-tier sig), 0=inconclusive, -1=unfaithful.
    """
    present = [k for k in (occ_key, cond_key, remove_and_retrain_key) if k in tier_means]
    shape = tier_means[present[0]].shape
    decision = np.zeros(shape, dtype=np.int8)

    # arbiter tier for significance (REMOVE_AND_RETRAIN > cond > occ)
    arb = remove_and_retrain_key if remove_and_retrain_key in tier_q else (cond_key if cond_key in tier_q else occ_key)
    q_arb = tier_q.get(arb, np.full(shape, np.nan))
    sig = q_arb < alpha

    # sign agreement across present tiers (treat ~0 / NaN as agnostic)
    signs = [_sign(tier_means[k]) for k in present]
    nz = [s != 0 for s in signs]
    # all non-zero signs must match
    agree = np.ones(shape, dtype=bool)
    base = None
    for s, m in zip(signs, nz):
        if base is None:
            base = np.where(m, s, 0)
        agree &= (~m) | (s == np.where(base != 0, base, s))
    disagree = ~agree

    # magnitude agreement REMOVE_AND_RETRAIN <-> cond
    mag_ok = np.ones(shape, dtype=bool)
    if remove_and_retrain_key in tier_means and cond_key in tier_means:
        dr, dc = tier_means[remove_and_retrain_key], tier_means[cond_key]
        tol = np.maximum(0.01, 0.5 * np.abs(dr))
        mag_ok = np.abs(dr - dc) <= tol

    # OOD guard tier1 <-> remove_and_retrain
    ood_ok = np.ones(shape, dtype=bool)
    if remove_and_retrain_key in tier_means and occ_key in tier_means:
        ood_ok = np.abs(tier_means[occ_key] - tier_means[remove_and_retrain_key]) <= OOD_TOL

    confirmed = sig & agree & mag_ok & ood_ok & (_sign(tier_means[arb]) > 0)
    suggestive = sig & agree & ~confirmed & (_sign(tier_means[arb]) > 0)
    unfaithful = disagree & sig

    decision[suggestive] = 1
    decision[confirmed] = 2
    decision[unfaithful] = -1
    return {"decision": decision, "arbiter": arb,
            "n_confirmed": int((decision == 2).sum()),
            "n_suggestive": int((decision == 1).sum()),
            "n_unfaithful": int((decision == -1).sum())}
