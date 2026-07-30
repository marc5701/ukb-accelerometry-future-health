"""Ablation engine: per-disease dC for marginal, conditional, and joint cell
ablations, across K permutation seeds, for one model seed.

Works for both ladder tiers that operate on a frozen model.
  * Tier 1: the Embedding model on its own L-moment cache (cells = AR/HA plus L-moment
    slabs that are column-slices of the model's input).
  * Tier 2: the purpose-built day/night stratified model on the day/night cache
    (cells = {day,night} x {AR,HA}); marginal/conditional/joint defined there.

Definitions (dC = C_full - C_ablated; positive means the cell carries signal).
  conditional(cell) : permute only this cell, keep the rest (unique contribution)
  marginal(cell)    : permute everything except this cell (standalone value)
  joint(cellA,cellB): permute both (interaction probe)

Outputs one npz per (run_prefix, model_seed):
  c_full        (n_disease,)
  cond          (n_cells, K, n_disease)
  marg          (n_cells, K, n_disease)        [if --marginal]
  joint         (n_pairs, K, n_disease)        [if --joint]
  cell_names, pair_names, phecodes, eids, perm_seeds
Per-subject full hazards are also saved (for later EID-clustered bootstrap).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np
import torch

from ukb_disease.baseline.config import resolve_oak_path
from ukb_disease.interpretability.ladder import attribution_common as ic
from ukb_disease.paths import UKB_ROOT

OUT_ROOT_DEFAULT = f"{UKB_ROOT}/interpretability/ablation"


def _cells_for(kind: str, n_bins: int = 2) -> dict:
    if kind == "primary_model":
        return ic.primary_model_cells()
    if kind == "day_night":
        return ic.day_night_cells(n_bins=n_bins)
    raise ValueError(kind)


def run_one_seed(run_dir: str, cache: str, cell_kind: str, window_size: int,
                 perm_seeds: list[int], do_marginal: bool, do_joint: bool,
                 device: str, batch_size: int, n_bins: int = 2,
                 save_ablated_hazards: bool = False) -> dict:
    saved = ic.load_saved_test(run_dir)
    outcomes = ic.make_outcomes_from_saved(saved)
    cov = ic.load_covariates()
    model, cfg, in_dim = ic.load_primary_model(run_dir, device=device)
    ds = ic.build_test_dataset(resolve_oak_path(cache), outcomes, cov,
                               window_size=window_size, split="test")
    phecodes = cfg["phecodes"]
    cells = _cells_for(cell_kind, n_bins=n_bins)
    cell_names = list(cells)
    K = len(perm_seeds)
    P = len(phecodes)

    full_res = ic.forward_aggregated(model, ds, device=device, batch_size=batch_size)
    c_full = ic.per_disease_c(full_res)
    print(f"[ablation] {os.path.basename(run_dir)}: full mean-C={ic.mean_c(c_full):.4f} "
          f"cells={len(cell_names)} K={K}")

    n_subj = len(full_res["eids"])
    cond = np.full((len(cell_names), K, P), np.nan, dtype=np.float64)
    marg = np.full((len(cell_names), K, P), np.nan, dtype=np.float64) if do_marginal else None
    cond_haz = (np.zeros((len(cell_names), n_subj, P), dtype=np.float32)
                if save_ablated_hazards else None)
    t0 = time.time()
    for ci, cname in enumerate(cell_names):
        spec = cells[cname]
        others = [cells[o] for o in cell_names if o != cname]
        for ki, ps in enumerate(perm_seeds):
            res_c = ic.hazards_with_ablation(model, ds, [spec], seed=ps,
                                             device=device, batch_size=batch_size)
            cond[ci, ki] = ic.per_disease_c(res_c)
            if save_ablated_hazards and ki == 0:
                cond_haz[ci] = res_c["hazards"].astype(np.float32)
            if do_marginal:
                marg[ci, ki] = ic.per_disease_c(
                    ic.hazards_with_ablation(model, ds, others, seed=ps,
                                             device=device, batch_size=batch_size))
        print(f"[ablation]  cell {cname}: cond mean dC="
              f"{ic.mean_c(c_full)-np.nanmean(cond[ci]):+.4f}  ({time.time()-t0:.0f}s)")

    pair_names, joint = [], None
    if do_joint:
        pairs = list(itertools.combinations(cell_names, 2))
        pair_names = [f"{a}+{b}" for a, b in pairs]
        joint = np.full((len(pairs), K, P), np.nan, dtype=np.float64)
        for pi, (a, b) in enumerate(pairs):
            for ki, ps in enumerate(perm_seeds):
                joint[pi, ki] = ic.per_disease_c(
                    ic.hazards_with_ablation(model, ds, [cells[a], cells[b]], seed=ps,
                                             device=device, batch_size=batch_size))
        print(f"[ablation]  {len(pairs)} joint pairs done ({time.time()-t0:.0f}s)")

    return {
        "c_full": c_full, "cond": cond,
        "marg": marg if marg is not None else np.zeros(0),
        "joint": joint if joint is not None else np.zeros(0),
        "cond_haz": cond_haz if cond_haz is not None else np.zeros(0),
        "cell_names": np.array(cell_names), "pair_names": np.array(pair_names),
        "phecodes": np.array(phecodes), "eids": full_res["eids"],
        "perm_seeds": np.array(perm_seeds),
        "full_hazards": full_res["hazards"].astype(np.float32),
        "full_event_times": full_res["event_times"].astype(np.float32),
        "full_is_event": full_res["is_event"].astype(np.float32),
        "full_eval_mask": full_res["eval_mask"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-prefix", default=ic.PRIMARY_MODEL_PREFIX,
                    help="run-dir prefix under runs_cox (e.g. embedding_disjoint_split, dn_strat).")
    ap.add_argument("--model-seed", type=int, required=True)
    ap.add_argument("--cache", required=True, help="day_mean_root the model reads.")
    ap.add_argument("--cell-kind", default="primary_model", choices=["primary_model", "day_night"])
    ap.add_argument("--n-bins", type=int, default=2)
    ap.add_argument("--window-size", type=int, default=3)
    ap.add_argument("--perm-seeds", type=int, nargs="+",
                    default=list(range(10)))
    ap.add_argument("--marginal", action="store_true")
    ap.add_argument("--joint", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--save-ablated-hazards", action="store_true",
                    help="Also save per-subject ablated hazards (perm-seed 0) per cell "
                         "for EID-clustered bootstrap CIs downstream.")
    ap.add_argument("--out-root", default=OUT_ROOT_DEFAULT)
    args = ap.parse_args()

    run_dir = f"{ic.RUNS_ROOT}/{args.run_prefix}_seed{args.model_seed}"
    res = run_one_seed(run_dir, args.cache, args.cell_kind, args.window_size,
                       args.perm_seeds, args.marginal, args.joint,
                       args.device, args.batch_size, n_bins=args.n_bins,
                       save_ablated_hazards=args.save_ablated_hazards)
    out_dir = resolve_oak_path(os.path.join(args.out_root, args.run_prefix))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"ablation_seed{args.model_seed}.npz")
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **res)
    os.rename(tmp, out_path)
    print(f"[ablation] wrote {out_path}")


if __name__ == "__main__":
    main()
