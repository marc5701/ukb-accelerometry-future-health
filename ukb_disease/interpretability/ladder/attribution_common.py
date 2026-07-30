"""Shared infrastructure for the faithfulness ladder.

Loads the frozen model(s), reconstructs the test outcomes directly from each
run's saved per-subject label arrays (no prevalence/survival re-derivation), and
provides on-manifold permutation-ablation plus per-disease dC scoring.

On-manifold ablation. To ablate a "cell" (a column block of the per-day
embedding) we permute that block across all valid (subject, day) slots in the
split. This preserves the cell's marginal distribution exactly (every value is a
real observed cell from some subject-day) while destroying its association with
this subject's outcome, i.e. permutation importance, the standard
remove-without-going-off-manifold operator. Zeroing, by contrast, pushes the
post-LayerNorm activation off the training manifold and conflates "lost
information" with "confused by an input never seen in training".

dC convention: dC = C_full - C_ablated (positive means the cell carries signal).
C-index sign matches per_disease_eval.c_index_lifelines (passes -hazards).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch

from ukb_disease.baseline.config import resolve_oak_path
from ukb_disease.baseline.cox_models import build_cox_model
from ukb_disease.baseline.window_dataset import WindowDataset, forward_subject_aggregated
from ukb_disease.baseline.outcomes_processing import load_covariates
from ukb_disease.baseline.train_rolling_window_cox import _aggregate_to_subject
from ukb_disease.baseline.per_disease_eval import c_index_lifelines
from ukb_disease.paths import UKB_ROOT

RUNS_ROOT = f"{UKB_ROOT}/runs_cox"
PRIMARY_MODEL_PREFIX = "embedding_disjoint_split"
SEEDS = [42, 123, 456, 789]
MEANC_MIN_EVAL = 10           # match the concordance benchmark / per_disease_eval evaluability gate
AR_L3_DIM = 768               # per-modality block = [L1||L2||L3] over 256


def primary_model_run_dir(seed: int) -> str:
    return f"{RUNS_ROOT}/{PRIMARY_MODEL_PREFIX}_seed{seed}"


def load_run_config(run_dir: str) -> dict:
    return json.load(open(resolve_oak_path(os.path.join(run_dir, "config.json"))))


def load_saved_test(run_dir: str) -> dict:
    d = resolve_oak_path(run_dir)
    return {k: np.load(os.path.join(d, f"test_{k}.npy"))
            for k in ("eids", "hazards", "event_times", "is_event", "eval_mask")}


def make_outcomes_from_saved(saved: dict) -> dict:
    """WindowDataset outcomes dict (eids unique-sorted; labels reordered to match)."""
    eids = saved["eids"].astype(np.int64)
    order = np.argsort(eids)
    return {
        "eids": eids[order],
        "event_times": saved["event_times"][order],
        "is_event": saved["is_event"][order],
        "eval_mask": saved["eval_mask"][order],
    }


def load_primary_model(run_dir: str, in_dim: int | None = None, device: str = "cpu",
                  random_init: bool = False):
    """Rebuild the model's exact arch from config.json and load best.pt.

    random_init=True keeps the freshly-initialized weights (skips best.pt), the
    model-parameter randomization sanity check (Adebayo et al.): a faithful
    attribution must collapse (dC near 0, C near 0.5) on a random model.
    """
    cfg = load_run_config(run_dir)
    a = cfg["args"]
    ck = torch.load(resolve_oak_path(os.path.join(run_dir, "best.pt")), map_location=device)
    sd = ck["model"]
    if in_dim is None:
        in_dim = int(sd["input_proj.weight"].shape[1])
    n_total = len(cfg["phecodes"]) + len(cfg.get("aux_phecodes", []) or [])
    model = build_cox_model(
        arch=a["arch"], window_size=a["window_size"], n_phecodes=n_total,
        n_covar=cfg.get("n_covar", 2), in_dim=in_dim, dropout=a.get("dropout", 0.1),
        n_layers=a.get("n_layers", 2), pe_type=a.get("pe_type", "learned"),
        input_norm=a.get("input_norm", False), output_rank=a.get("output_rank", 0),
        max_days=8, cov_bottleneck_dim=a.get("cov_bottleneck_dim", 0),
    )
    if not random_init:
        model.load_state_dict(sd)
    model.to(device).eval()
    # The TransformerEncoder nested-tensor fast path and TF32 give wrong ablation
    # forwards on cuda (permuted inputs collapse every cell to the age+sex floor,
    # confirmed against the CPU path). Disable both. Ablation forwards run on CPU
    # regardless; this also makes any cuda use safe.
    enc = getattr(model, "encoder", None)
    if enc is not None and hasattr(enc, "enable_nested_tensor"):
        enc.enable_nested_tensor = False
    if "cuda" in str(device):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return model, cfg, in_dim


def build_test_dataset(day_mean_root: str, outcomes: dict, covariates: pd.DataFrame,
                       window_size: int, split: str = "test", eid_filter=None):
    return WindowDataset(
        split=split, window_size=window_size, outcomes=outcomes, covariates=covariates,
        day_mean_root=day_mean_root, eager_load=True, stratified=False,
        eid_filter=eid_filter,
    )


# Cell specs: (tensor, start, end) where tensor is "ar" or "ha", indexing the
# dataset's separately-stored _ar/_ha blocks (window = concat([ar, ha])).
def primary_model_cells() -> dict:
    """Cells expressible as column slices of the model's 1536-d window."""
    return {
        "AR": ("ar", 0, AR_L3_DIM), "HA": ("ha", 0, AR_L3_DIM),
        "AR_L1": ("ar", 0, 256), "AR_L2": ("ar", 256, 512), "AR_L3": ("ar", 512, 768),
        "HA_L1": ("ha", 0, 256), "HA_L2": ("ha", 256, 512), "HA_L3": ("ha", 512, 768),
    }


def day_night_cells(n_bins: int = 2) -> dict:
    """Cells of the day/night stratified cache (ar/ha = [bin0_l3 || bin1_l3 || ...])."""
    labels = ["day", "night"] if n_bins == 2 else \
        ["late_night", "early_morning", "midday", "evening"]
    cells = {}
    for b, lab in enumerate(labels):
        s = b * AR_L3_DIM
        cells[f"AR_{lab}"] = ("ar", s, s + AR_L3_DIM)
        cells[f"HA_{lab}"] = ("ha", s, s + AR_L3_DIM)
    return cells


def _block_refs(dataset, spec):
    tname, s, e = spec
    arr = dataset._ar if tname == "ar" else dataset._ha
    return arr, s, e


def apply_permutation_ablation(dataset, specs: list, rng) -> dict:
    """Permute the given cell blocks across all valid (subject,day) slots, IN PLACE.

    Returns a restore dict so the dataset tensors can be reverted. All specs share
    ONE permutation of the valid-slot order, so a multi-cell ablation removes them
    jointly with a single coherent reshuffle.
    """
    valid = np.asarray(dataset._mask, dtype=bool)
    vi = np.argwhere(valid)                       # (n_valid, 2)
    perm = rng.permutation(len(vi))
    # Snapshot all original blocks before any modification. Specs may overlap in
    # column range (e.g. AR contains AR_L1/L2/L3). If we snapshotted during the
    # apply loop, a later overlapping spec would capture already-permuted values and
    # restore would write them back, corrupting the dataset for every subsequent
    # ablation (this produced a uniform-dC artifact across all cells). Two-pass
    # (snapshot-all, then apply-all from the true originals) makes restore exact.
    saved = []
    for spec in specs:
        arr, s, e = _block_refs(dataset, spec)
        # Guard against a cell-spec / cache-dim mismatch silently slicing the wrong
        # columns (numpy truncates out-of-range slices instead of erroring).
        if arr is None or e > arr.shape[-1]:
            raise ValueError(f"cell spec {spec} exceeds block dim "
                             f"{None if arr is None else arr.shape[-1]} (cache/cell-kind mismatch)")
        saved.append((arr, s, e, arr[vi[:, 0], vi[:, 1], s:e].copy()))
    for arr, s, e, orig in saved:
        arr[vi[:, 0], vi[:, 1], s:e] = orig[perm]
    return {"saved": saved, "vi": vi}


def restore_ablation(state: dict) -> None:
    vi = state["vi"]
    for arr, s, e, block in state["saved"]:
        arr[vi[:, 0], vi[:, 1], s:e] = block


def forward_aggregated(model, dataset, device: str = "cpu", batch_size: int = 256,
                       stride: int = 1) -> dict:
    """Frozen forward over all windows, then per-subject mean + collapse to subject."""
    out = forward_subject_aggregated(model, dataset, device=device,
                                     batch_size=batch_size, stride=stride)
    eids, haz, et, ie, em = _aggregate_to_subject(
        out["eids"], out["hazards"], out["event_times"], out["is_event"], out["eval_mask"])
    return {"eids": eids, "hazards": haz, "event_times": et, "is_event": ie, "eval_mask": em}


def hazards_with_ablation(model, dataset, specs: list | None, seed: int,
                          device: str = "cpu", batch_size: int = 256) -> dict:
    """Per-subject hazards under an (optional) permutation ablation of `specs`."""
    if not specs:
        return forward_aggregated(model, dataset, device=device, batch_size=batch_size)
    rng = np.random.default_rng(seed)
    state = apply_permutation_ablation(dataset, specs, rng)
    try:
        res = forward_aggregated(model, dataset, device=device, batch_size=batch_size)
    finally:
        restore_ablation(state)
    return res


def per_disease_c(res: dict, min_eval: int = MEANC_MIN_EVAL) -> np.ndarray:
    """Per-disease Harrell C from an aggregated result dict. NaN where ungated."""
    haz, et, ie, em = res["hazards"], res["event_times"], res["is_event"], res["eval_mask"]
    P = haz.shape[1]
    c = np.full(P, np.nan, dtype=np.float64)
    for j in range(P):
        m = em[:, j].astype(bool)
        if int(m.sum()) < min_eval:
            continue
        evj = ie[m, j]
        if int((evj > 0).sum()) < 1:
            continue
        c[j] = c_index_lifelines(haz[m, j], et[m, j], evj)
    return c


def mean_c(c: np.ndarray) -> float:
    return float(np.nanmean(c))
