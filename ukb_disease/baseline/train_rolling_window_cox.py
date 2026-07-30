"""Rolling-window Cox trainer: rolling-window sequence head over per-day mean embeddings.

Trains one cell of the 18-cell sequence grid:
    arch       ∈ {mlp_flat, lstm, transformer}
    window     ∈ {3, 5, 7} days
    curriculum ∈ {uniform, pos_enriched}

Plus 1 control run handled by `run_baseline.py --curriculum pos_enriched`.

Pipeline: WindowDataset (random valid N-day window per subject per epoch
during training; all valid windows averaged at inference), then
cox_models.build_cox_model, then AdamW + cosine LR + multilabel CoxPH
loss (in-batch `cox_ph_loss`). Per-epoch checkpoint, SIGTERM-resilient.

Outputs per run at `<out_root>/<run_name>/`:
    best.pt              best-val checkpoint (state_dict + meta)
    latest.pt            most-recent epoch (for preempt resume)
    test_hazards.npy     (N_test, P) float32 average log-hazards
    test_event_times.npy (N_test, P) float32
    test_is_event.npy    (N_test, P) float32
    test_eval_mask.npy   (N_test, P) bool
    test_eids.npy        (N_test,) int64
    train_history.csv    per-epoch loss + lr + patience
    config.json          args + phecode list + stats
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Cox partial-likelihood loss is implemented locally as _cox_ph_loss_weighted (below).

from ukb_disease.paths import UKB_ROOT  # noqa: E402
from ukb_disease.baseline.config import (  # noqa: E402
    CANONICAL_DAY_MEAN_ROOT,
    CIRCADIAN_COV_ROOT,
    DAY_MEAN_ROOT,
    DAY_MEAN_STRAT_ROOT,
    MIN_PREVALENCE,
    MISSINGNESS_COV_ROOT,
    RUNS_COX_ROOT,
    resolve_oak_path,
)
from ukb_disease.baseline.outcomes_processing import (  # noqa: E402
    build_survival_arrays,
    filter_phecodes_by_prevalence,
    load_covariates,
    load_phecodes,
)
from ukb_disease.baseline.cox_models import build_cox_model  # noqa: E402
from ukb_disease.baseline.loss_ranking import multilabel_ranking_loss  # noqa: E402
from ukb_disease.baseline.loss_discrete import (  # noqa: E402
    build_time_bins,
    discrete_hazard_loss,
    logits_to_risk,
)
from ukb_disease.baseline.hierarchy_labels import build_aux_columns  # noqa: E402
from ukb_disease.baseline.sampler import (  # noqa: E402
    make_per_disease_balanced_sampler,
    make_positive_enriched_sampler,
)
from ukb_disease.baseline.window_dataset import (  # noqa: E402
    WindowDataset,
    collate_fn,
    forward_subject_aggregated,
)
from ukb_disease.baseline.flex_window_dataset import (  # noqa: E402
    FlexibleWindowDataset,
    forward_subject_flex,
)


SIGTERM_RECEIVED = {"flag": False}


def _install_sigterm_handler() -> None:
    def _h(signum, frame):
        SIGTERM_RECEIVED["flag"] = True
        print(f"[train_rolling_window_cox] caught signal {signum}; will save+exit at next epoch boundary",
              flush=True)
    signal.signal(signal.SIGTERM, _h)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _atomic_torch_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.rename(tmp, path)


def default_run_name(args) -> str:
    if args.curriculum == "pos_enriched":
        short_curric = "posenr"
    elif args.curriculum == "per_disease_balanced":
        tgt = (args.target_phecode or "none").replace(".", "p").replace("'", "p")
        short_curric = f"pdbal_{tgt}"
    else:
        short_curric = "uniform"
    return f"{args.arch}_w{args.window_size}d_{short_curric}"


@torch.no_grad()
def evaluate_dataset(
    model: torch.nn.Module,
    dataset: WindowDataset,
    device: str,
    batch_size: int = 64,
    stride: int = 1,
    *,
    discrete: bool = False,
    n_leaf: int | None = None,
    n_total: int | None = None,
    n_bins: int = 1,
    cuts_t: torch.Tensor | None = None,
    flexible: bool = False,
) -> tuple[float, np.ndarray]:
    """Forward all valid windows per subject (averaged).

    Returns (leaf_val_loss, per_phecode_score) where per_phecode_score is
    (N, n_total): the aggregated Cox log-hazard, or for discrete the cumulative
    -hazard risk score `logits_to_risk`. Early stopping uses the LEAF loss only
    (aux heads exist purely to shape the trunk).
    """
    n_total = n_total if n_total is not None else dataset.n_phecodes
    n_leaf = n_leaf if n_leaf is not None else n_total
    out_dim = (n_total * n_bins) if discrete else n_total
    if flexible:
        pack = forward_subject_flex(
            model, dataset, device=device, batch_size=batch_size, out_dim=out_dim,
        )
    else:
        pack = forward_subject_aggregated(
            model, dataset, device=device, batch_size=batch_size, stride=stride,
            out_dim=out_dim,
        )
    haz = torch.from_numpy(pack["hazards"]).to(device)      # (N, out_dim)
    t = torch.from_numpy(pack["event_times"]).to(device)    # (N, n_total)
    e = torch.from_numpy(pack["is_event"]).to(device)       # (N, n_total)
    if discrete:
        h3 = haz.view(haz.shape[0], n_total, n_bins)
        leaf_loss = float(discrete_hazard_loss(
            h3[:, :n_leaf], t[:, :n_leaf], e[:, :n_leaf], cuts_t).item())
        risk = logits_to_risk(pack["hazards"], n_bins)      # (N, n_total)
        if risk.ndim == 1:  # single-target (n_total==1): logits_to_risk squeezes to (N,)
            risk = risk[:, None]
        return leaf_loss, risk
    leaf_loss = float(_cox_ph_loss_weighted(haz[:, :n_leaf], t[:, :n_leaf], e[:, :n_leaf]).item())
    return leaf_loss, pack["hazards"]


def _build_scheduler(args, optim, n_batches_per_epoch: int):
    """LR-scheduler factory dispatched on --schedule.

    Returns (scheduler, step_mode) where step_mode ∈ {"epoch", "batch",
    "plateau"} tells the train loop when/how to step.
    """
    T_max_epochs = args.schedule_T_max if args.schedule_T_max > 0 else args.epochs
    if args.schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=T_max_epochs)
        return sched, "epoch"
    if args.schedule == "cosine_warmup":
        warmup_steps = max(1, int(args.warmup_steps))
        total_steps = T_max_epochs * max(1, int(n_batches_per_epoch))
        if warmup_steps >= total_steps:
            raise ValueError(
                f"warmup_steps {warmup_steps} >= total_steps {total_steps}"
            )
        post_steps = total_steps - warmup_steps
        warmup = torch.optim.lr_scheduler.LinearLR(
            optim, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=post_steps)
        sched = torch.optim.lr_scheduler.SequentialLR(
            optim, schedulers=[warmup, cosine], milestones=[warmup_steps],
        )
        return sched, "batch"
    if args.schedule == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode="min", factor=0.5, patience=3, threshold=1e-4,
        )
        return sched, "plateau"
    raise ValueError(f"Unknown schedule '{args.schedule}'")


def _current_lr(optim) -> float:
    """Get the current optimizer LR. Works for all schedulers (including
    ReduceLROnPlateau, which lacks get_last_lr in some PyTorch versions)."""
    return float(optim.param_groups[0]["lr"])


def _aggregate_to_subject(eids, hazards, event_times, is_event, eval_mask):
    """Collapse per-recording rows to one row per subject for held-out scoring.

    With the `_all` cache a subject contributes one row per recording (duplicate
    eids), which over-counts multi-recording subjects in the C-index. Mean the
    hazards across a subject's recordings; labels are per-subject so take the
    first row's. Returns arrays sorted by eid; a no-op when eids are unique.
    """
    eids = np.asarray(eids)
    uniq, first = np.unique(eids, return_index=True)
    if len(uniq) == len(eids):
        return eids, hazards, event_times, is_event, eval_mask
    inv = np.searchsorted(uniq, eids)
    haz_sum = np.zeros((len(uniq), hazards.shape[1]), dtype=np.float64)
    cnt = np.zeros(len(uniq), dtype=np.float64)
    np.add.at(haz_sum, inv, hazards.astype(np.float64))
    np.add.at(cnt, inv, 1.0)
    haz_u = (haz_sum / cnt[:, None]).astype(np.float32)
    return uniq, haz_u, event_times[first], is_event[first], eval_mask[first]


def _apply_freeze(model: torch.nn.Module, mode: str) -> None:
    """Partial-freeze for single-target specialization fine-tuning.

    Freezes everything, then unfreezes the final head (always) plus, per `mode`,
    the last transformer encoder block(s) + the final LayerNorm. Targets the
    TransformerSeqHead module names (`head`, `norm`, `encoder.layers[-N:]`); a
    no-op for arches lacking those (only `head` is unfrozen).
    """
    for p in model.parameters():
        p.requires_grad = False

    def _unfreeze(mod):
        if mod is None:
            return
        for p in mod.parameters():
            p.requires_grad = True

    _unfreeze(getattr(model, "head", None))  # head always trains
    if mode == "head_only":
        return
    _unfreeze(getattr(model, "norm", None))
    enc = getattr(model, "encoder", None)
    layers = getattr(enc, "layers", None) if enc is not None else None
    if layers is not None and len(layers) > 0:
        if mode == "last_block":
            _unfreeze(layers[-1])
        elif mode == "last2_block":
            for L in list(layers)[-2:]:
                _unfreeze(L)


def _warm_start(model: torch.nn.Module, ckpt_path: str, device: str,
                target_phecode: str | None, single_output: bool) -> None:
    """Shape-filtered warm-start from a a pretrained best.pt.

    Copies every source tensor whose name + shape matches the destination; skips
    the rest (e.g. a 283-wide final head when the destination is single-output).
    For a single-output head, warm-initializes the lone output row from the
    source's `target_phecode` column so the head starts hot rather than random.
    """
    ck = torch.load(resolve_oak_path(ckpt_path), map_location=device)
    src_sd = ck["model"]
    src_phecodes = list(ck.get("phecodes", []))
    own = model.state_dict()
    loaded, skipped = [], []
    for k, v in src_sd.items():
        if k in own and own[k].shape == v.shape:
            own[k] = v
            loaded.append(k)
        else:
            skipped.append(k)
    model.load_state_dict(own)
    warm_row = None
    if (single_output and target_phecode and target_phecode in src_phecodes
            and isinstance(getattr(model, "head", None), torch.nn.Sequential)):
        tcol = src_phecodes.index(target_phecode)
        final = model.head[-1]
        sw = src_sd.get("head.4.weight")
        sb = src_sd.get("head.4.bias")
        if (isinstance(final, torch.nn.Linear) and final.weight.shape[0] == 1
                and sw is not None and sw.shape[0] > tcol):
            with torch.no_grad():
                final.weight.copy_(sw[tcol:tcol + 1].to(final.weight.device))
                if sb is not None and final.bias is not None:
                    final.bias.copy_(sb[tcol:tcol + 1].to(final.bias.device))
            warm_row = tcol
    print(f"[train_rolling_window_cox] warm-start from {ckpt_path}: loaded {len(loaded)} / "
          f"skipped {len(skipped)} tensors (skipped e.g. {skipped[:3]}); "
          f"head_warm_row={warm_row}")


def _cox_ph_loss_weighted(hazards, event_times, is_event, label_weights=None):
    """In-batch Cox partial-likelihood with optional PER-LABEL weights.

    Reduces to the plain in-batch `cox_ph_loss` when `label_weights is None`
    (unweighted mean over labels). With `label_weights` (a (P,) tensor), returns
    the weighted mean of per-label losses, used to UPWEIGHT a target disease
    inside the joint multilabel model (keeping the other labels as regularizers).
    """
    et, idx = event_times.sort(dim=0, descending=True)
    h = hazards.gather(0, idx)
    ev = is_event.gather(0, idx)
    lch = torch.logcumsumexp(h.float(), dim=0)
    losses = -((h - lch) * ev)
    label_loss = losses.sum(dim=0) / (ev.sum(dim=0) + 1e-9)
    if label_weights is None:
        return label_loss.mean()
    return (label_loss * label_weights).sum() / label_weights.sum()


def train_one_run(args) -> dict:
    # PREDICT-ONLY: rebuild the source run's exact architecture by overriding model/data args
    # from its config.json (so best.pt loads cleanly), keeping only our output/device flags.
    if getattr(args, "predict_only", False) and getattr(args, "init_from", None):
        _init = Path(resolve_oak_path(args.init_from))
        _saved = json.load(open(_init / "config.json"))["args"]
        _keep = {"out_root", "run_name", "seed", "device", "predict_only", "init_from",
                 "save_val_artifacts"}
        for _k, _v in _saved.items():
            if _k not in _keep and hasattr(args, _k):
                setattr(args, _k, _v)
        print(f"[train_rolling_window_cox] PREDICT-ONLY: arch/data args loaded from {_init}/config.json")
    set_seed(args.seed)
    device = args.device
    run_name = args.run_name or default_run_name(args)
    out_dir = Path(resolve_oak_path(args.out_root)) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train_rolling_window_cox] arch={args.arch} window={args.window_size} "
          f"curriculum={args.curriculum} run_name={run_name}")
    print(f"[train_rolling_window_cox] out_dir={out_dir}")
    print(f"[train_rolling_window_cox] day_mean_root={args.day_mean_root}")

    # --- Load labels + covariates
    df_pheco = load_phecodes()
    if args.no_demographics and (args.missingness_covariates or args.circadian_covariates):
        raise SystemExit("--no-demographics is incompatible with --missingness-covariates "
                         "/ --circadian-covariates (those covariates would be dropped).")
    if args.no_demographics and args.keep_covariates:
        raise SystemExit("--no-demographics and --keep-covariates are mutually exclusive.")
    df_cov = load_covariates(
        missingness_csv=(resolve_oak_path(os.path.join(MISSINGNESS_COV_ROOT, "all.csv"))
                         if args.missingness_covariates else None),
        missingness_cols=args.missingness_cols,
        circadian_csv=(resolve_oak_path(args.circadian_csv
                                        or os.path.join(CIRCADIAN_COV_ROOT, "all.csv"))
                       if args.circadian_covariates else None),
        circadian_cols=args.circadian_cols,
    )
    if args.add_bmi:
        # Join clinical BMI (UKB field 21001) as a covariate for the
        # clinical+embedding model. z-score, then impute NaN to 0 (the mean
        # after centering).
        bmi_path = resolve_oak_path(f"{UKB_ROOT}/metadata/BMI_yes.csv")
        bmi = (pd.read_csv(bmi_path, usecols=["eid", "21001-0.0"])
               .rename(columns={"21001-0.0": "bmi"}).groupby("eid").mean())
        bmi_z = (bmi["bmi"] - bmi["bmi"].mean()) / (bmi["bmi"].std() + 1e-9)
        df_cov = df_cov.join(bmi_z.rename("bmi"))
        df_cov["bmi"] = df_cov["bmi"].fillna(0.0)
        print(f"[train_rolling_window_cox] --add-bmi: joined BMI (coverage {bmi['bmi'].notna().mean():.2f}), "
              f"cov cols now {list(df_cov.columns)}")
    if args.no_demographics:
        # Wrist-only: keep the eid index (so subjects aren't dropped by the
        # covariate filter) but drop all covariate columns for n_covar=0.
        df_cov = df_cov.iloc[:, :0]
        print("[train_rolling_window_cox] --no-demographics: wrist-only model, n_covar=0")
    elif args.keep_covariates:
        # Covariate-subset ablation (e.g. age-only / sex-only): keep ONLY the
        # named columns so n_covar = len(kept). Used to attribute the demographic
        # contribution between age and sex (n_covar auto-flows via window_dataset).
        missing = [c for c in args.keep_covariates if c not in df_cov.columns]
        if missing:
            raise SystemExit(f"--keep-covariates {missing} not in covariate columns "
                             f"{list(df_cov.columns)}")
        df_cov = df_cov[list(args.keep_covariates)]
        print(f"[train_rolling_window_cox] --keep-covariates: kept {list(args.keep_covariates)}, "
              f"n_covar={df_cov.shape[1]}")
    # --- Optional non-linear age featurisation (head ablation, encoder frozen).
    #     Replaces the raw scalar 'age' covariate with a non-linear basis so the
    #     head can learn a non-linear age MARGINAL. Default 'none' is unchanged. ---
    if getattr(args, "age_basis", "none") != "none":
        if "age" not in df_cov.columns:
            raise SystemExit("--age-basis requires an 'age' covariate column "
                             "(incompatible with --no-demographics).")
        _a = df_cov["age"].to_numpy(float)
        _z = (_a - np.nanmean(_a)) / (np.nanstd(_a) + 1e-9)
        _k = int(args.age_basis_k)
        if args.age_basis == "bins":
            edges = np.nanquantile(_z, np.linspace(0, 1, _k + 1))
            edges[0] -= 1e-6; edges[-1] += 1e-6
            b = np.clip(np.digitize(_z, edges[1:-1]), 0, _k - 1)
            Bm = np.zeros((len(_z), _k), np.float32); Bm[np.arange(len(_z)), b] = 1.0
        else:  # rbf
            centers = np.linspace(np.nanmin(_z), np.nanmax(_z), _k)
            width = (centers[1] - centers[0]) if _k > 1 else 1.0
            Bm = np.exp(-0.5 * ((_z[:, None] - centers[None, :]) / (width + 1e-9)) ** 2).astype(np.float32)
        Bm[~np.isfinite(_z)] = 0.0
        cols = [f"age_b{i}" for i in range(Bm.shape[1])]
        basis = pd.DataFrame(Bm, index=df_cov.index, columns=cols)
        others = [c for c in df_cov.columns if c != "age"]
        df_cov = pd.concat([basis, df_cov[others]], axis=1)
        print(f"[train_rolling_window_cox] --age-basis {args.age_basis} k={_k}: age -> {len(cols)} cols, "
              f"n_covar={df_cov.shape[1]}")
    print(f"[train_rolling_window_cox] phecode_csv rows: {len(df_pheco)}  cov_csv rows: {len(df_cov)}  "
          f"n_covar_cols: {df_cov.shape[1]}")

    # --- Pre-load day_mean index for train/test to enumerate eligible eids
    def _load_index(split: str) -> pd.DataFrame:
        p = resolve_oak_path(os.path.join(args.day_mean_root, split, "index.parquet"))
        return pd.read_parquet(p)

    train_idx_df = _load_index(args.train_split)
    test_idx_df = _load_index(args.test_split)
    train_eids_all = np.intersect1d(
        np.intersect1d(train_idx_df["eid"].to_numpy(dtype=np.int64),
                       df_pheco.index.to_numpy()),
        df_cov.index.to_numpy(),
    )
    test_eids_all = np.intersect1d(
        np.intersect1d(test_idx_df["eid"].to_numpy(dtype=np.int64),
                       df_pheco.index.to_numpy()),
        df_cov.index.to_numpy(),
    )
    # --- Explicit val partition (subject-disjoint), if requested
    val_eids_all = None
    if args.use_holdout_val:
        val_dir = resolve_oak_path(os.path.join(args.day_mean_root, "val"))
        if not os.path.isdir(val_dir):
            raise SystemExit(f"--use-holdout-val needs a 'val' subdir under "
                             f"{args.day_mean_root} (not found: {val_dir}). "
                             "Build it via prepare_fixed_split_npz + aggregate_day_mean.")
        val_idx_df = _load_index("val")
        val_eids_all = np.intersect1d(
            np.intersect1d(val_idx_df["eid"].to_numpy(dtype=np.int64),
                           df_pheco.index.to_numpy()),
            df_cov.index.to_numpy(),
        )
        print(f"[train_rolling_window_cox] explicit val split: {len(val_eids_all)} subjects "
              f"(early-stopping; full train pool retained)")
    if args.max_train:
        rng = np.random.default_rng(args.seed)
        train_eids_all = rng.permutation(train_eids_all)[: args.max_train]
        train_eids_all.sort()
    print(f"[train_rolling_window_cox] resolved subjects (pre-window-filter): "
          f"train={len(train_eids_all)} test={len(test_eids_all)}")

    # --- Filter phecodes by prevalence on train pool
    phecodes, stats = filter_phecodes_by_prevalence(
        df_pheco,
        train_eids_all,
        min_prevalence=args.min_prevalence,
        include_prevalent_in_count=not args.incident_only,
    )
    if args.max_phecodes:
        phecodes = phecodes[: args.max_phecodes]
    # Optional disease-category subset filter (e.g. GI-only fine-tune).
    if args.phecode_prefix_filter:
        before = len(phecodes)
        phecodes = [p for p in phecodes if p.startswith(args.phecode_prefix_filter)]
        print(f"[train_rolling_window_cox] phecode prefix filter '{args.phecode_prefix_filter}': "
              f"{before} -> {len(phecodes)} kept")
    # Single-target specialization: restrict the head to EXACTLY one phecode.
    if args.phecode_exact:
        before = len(phecodes)
        if args.phecode_exact not in phecodes:
            raise RuntimeError(
                f"--phecode-exact '{args.phecode_exact}' did not survive the "
                f"prevalence/prefix filter (kept {before}). Check spelling, "
                f"--min-prevalence, or --phecode-prefix-filter.")
        phecodes = [args.phecode_exact]
        print(f"[train_rolling_window_cox] phecode exact filter '{args.phecode_exact}': "
              f"{before} -> 1 kept")
    print(f"[train_rolling_window_cox] phecodes kept: {len(phecodes)}  stats={stats}")
    if len(phecodes) == 0:
        raise RuntimeError("No phecodes passed prevalence filter; relax --min-prevalence "
                            "or check --phecode-prefix-filter.")

    # --- Phecode-hierarchy auxiliary columns (trained jointly, dropped at
    #     inference). Built from the FULL df_pheco BEFORE trimming to leaves.
    leaf_phecodes = list(phecodes)
    n_leaf = len(leaf_phecodes)
    aux_names: list[str] = []
    if args.aux_hierarchy:
        aux_names, df_aux = build_aux_columns(
            df_pheco, train_eids_all,
            min_prevalence=args.aux_min_prevalence,
            categories_only=args.aux_categories_only,
            seven_day_years=args.buffer_years,
        )
        print(f"[train_rolling_window_cox] aux-hierarchy: {len(aux_names)} aux columns "
              f"(categories_only={args.aux_categories_only}, "
              f"min_prev={args.aux_min_prevalence})")
        df_combined = pd.concat([df_pheco[leaf_phecodes], df_aux[aux_names]], axis=1)
    else:
        df_combined = df_pheco[leaf_phecodes].copy()
    phecodes_all = leaf_phecodes + aux_names
    n_total = len(phecodes_all)
    del df_pheco
    import gc; gc.collect()

    # --- Survival arrays (per-split rules from outcomes_processing.py). The aux
    #     aggregate columns get the exact same incident/zone/prevalent/censored
    #     treatment as leaves (their cell value is the per-subject nanmin over
    #     children's raw times).
    arrs = build_survival_arrays(df_combined, phecodes_all, train_eids_all, test_eids_all,
                                 seven_day_years=args.buffer_years, val_eids=val_eids_all)
    print(f"[train_rolling_window_cox] max_follow_up = {arrs['max_follow_up']:.3f} years  "
          f"n_leaf={n_leaf} n_aux={len(aux_names)} n_total={n_total}")

    # Negative control: permute the target phecode's (is_event, event_time) pairs
    # across TRAIN subjects (val/test untouched). A model trained on this must NOT
    # beat the baseline on the target, confirming the eval/pipeline isn't leaking.
    if args.permute_target_phecode:
        if args.permute_target_phecode not in phecodes_all:
            raise SystemExit(
                f"--permute-target-phecode '{args.permute_target_phecode}' not in "
                f"kept phecodes ({len(phecodes_all)}).")
        pcol = phecodes_all.index(args.permute_target_phecode)
        prng = np.random.default_rng(args.permute_seed)
        perm = prng.permutation(arrs["train"]["is_event"].shape[0])
        arrs["train"]["is_event"][:, pcol] = arrs["train"]["is_event"][perm, pcol]
        arrs["train"]["event_times"][:, pcol] = arrs["train"]["event_times"][perm, pcol]
        print(f"[train_rolling_window_cox] PERMUTED train labels for "
              f"{args.permute_target_phecode} (col {pcol}, seed {args.permute_seed}); "
              f"negative control; val/test labels untouched")

    # --- Discrete-time bins (from train incident events). n_out_per_phecode
    #     = T for the discrete head, 1 for Cox.
    discrete = (args.loss == "discrete")
    n_bins = int(args.n_time_bins) if discrete else 1
    cuts_np = None
    cuts_t = None
    if discrete:
        cuts_np = build_time_bins(
            arrs["train"]["event_times"], arrs["train"]["is_event"],
            n_bins=n_bins, seven_day_years=args.buffer_years,
        )
        cuts_t = torch.from_numpy(cuts_np).to(args.device).float()
        print(f"[train_rolling_window_cox] discrete-time: n_bins={n_bins} "
              f"cuts={np.round(cuts_np, 3).tolist()}")

    # --- Validation split: explicit partition, or seeded 10% carve-out
    if args.use_holdout_val:
        tr_eids = np.asarray(train_eids_all, dtype=np.int64)
        val_eids = np.asarray(val_eids_all, dtype=np.int64)
        _ov = np.intersect1d(tr_eids, val_eids)
        assert _ov.size == 0, f"train∩val leak: {_ov.size} subjects (e.g. {_ov[:5].tolist()})"
        assert np.intersect1d(val_eids, test_eids_all).size == 0, "val∩test leak"
    else:
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(train_eids_all))
        n_val = max(1, int(len(train_eids_all) * args.val_frac))
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        val_eids = np.asarray(train_eids_all)[val_idx]
        tr_eids = np.asarray(train_eids_all)[tr_idx]

    # --- Load the OOF ar_emb target (train-only aux regression), with guards.
    ar_emb_series = None
    if args.aux_age_target:
        if args.loss != "cox":
            raise SystemExit("--aux-age-target is implemented for --loss cox only")
        if args.aux_hierarchy:
            raise SystemExit("--aux-age-target is incompatible with --aux-hierarchy")
        if args.flexible:
            raise SystemExit("--aux-age-target is incompatible with --flexible")
        ar_emb_series = pd.read_parquet(resolve_oak_path(args.aux_age_parquet)).set_index("eid")["ar_emb_z"]
        print(f"[train_rolling_window_cox] aux-age: {len(ar_emb_series)} OOF ar_emb targets "
              f"(weight={args.aux_age_weight})")

    # --- Datasets (drops subjects with <window_size valid days)
    ds_kwargs = dict(
        window_size=args.window_size,
        covariates=df_cov,
        day_mean_root=args.day_mean_root,
        rng_seed=args.seed,
        stratified=args.stratified,
    )
    DS = FlexibleWindowDataset if args.flexible else WindowDataset
    _aux_kw = {"aux_age": ar_emb_series} if args.aux_age_target else {}
    train_ds = DS(
        split=args.train_split, outcomes=arrs["train"], eid_filter=tr_eids, **_aux_kw, **ds_kwargs,
    )
    val_ds = DS(
        split=("val" if args.use_holdout_val else args.train_split),
        outcomes=(arrs["val"] if args.use_holdout_val else arrs["train"]),
        eid_filter=val_eids, **ds_kwargs,
    )
    test_ds = DS(
        split=args.test_split, outcomes=arrs["test"], **ds_kwargs,
    )
    model_max_days = int(getattr(train_ds, "d_max", 8)) if args.flexible else 8
    print(f"[train_rolling_window_cox] datasets (post window-filter): "
          f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    # --- Locate the interleaved covariance columns (the last C of EACH of the
    #     two modality blocks in the AR|HA concat). For meanstd_cov each block
    #     is [mean256, std256, cov136], so cov = [512:648] and [1160:1296], for
    #     the split-projection bottleneck. cov_bottleneck_dim==0 leaves cov_idx None.
    cov_idx = None
    if args.cov_bottleneck_dim > 0:
        _in_dim = int(train_ds.in_dim)
        _C = int(args.cov_cols_per_modality)
        if _in_dim % 2 != 0 or _in_dim // 2 <= _C:
            raise ValueError(
                f"--cov-bottleneck-dim set but in_dim={_in_dim} is not splittable into "
                f"two modality blocks each wider than --cov-cols-per-modality={_C}.")
        _per_mod = _in_dim // 2
        cov_idx = list(range(_per_mod - _C, _per_mod)) + list(range(_in_dim - _C, _in_dim))
        print(f"[train_rolling_window_cox] cov-bottleneck: dim={args.cov_bottleneck_dim} "
              f"cov_cols={len(cov_idx)} (last {_C}/modality) "
              f"base_cols={_in_dim - len(cov_idx)}")

    # --- Model
    model = build_cox_model(
        arch=args.arch,
        window_size=args.window_size,
        n_phecodes=n_total,
        n_covar=train_ds.n_covar,
        in_dim=train_ds.in_dim,  # 512 (mean) / 1024 (distrib or stratified)
        dropout=args.dropout,
        n_layers=args.n_layers,
        pe_type=args.pe_type,
        drop_path=args.drop_path,
        n_out_per_phecode=n_bins,
        input_norm=args.input_norm,
        output_rank=args.output_rank,
        max_days=model_max_days,
        cov_bottleneck_dim=args.cov_bottleneck_dim,
        cov_idx=cov_idx,
        across_day_pool=args.across_day_pool,
        pma_seeds_day=args.pma_seeds_day,
        aux_age=args.aux_age_target,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train_rolling_window_cox] model params: {n_params:,}  in_dim={train_ds.in_dim}  "
          f"head_out={n_total * n_bins}")

    # --- Warm-start (shape-filtered) from a pretrained ckpt,
    #     then optional partial freeze for single-target specialization. Done
    #     BEFORE the optimizer is built so frozen params are excluded.
    if args.warm_start_ckpt:
        _warm_start(model, args.warm_start_ckpt, device,
                    target_phecode=args.phecode_exact,
                    single_output=(n_total == 1 and n_bins == 1))
    if args.freeze_mode and args.freeze_mode != "none":
        _apply_freeze(model, args.freeze_mode)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[train_rolling_window_cox] freeze_mode={args.freeze_mode}: "
              f"trainable params={n_trainable:,} / {n_params:,}")

    # --- Regularized target-upweight: weight ONE leaf phecode's Cox loss by alpha
    #     inside the JOINT model (the other leaves stay weight 1 = regularizers).
    leaf_weights = None
    if args.target_upweight:
        if args.loss != "cox":
            raise SystemExit("--target-upweight is implemented for --loss cox only")
        if args.target_upweight not in leaf_phecodes:
            raise SystemExit(f"--target-upweight '{args.target_upweight}' not in kept "
                             f"leaf phecodes ({len(leaf_phecodes)}).")
        if n_leaf < 2:
            raise SystemExit("--target-upweight needs >=2 leaf phecodes (it upweights "
                             "the target WITHIN a joint model). Don't combine with "
                             "--phecode-exact.")
        leaf_weights = torch.ones(n_leaf, device=device)
        leaf_weights[leaf_phecodes.index(args.target_upweight)] = float(args.upweight_alpha)
        print(f"[train_rolling_window_cox] target-upweight: {args.target_upweight} "
              f"(leaf idx {leaf_phecodes.index(args.target_upweight)}) alpha="
              f"{args.upweight_alpha} over {n_leaf} leaves")

    # --- Ranking-loss guard: the pairwise concordance term ranks raw per-label
    #     log-hazards, so it is Cox-only (the discrete-time head emits per-bin
    #     logits instead).
    if args.ranking_weight > 0.0 and args.loss != "cox":
        raise SystemExit("--ranking-weight is implemented for --loss cox only")

    # --- Unified loss: leaf (+ weighted aux) under Cox or discrete-time. The
    #     model emits a flat (B, n_total*n_bins); reshape to (B, n_total, n_bins)
    #     for discrete. eval_mask is None during training (train mask all-True).
    def _total_loss(haz_flat, t, e, eval_mask=None, ar_emb_target=None):
        if discrete:
            haz = haz_flat.view(haz_flat.shape[0], n_total, n_bins)
            leaf = discrete_hazard_loss(
                haz[:, :n_leaf], t[:, :n_leaf], e[:, :n_leaf], cuts_t,
                eval_mask[:, :n_leaf] if eval_mask is not None else None,
            )
            loss = leaf
            if n_leaf < n_total:
                aux = discrete_hazard_loss(
                    haz[:, n_leaf:], t[:, n_leaf:], e[:, n_leaf:], cuts_t,
                    eval_mask[:, n_leaf:] if eval_mask is not None else None,
                )
                loss = leaf + args.aux_weight * aux
            return loss, leaf
        # Cox (target-upweighted if leaf_weights is set; else == plain cox_ph_loss)
        leaf = _cox_ph_loss_weighted(haz_flat[:, :n_leaf], t[:, :n_leaf],
                                     e[:, :n_leaf], leaf_weights)
        loss = leaf
        if n_leaf < n_total:
            aux = _cox_ph_loss_weighted(haz_flat[:, n_leaf:], t[:, n_leaf:], e[:, n_leaf:])
            loss = leaf + args.aux_weight * aux
        # DeepHit-style pairwise concordance term on the LEAF hazards only
        # (additive; --ranking-weight 0 gives an exact no-op). eval_mask is None in training.
        if args.ranking_weight > 0.0:
            l_rank = multilabel_ranking_loss(
                haz_flat[:, :n_leaf], t[:, :n_leaf], e[:, :n_leaf],
                valid_mask=(eval_mask[:, :n_leaf] if eval_mask is not None else None),
                phi=args.ranking_phi, margin=args.ranking_margin, sigma=args.ranking_sigma,
            )
            loss = loss + args.ranking_weight * l_rank
        # ar_emb aux regression (appended last column of haz_flat), masked to subjects
        # with an OOF target. Dropped at inference (eval calls with ar_emb_target=None).
        if args.aux_age_target and ar_emb_target is not None:
            ar_emb_pred = haz_flat[:, n_total]
            m = ~torch.isnan(ar_emb_target)
            if bool(m.any()):
                loss = loss + args.aux_age_weight * ((ar_emb_pred[m] - ar_emb_target[m]) ** 2).mean()
        return loss, leaf

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay)

    # --- Curriculum-aware DataLoader (built before scheduler so cosine_warmup
    # can size its per-step budget from len(train_loader)).
    if args.curriculum == "pos_enriched":
        sampler = make_positive_enriched_sampler(
            is_event=train_ds.is_event,
            eval_mask=train_ds.eval_mask,
            balance=args.pos_balance,
            seed=args.seed,
        )
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=sampler,
            collate_fn=collate_fn, drop_last=False, num_workers=args.num_workers,
            persistent_workers=False,
        )
        print(f"[train_rolling_window_cox] curriculum=pos_enriched balance={args.pos_balance}")
    elif args.curriculum == "per_disease_balanced":
        if not args.target_phecode:
            raise ValueError(
                "--target-phecode is required when --curriculum=per_disease_balanced"
            )
        try:
            target_col = phecodes.index(args.target_phecode)
        except ValueError:
            raise ValueError(
                f"--target-phecode '{args.target_phecode}' not in the kept phecode "
                f"list ({len(phecodes)} phecodes). Check prevalence filter or spelling."
            )
        n_pos_pool = int(
            ((train_ds.is_event[:, target_col] > 0.5) & train_ds.eval_mask[:, target_col]).sum()
        )
        batch_sampler = make_per_disease_balanced_sampler(
            is_event=train_ds.is_event,
            eval_mask=train_ds.eval_mask,
            target_col=target_col,
            batch_size=args.batch_size,
            n_pos_per_batch=args.n_pos_per_batch,
            seed=args.seed,
        )
        train_loader = DataLoader(
            train_ds, batch_sampler=batch_sampler,
            collate_fn=collate_fn, num_workers=args.num_workers,
            persistent_workers=False,
        )
        print(f"[train_rolling_window_cox] curriculum=per_disease_balanced "
              f"target={args.target_phecode} col={target_col} "
              f"n_pos_per_batch={args.n_pos_per_batch} n_pos_pool={n_pos_pool} "
              f"n_batches_per_epoch={len(batch_sampler)}")
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_fn, drop_last=False, num_workers=args.num_workers,
            persistent_workers=False,
        )
        print(f"[train_rolling_window_cox] curriculum=uniform")
    n_batches_per_epoch = max(1, len(train_loader))

    sched, sched_step_mode = _build_scheduler(args, optim, n_batches_per_epoch)
    print(f"[train_rolling_window_cox] schedule={args.schedule} step_mode={sched_step_mode} "
          f"T_max_epochs={args.schedule_T_max or args.epochs} "
          f"warmup_steps={args.warmup_steps} n_batches_per_epoch={n_batches_per_epoch}")

    # --- Resume from latest.pt if present
    latest_path = out_dir / "latest.pt"
    best_path = out_dir / "best.pt"
    start_epoch = 0
    best_val = float("inf")
    best_epoch = -1
    patience_counter = 0
    history: list[dict] = []
    # SWA running weight-average state (preempt-safe via latest.pt).
    swa_active = bool(args.swa) and args.swa_start_epoch >= 0
    swa_state: dict | None = None
    swa_n = 0
    if getattr(args, "predict_only", False):
        _init = Path(resolve_oak_path(args.init_from)) if args.init_from else out_dir
        ckpt = torch.load(_init / "best.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = args.epochs  # range(start_epoch, epochs) is empty -> no training
        print(f"[train_rolling_window_cox] PREDICT-ONLY: loaded {_init}/best.pt; skipping training loop")
    elif latest_path.exists():
        try:
            ckpt = torch.load(latest_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            optim.load_state_dict(ckpt["optim"])
            sched.load_state_dict(ckpt["sched"])
            start_epoch = int(ckpt["epoch"]) + 1
            best_val = float(ckpt["best_val"])
            best_epoch = int(ckpt["best_epoch"])
            patience_counter = int(ckpt["patience_counter"])
            history = list(ckpt.get("history", []))
            swa_state = ckpt.get("swa_state", None)
            swa_n = int(ckpt.get("swa_n", 0))
            print(f"[train_rolling_window_cox] RESUMED from latest.pt epoch={ckpt['epoch']} "
                  f"best_val={best_val:.4f} best_epoch={best_epoch} "
                  f"patience={patience_counter}/{args.patience}")
        except Exception as ex:
            print(f"[train_rolling_window_cox] failed to resume from latest.pt ({type(ex).__name__}: {ex}); "
                  f"starting fresh")

    # --- Train loop
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        tr_loss_sum = 0.0
        n_tr_batches = 0
        _ga = max(1, int(args.grad_accum_steps))
        optim.zero_grad()
        _accum = 0
        for windows, masks, t, e, c in train_loader:
            windows = windows.to(device)
            masks = masks.to(device)
            t = t.to(device)
            e = e.to(device)
            c = c.to(device)
            ar_emb_t = c[:, train_ds.n_covar] if args.aux_age_target else None
            haz = model(windows, masks, c)
            loss, _ = _total_loss(haz, t, e, ar_emb_target=ar_emb_t)
            (loss / _ga).backward()
            _accum += 1
            if _accum == _ga:
                optim.step()
                optim.zero_grad()
                _accum = 0
                if sched_step_mode == "batch":
                    sched.step()
            tr_loss_sum += float(loss.item())
            n_tr_batches += 1
        # flush a partial accumulation window at epoch end (effective batch < target)
        if _accum > 0:
            optim.step()
            optim.zero_grad()
            if sched_step_mode == "batch":
                sched.step()
        tr_loss = tr_loss_sum / max(n_tr_batches, 1)

        # Val: subject-aggregated forward (matches test protocol)
        val_loss, _ = evaluate_dataset(
            model, val_ds, device=device, batch_size=args.batch_size,
            stride=args.inference_stride,
            discrete=discrete, n_leaf=n_leaf, n_total=n_total,
            n_bins=n_bins, cuts_t=cuts_t, flexible=args.flexible,
        )
        if sched_step_mode == "epoch":
            sched.step()
        elif sched_step_mode == "plateau":
            sched.step(val_loss)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            best_epoch = epoch
            _atomic_torch_save({
                "model": model.state_dict(),
                "best_epoch": best_epoch,
                "best_val": best_val,
                "phecodes": phecodes,
                "args": vars(args),
            }, best_path)
            patience_counter = 0
        else:
            patience_counter += 1

        history.append({
            "epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss,
            "lr": _current_lr(optim),
            "patience": patience_counter,
        })

        # SWA: accumulate a running mean of the late-trajectory weights. Only
        # floating tensors are averaged (no BN buffers in this LayerNorm-only arch).
        if swa_active and epoch >= args.swa_start_epoch:
            sd = model.state_dict()
            if swa_state is None:
                swa_state = {k: v.detach().float().cpu().clone() for k, v in sd.items()}
                swa_n = 1
            else:
                swa_n += 1
                for k, v in sd.items():
                    if swa_state[k].is_floating_point():
                        swa_state[k].mul_(1.0 - 1.0 / swa_n).add_(
                            v.detach().float().cpu(), alpha=1.0 / swa_n)

        # latest.pt for preempt resume
        _atomic_torch_save({
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "sched": sched.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "best_epoch": best_epoch,
            "patience_counter": patience_counter,
            "history": history,
            "swa_state": swa_state,
            "swa_n": swa_n,
        }, latest_path)

        if epoch % args.log_every == 0 or improved or epoch == start_epoch:
            print(f"[train_rolling_window_cox] ep {epoch:3d}  train {tr_loss:.4f}  val {val_loss:.4f}  "
                  f"pat {patience_counter}/{args.patience}"
                  + ("  * best+saved" if improved else ""), flush=True)

        if SIGTERM_RECEIVED["flag"]:
            print(f"[train_rolling_window_cox] SIGTERM received; saved latest.pt epoch={epoch}, exiting 75 for requeue",
                  flush=True)
            sys.exit(75)

        if patience_counter >= args.patience:
            print(f"[train_rolling_window_cox] early stop at epoch {epoch}")
            break

    print(f"[train_rolling_window_cox] best epoch {best_epoch}  val_loss {best_val:.4f}  "
          f"total time {(time.time()-t0)/60:.1f}min")

    # --- Load best.pt for test forward pass
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])

    test_loss, test_score = evaluate_dataset(
        model, test_ds, device=device, batch_size=args.batch_size,
        stride=args.inference_stride,
        discrete=discrete, n_leaf=n_leaf, n_total=n_total,
        n_bins=n_bins, cuts_t=cuts_t, flexible=args.flexible,
    )
    print(f"[train_rolling_window_cox] test leaf {args.loss} loss (subject-aggregated): {test_loss:.4f}")

    # Save LEAF columns only; aux hierarchy heads are dropped at inference, and
    # the discrete risk score is already a per-phecode Cox-convention score.
    # per_disease_eval reads config['phecodes'] (= leaf) and aligns to these arrays.
    # Collapse per-recording rows to per-subject (mean hazards) before scoring, so
    # multi-recording subjects aren't over-counted in the held-out C-index.
    s_eids, s_haz, s_et, s_ie, s_em = _aggregate_to_subject(
        test_ds.eids, test_score[:, :n_leaf], test_ds.event_times[:, :n_leaf],
        test_ds.is_event[:, :n_leaf], test_ds.eval_mask[:, :n_leaf])
    print(f"[train_rolling_window_cox] test scored over {len(s_eids)} subjects "
          f"(from {len(test_ds.eids)} recordings)")
    np.save(out_dir / "test_hazards.npy", np.ascontiguousarray(s_haz))
    np.save(out_dir / "test_event_times.npy", np.ascontiguousarray(s_et))
    np.save(out_dir / "test_is_event.npy", np.ascontiguousarray(s_ie))
    np.save(out_dir / "test_eval_mask.npy", np.ascontiguousarray(s_em))
    np.save(out_dir / "test_eids.npy", s_eids)

    # --- Selection artifact: per-subject VAL hazards under TEST-rule masking
    #     (effective-prevalent excluded from eval) so the val C-index is an
    #     unbiased proxy for the frozen-test metric and can drive model SELECTION
    #     without ever reading the test arrays. Requires --use-holdout-val.
    if args.save_val_artifacts:
        if not args.use_holdout_val:
            raise SystemExit("--save-val-artifacts requires --use-holdout-val")
        from ukb_disease.baseline._per_phecode_utils import c_index as _cidx
        _, val_score = evaluate_dataset(
            model, val_ds, device=device, batch_size=args.batch_size,
            stride=args.inference_stride, discrete=discrete, n_leaf=n_leaf,
            n_total=n_total, n_bins=n_bins, cuts_t=cuts_t, flexible=args.flexible,
        )
        # Test-rule eval arrays for the val subjects: pass val_eids as the "test"
        # split arg so effective-prevalent are excluded from eval (eval_mask=False).
        val_tr = build_survival_arrays(
            df_combined, phecodes_all, train_eids_all, val_eids_all,
            seven_day_years=args.buffer_years)["test"]
        vpos = {int(e): i for i, e in enumerate(val_tr["eids"])}
        vrows = np.array([vpos[int(e)] for e in val_ds.eids], dtype=np.int64)
        vt = val_tr["event_times"][vrows][:, :n_leaf]
        vi = val_tr["is_event"][vrows][:, :n_leaf]
        vm = val_tr["eval_mask"][vrows][:, :n_leaf]
        vs_eids, vs_haz, vs_et, vs_ie, vs_em = _aggregate_to_subject(
            val_ds.eids, val_score[:, :n_leaf], vt, vi, vm)
        np.save(out_dir / "val_hazards.npy", np.ascontiguousarray(vs_haz))
        np.save(out_dir / "val_event_times.npy", np.ascontiguousarray(vs_et))
        np.save(out_dir / "val_is_event.npy", np.ascontiguousarray(vs_ie))
        np.save(out_dir / "val_eval_mask.npy", np.ascontiguousarray(vs_em))
        np.save(out_dir / "val_eids.npy", vs_eids)
        val_c, val_npos = {}, {}
        for j, pc in enumerate(leaf_phecodes):
            mm = vs_em[:, j]
            val_npos[pc] = int(vs_ie[mm, j].sum())
            if mm.sum() >= 2 and val_npos[pc] > 0:
                val_c[pc] = _cidx(vs_haz[mm, j], vs_et[mm, j], vs_ie[mm, j])
        with open(out_dir / "val_scores.json", "w") as f:
            json.dump({"run_name": run_name, "val_c": val_c,
                       "val_n_pos": val_npos, "n_val_subjects": int(len(vs_eids))},
                      f, indent=2)
        print(f"[train_rolling_window_cox] saved val artifacts + val_scores.json: val_c={val_c}")

    # SWA: evaluate the averaged-weight model; save its hazards alongside the
    # best-val ones so the gate can isolate the selection-bias effect. The arch is
    # LayerNorm-only, so no BN running stats to recompute (no update_bn needed).
    swa_test_loss = None
    if swa_active and swa_state is not None:
        swa_model = build_cox_model(
            arch=args.arch, window_size=args.window_size, n_phecodes=n_total,
            n_covar=train_ds.n_covar, in_dim=train_ds.in_dim, dropout=args.dropout,
            n_layers=args.n_layers, pe_type=args.pe_type, drop_path=args.drop_path,
            n_out_per_phecode=n_bins, input_norm=args.input_norm,
            output_rank=args.output_rank, max_days=model_max_days,
            cov_bottleneck_dim=args.cov_bottleneck_dim, cov_idx=cov_idx,
            across_day_pool=args.across_day_pool, pma_seeds_day=args.pma_seeds_day,
        ).to(device)
        swa_model.load_state_dict({k: v.to(device) for k, v in swa_state.items()})
        swa_test_loss, swa_score = evaluate_dataset(
            swa_model, test_ds, device=device, batch_size=args.batch_size,
            stride=args.inference_stride, discrete=discrete, n_leaf=n_leaf,
            n_total=n_total, n_bins=n_bins, cuts_t=cuts_t, flexible=args.flexible,
        )
        _, swa_haz_s, _, _, _ = _aggregate_to_subject(
            test_ds.eids, swa_score[:, :n_leaf], test_ds.event_times[:, :n_leaf],
            test_ds.is_event[:, :n_leaf], test_ds.eval_mask[:, :n_leaf])
        np.save(out_dir / "test_hazards_swa.npy", np.ascontiguousarray(swa_haz_s))
        print(f"[train_rolling_window_cox] SWA: averaged {swa_n} snapshots; test leaf loss {swa_test_loss:.4f}")

    pd.DataFrame(history).to_csv(out_dir / "train_history.csv", index=False)
    with open(out_dir / "config.json", "w") as f:
        json.dump({
            "args": vars(args),
            "run_name": run_name,
            "phecodes": leaf_phecodes,
            "aux_phecodes": aux_names,
            "loss": args.loss,
            "n_time_bins": n_bins,
            "time_bin_cuts": (cuts_np.tolist() if cuts_np is not None else None),
            "phecode_stats": stats,
            "max_follow_up": arrs["max_follow_up"],
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "test_leaf_loss": test_loss,
            "n_leaf": n_leaf,
            "n_aux": len(aux_names),
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "n_test": len(test_ds),
            "n_params": n_params,
            "output_rank": args.output_rank,
            "swa": bool(args.swa),
            "swa_start_epoch": args.swa_start_epoch,
            "swa_n": swa_n,
            "swa_test_loss": swa_test_loss,
            "n_covar": int(train_ds.n_covar),
        }, f, indent=2, default=str)
    print(f"[train_rolling_window_cox] saved to {out_dir}")
    return {"out_dir": str(out_dir), "best_val": best_val, "best_epoch": best_epoch}


def main() -> None:
    _install_sigterm_handler()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", required=True,
                    choices=["mlp_flat", "lstm", "transformer", "mamba"])
    ap.add_argument("--window-size", type=int, required=True,
                    help="Days per window. Common values 3/5/7; longer (14/21/28) "
                         "supported but drops subjects without enough valid days.")
    ap.add_argument("--curriculum", required=True,
                    choices=["uniform", "pos_enriched", "per_disease_balanced"])
    ap.add_argument("--target-phecode", default=None,
                    help="Phecode column (e.g. 'NS_324.11') to oversample in each "
                         "batch when --curriculum=per_disease_balanced. The pool is "
                         "every train subject with is_event==1 AND eval_mask on this "
                         "column, i.e. exactly the rows the Cox loss already treats "
                         "as positive events for this disease.")
    ap.add_argument("--n-pos-per-batch", type=int, default=32,
                    help="Positives per batch when --curriculum=per_disease_balanced. "
                         "Default 32 paired with --batch-size 64 = 50/50.")
    ap.add_argument("--stratified", action="store_true",
                    help="Use sleep/wake-stratified per-day means (4×256-d per day = 1024-d). "
                         "Requires day_mean_strat_v3/ produced by compute_day_mean_stratified.")
    ap.add_argument("--day-means-root", default=CANONICAL_DAY_MEAN_ROOT,
                    help=f"Per-day means root (default {CANONICAL_DAY_MEAN_ROOT}). "
                         f"For --stratified, pass {DAY_MEAN_STRAT_ROOT}; for a "
                         f"different split cache, pass it explicitly.")
    ap.add_argument("--out-root", default=RUNS_COX_ROOT,
                    help="Where to save the run directory (default: the runs_cox root)")
    ap.add_argument("--run-name", default=None,
                    help="Override the auto-derived run dir name")
    ap.add_argument("--train-split", default="train", choices=["audit", "test", "train"])
    ap.add_argument("--test-split", default="test", choices=["audit", "test", "train"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--use-holdout-val", action="store_true",
                    help="Use the explicit 'val' day-means split (a subject-disjoint "
                         "validation partition) for early stopping instead of carving a "
                         "seeded --val-frac fraction out of train. Requires a 'val' subdir "
                         "under --day-means-root (train is used in full).")
    ap.add_argument("--no-demographics", action="store_true",
                    help="Drop age+sex (and any missingness/circadian) covariates for "
                         "n_covar=0, a fully wrist-only model. Head becomes "
                         "Linear(d_model, ...). Use for the wrist-only and neuro arms.")
    ap.add_argument("--keep-covariates", nargs="+", default=None, metavar="COL",
                    help="Keep ONLY these covariate columns (e.g. 'age', 'sex', or "
                         "'age sex') so n_covar=len(kept). For the age-only/sex-only "
                         "demographic-attribution ablation. Mutually exclusive with "
                         "--no-demographics.")
    ap.add_argument("--add-bmi", action="store_true",
                    help="Join clinical BMI (UKB field 21001) as an extra covariate for "
                         "n_covar=3 (age, sex, bmi). The clinical+embedding model for the "
                         "baseline comparison.")
    ap.add_argument("--age-basis", choices=["none", "bins", "rbf"], default="none",
                    help="Expand the raw 'age' covariate into a non-linear basis "
                         "(quantile one-hot bins or gaussian RBFs) fed to the head. "
                         "Tests whether a non-linear age MARGINAL helps. Default 'none' "
                         "is unchanged. n_covar auto-flows via window_dataset.")
    ap.add_argument("--age-basis-k", type=int, default=8,
                    help="Number of age basis functions for --age-basis.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--pos-balance", type=float, default=0.5,
                    help="Positive-subject mass for the curriculum sampler.")
    ap.add_argument("--min-prevalence", type=float, default=MIN_PREVALENCE)
    ap.add_argument("--incident-only", action="store_true")
    ap.add_argument("--buffer-years", type=float, default=7 / 365.25,
                    help="Prevalent-exclusion buffer in years (default 7/365.25 = 7d).")
    ap.add_argument("--max-phecodes", type=int, default=None)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers. 0 = main process only (lowest memory).")
    ap.add_argument("--log-every", type=int, default=1)
    # 7-day retune additions (backward-compatible defaults).
    ap.add_argument("--schedule", default="cosine",
                    choices=["cosine", "cosine_warmup", "plateau"],
                    help="LR schedule. cosine = default; cosine_warmup = "
                         "linear warmup then cosine (stepped per batch); plateau = "
                         "ReduceLROnPlateau (factor=0.5, patience=3) on val_loss.")
    ap.add_argument("--schedule-T-max", type=int, default=0,
                    help="T_max for cosine/cosine_warmup, in epochs. 0 = use --epochs.")
    ap.add_argument("--warmup-steps", type=int, default=0,
                    help="Linear-warmup optimizer steps (batches) for --schedule "
                         "cosine_warmup. 0 means no warmup.")
    ap.add_argument("--drop-path", type=float, default=0.0,
                    help="Stochastic-depth probability on each transformer encoder "
                         "layer's residual paths. Default 0.0 (off).")
    ap.add_argument("--pe-type", default="learned", choices=["learned", "sinusoidal"],
                    help="Positional encoding for the transformer head. "
                         "learned = nn.Embedding (default); sinusoidal = "
                         "parameter-free sin/cos table.")
    ap.add_argument("--n-layers", type=int, default=2,
                    help="Transformer encoder layer count. Default 2.")
    ap.add_argument("--inference-stride", type=int, default=1,
                    help="Subsample valid window starts by this stride during "
                         "forward_subject_aggregated (val + test). Default 1.")
    ap.add_argument("--phecode-prefix-filter", default=None,
                    help="If set, restrict the phecode set to columns whose name "
                         "starts with this prefix (e.g. 'GI_' for gastrointestinal "
                         "phecodes only). Applied AFTER prevalence filter. The "
                         "Cox head will only predict + be trained on this subset. "
                         "Mortality (time_to_death) is preserved unless excluded "
                         "by the prefix.")
    # --- Single-target specialization (dedicated-disease fine-tune)
    ap.add_argument("--phecode-exact", default=None,
                    help="Restrict the head to EXACTLY this one phecode column "
                         "(single-target specialization). Applied after the "
                         "prevalence + prefix filters; errors if the column did not "
                         "survive the prevalence filter. Cleaner than "
                         "--phecode-prefix-filter for a single disease (avoids "
                         "startswith over-matching, e.g. CV_401.1 vs CV_401.11).")
    ap.add_argument("--warm-start-ckpt", default=None,
                    help="Path to a a pretrained best.pt to warm-start from (shape-filtered "
                         "load: every matching-shape tensor copied, mismatches skipped). "
                         "For a single-output head, the lone output row is warm-"
                         "initialized from the source's --phecode-exact column.")
    ap.add_argument("--freeze-mode", default="none",
                    choices=["none", "head_only", "last_block", "last2_block"],
                    help="Partial-freeze regime for fine-tuning: none=full FT; "
                         "head_only=train only the final head; last_block / last2_block "
                         "also unfreeze the last 1/2 transformer encoder block(s) + the "
                         "final LayerNorm.")
    ap.add_argument("--predict-only", action="store_true",
                    help="Skip training; load best.pt from --init-from and run eval only "
                         "(reproduces test_hazards; saves val artifacts with --save-val-artifacts). "
                         "Non-destructive: reads --init-from, writes to --out-root/--run-name.")
    ap.add_argument("--init-from", default=None,
                    help="Source run dir (best.pt + config.json) for --predict-only.")
    ap.add_argument("--save-val-artifacts", action="store_true",
                    help="Save per-subject VAL hazards + labels under TEST-rule masking "
                         "and a val_scores.json (per-target val C-index) for model "
                         "SELECTION. Requires --use-holdout-val. Never reads test arrays.")
    ap.add_argument("--permute-target-phecode", default=None,
                    help="Negative control: permute this phecode's (is_event, time) "
                         "pairs across TRAIN subjects (val/test untouched). A model "
                         "trained on permuted labels must NOT beat baseline on it.")
    ap.add_argument("--permute-seed", type=int, default=0,
                    help="RNG seed for --permute-target-phecode.")
    ap.add_argument("--target-upweight", default=None,
                    help="Regularized specialization: upweight this ONE leaf phecode's "
                         "Cox loss by --upweight-alpha inside the JOINT multilabel model "
                         "(the other leaves stay weight 1 as regularizers). Addresses the "
                         "single-target overfitting failure mode. Cox-only; needs >=2 leaves "
                         "(do NOT combine with --phecode-exact).")
    ap.add_argument("--upweight-alpha", type=float, default=1.0,
                    help="Per-label loss weight on --target-upweight (alpha=1 == joint "
                         "baseline; larger = more target gradient).")
    # --- Discrete-time logistic-hazard loss
    ap.add_argument("--loss", default="cox", choices=["cox", "discrete"],
                    help="cox = in-batch CoxPHLoss (default). discrete = "
                         "per-time-bin logistic hazard (Nnet-survival): dense "
                         "gradient on every head every batch, no risk set.")
    ap.add_argument("--n-time-bins", type=int, default=20,
                    help="Number of KM-quantile time bins T for --loss discrete.")
    # --- Concordance (DeepHit-style) pairwise ranking loss
    ap.add_argument("--ranking-weight", type=float, default=0.0,
                    help="lambda for the additive pairwise concordance term on the Cox "
                         "leaf hazards (0 = off, exact no-op). Wrapper in "
                         "baseline/loss_ranking.py. --loss cox only.")
    ap.add_argument("--ranking-phi", default="sigmoid",
                    choices=["sigmoid", "softplus", "margin"],
                    help="pair penalty: sigmoid/softplus (DeepHit smooth) or margin (hinge).")
    ap.add_argument("--ranking-margin", type=float, default=0.0,
                    help="hinge margin m (only used when --ranking-phi margin).")
    ap.add_argument("--ranking-sigma", type=float, default=1.0,
                    help="sigmoid sharpness (only used when --ranking-phi sigmoid).")
    ap.add_argument("--grad-accum-steps", type=int, default=1,
                    help="accumulate N microbatches before optim.step (effective batch "
                         "= N*batch-size); keeps the B^2*P ranking pair matrix bounded "
                         "while giving rare diseases more comparable pairs per step.")
    # --- Accelerated-aging residual (ar_emb) as an aux regression target
    ap.add_argument("--aux-age-target", action="store_true",
                    help="Add an auxiliary head regressing the per-subject OOF ar_emb (wrist "
                         "age-gap), dropped at inference. --loss cox only; not with --aux-hierarchy/--flexible.")
    ap.add_argument("--aux-age-weight", type=float, default=0.5,
                    help="weight alpha on the ar_emb MSE aux loss (leaf + alpha*MSE).")
    ap.add_argument("--aux-age-parquet",
                    default=f"{UKB_ROOT}/ar_emb_oof_disjoint_split.parquet",
                    help="parquet with columns eid, ar_emb_z (z-scored OOF ar_emb target).")
    # --- Phecode-hierarchy auxiliary heads (dropped at inference)
    ap.add_argument("--aux-hierarchy", action="store_true",
                    help="Add category + high-prevalence parent aggregate phecode "
                         "columns as auxiliary survival heads (shape the shared "
                         "trunk with dense gradient; never scored at inference).")
    ap.add_argument("--aux-weight", type=float, default=0.3,
                    help="Weight on the auxiliary-head loss (leaf + w*aux).")
    ap.add_argument("--aux-min-prevalence", type=float, default=0.01,
                    help="Keep an integer-parent aggregate only if its any-positive "
                         "train prevalence >= this (categories always kept).")
    ap.add_argument("--aux-categories-only", action="store_true",
                    help="Build ONLY the 13 category roll-ups (densest, highest-EV "
                         "subset); a reasonable first configuration.")
    # --- Distributional pooling (input scale normalization)
    ap.add_argument("--input-norm", action="store_true",
                    help="LayerNorm over the per-day input feature dim. Recommended "
                         "with the distributional ([mean, std]) day cache to tame "
                         "the std-channel scale mismatch.")
    # --- Flexible-length DAY sequence (variable valid days + real mask)
    ap.add_argument("--flexible", action="store_true",
                    help="Use FlexibleWindowDataset: feed each subject's FULL valid-day "
                         "sequence (1..d_max days, real padding mask) instead of fixed "
                         "N-day windows averaged at inference. Day-level sanity check.")
    # --- Low-rank factorized output head (diagnostic)
    ap.add_argument("--output-rank", type=int, default=0,
                    help="Factorize the final hazard Linear into "
                         "Linear(hidden,rank)->Linear(rank,n_out). 0 = dense (default).")
    # --- SWA / snapshot weight averaging
    ap.add_argument("--swa", action="store_true",
                    help="Maintain a running mean of weights over epochs >= "
                         "--swa-start-epoch; emit test_hazards_swa.npy beside best-val.")
    ap.add_argument("--swa-start-epoch", type=int, default=-1,
                    help="Epoch to begin SWA averaging. -1 = off. Pair with a raised "
                         "--patience so the late trajectory is actually traversed.")
    # --- Informative-missingness covariates (pre-screen first)
    ap.add_argument("--missingness-covariates", action="store_true",
                    help="Append wear-pattern descriptors to [age,sex] covariates. "
                         "Reads MISSINGNESS_COV_ROOT/all.csv.")
    ap.add_argument("--missingness-cols", nargs="+", default=None,
                    help="Subset of descriptor columns to use (default: all in the CSV).")
    # --- Covariance-channel bottleneck (split input projection; transformer only)
    # --- Learnable across-day pooling (PMA) vs CLS
    ap.add_argument("--across-day-pool", default="cls", choices=["cls", "pma"],
                    help="Across-day aggregator: 'cls' (default) or 'pma' "
                         "(Pooling-by-Multihead-Attention over the N day tokens).")
    ap.add_argument("--pma-seeds-day", type=int, default=1,
                    help="Number of PMA seed queries for --across-day-pool pma "
                         "(outputs mean-reduced over seeds to keep head width fixed).")
    ap.add_argument("--cov-bottleneck-dim", type=int, default=0,
                    help="Project the within-day covariance tail channels through a "
                         "Linear(cov,k)->Linear(k,d_model) bottleneck of rank k before "
                         "adding to the base mean-std projection (transformer only). "
                         "0 = single input_proj over all in_dim (the default).")
    ap.add_argument("--cov-cols-per-modality", type=int, default=136,
                    help="Covariance columns at the END of each modality block "
                         "(r(r+1)/2; 136 for r=16). Used with --cov-bottleneck-dim to "
                         "locate the interleaved cov channels in the AR|HA concat.")
    # --- Circadian rest-activity-rhythm covariates (pre-screen first)
    ap.add_argument("--circadian-covariates", action="store_true",
                    help="Append full-recording RAR descriptors (IS/IV/RA/M10/L5/cosinor) "
                         "to [age,sex] covariates. Reads CIRCADIAN_COV_ROOT/all.csv.")
    ap.add_argument("--circadian-csv", default=None,
                    help="Override path to the combined circadian covariate CSV "
                         "(default: CIRCADIAN_COV_ROOT/all.csv).")
    ap.add_argument("--circadian-cols", nargs="+", default=None,
                    help="Subset of circadian descriptor columns to use (default: all in "
                         "the CSV). E.g. 'enmo_IS enmo_IV enmo_RA ...' for the ENMO family.")
    args = ap.parse_args()
    # If --stratified is set and the user kept the default --day-means-root, auto-switch
    # to the stratified root for convenience.
    if args.stratified and args.day_mean_root == CANONICAL_DAY_MEAN_ROOT:
        args.day_mean_root = DAY_MEAN_STRAT_ROOT
    train_one_run(args)


if __name__ == "__main__":
    main()
