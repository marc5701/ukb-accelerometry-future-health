"""Embedding extraction.

The correct entry point for our H5 schema is get_inputs_main.py (not
make_embeddings_main.py). Nonwear is applied live: NaN-fill the input signal
and let the model's own NaN propagation handle the rest.

What this script does
---------------------

1. Loads models from torch.hub (no local checkpoints):
   - accelerest_multihead with all three heads (linear_sleepstages,
     lstm_sleepstages, linear_respevents) bundled on a shared encoder.
   - activity_mae(pretrained_model='mae_lmm').

2. Reads the preprocessed H5 (/data/accelerometry plus
   /masks/detected_nonwear_actipy), then NaN-fills the input signal where the
   nonwear mask is 1 (the apply-the-mask-live pattern).

3. Runs sliding-window inference with shift=1 patch. At each patch position the
   embedding is averaged across all overlapping windows that cover it. Any
   window touching a NaN sample produces a NaN output, so the averaged
   embedding for that position is NaN, which is the intended behaviour.

4. No clock-anchored day-splitting at extraction time. The full 7-day recording
   is processed in one go; start_time is saved in meta.json for downstream
   day-splitting in the pooling and modelling stages.

5. Output structure per recording. AcceleRest has 3 heads but only one encoder,
   so we save a single accelerest_embeddings.npy rather than three identical
   copies:

       cache/<split>/<eid>_<run>/
         meta.json                                      start_time, source, commits, etc.
         nonwear_mask.npy                               (n_samples,) u8, source mask (for reference)
         accelerest_embeddings.npy                      (n_patches_30s, 256) f32, NaN-aware
         accelerest_linear_sleepstages_soft_preds.npy   (n_patches_30s, 4) f32
         accelerest_lstm_sleepstages_soft_preds.npy     (n_patches_30s, 4) f32
         accelerest_linear_respevents_soft_preds.npy    (n_patches_30s, 2) f32
         activity_mae_embeddings.npy                    (n_patches_10s, 256) f32
         activity_mae_soft_preds.npy                    (n_patches_10s, 6) f32

torch.hub on a cluster
----------------------

Compute nodes without internet cannot fetch from torch.hub, so pre-warm the
cache from a login node:

    cd ~ && python3 -c "
    import torch
    torch.hub.load('NielsRLorenzen/AcceleRest', 'accelerest_multihead',
        linear_sleepstage=True, lstm_sleepstage=True, linear_respevent=True,
        trust_repo='check', force_reload=True)
    torch.hub.load('NielsRLorenzen/activity-mae', 'activity_mae',
        pretrained_model='mae_lmm', trust_repo='check', force_reload=True)
    "

After this, compute-node jobs can load with force_reload=False (the default here).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import softmax

from ukb_disease.extraction import config as C


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_models(device: str, force_reload: bool = False) -> dict[str, torch.nn.Module]:
    """Load the encoder models from torch.hub.

    Returns a dict keyed by model name; each model exposes
    .patch_size (samples per patch), .max_seq_len (patches per window),
    .get_embeddings(x), .get_predictions(embeds).
    """
    print(f"[load_models] force_reload={force_reload}")
    accelerest = torch.hub.load(
        "NielsRLorenzen/AcceleRest",
        "accelerest_multihead",
        linear_sleepstage=True,
        lstm_sleepstage=True,
        linear_respevent=True,
        trust_repo=True,  # auto-trust; non-interactive batch jobs can't answer the prompt
        force_reload=force_reload,
    )
    activity_mae = torch.hub.load(
        "NielsRLorenzen/activity-mae",
        "activity_mae",
        pretrained_model="mae_lmm",
        trust_repo=True,  # auto-trust; non-interactive batch jobs can't answer the prompt
        force_reload=force_reload,
    )
    for name, m in [("accelerest", accelerest), ("activity_mae", activity_mae)]:
        m.eval().to(device)
        n = sum(p.numel() for p in m.parameters())
        print(f"  loaded {name}: patch_size={m.patch_size}  max_seq_len={m.max_seq_len}  params={n/1e6:.2f}M")
    return {"accelerest": accelerest, "activity_mae": activity_mae}


# ---------------------------------------------------------------------------
# Source H5 reader with live nonwear masking
# ---------------------------------------------------------------------------


def load_and_mask_signal(source_h5: str) -> tuple[np.ndarray, np.ndarray, str, int]:
    """Read our preprocessed H5 and NaN-fill the signal where nonwear==1.

    Returns:
        signal_nan:  (3, N) float32, NaN where the underlying sample is nonwear
        nonwear:     (N,) uint8, original 30 Hz mask, kept for reference
        start_time:  ISO-formatted recording start time
        fs:          sample rate (30)
    """
    src = C.resolve_oak_path(source_h5)
    with h5py.File(src, "r") as f:
        signal = f["/data/accelerometry"][:].astype(np.float32)
        nonwear = f["/masks/detected_nonwear_actipy"][:].astype(np.uint8)
        start_str = f.attrs["start_time"]
        if isinstance(start_str, bytes):
            start_str = start_str.decode()
        if "output_fs" in f.attrs:
            fs = int(f.attrs["output_fs"])
        elif "resample_ResampleRate" in f.attrs:
            fs = int(f.attrs["resample_ResampleRate"])
        elif "fs" in f.attrs:
            fs = int(f.attrs["fs"])
        else:
            raise KeyError(f"No sample-rate attr in {source_h5}")

    # Live mask application: NaN-fill every nonwear sample. The model natively
    # propagates NaN through PatchEmbedding (Linear over 2700-sample patch to
    # 256-d), RMSNorm, and softmax-attention, so any window touching a nonwear
    # sample produces a NaN output. With shift=1 averaging, only positions whose
    # entire covering-window set is clean retain a finite embedding.
    signal[:, nonwear.astype(bool)] = np.nan
    return signal, nonwear, start_str, fs


# ---------------------------------------------------------------------------
# Sliding-window inference (replicates get_inputs_main.py's eval_single)
# ---------------------------------------------------------------------------


def run_single_model(
    model: torch.nn.Module,
    model_name: str,
    signal_nan: np.ndarray,
    device: str,
    batch_size: int,
    shift_patches: int,
) -> dict:
    """Run one model with overlapping-window inference plus averaging.

    Mirrors get_inputs_main.py:eval_single: unfold the signal into windows of
    max_seq_len patches stepping by shift_patches, run the encoder and each
    head, accumulate per-patch sum_logits, sum_embeddings, and position_count,
    then divide to average across overlapping windows.

    Returns dict with:
        embeddings: (n_patches, embed_dim) float32, averaged across overlapping windows
        soft_preds: dict head_name to (n_patches, n_classes) float32
        n_patches:  int
    """
    patch_size = model.patch_size
    max_seq_len = model.max_seq_len
    window_samples = patch_size * max_seq_len
    step_samples = patch_size * shift_patches

    n_samples = signal_nan.shape[1]
    n_patches = n_samples // patch_size

    # Unfold (3, N) to (3, n_windows, window_samples) to (n_windows, 3, window_samples)
    sig_t = torch.from_numpy(signal_nan)
    windows = sig_t.unfold(-1, window_samples, step_samples).permute(1, 0, 2)
    n_windows = windows.shape[0]
    print(
        f"  [{model_name}] n_samples={n_samples}  n_patches={n_patches}  "
        f"n_windows={n_windows}  patch={patch_size}  max_seq_len={max_seq_len}  shift={shift_patches}"
    )

    # Probe embed_dim and head signatures with a tiny forward pass on the first
    # (possibly NaN-clean) window. Using random data avoids depending on signal.
    # use_sdpa=False for consistency with the main loop (see below).
    with torch.no_grad():
        probe = torch.zeros(1, 3, window_samples, device=device)
        probe_emb = model.get_embeddings(probe, use_sdpa=False)
        embed_dim = probe_emb.shape[-1]
        probe_pred = model.get_predictions(probe_emb)
        if isinstance(probe_pred, dict):
            heads = {f"{model_name}_{k}": v.shape[-1] for k, v in probe_pred.items()}
        else:
            heads = {model_name: probe_pred.shape[-1]}

    sum_embeds = torch.zeros((n_patches, embed_dim), dtype=torch.float32)
    sum_logits = {h: torch.zeros((n_patches, c), dtype=torch.float32) for h, c in heads.items()}
    position_count = torch.zeros((n_patches,), dtype=torch.float32)

    # use_sdpa=False forces the standard (non-fused) attention kernel. This is
    # ~10-15% slower than the fused SDPA path but works on all CUDA GPUs. Some
    # heterogeneous GPUs throw "CUDA error: misaligned address" from the fused
    # kernel, so the non-fused path is used for portability.
    with torch.no_grad():
        for b_start in range(0, n_windows, batch_size):
            batch = windows[b_start : b_start + batch_size].to(device, non_blocking=True)
            embeds = model.get_embeddings(batch, use_sdpa=False)   # (B, max_seq_len, embed_dim)
            preds_raw = model.get_predictions(embeds)              # dict OR tensor

            if isinstance(preds_raw, dict):
                preds_norm = {f"{model_name}_{k}": v for k, v in preds_raw.items()}
            else:
                preds_norm = {model_name: preds_raw}

            embeds_cpu = embeds.cpu()
            preds_cpu = {h: t.cpu() for h, t in preds_norm.items()}

            B = embeds_cpu.shape[0]
            for j in range(B):
                widx = b_start + j
                p_start = widx * shift_patches
                p_end = min(p_start + max_seq_len, n_patches)
                used = p_end - p_start
                sum_embeds[p_start:p_end] += embeds_cpu[j, :used]
                for h, t in preds_cpu.items():
                    sum_logits[h][p_start:p_end] += t[j, :used]
                position_count[p_start:p_end] += 1

    count_clamped = position_count.clamp_min(1).unsqueeze(-1)
    avg_embeds = (sum_embeds / count_clamped).numpy().astype(np.float32)
    soft_preds = {
        h: softmax(t / count_clamped, dim=-1).numpy().astype(np.float32)
        for h, t in sum_logits.items()
    }
    return {
        "embeddings": avg_embeds,
        "soft_preds": soft_preds,
        "n_patches": n_patches,
    }


# ---------------------------------------------------------------------------
# Per-recording orchestrator
# ---------------------------------------------------------------------------


def git_commit_or_unknown() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(__file__).resolve().parent.parent.parent),
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def extract_one(
    source_h5: str,
    eid_run: str,
    cache_subj_dir: Path,
    models: dict[str, torch.nn.Module],
    device: str,
    batch_size: int,
    shift_patches: int,
    code_commit: str,
) -> None:
    cache_subj_dir.mkdir(parents=True, exist_ok=True)
    signal_nan, nonwear, start_time, fs = load_and_mask_signal(source_h5)

    nw_frac = float(nonwear.mean())
    print(
        f"  source={source_h5}  start={start_time}  fs={fs}  "
        f"n_samples={signal_nan.shape[1]}  nonwear_frac={nw_frac:.3f}"
    )

    meta_models = {}
    for model_name, model in models.items():
        result = run_single_model(
            model, model_name, signal_nan, device, batch_size, shift_patches,
        )
        np.save(cache_subj_dir / f"{model_name}_embeddings.npy", result["embeddings"])
        for head_name, sp in result["soft_preds"].items():
            np.save(cache_subj_dir / f"{head_name}_soft_preds.npy", sp)
        nan_row_frac = float(np.isnan(result["embeddings"]).any(axis=-1).mean())
        meta_models[model_name] = {
            "n_patches": int(result["n_patches"]),
            "patch_size_samples": int(model.patch_size),
            "max_seq_len": int(model.max_seq_len),
            "nan_row_frac": nan_row_frac,
            "head_names": sorted(result["soft_preds"].keys()),
        }

    np.save(cache_subj_dir / "nonwear_mask.npy", nonwear.astype(np.uint8))

    meta = {
        "eid_run": eid_run,
        "source_h5": source_h5,
        "start_time": start_time,
        "fs": fs,
        "n_samples": int(signal_nan.shape[1]),
        "nonwear_frac": nw_frac,
        "shift_patches": int(shift_patches),
        "code_commit": code_commit,
        "models": meta_models,
    }
    (cache_subj_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def run_extraction(
    manifest_path: str,
    output_root: str | None = None,
    max_subjects: int | None = None,
    device: str = "cpu",
    batch_size: int = 32,
    shift_patches: int = 1,
    overwrite: bool = False,
    force_reload_hub: bool = False,
    task_id: int = 0,
    num_tasks: int = 1,
) -> int:
    manifest = pd.read_parquet(C.resolve_oak_path(manifest_path))
    if num_tasks > 1:
        manifest = manifest.iloc[task_id::num_tasks].reset_index(drop=True)
        print(f"[run_extraction] strided slice: task={task_id}/{num_tasks}  rows={len(manifest)}")
    if max_subjects is not None:
        manifest = manifest.iloc[:max_subjects]
    print(
        f"[run_extraction] manifest={manifest_path}  n={len(manifest)}  device={device}  "
        f"shift={shift_patches}  batch={batch_size}"
    )

    models = load_models(device, force_reload=force_reload_hub)
    code_commit = git_commit_or_unknown()
    print(f"[run_extraction] code_commit={code_commit}")

    n_done = n_skip = n_fail = 0
    for _, row in manifest.iterrows():
        eid_run = row["eid_run"]
        # Output dir: a per-recording subdir under output_root (or the
        # manifest's cache_h5 column dropped to a dirname). The manifest's
        # cache_h5 column from build_manifest.py points at a .h5 file (v1
        # schema). For v2 we treat that as a *directory* with the same stem
        # under audit/test/train.
        if output_root is not None:
            base = Path(C.resolve_oak_path(output_root)) / row["split"]
        else:
            cache_h5 = row["cache_h5"]
            if not cache_h5:
                raise ValueError(
                    f"manifest row {eid_run} has empty cache_h5 and no --output-root "
                    "given; pass --output-root to avoid writing to the filesystem root.")
            base = Path(C.resolve_oak_path(os.path.dirname(cache_h5)))
        cache_subj_dir = base / eid_run

        if (cache_subj_dir / "meta.json").exists() and not overwrite:
            n_skip += 1
            continue
        try:
            extract_one(
                row["source_h5"], eid_run, cache_subj_dir,
                models, device, batch_size, shift_patches, code_commit,
            )
            n_done += 1
            print(f"  [{n_done + n_skip + n_fail}/{len(manifest)}] {eid_run} at {cache_subj_dir}")
        except Exception as e:
            n_fail += 1
            print(f"  [FAIL] {eid_run}: {type(e).__name__}: {e}")

    print(f"[run_extraction] done. n_extracted={n_done}  n_skipped(existing)={n_skip}  n_failed={n_fail}")
    return 0 if n_fail == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", required=True, help="Parquet from build_manifest.py")
    p.add_argument("--output-root", default=None,
                   help="Override cache root (default: derived from manifest's cache_h5 column).")
    p.add_argument("--max-subjects", type=int, default=None)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--batch-size", type=int, default=32,
                   help="Batch size for windowed inference (default 32).")
    p.add_argument("--shift-patches", type=int, default=1,
                   help="Window stride in patches (default 1 = full overlap).")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--force-reload-hub", action="store_true",
                   help="Re-download torch.hub repos. Use on login node to warm cache; "
                        "leave off on compute nodes (no internet).")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--num-tasks", type=int, default=1)
    args = p.parse_args()
    return run_extraction(
        manifest_path=args.manifest,
        output_root=args.output_root,
        max_subjects=args.max_subjects,
        device=args.device,
        batch_size=args.batch_size,
        shift_patches=args.shift_patches,
        overwrite=args.overwrite,
        force_reload_hub=args.force_reload_hub,
        task_id=args.task_id,
        num_tasks=args.num_tasks,
    )


if __name__ == "__main__":
    sys.exit(main())
