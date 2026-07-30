"""Per-day distributional summary embeddings (mean, std) per subject.

Instead of collapsing each day's valid epoch embeddings to a single mean, this
keeps a richer per-dimension distributional summary [mean, std] (and optionally
L-moments) so the within-day spread survives the daily reduction. That spread is
where AR's rare epoch-specific signal (respiratory events during sleep) lives,
and the plain daily mean washes it out. On an audit set the within-day AR std is
roughly 3.4x the across-day-mean variation and largely orthogonal to the mean
(median absolute correlation 0.22).

Output is one .npz per subject with the same schema as compute_day_mean so the
existing aggregate step consumes it:

    <out_root>/<split>/<eid_run>.npz
        ar:        (n_days, 2*256) float32, [mean, std] per dim, AR
        ha:        (n_days, 2*256) float32, [mean, std] per dim, HA
        day_codes: (n_days,) int64

A day is kept only if it has sufficient valid patches in BOTH modalities, same
as compute_day_mean. The std channels are population std (ddof=0) over the
day's valid epochs; per-dim standardization is applied downstream in the model's
input projection (the std channels have a different scale than the mean).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from ukb_disease.baseline.config import (
    AR_PATCH_SECONDS,
    CACHE_ROOT_V2,
    HA_PATCH_SECONDS,
    resolve_oak_path,
)
from ukb_disease.baseline.compute_day_mean import MODEL_FILE, PATCH_SECONDS, list_subjects
from ukb_disease.paths import UKB_ROOT

DAY_DISTRIB_ROOT_DEFAULT = f"{UKB_ROOT}/day_distrib_v3"

# Physiology-weighted moments use AcceleRest per-epoch respiratory-event softpreds
# (n_epochs, 2); column 1 is P(respiratory event), aligned 1:1 with AR epochs.
AR_RESPEVENTS_FILE = "accelerest_linear_respevents_soft_preds.npy"
RESP_EVENT_COL = 1

# Circadian spectral pooling: harmonics probed by Lomb-Scargle (hours).
_CIRC_PERIODS_H = np.array([24.0, 12.0, 8.0, 6.0], dtype=np.float64)


def _l_moments(day_emb: np.ndarray) -> np.ndarray:
    """First 3 sample L-moments per dimension over a day's epochs.

    L-moments (Hosking) are linear combinations of order statistics, far less
    collinear and more outlier-robust than raw central moments, so they are a
    natural escalation beyond mean and std. l1 = mean (location), l2 = scale
    (approx. Gini mean difference), l3 = L-skewness scale. Returns concat
    [l1, l2, l3].

    Args:
        day_emb: (n_epochs, D) float64.
    Returns:
        (3*D,) float64.
    """
    n = day_emb.shape[0]
    xs = np.sort(day_emb, axis=0)  # ascending per dimension
    i = np.arange(n, dtype=np.float64)
    b0 = xs.mean(axis=0)
    b1 = ((i / (n - 1))[:, None] * xs).sum(axis=0) / n
    b2 = ((i * (i - 1) / ((n - 1) * (n - 2)))[:, None] * xs).sum(axis=0) / n
    l1 = b0
    l2 = 2.0 * b1 - b0
    l3 = 6.0 * b2 - 6.0 * b1 + b0
    return np.concatenate([l1, l2, l3])


def _within_day_cov_vech(day_emb: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Off-diagonal coupling: vech of the within-day covariance in a fixed basis.

    Projects the day's valid epochs onto a fixed train-fit top-r PCA basis,
    estimates the r x r covariance with Ledoit-Wolf shrinkage toward the diagonal
    (robust for patch-sparse days), and returns the upper triangle (including the
    diagonal), r(r+1)/2 values. std is the diagonal of the full-D covariance; this
    adds the off-diagonal coupling the marginal pooling discards.

    Args:
        day_emb: (n_epochs, D) float64 valid in-day epochs.
        mean:    (D,) PCA mean (train-fit).
        basis:   (D, r) PCA components.T (train-fit).
    Returns:
        (r(r+1)/2,) float64. For r=16 this is 136.
    """
    from sklearn.covariance import LedoitWolf

    z = (day_emb - mean) @ basis                      # (n, r)
    cov = LedoitWolf(assume_centered=False).fit(z).covariance_
    iu = np.triu_indices(cov.shape[0])
    return cov[iu]


def _resp_weighted_meanstd(day_emb: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Respiratory-event-weighted per-dim [mean, std] over a day's epochs.

    Re-weights the daily moments toward the rare respiratory-event epochs where
    AR's unique, epoch-localized signal lives, countering the bulk-wake dilution
    the plain daily mean suffers. Falls back to unweighted moments when the day
    carries no resp-event mass (sleep-free or all-NaN softpreds).

    Args:
        day_emb: (n_epochs, D) float64 valid in-day epochs.
        w:       (n_epochs,) float64 P(resp event), sliced to the same mask,
                 already NaN-zeroed and clipped to [0, 1] by the caller.
    Returns:
        (2*D,) float64.
    """
    ws = float(w.sum())
    if ws <= 0.0:
        return np.concatenate([day_emb.mean(axis=0), day_emb.std(axis=0)])
    wn = (w / ws)[:, None]
    mu = (wn * day_emb).sum(axis=0)
    var = (wn * (day_emb - mu) ** 2).sum(axis=0)
    sd = np.sqrt(np.maximum(var, 0.0))
    return np.concatenate([mu, sd])


def _lombscargle_amplitudes(day_emb: np.ndarray, t_hours: np.ndarray) -> np.ndarray:
    """Per-dim Lomb-Scargle amplitude at circadian harmonics (irregular t).

    The L-moment pooling is permutation-invariant over epochs, so it discards
    when values occur. This recovers per-dim within-day rhythm amplitude at
    24/12/8/6h from the (gappy, nonwear-irregular) epoch timestamps via
    Lomb-Scargle. Amplitude only; phase coherence is not computed here.

    Args:
        day_emb: (n, D) float64 valid in-day epochs (time-ordered).
        t_hours: (n,) float64 within-day epoch times in hours (irregular).
    Returns:
        (H*D,) float64 = [amp(24h), amp(12h), amp(8h), amp(6h)], each block (D,).
    """
    from scipy.signal import lombscargle

    ang = 2.0 * np.pi / _CIRC_PERIODS_H               # angular freqs (rad/hr)
    n, D = day_emb.shape
    out = np.empty((len(ang), D), dtype=np.float64)
    for j in range(D):
        x = day_emb[:, j]
        x = x - x.mean()                              # lombscargle needs zero-mean
        if not np.any(x):
            out[:, j] = 0.0
            continue
        pgram = lombscargle(t_hours, x, ang, normalize=False)
        out[:, j] = np.sqrt(np.maximum(pgram, 0.0) * 2.0 / n)   # ~amplitude
    return out.reshape(-1)


def per_modality_day_distrib(
    emb_path: str,
    start_time: pd.Timestamp,
    patch_sec: float,
    min_day_valid_patches: int,
    moments: str = "meanstd",
    pca: dict | None = None,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-day distributional summary for one modality.

    `moments`:
      "meanstd"     : [mean, std] (2*D)
      "l3"          : first 3 L-moments (3*D)
      "meanstd_cov" : [mean, std, vech(cov_r)] (off-diagonal coupling; needs `pca`)
      "l3_physio"   : [l1, l2, l3] + resp-weighted [mean, std] if `weights` given,
                      else [l1, l2, l3] (AR-only extra; pass weights for AR, None for HA)
      "l3_circ"     : [l1, l2, l3, LS circadian amplitudes]

    `pca`:     {"mean": (D,), "basis": (D,r)} for "meanstd_cov" (else None).
    `weights`: (n_epochs,) per-epoch resp-event prob for "l3_physio" (AR only).

    Returns (day_codes (n,), per_day_distrib (n, K*D)) or None.
    """
    emb = np.load(emb_path)
    valid = ~np.isnan(emb[:, 0])
    if not valid.any():
        return None

    all_idx = np.arange(emb.shape[0])
    offsets = pd.to_timedelta(np.arange(emb.shape[0]) * patch_sec, unit="s")
    ts = start_time + offsets
    day_floor = (ts - pd.Timedelta(hours=12)).floor("D")
    day_codes_all = day_floor.astype("int64").to_numpy()
    unique_days, day_idx = np.unique(day_codes_all, return_inverse=True)

    kept_codes: list[int] = []
    kept_vecs: list[np.ndarray] = []
    for d in range(len(unique_days)):
        mask = (day_idx == d) & valid
        n_valid = int(mask.sum())
        if n_valid < min_day_valid_patches:
            continue
        day_emb = emb[mask].astype(np.float64)
        if moments == "l3":
            vec = _l_moments(day_emb)
        elif moments == "meanstd_cov":
            base = np.concatenate([day_emb.mean(axis=0), day_emb.std(axis=0)])
            cov = _within_day_cov_vech(day_emb, pca["mean"], pca["basis"])
            vec = np.concatenate([base, cov])
        elif moments == "l3_physio":
            base = _l_moments(day_emb)
            if weights is not None:
                w_day = np.clip(np.nan_to_num(weights[mask], nan=0.0), 0.0, 1.0)
                vec = np.concatenate([base, _resp_weighted_meanstd(day_emb, w_day)])
            else:
                vec = base
        elif moments == "l3_circ":
            base = _l_moments(day_emb)
            idx = all_idx[mask].astype(np.float64)
            t_hours = (idx - idx.min()) * patch_sec / 3600.0
            vec = np.concatenate([base, _lombscargle_amplitudes(day_emb, t_hours)])
        else:
            vec = np.concatenate([day_emb.mean(axis=0), day_emb.std(axis=0)])
        kept_codes.append(int(unique_days[d]))
        kept_vecs.append(vec)

    if not kept_codes:
        return None
    return (
        np.asarray(kept_codes, dtype=np.int64),
        np.stack(kept_vecs).astype(np.float32),
    )


def per_subject_day_distrib(
    subj_dir: str,
    min_day_valid_patches: int,
    min_subject_valid_frac: float,
    moments: str = "meanstd",
    pca: dict | None = None,
) -> dict | None:
    with open(os.path.join(subj_dir, "meta.json")) as f:
        meta = json.load(f)
    start_time = pd.to_datetime(meta["start_time"])

    out: dict[str, np.ndarray] = {}
    for model in ("ar", "ha"):
        emb_path = os.path.join(subj_dir, MODEL_FILE[model])
        if not os.path.exists(emb_path):
            return None
        try:
            emb_mm = np.load(emb_path, mmap_mode="r")
            valid_frac = float((~np.isnan(emb_mm[:, 0])).mean()) if emb_mm.size else 0.0
            n_epochs = emb_mm.shape[0] if emb_mm.size else 0
            del emb_mm
        except Exception:
            return None
        if valid_frac < min_subject_valid_frac:
            return None

        # Per-modality PCA basis for meanstd_cov.
        pca_m = None
        if moments == "meanstd_cov":
            if pca is None:
                return None
            pca_m = {"mean": pca[f"{model}_mean"], "basis": pca[f"{model}_basis"]}

        # AR-only respiratory-event weights for l3_physio (HA stays unweighted).
        weights = None
        if moments == "l3_physio" and model == "ar":
            resp_path = os.path.join(subj_dir, AR_RESPEVENTS_FILE)
            if not os.path.exists(resp_path):
                return None
            soft = np.load(resp_path)
            if soft.shape[0] != n_epochs:
                return None
            weights = soft[:, RESP_EVENT_COL].astype(np.float64)

        res = per_modality_day_distrib(
            emb_path,
            start_time=start_time,
            patch_sec=PATCH_SECONDS[model],
            min_day_valid_patches=min_day_valid_patches,
            moments=moments,
            pca=pca_m,
            weights=weights,
        )
        if res is None:
            return None
        codes, vecs = res
        out[f"{model}_codes"] = codes
        out[f"{model}_vecs"] = vecs

    common, ar_idx, ha_idx = np.intersect1d(
        out["ar_codes"], out["ha_codes"], return_indices=True
    )
    if common.size == 0:
        return None
    return {
        "day_codes": common,
        "ar": out["ar_vecs"][ar_idx].astype(np.float32),
        "ha": out["ha_vecs"][ha_idx].astype(np.float32),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-root", default=CACHE_ROOT_V2)
    ap.add_argument("--out-root", default=DAY_DISTRIB_ROOT_DEFAULT)
    ap.add_argument("--split", required=True, choices=["audit", "test", "train"])
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--min-day-valid-patches", type=int, default=100)
    ap.add_argument("--min-subject-valid-frac", type=float, default=0.05)
    ap.add_argument("--moments", default="meanstd",
                    choices=["meanstd", "l3", "meanstd_cov", "l3_physio", "l3_circ"],
                    help="meanstd = [mean, std]; l3 = 3 L-moments; "
                         "meanstd_cov = + within-day cov vech (needs --pca-basis-path); "
                         "l3_physio = l3 + AR resp-weighted [mean, std]; "
                         "l3_circ = l3 + Lomb-Scargle circadian amplitudes.")
    ap.add_argument("--pca-basis-path", default=None,
                    help="meanstd_cov: path to the train-only PCA basis .npz "
                         "(keys ar_mean/ar_basis/ha_mean/ha_basis from fit_pca_basis.py).")
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    # Load the fixed train-only PCA basis once (never refit per split).
    pca = None
    if args.moments == "meanstd_cov":
        if not args.pca_basis_path:
            raise SystemExit("--pca-basis-path is required for --moments meanstd_cov")
        pca = {k: v for k, v in np.load(resolve_oak_path(args.pca_basis_path)).items()}
        r = pca["ar_basis"].shape[1]
        if args.min_day_valid_patches < r + 1:
            raise SystemExit(
                f"--min-day-valid-patches {args.min_day_valid_patches} < r+1 ({r + 1}), "
                f"covariance under-determined.")
        print(f"[compute_day_lmoments] loaded PCA basis r={r} from {args.pca_basis_path}")

    cache_split = resolve_oak_path(os.path.join(args.cache_root, args.split))
    out_split = resolve_oak_path(os.path.join(args.out_root, args.split))
    os.makedirs(out_split, exist_ok=True)
    print(f"[compute_day_lmoments] cache: {cache_split}")
    print(f"[compute_day_lmoments] out:   {out_split}")

    subjects = list_subjects(cache_split)
    print(f"[compute_day_lmoments] total subjects in split: {len(subjects)}")
    if args.n_tasks > 1:
        subjects = [s for i, s in enumerate(subjects) if i % args.n_tasks == args.task_id]
        print(f"[compute_day_lmoments] task {args.task_id}/{args.n_tasks}: {len(subjects)} subjects")
    if args.max_subjects:
        subjects = subjects[: args.max_subjects]

    n_done = n_skip_exist = n_skip_low = n_err = 0
    t0 = time.time()
    for i, (eid_run, subj_dir) in enumerate(subjects):
        out_path = os.path.join(out_split, f"{eid_run}.npz")
        if not args.no_resume and os.path.exists(out_path):
            n_skip_exist += 1
            continue
        try:
            res = per_subject_day_distrib(
                subj_dir,
                min_day_valid_patches=args.min_day_valid_patches,
                min_subject_valid_frac=args.min_subject_valid_frac,
                moments=args.moments,
                pca=pca,
            )
            if res is None:
                n_skip_low += 1
                continue
            tmp = out_path + ".tmp"
            with open(tmp, "wb") as fh:
                np.savez(fh, **res)
            os.rename(tmp, out_path)
            n_done += 1
        except Exception as e:
            n_err += 1
            print(f"[compute_day_lmoments] ERROR on {eid_run}: {type(e).__name__}: {e}")
        if (i + 1) % 200 == 0:
            el = time.time() - t0
            rate = (i + 1) / el if el > 0 else 0
            print(f"[compute_day_lmoments] {i+1}/{len(subjects)} done={n_done} "
                  f"skip_exist={n_skip_exist} skip_low={n_skip_low} err={n_err} "
                  f"rate={rate:.1f}/s eta={(len(subjects)-(i+1))/max(rate,1e-3)/60:.1f}min")

    print(f"[compute_day_lmoments] DONE in {(time.time()-t0)/60:.1f}min: "
          f"done={n_done} skip_exist={n_skip_exist} skip_low={n_skip_low} err={n_err}")


if __name__ == "__main__":
    main()
