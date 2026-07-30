"""Core data layer for the BMI-from-actigraphy analysis.

Loads frozen AR+HA daily embeddings from the split caches, pools to a
per-subject vector (mask-mean over valid days, then mean over a subject's
recordings), and joins the BMI label (UKB field 21001-0.0, instance-0 baseline)
plus demographics (age_at_first_run, sex) and age-at-recruitment (21022-0.0, for
the stale-label gap). Includes the sealed-test guard: test embeddings cannot be
loaded without allow_test=True, and every such load is logged.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

from ukb_disease.paths import UKB_ROOT
from ukb_disease.baseline.config import (
    CANONICAL_DAY_LMOMENT_ROOT,
    CANONICAL_DAY_MEAN_ROOT,
    resolve_oak_path,
)

# --- inputs (read-only) ----------------------------------------------------
BMI_CSV = f"{UKB_ROOT}/metadata/BMI_yes.csv"
DEM_CSV = f"{UKB_ROOT}/metadata/acc_dem.csv"
POOL_ROOTS = {"mean": CANONICAL_DAY_MEAN_ROOT, "l3": CANONICAL_DAY_LMOMENT_ROOT}

# --- outputs ---------------------------------------------------------------
LOOP_DIR = f"{UKB_ROOT}/bmi_loop"
RUNS_DIR = os.path.join(LOOP_DIR, "runs")
LEDGER = os.path.join(LOOP_DIR, "bmi_ledger.jsonl")
TEST_MANIFEST = os.path.join(LOOP_DIR, "test_manifest.json")
TEST_LOOKS = os.path.join(LOOP_DIR, "test_looks.jsonl")
os.makedirs(RUNS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# pooling + subject aggregation
# ---------------------------------------------------------------------------
def _pool_recordings(pooling: str, split: str, modality: str):
    """Load a split cache and mask-mean over valid days -> per-recording vectors.

    Returns (eids[int64] (N,), X[float32] (N, F)). modality in {ar, ha, both}.
    """
    root = POOL_ROOTS[pooling]
    d = os.path.join(resolve_oak_path(root), split)
    mask = np.load(os.path.join(d, "valid_day_mask.npy")).astype(np.float32)  # (N, D)
    idx = pd.read_parquet(os.path.join(d, "index.parquet"))
    mods = {"ar": ["ar"], "ha": ["ha"], "both": ["ar", "ha"]}[modality]
    mats = []
    for name in mods:
        emb = np.load(os.path.join(d, f"{name}.npy"))  # (N, D, F)
        finite = np.isfinite(emb)
        w = mask[:, :, None] * finite  # (N, D, F) weight per day per feature
        num = (np.where(finite, emb, 0.0) * w).sum(axis=1)  # (N, F)
        den = w.sum(axis=1)  # (N, F)
        mats.append((num / np.maximum(den, 1e-8)).astype(np.float32))
    X = np.concatenate(mats, axis=1) if len(mats) > 1 else mats[0]
    return idx["eid"].to_numpy(np.int64), X


_SPLIT_EIDSETS: dict = {}


def _split_eidset(pooling: str, split: str) -> set:
    """Cached set of unique subject eids for a split (from its index.parquet)."""
    key = (pooling, split)
    if key not in _SPLIT_EIDSETS:
        d = os.path.join(resolve_oak_path(POOL_ROOTS[pooling]), split)
        idx = pd.read_parquet(os.path.join(d, "index.parquet"), columns=["eid"])
        _SPLIT_EIDSETS[key] = set(int(e) for e in idx["eid"].unique())
    return _SPLIT_EIDSETS[key]


def _aggregate_subject(eids: np.ndarray, X: np.ndarray):
    """Mean-pool a subject's multiple recordings -> one vector per eid."""
    order = np.argsort(eids, kind="stable")
    eids_s, Xs = eids[order], X[order]
    uniq, start = np.unique(eids_s, return_index=True)
    sums = np.add.reduceat(Xs, start, axis=0)
    counts = np.diff(np.append(start, len(eids_s))).astype(np.float32)
    return uniq, (sums / counts[:, None]).astype(np.float32)


# ---------------------------------------------------------------------------
# labels + demographics
# ---------------------------------------------------------------------------
def load_bmi(eids):
    """BMI (field 21001-0.0) and age-at-recruitment (21022-0.0), aligned to eids."""
    df = (
        pd.read_csv(resolve_oak_path(BMI_CSV), usecols=["eid", "21001-0.0", "21022-0.0"])
        .rename(columns={"21001-0.0": "bmi", "21022-0.0": "age_recruit"})
        .groupby("eid")
        .mean()
    )
    bmi = df.reindex(eids)["bmi"].to_numpy(float)
    age_recruit = df.reindex(eids)["age_recruit"].to_numpy(float)
    return bmi, age_recruit


def load_demo(eids):
    """age_at_first_run (age at wear) and sex (0/1), aligned to eids."""
    dem = pd.read_csv(resolve_oak_path(DEM_CSV)).set_index("eid")
    age = dem.reindex(eids)["age_at_first_run"].to_numpy(float)
    sex = dem.reindex(eids)["sex"].to_numpy(float)
    return age, sex


# ---------------------------------------------------------------------------
# sealed-test guard
# ---------------------------------------------------------------------------
def _eids_sha256(eids) -> str:
    arr = np.asarray(sorted(int(e) for e in eids), dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _vals_sha256(eids, vals) -> str:
    order = np.argsort(np.asarray(eids, dtype=np.int64))
    arr = np.asarray(vals, dtype=np.float64)[order]
    return hashlib.sha256(np.round(arr, 6).tobytes()).hexdigest()


def seal_test() -> dict:
    """Write the immutable test descriptor ONCE (eids + BMI labels + sha256).

    Reads only eids/labels (no embeddings, no predictions), so it is not a look.
    """
    eids, _ = _pool_recordings("mean", "test", "ar")
    seids, _ = _aggregate_subject(eids, np.zeros((len(eids), 1), np.float32))
    bmi, _ = load_bmi(seids)
    keep = np.isfinite(bmi)
    seids, bmi = seids[keep], bmi[keep]
    manifest = {
        "split": "disjoint_split_v1",
        "n_test_subjects": int(len(seids)),
        "test_eids_sha256": _eids_sha256(seids),
        "test_labels_sha256": _vals_sha256(seids, bmi),
        "bmi_mean": float(np.mean(bmi)),
        "bmi_sd": float(np.std(bmi)),
        "created_unix": time.time(),
        "split_fn": "_split_hash.hash_split",
    }
    with open(TEST_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def assert_test_unseen() -> None:
    """Refuse to proceed if a headline test look already happened."""
    if os.path.exists(TEST_LOOKS):
        with open(TEST_LOOKS) as f:
            for line in f:
                if line.strip() and json.loads(line).get("purpose") == "headline":
                    raise RuntimeError(
                        "A headline test look already exists in test_looks.jsonl; "
                        "the test set is spent for this candidate."
                    )


def log_test_look(record: dict) -> None:
    record = {"timestamp_unix": time.time(), **record}
    with open(TEST_LOOKS, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# top-level dataset assembly
# ---------------------------------------------------------------------------
def load_dataset(split: str, pooling: str = "mean", modality: str = "both",
                 *, allow_test: bool = False) -> dict:
    """Subject-level (X, bmi, age, sex, age_recruit) for a split, BMI-complete.

    TEST embeddings require allow_test=True (sealed-test guard).
    """
    if split == "test" and not allow_test:
        raise RuntimeError(
            "TEST embeddings requested in a test-blind context. "
            "Pass allow_test=True only from the final-gate entry point."
        )
    eids_r, X_r = _pool_recordings(pooling, split, modality)
    eids, X = _aggregate_subject(eids_r, X_r)
    bmi, age_recruit = load_bmi(eids)
    age, sex = load_demo(eids)
    keep = np.isfinite(bmi) & np.isfinite(X).all(axis=1)
    eids, X, bmi, age, sex, age_recruit = (
        eids[keep], X[keep], bmi[keep], age[keep], sex[keep], age_recruit[keep]
    )
    # Leakage guard: this split's subjects must be disjoint from the other two.
    # The split is a hybrid of a base split plus md5-hashed neuro subjects, so
    # per-eid hash_split membership does not hold, but cross-split disjointness
    # does.
    others = {"train", "val", "test"} - {split}
    here = set(int(e) for e in eids)
    for o in others:
        overlap = here & _split_eidset(pooling, o)
        if overlap:
            raise RuntimeError(
                f"split leakage: {len(overlap)} subjects in '{split}' also in '{o}'"
            )
    return {
        "split": split, "pooling": pooling, "modality": modality,
        "eids": eids, "X": X, "bmi": bmi, "age": age, "sex": sex,
        "age_recruit": age_recruit, "n": int(len(eids)),
    }
