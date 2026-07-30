"""Paths and model specs for embedding extraction.

Paths are expressed relative to ``UKB_ROOT`` (see ``ukb_disease.paths``). No data
or model weights ship with this repository; the two encoders are obtained from
their published sources (loaded via torch.hub in ``run_extraction``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ukb_disease.paths import UKB_ROOT, resolve_oak_path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Legacy split lists (flat JSON lists of <eid>_<run>.h5); superseded by the
# subject-disjoint split below but kept for reference.
LEGACY_TRAIN_JSON = f"{UKB_ROOT}/splits/ukb_train_files.json"
LEGACY_TEST_JSON = f"{UKB_ROOT}/splits/ukb_test_files.json"

# Preprocessed UKB recordings (H5 source files for extraction).
UKB_PP_DIR = f"{UKB_ROOT}/preprocessed_h5"
UKB_EXCLUDED_DIR = f"{UKB_ROOT}/excluded_h5"  # calibration failures; never read embeddings from here

# Subject-disjoint split: train/val/test are disjoint at the subject level, with a
# neuro hold-out (PD/AD/dementia) also available. Each JSON is a flat list of
# <eid>_<run>.h5 paths; only the basenames (IDs) are used.
UKB_SPLIT_FULL_DIR = f"{UKB_ROOT}/splits/ukb_split_full"
NEW_SPLIT_JSON = {
    "train": f"{UKB_SPLIT_FULL_DIR}/ukb_train_files_all.json",
    "val":   f"{UKB_SPLIT_FULL_DIR}/ukb_val_files_all.json",
    "test":  f"{UKB_SPLIT_FULL_DIR}/ukb_test_files_all.json",
    "neuro": f"{UKB_SPLIT_FULL_DIR}/ukb_neuro_files.json",
}

# Extraction output cache.
CACHE_ROOT = f"{UKB_ROOT}/embeddings"

# Code root (this package) and extraction manifests.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
MANIFEST_DIR = f"{UKB_ROOT}/manifests"

# Encoder checkpoints. Obtained externally (see run_extraction / torch.hub); paths
# are provided for the local-load path only.
AR_AUTOENCODER_PT = f"{UKB_ROOT}/encoders/accelerest_autoencoder.pt"
HA_AUTOENCODER_PT = f"{UKB_ROOT}/encoders/mae_lmm.pt"
AR_SLEEPSTAGE_LINEAR_PT = f"{UKB_ROOT}/encoders/accelerest_sleepstage_linear.pt"
AR_RESPEVENT_LINEAR_PT = f"{UKB_ROOT}/encoders/accelerest_respevent_linear.pt"

# Demographics and labels (used in the modelling phase; listed here for completeness).
ACC_DEM_CSV = f"{UKB_ROOT}/metadata/acc_dem.csv"
MORTALITY_PHECODES_CSV = f"{UKB_ROOT}/metadata/mortality_and_phecodes.csv"

# ---------------------------------------------------------------------------
# Model specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoundationModelSpec:
    name: str
    weights_path: str
    window_size_min: int   # context window length (minutes)
    patch_size_sec: int    # patch / epoch length (seconds)
    embed_dim: int         # output embedding dimensionality

    @property
    def patches_per_window(self) -> int:
        return self.window_size_min * 60 // self.patch_size_sec

    @property
    def epochs_per_day(self) -> int:
        return 24 * 60 * 60 // self.patch_size_sec


ACCELEREST = FoundationModelSpec(
    name="accelerest",
    weights_path=AR_AUTOENCODER_PT,
    window_size_min=128,
    patch_size_sec=30,
    embed_dim=256,
)

HA_MODEL = FoundationModelSpec(
    name="ha",
    weights_path=HA_AUTOENCODER_PT,
    window_size_min=50,
    patch_size_sec=10,
    embed_dim=256,
)

FM_SPECS = {"accelerest": ACCELEREST, "ha": HA_MODEL}

# ---------------------------------------------------------------------------
# Extraction parameters
# ---------------------------------------------------------------------------

SEGMENT_LENGTH_MIN = 1450   # 24h10m per segment
START_HOUR = 8              # clock-anchored start
START_MINUTE = 55
SEED = 42

# Nonwear masking threshold: an epoch is nonwear if the mean of the underlying
# 30 Hz mask is >= this value.
NONWEAR_EPOCH_THRESHOLD = 0.5

# Storage precision for embeddings.
EMBEDDING_DTYPE = "float32"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def map_split_path_to_local(split_path: str) -> str:
    """Map a split-list path to the local preprocessed_h5 pool by basename."""
    return os.path.join(UKB_PP_DIR, os.path.basename(split_path))


def is_calib_failed(local_path: str) -> bool:
    """True if the file is in the excluded_h5/ directory (calibration failure)."""
    basename = os.path.basename(local_path)
    candidate = os.path.join(UKB_EXCLUDED_DIR, basename)
    return os.path.exists(resolve_oak_path(candidate))


def cache_path(split: str, eid_run: str) -> str:
    """Per-recording cache H5 path."""
    return os.path.join(CACHE_ROOT, split, f"{eid_run}.h5")


def cache_split_dir(split: str) -> str:
    """Per-split cache directory."""
    return os.path.join(CACHE_ROOT, split)


def path_exists_resolved(path: str) -> bool:
    """os.path.exists on a resolved path."""
    return os.path.exists(resolve_oak_path(path))


def eid_run_from_h5(h5_path: str) -> str:
    """Extract the '<eid>_<run>' stem from an H5 path."""
    return os.path.basename(h5_path).replace(".h5", "")


__all__ = [
    "ACC_DEM_CSV",
    "AR_AUTOENCODER_PT",
    "AR_RESPEVENT_LINEAR_PT",
    "AR_SLEEPSTAGE_LINEAR_PT",
    "ACCELEREST",
    "CACHE_ROOT",
    "EMBEDDING_DTYPE",
    "FM_SPECS",
    "FoundationModelSpec",
    "HA_AUTOENCODER_PT",
    "HA_MODEL",
    "MANIFEST_DIR",
    "MORTALITY_PHECODES_CSV",
    "LEGACY_TEST_JSON",
    "LEGACY_TRAIN_JSON",
    "NONWEAR_EPOCH_THRESHOLD",
    "PROJECT_ROOT",
    "SEED",
    "SEGMENT_LENGTH_MIN",
    "START_HOUR",
    "START_MINUTE",
    "UKB_EXCLUDED_DIR",
    "UKB_PP_DIR",
    "cache_path",
    "cache_split_dir",
    "eid_run_from_h5",
    "is_calib_failed",
    "map_split_path_to_local",
    "path_exists_resolved",
    "resolve_oak_path",
]
