"""Deterministic, reproducible subject -> split hashing for the participant-disjoint split.

Keyed on the INTEGER eid (never the eid_run stem) so every recording of a subject
hashes to the SAME split, giving subject-level disjointness for free. Single
source of truth for the md5 cut points so they cannot drift between scripts,
which would be the key leakage risk.

The hash is salt-free and depends only on the eid, so it is identical across
machines, runs and Python versions.
"""
from __future__ import annotations

import hashlib

# train / val / test ratios matching the main ukb_split_full proportions (~89.5/5/5.5).
HASH_RATIOS = (0.895, 0.05, 0.055)
_CUT_TRAIN = HASH_RATIOS[0]
_CUT_VAL = HASH_RATIOS[0] + HASH_RATIOS[1]


def hash_unit(eid: int) -> float:
    """Stable uniform-in-[0,1) value from an integer eid."""
    # Guard: must be a bare integer eid, NEVER an 'eid_run' stem. Python's int()
    # treats '_' as a digit separator (int('1000001_1') == 10000011), so passing a
    # stem would silently hash two recordings of one subject differently => split a
    # subject across train/val/test = leakage. Fail loudly so a future caller cannot
    # regress this (all current callers pass _eid(stem) = a bare int).
    if isinstance(eid, str) and "_" in eid:
        raise ValueError(f"hash_unit expects a bare integer eid, got stem {eid!r}")
    return int(hashlib.md5(str(int(eid)).encode()).hexdigest(), 16) / float(1 << 128)


def hash_split(eid: int) -> str:
    """Deterministically assign a subject (by eid) to 'train' / 'val' / 'test'."""
    u = hash_unit(eid)
    if u < _CUT_TRAIN:
        return "train"
    if u < _CUT_VAL:
        return "val"
    return "test"
