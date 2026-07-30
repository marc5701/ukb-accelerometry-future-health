"""Filesystem roots for the analysis code.

Every input (embeddings, per-day summaries, phecode maps, demographics, model
run outputs) lives under a single root so the pipeline is portable. Set the
``UKB_ROOT`` environment variable to point at your local copy; it defaults to a
``data`` directory beside this package.

No data ships with this repository. UK Biobank inputs are available to approved
researchers through the UK Biobank Access Management System (this study used
Application 62249). The two accelerometer foundation-model encoders are obtained
separately from their published sources.
"""
from __future__ import annotations

import os
from pathlib import Path

UKB_ROOT = Path(os.environ.get("UKB_ROOT", Path(__file__).resolve().parent.parent / "data"))


def resolve_oak_path(path) -> str:
    """Return *path* as a string.

    A single call site for path resolution; constants are already expressed
    relative to ``UKB_ROOT``, so this is an identity apart from normalising to
    ``str``.
    """
    return str(path)
