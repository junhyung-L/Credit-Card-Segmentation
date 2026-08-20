"""Portable paths for legacy segmentation experiments.

Set ``CCS_DATA_DIR`` to the original dataset root.  It must contain the legacy
monthly folders plus any CSV inputs referenced by the exported experiments.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("CCS_DATA_DIR", PROJECT_ROOT / "data"))
ARTIFACT_DIR = Path(os.getenv("CCS_ARTIFACT_DIR", PROJECT_ROOT / "artifacts"))

TRAIN_CSV = Path(os.getenv("CCS_TRAIN_CSV", DATA_DIR / "train_df.csv"))
TEST_CSV = Path(os.getenv("CCS_TEST_CSV", DATA_DIR / "test_df.csv"))
TRAIN_SAMPLE_CSV = Path(os.getenv("CCS_TRAIN_SAMPLE_CSV", ARTIFACT_DIR / "train_sampled_df.csv"))
EVALUATION_CSV = Path(os.getenv("CCS_EVALUATION_CSV", ARTIFACT_DIR / "evaluation_df.csv"))


def artifact_path(filename: str) -> Path:
    """Return an output path and create its parent directory when needed."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACT_DIR / filename
