"""Filesystem locations.

The data (13 GB raw) lives outside Dropbox so it is never synced. Override the
root with the TFB_DATA environment variable.
"""
from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("TFB_DATA", Path.home() / "data" / "tfb")).expanduser()
RELEASE = DATA_ROOT / "kaggle_public"  # the unpacked Kaggle release; the scripts' --release-root
DENSE = DATA_ROOT / "dense"  # per-panel dense arrays
CACHE = DATA_ROOT / "cache"
SUBMISSIONS = DATA_ROOT / "submissions"

PROJECT = Path(__file__).resolve().parents[2]
ORGANIZER_REPO = PROJECT / "organizer-repo"

PANELS = (
    "D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N",
    "D7_I405_S", "D12_I5_N", "D12_I5_S", "D12_I405_N", "D12_I405_S",
)
TASK2_EXCLUDED = ("D12_I405_N", "D12_I405_S")
SPLITS = ("train", "validation", "private")
