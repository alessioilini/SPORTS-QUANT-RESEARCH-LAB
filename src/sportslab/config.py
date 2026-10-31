"""Paths and global research constants."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("SPORTSLAB_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = Path(os.environ.get("SPORTSLAB_DB", DATA_DIR / "sportslab.sqlite"))
REPORTS_DIR = ROOT / "reports"
ARTIFACTS_DIR = REPORTS_DIR / "artifacts"
CONFIG_DIR = ROOT / "config"

GLOBAL_SEED = 20261031

# Value provenance labels (stage 3 requirement: separate observed / model / unknown).
OBSERVED = "OBSERVED"   # fetched from an external source, unmodified
MODEL = "MODEL"         # computed by this lab
UNKNOWN = "UNKNOWN"     # not available from any verified source
VALUE_KINDS = (OBSERVED, MODEL, UNKNOWN)
