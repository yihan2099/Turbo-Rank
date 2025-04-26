"""Centralised project paths & defaults."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # turbo_rank/
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "small"  # memmap cache
MLFLOW_EXPERIMENT = "turbo_rank_experiments"

# Lazily ensure directories exist (only when imported in a writer context).
DATA_DIR.mkdir(parents=True, exist_ok=True)
