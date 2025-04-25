#!/usr/bin/env python3
"""
Re-build NRMS memmap caches (Spark version) for both 'train' & 'dev'.
Usage:
    PYSPARK_PYTHON=$(which python) spark-submit --driver-memory 100g \
        scripts/rebuild_nrms_cache_spark.py
"""
from pathlib import Path
import logging
from turbo_rank.data.preprocess_spark import load_and_prepare_nrms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "large"

for split in ("train", "dev"):
    cand, hist, lbl, vocab = load_and_prepare_nrms(DATA_DIR, split=split, regen=True)
    logging.info(
        "✔️  %s cache rebuilt | cand %s | hist %s | #labels %d | vocab %d | dir %s",
        split,
        cand.shape,
        hist.shape,
        len(lbl),
        vocab,
        (DATA_DIR / split / 'cache').resolve(),
    )