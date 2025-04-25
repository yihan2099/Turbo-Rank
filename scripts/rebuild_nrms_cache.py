#!/usr/bin/env python3
"""
Re-build NRMS memmap caches for both 'train' and 'dev' splits.
Run:  python scripts/rebuild_nrms_cache.py
"""

from pathlib import Path
import logging
from turbo_rank.data.preprocess import load_and_prepare_nrms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "large"  # memmap cache

for split in ("train", "dev"):
    cand, hist, lbl, vocab = load_and_prepare_nrms(
        DATA_DIR,
        split=split,
        regen=True,           # ⇦ force rebuild
    )
    logging.info(
        "✔️  %s cache rebuilt | cand %s | hist %s | #labels %d | vocab %d | dir %s",
        split,
        cand.shape,
        hist.shape,
        len(lbl),
        vocab,
        (DATA_DIR / split / "cache").resolve(),
    )