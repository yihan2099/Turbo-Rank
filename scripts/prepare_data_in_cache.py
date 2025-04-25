#!/usr/bin/env python3
from pathlib import Path
import logging

# Make sure we see INFO logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from turbo_rank.data.preprocess import load_and_prepare_nrms

# <-- Point this at the parent of your train/ directory
data_dir = Path(__file__).resolve().parents[1] / "data" / "processed"

# Force cache rebuild
cand, hist, lbls, vocab_size = load_and_prepare_nrms(data_dir, regen=True)

print("✔️  Built cache:")
print("  • cand shape       :", cand.shape)
print("  • hist shape       :", hist.shape)
print("  • # labels         :", len(lbls))
print("  • vocab size       :", vocab_size)
print("  • cache directory  :", (data_dir / "cache").resolve())