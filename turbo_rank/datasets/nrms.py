"""Memory-efficient Dataset for NRMS."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class NRMSDataset(Dataset):
    def __init__(
        self,
        cand: np.ndarray | Sequence,
        hist: np.ndarray | Sequence,
        labels: np.ndarray | Sequence,
    ):
        assert len(cand) == len(hist) == len(labels)
        self.cand, self.hist, self.labels = cand, hist, labels

    # -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # torch.as_tensor() makes a *writable* copy only when necessary,
        # suppressing the “NumPy array is not writable” warning.
        return {
            "cand": torch.as_tensor(self.cand[idx], dtype=torch.long),
            "hist": torch.as_tensor(self.hist[idx], dtype=torch.long),
            "label": torch.as_tensor(self.labels[idx], dtype=torch.float32),
        }
