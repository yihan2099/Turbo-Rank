"""Tensor-friendly NRMS dataset."""
import torch
from torch.utils.data import Dataset

class NRMSDataset(Dataset):
    def __init__(self, cand, hist, labels):
        self.cand  = torch.tensor(cand,  dtype=torch.long)          # (N, L)
        self.hist  = torch.tensor(hist,  dtype=torch.long)          # (N, H, L)
        self.label = torch.tensor(labels,dtype=torch.float32)       # (N)

    def __len__(self): return len(self.label)

    def __getitem__(self, idx):
        return {"cand": self.cand[idx],
                "hist": self.hist[idx],
                "label": self.label[idx]}