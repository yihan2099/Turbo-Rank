"""PyTorch dataset (user, item, label) for Two-Tower."""

import torch
from torch.utils.data import Dataset


class MindInteractionDataset(Dataset):
    def __init__(self, df):
        self.u = torch.tensor(df["user_id"].values, dtype=torch.long)
        self.i = torch.tensor(df["item_id"].values, dtype=torch.long)
        self.y = torch.tensor(df["label"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"user": self.u[idx], "item": self.i[idx], "label": self.y[idx]}
