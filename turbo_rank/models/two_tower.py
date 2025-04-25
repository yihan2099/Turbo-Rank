"""Two-Tower recommender (unchanged logic, now reusable)."""

import torch
import torch.nn as nn


class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_items, emb_dim: int = 64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, user, item):
        u, i = self.user_emb(user), self.item_emb(item)
        return self.fc(torch.cat([u, i], dim=-1)).squeeze(-1)
