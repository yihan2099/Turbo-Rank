# turbo_rank/modeling/models/nrms.py
"""Neural News Recommendation with Multi‑Head Self‑Attention (NRMS)
------------------------------------------------------------------
This implementation follows the original NRMS paper (Wu et al., 2019)
with a few pragmatic simplifications so that it integrates cleanly
into the existing *turbo_rank* code‑base and runs on the MIND dataset.

The model is decomposed into three reusable blocks:

1. **NewsEncoder**   – encodes a single news title/abstract sequence
   into a fixed‑length vector using word embeddings + multi‑head self‑attention.
2. **UserEncoder**   – aggregates a user’s clicked‑news sequence into a
   single vector, again via multi‑head self‑attention.
3. **NRMSModel**     – scores a *candidate* news article against the user
   representation with a dot‑product (or learnable linear layer).

All PyTorch modules are torchscript‑friendly and free of
third‑party dependencies beyond `torch>=2.2`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Attention utilities
# -----------------------------------------------------------------------------


class _ScaledDotSelfAttention(nn.Module):
    """A thin wrapper around :class:`torch.nn.MultiheadAttention` (batch‑first).

    This is factored out so the same attention block can be reused in both the
    news‑ and user‑encoders.  It returns **only** the transformed values – the
    attention map is not required for inference and is therefore discarded.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, D)
        y, _ = self.attn(x, x, x, need_weights=False)
        return y


# -----------------------------------------------------------------------------
# Encoder blocks
# -----------------------------------------------------------------------------


class NewsEncoder(nn.Module):
    """Encodes a single news article (token ids) into a fixed‑length vector."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        num_heads: int = 8,
        max_len: int = 30,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=padding_idx,
        )
        self.self_attn = _ScaledDotSelfAttention(embed_dim, num_heads)
        # Additive attention – weights each token vector
        self.additive = nn.Linear(embed_dim, 1, bias=False)
        self.max_len = max_len

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # (B, L)
        # Shape bookkeeping ----------------------------------------------------
        if tokens.dim() != 2:
            raise ValueError("Expected shape (batch, seq_len)")

        if tokens.size(1) > self.max_len:
            tokens = tokens[:, : self.max_len]

        x = self.embedding(tokens)  # (B, L, D)
        h = self.self_attn(x)  # (B, L, D)
        w = self.additive(h).squeeze(-1)  # (B, L)
        alpha = F.softmax(w, dim=-1)  # (B, L)
        v = torch.sum(h * alpha.unsqueeze(-1), dim=1)  # (B, D)
        return v


class UserEncoder(nn.Module):
    """Aggregates the sequence of clicked news vectors into a user vector."""

    def __init__(self, embed_dim: int = 300, num_heads: int = 8):
        super().__init__()
        self.self_attn = _ScaledDotSelfAttention(embed_dim, num_heads)
        self.additive = nn.Linear(embed_dim, 1, bias=False)

    def forward(self, news_vecs: torch.Tensor) -> torch.Tensor:  # (B, H, D)
        h = self.self_attn(news_vecs)  # (B, H, D)
        w = self.additive(h).squeeze(-1)  # (B, H)
        alpha = F.softmax(w, dim=-1)  # (B, H)
        u = torch.sum(h * alpha.unsqueeze(-1), dim=1)  # (B, D)
        return u


# -----------------------------------------------------------------------------
# NRMS model
# -----------------------------------------------------------------------------


class NRMSModel(nn.Module):
    """Full NRMS – score candidate news given user click history."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        num_heads: int = 8,
        max_news_len: int = 30,
        max_hist_len: int = 50,
    ) -> None:
        super().__init__()
        self.max_hist_len = max_hist_len
        self.news_encoder = NewsEncoder(vocab_size, embed_dim, num_heads, max_news_len)
        self.user_encoder = UserEncoder(embed_dim, num_heads)

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------

    def forward(
        self,
        candidate_tokens: torch.Tensor,  # (B, L)
        history_tokens: torch.Tensor,  # (B, H, L)
    ) -> torch.Tensor:  # (B)
        B, H, L = history_tokens.size()

        # Encode candidate ------------------------------------------------------
        v_c = self.news_encoder(candidate_tokens)  # (B, D)

        # Encode history – flatten so we can reuse NewsEncoder ------------------
        history_tokens = history_tokens.view(B * H, L)
        v_h = self.news_encoder(history_tokens)  # (B*H, D)
        v_h = v_h.view(B, H, -1)  # (B, H, D)
        u = self.user_encoder(v_h)  # (B, D)

        # ---- stabilise -------------------------------------------------------------
        u   = F.normalize(u,   dim=-1)             #  ❰ new ❱  L2-normalise
        v_c = F.normalize(v_c, dim=-1)             #  ❰ new ❱
        scores = torch.sum(u * v_c, dim=-1)        # (B)     ~cosine similarity
        return scores

    # ---------------------------------------------------------------------
    # Convenient checkpoints / export helpers
    # ---------------------------------------------------------------------

    def save_pretrained(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load_pretrained(cls, path: str | Path, **kwargs) -> "NRMSModel":
        model = cls(**kwargs)
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        return model
