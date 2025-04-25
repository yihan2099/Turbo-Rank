"""Neural News Recommendation (NRMS) – optimised implementation."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _xavier_init(m: nn.Module) -> None:
    if isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.xavier_uniform_(m.weight)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
class _ScaledDotSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, D)
        y, _ = self.attn(x, x, x, need_weights=False)
        return y


# ---------------------------------------------------------------------------
class NewsEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        num_heads: int = 8,
        max_len: int = 30,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.self_attn = _ScaledDotSelfAttention(embed_dim, num_heads)
        self.additive = nn.Linear(embed_dim, 1, bias=False)
        self.max_len = max_len
        self.apply(_xavier_init)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # (B, L)
        if tokens.size(1) > self.max_len:
            tokens = tokens[:, : self.max_len]

        x = self.embedding(tokens)
        h = self.self_attn(x)
        w = self.additive(h).squeeze(-1)
        alpha = F.softmax(w, dim=-1)
        v = torch.sum(h * alpha.unsqueeze(-1), dim=1)
        return v


# ---------------------------------------------------------------------------
class UserEncoder(nn.Module):
    def __init__(self, embed_dim: int = 300, num_heads: int = 8):
        super().__init__()
        self.self_attn = _ScaledDotSelfAttention(embed_dim, num_heads)
        self.additive = nn.Linear(embed_dim, 1, bias=False)
        self.apply(_xavier_init)

    def forward(self, news_vecs: torch.Tensor) -> torch.Tensor:  # (B, H, D)
        h = self.self_attn(news_vecs)
        w = self.additive(h).squeeze(-1)
        alpha = F.softmax(w, dim=-1)
        u = torch.sum(h * alpha.unsqueeze(-1), dim=1)
        return u


# ---------------------------------------------------------------------------
class NRMSModel(nn.Module):
    """Full NRMS with optional `torch.compile()` for acceleration."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        num_heads: int = 8,
        max_news_len: int = 30,
        max_hist_len: int = 50,
        compile_model: bool = False,
    ):
        super().__init__()
        self.max_hist_len = max_hist_len
        self.news_encoder = NewsEncoder(vocab_size, embed_dim, num_heads, max_news_len)
        self.user_encoder = UserEncoder(embed_dim, num_heads)

        if compile_model and torch.cuda.is_available():
            self.news_encoder = torch.compile(self.news_encoder)
            self.user_encoder = torch.compile(self.user_encoder)

    # ---------------------------------------------------------------------
    def forward(
        self, candidate_tokens: torch.Tensor, history_tokens: torch.Tensor
    ) -> torch.Tensor:
        B, H, L = history_tokens.size()

        v_c = self.news_encoder(candidate_tokens)

        history_tokens = history_tokens.view(B * H, L)
        v_h = self.news_encoder(history_tokens).view(B, H, -1)
        u = self.user_encoder(v_h)

        u = F.normalize(u, dim=-1, eps=1e-8)
        v_c = F.normalize(v_c, dim=-1, eps=1e-8)
        return torch.sum(u * v_c, dim=-1)

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
