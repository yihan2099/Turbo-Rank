"""Neural News Recommendation (NRMS) – optimised implementation."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
        # FFT-ready init
        self.attn._reset_parameters()

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        # Optionally skip padded tokens if all lengths > 0
        if lengths is not None:
            lengths_cpu = lengths.cpu()
            if torch.all(lengths_cpu > 0):
                packed = pack_padded_sequence(
                    x, lengths_cpu, batch_first=True, enforce_sorted=False
                )
                padded, _ = pad_packed_sequence(packed, batch_first=True, total_length=x.size(1))
                x = padded
        # Self-attention
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
        self.padding_idx = padding_idx
        self.max_len = max_len

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.self_attn = _ScaledDotSelfAttention(embed_dim, num_heads)
        self.additive = nn.Linear(embed_dim, 1, bias=False)

        # Custom initialization
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Xavier init for embeddings and additive
        _xavier_init(self.embedding)
        _xavier_init(self.additive)
        # Attention parameters already initialized in its constructor

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, L)
        B, L = tokens.size()
        lengths = (tokens != self.padding_idx).sum(dim=1)

        # Truncate sequences longer than max_len
        if L > self.max_len:
            tokens = tokens[:, : self.max_len]
            lengths = lengths.clamp(max=self.max_len)

        x = self.embedding(tokens)  # (B, L, D)
        # h is your per-token, contextualized embeddings (B, L, D)
        h = self.self_attn(x, lengths)  # (B, L, D)

        # w/α compute attention weights across the L tokens
        w = self.additive(h).squeeze(-1)  # (B, L)
        alpha = torch.softmax(w, dim=-1)
        # v is the attention-weighted sum of those L vectors, giving you one (B, D) output
        v = torch.sum(h * alpha.unsqueeze(-1), dim=1)  # (B, D)
        return v


# ---------------------------------------------------------------------------
class UserEncoder(nn.Module):
    def __init__(self, embed_dim: int = 300, num_heads: int = 8):
        super().__init__()
        self.self_attn = _ScaledDotSelfAttention(embed_dim, num_heads)
        self.additive = nn.Linear(embed_dim, 1, bias=False)

        # Custom initialization
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _xavier_init(self.additive)
        # Attention parameters already initialized in its constructor

    def forward(self, news_vecs: torch.Tensor) -> torch.Tensor:
        # news_vecs: (B, H, D)
        lengths = torch.full(
            (news_vecs.size(0),), news_vecs.size(1), dtype=torch.long, device=news_vecs.device
        )
        h = self.self_attn(news_vecs, lengths)  # (B, H, D)

        w = self.additive(h).squeeze(-1)  # (B, H)
        alpha = torch.softmax(w, dim=-1)
        u = torch.sum(h * alpha.unsqueeze(-1), dim=1)  # (B, D)
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

        # Optional compilation (PyTorch 2.1+)
        if compile_model and torch.cuda.is_available():
            self.news_encoder = torch.compile(self.news_encoder)
            self.user_encoder = torch.compile(self.user_encoder)

    # ---------------------------------------------------------------------
    def forward(
        self,
        candidate_tokens: torch.Tensor,
        history_tokens: torch.Tensor,
    ) -> torch.Tensor:
        B, H, L = history_tokens.size()

        # Encode candidate news
        v_c = self.news_encoder(candidate_tokens)  # (B, D)

        # Encode user history
        history_tokens = history_tokens.view(B * H, L)
        v_h = self.news_encoder(history_tokens).view(B, H, -1)  # (B*H, D) => (B, H, D)
        u = self.user_encoder(v_h)  # (B, D)

        # Safe (out-of-place) normalization
        norm_u = torch.linalg.vector_norm(u, dim=-1, keepdim=True).add(1e-8)
        norm_v = torch.linalg.vector_norm(v_c, dim=-1, keepdim=True).add(1e-8)
        u = u / norm_u
        v_c = v_c / norm_v

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
