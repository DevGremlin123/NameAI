"""NameFormer encoder: processes BPE-tokenized text."""

from __future__ import annotations

import torch
import torch.nn as nn

from nameai.model.layers import FeedForward, MultiHeadAttention, RMSNorm
from nameai.model.positional import HybridPositionalEncoding


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), mask=mask)
        x = x + self.ffn(self.norm2(x))
        return x


class DescriptionEncoder(nn.Module):
    """BPE-input encoder. Used for both pre-training (Wikipedia) and fine-tuning (names)."""

    def __init__(
        self, vocab_size: int, d_model: int, n_heads: int, n_layers: int,
        d_ff: int, max_seq_len: int, dropout: float,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = HybridPositionalEncoding(d_model, max_seq_len, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.final_norm = RMSNorm(d_model)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, std=0.02)
        with torch.no_grad():
            self.embedding.weight[0].zero_()

    def forward(self, token_ids: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embedding(token_ids) * (self.d_model ** 0.5)
        x = self.pos_encoding(x)
        attn_mask = None
        if padding_mask is not None:
            attn_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        for layer in self.layers:
            x = layer(x, mask=attn_mask)
        return self.final_norm(x)
