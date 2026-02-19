"""NameFormer decoder: character-level name generation with cross-attention."""

from __future__ import annotations

import torch
import torch.nn as nn

from nameai.model.layers import CrossAttention, FeedForward, MultiHeadAttention, RMSNorm
from nameai.model.positional import LearnedPositionalEncoding


class DecoderLayer(nn.Module):
    def __init__(self, d_dec: int, d_enc: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_dec)
        self.self_attn = MultiHeadAttention(d_dec, n_heads, dropout)
        self.norm2 = RMSNorm(d_dec)
        self.cross_attn = CrossAttention(d_dec, d_enc, n_heads, dropout)
        self.norm3 = RMSNorm(d_dec)
        self.ffn = FeedForward(d_dec, d_ff, dropout)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor, encoder_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), is_causal=True)
        x = x + self.cross_attn(self.norm2(x), encoder_out, encoder_mask)
        x = x + self.ffn(self.norm3(x))
        return x


class NameDecoder(nn.Module):
    """Character-level decoder — generates brand names one letter at a time."""

    def __init__(
        self, vocab_size: int, d_model: int, d_enc: int, n_heads: int,
        n_layers: int, d_ff: int, max_seq_len: int, dropout: float,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = LearnedPositionalEncoding(d_model, max_seq_len, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, d_enc, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj.weight = self.embedding.weight  # Weight tying
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, std=0.02)
        with torch.no_grad():
            self.embedding.weight[0].zero_()

    def forward(self, char_ids: torch.Tensor, encoder_out: torch.Tensor, encoder_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.embedding(char_ids) * (self.d_model ** 0.5)
        x = self.pos_encoding(x)
        enc_attn_mask = None
        if encoder_mask is not None:
            enc_attn_mask = encoder_mask.unsqueeze(1).unsqueeze(2)
        for layer in self.layers:
            x = layer(x, encoder_out, enc_attn_mask)
        x = self.final_norm(x)
        return self.output_proj(x)
