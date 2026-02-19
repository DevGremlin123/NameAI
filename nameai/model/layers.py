"""Core transformer building blocks: RMSNorm, MultiHeadAttention, FeedForward."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional causal masking."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if key is None:
            key = query
        if value is None:
            value = key

        B, T_q, _ = query.shape
        T_k = key.shape[1]

        q = self.q_proj(query).view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        if is_causal:
            causal_mask = torch.triu(
                torch.ones(T_q, T_k, dtype=torch.bool, device=query.device), diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        elif mask is not None:
            attn_weights = attn_weights.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        return self.out_proj(out)


class CrossAttention(nn.Module):
    """Cross-attention: Q from decoder, K/V from encoder (handles dim mismatch)."""

    def __init__(self, d_dec: int, d_enc: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_dec % n_heads == 0
        self.d_dec = d_dec
        self.n_heads = n_heads
        self.head_dim = d_dec // n_heads

        self.q_proj = nn.Linear(d_dec, d_dec)
        self.k_proj = nn.Linear(d_enc, d_dec)
        self.v_proj = nn.Linear(d_enc, d_dec)
        self.out_proj = nn.Linear(d_dec, d_dec)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, query: torch.Tensor, encoder_out: torch.Tensor, encoder_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T_q, _ = query.shape
        T_k = encoder_out.shape[1]

        q = self.q_proj(query).view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(encoder_out).view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(encoder_out).view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

        if encoder_mask is not None:
            attn_weights = attn_weights.masked_fill(encoder_mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_dec)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.w_gate = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w_gate(x)))
