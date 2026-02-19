"""NameFormer: custom encoder-decoder trained from scratch.

Phase 1: Pre-train encoder on Wikipedia (learns English)
Phase 2: Fine-tune full model on (description → brand name) pairs
"""

from __future__ import annotations

import torch
import torch.nn as nn

from nameai.model.decoder import NameDecoder
from nameai.model.encoder import DescriptionEncoder


class NameFormer(nn.Module):
    def __init__(
        self,
        enc_vocab_size: int = 16000,
        enc_d_model: int = 512,
        enc_n_heads: int = 8,
        enc_n_layers: int = 10,
        enc_d_ff: int = 2816,
        enc_max_seq_len: int = 512,
        dec_vocab_size: int = 76,
        dec_d_model: int = 384,
        dec_n_heads: int = 6,
        dec_n_layers: int = 7,
        dec_d_ff: int = 1792,
        dec_max_seq_len: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.encoder = DescriptionEncoder(
            vocab_size=enc_vocab_size, d_model=enc_d_model, n_heads=enc_n_heads,
            n_layers=enc_n_layers, d_ff=enc_d_ff, max_seq_len=enc_max_seq_len, dropout=dropout,
        )
        self.decoder = NameDecoder(
            vocab_size=dec_vocab_size, d_model=dec_d_model, d_enc=enc_d_model,
            n_heads=dec_n_heads, n_layers=dec_n_layers, d_ff=dec_d_ff,
            max_seq_len=dec_max_seq_len, dropout=dropout,
        )

        # MLM head for pre-training (predicts masked BPE tokens)
        self.mlm_head = nn.Linear(enc_d_model, enc_vocab_size, bias=False)
        self.mlm_head.weight = self.encoder.embedding.weight  # Tie weights

    def forward(self, src_ids, tgt_ids, src_padding_mask=None):
        """Full encoder-decoder forward for name generation fine-tuning."""
        encoder_out = self.encoder(src_ids, src_padding_mask)
        logits = self.decoder(tgt_ids, encoder_out, src_padding_mask)
        return logits

    def forward_mlm(self, token_ids, padding_mask=None):
        """Encoder-only forward for masked language model pre-training."""
        encoder_out = self.encoder(token_ids, padding_mask)
        return self.mlm_head(encoder_out)

    def encode(self, src_ids, src_padding_mask=None):
        return self.encoder(src_ids, src_padding_mask)

    def decode_step(self, tgt_ids, encoder_out, encoder_mask=None):
        logits = self.decoder(tgt_ids, encoder_out, encoder_mask)
        return logits[:, -1, :]

    def count_parameters(self):
        enc = sum(p.numel() for p in self.encoder.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {"encoder": enc, "decoder": dec, "total": total}

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, device: str = "cpu", **kwargs) -> "NameFormer":
        model = cls(**kwargs)
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        return model.to(device)
