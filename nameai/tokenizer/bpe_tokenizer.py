"""BPE tokenizer for description encoding (wraps SentencePiece)."""

from __future__ import annotations

from pathlib import Path

import sentencepiece as spm

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
MASK_ID = 4  # For MLM pre-training


class BPETokenizer:
    def __init__(self, model_path: str | Path | None = None) -> None:
        self.sp = spm.SentencePieceProcessor()
        self._loaded = False
        if model_path is not None:
            self.load(model_path)

    def load(self, model_path: str | Path) -> None:
        self.sp.Load(str(model_path))
        self._loaded = True

    @property
    def vocab_size(self) -> int:
        return self.sp.GetPieceSize()

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def mask_id(self) -> int:
        return MASK_ID

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        ids = self.sp.Encode(text)
        if add_special:
            ids = [BOS_ID] + ids + [EOS_ID]
        return ids

    def decode(self, ids: list[int]) -> str:
        filtered = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID, MASK_ID)]
        return self.sp.Decode(filtered)

    def pad_sequence(self, ids: list[int], max_len: int) -> list[int]:
        if len(ids) >= max_len:
            return ids[:max_len]
        return ids + [PAD_ID] * (max_len - len(ids))

    @staticmethod
    def train(input_file: str | Path, model_prefix: str = "bpe_tokenizer",
              vocab_size: int = 16000) -> Path:
        spm.SentencePieceTrainer.Train(
            input=str(input_file),
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=0.9995,
            pad_id=PAD_ID, bos_id=BOS_ID, eos_id=EOS_ID, unk_id=UNK_ID,
            user_defined_symbols=["<mask>"],
            normalization_rule_name="nmt_nfkc_cf",
        )
        return Path(f"{model_prefix}.model")
