"""Character-level tokenizer for brand name generation.

Maps individual characters to integer IDs. This is the output tokenizer —
generating names character-by-character lets the model invent novel words
rather than recombining known subwords.
"""

from __future__ import annotations


# Special tokens
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

# Printable ASCII characters for brand names
CHARS = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " -_.&'!+"
)


class CharTokenizer:
    """Character-level tokenizer for name generation."""

    def __init__(self) -> None:
        self.special_tokens = list(SPECIAL_TOKENS)
        self.chars = list(CHARS)
        self.vocab = self.special_tokens + self.chars

        self.char_to_id: dict[str, int] = {c: i for i, c in enumerate(self.vocab)}
        self.id_to_char: dict[int, str] = {i: c for i, c in enumerate(self.vocab)}

        self.pad_id = self.char_to_id[PAD_TOKEN]
        self.bos_id = self.char_to_id[BOS_TOKEN]
        self.eos_id = self.char_to_id[EOS_TOKEN]
        self.unk_id = self.char_to_id[UNK_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        """Encode a name string to a list of token IDs."""
        ids = [self.char_to_id.get(c, self.unk_id) for c in text]
        if add_special:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int], strip_special: bool = True) -> str:
        """Decode a list of token IDs back to a string."""
        chars = []
        for token_id in ids:
            token = self.id_to_char.get(token_id, "")
            if strip_special and token in self.special_tokens:
                if token == EOS_TOKEN:
                    break
                continue
            chars.append(token)
        return "".join(chars)

    def pad_sequence(self, ids: list[int], max_len: int) -> list[int]:
        """Pad or truncate a sequence to max_len."""
        if len(ids) >= max_len:
            return ids[:max_len]
        return ids + [self.pad_id] * (max_len - len(ids))
