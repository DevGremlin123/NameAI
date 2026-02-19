"""Train BPE tokenizer on Wikipedia text.

Usage:
    python -m nameai.tokenizer.train_tokenizer
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nameai.tokenizer.bpe_tokenizer import BPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument("--input", type=str, default="data/wikipedia/wiki_text.txt")
    parser.add_argument("--vocab-size", type=int, default=16000)
    parser.add_argument("--output-prefix", type=str, default="data/bpe_tokenizer")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Input file not found: {args.input}")
        print("Run `python -m nameai.data.download_wikipedia` first.")
        return

    print(f"Training BPE tokenizer (vocab_size={args.vocab_size})...")
    model_path = BPETokenizer.train(
        input_file=args.input,
        model_prefix=args.output_prefix,
        vocab_size=args.vocab_size,
    )
    print(f"Saved to {model_path}")

    # Quick test
    tok = BPETokenizer(model_path)
    test = "A streaming service for watching movies and TV shows on demand"
    encoded = tok.encode(test)
    decoded = tok.decode(encoded)
    print(f"Test: '{test}'")
    print(f"Tokens: {len(encoded)}, Decoded: '{decoded}'")
    print(f"Vocab size: {tok.vocab_size}")


if __name__ == "__main__":
    main()
