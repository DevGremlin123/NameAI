"""Download and extract Wikipedia text for pre-training.

Downloads ~2-3GB of clean English Wikipedia text using HuggingFace datasets.
Extracts plain text, splits into chunks, and saves for BPE training + MLM.

Usage:
    python -m nameai.data.download_wikipedia
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Wikipedia for pre-training")
    parser.add_argument("--output-dir", type=str, default="data/wikipedia")
    parser.add_argument("--max-articles", type=int, default=None,
                        help="Limit articles (None = all ~6M)")
    parser.add_argument("--chunk-size", type=int, default=512,
                        help="Max words per text chunk")
    args = parser.parse_args()

    from datasets import load_dataset

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading English Wikipedia (this takes a few minutes)...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")

    text_path = output_dir / "wiki_text.txt"
    chunk_count = 0

    print(f"Extracting text to {text_path}...")
    with open(text_path, "w", encoding="utf-8") as out:
        for i, article in enumerate(tqdm(ds, desc="Processing articles")):
            if args.max_articles and i >= args.max_articles:
                break

            text = article["text"].strip()
            if len(text) < 100:
                continue

            # Split into chunks of ~chunk_size words
            words = text.split()
            for start in range(0, len(words), args.chunk_size):
                chunk = " ".join(words[start : start + args.chunk_size])
                if len(chunk) > 50:
                    out.write(chunk + "\n")
                    chunk_count += 1

    size_mb = text_path.stat().st_size / 1e6
    print(f"\nDone: {chunk_count:,} text chunks, {size_mb:.0f} MB")
    print(f"Saved to {text_path}")


if __name__ == "__main__":
    main()
