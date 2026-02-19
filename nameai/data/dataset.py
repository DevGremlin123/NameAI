"""Data loading utilities for name generation training.

Handles JSONL files with {name, description, quality} records.
Tokenization is done in the trainer using HuggingFace's Dataset.map().
"""

from __future__ import annotations

import json
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    """Load records from a JSONL file."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: str | Path) -> None:
    """Save records to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
