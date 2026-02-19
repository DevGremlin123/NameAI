"""Hand-curated iconic brand names with quality weights.

This module provides the curated dataset of iconic brands that receive
higher training weight to steer the model toward creative, memorable names.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CURATED_PATH = Path("data/curated/iconic_brands.jsonl")


@dataclass
class BrandEntry:
    name: str
    description: str
    quality: float = 1.0
    category: str = ""


def load_curated_brands(path: str | Path = DEFAULT_CURATED_PATH) -> list[BrandEntry]:
    """Load curated brand entries from a JSONL file."""
    entries = []
    path = Path(path)
    if not path.exists():
        return entries
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            entries.append(
                BrandEntry(
                    name=data["name"],
                    description=data["description"],
                    quality=data.get("quality", 1.0),
                    category=data.get("category", ""),
                )
            )
    return entries


def get_quality_weights(entries: list[BrandEntry], alpha: float = 2.0) -> list[float]:
    """Compute per-example training weights based on quality scores.

    Higher alpha = more emphasis on high-quality examples.
    A quality=1.0 brand with alpha=2.0 gets weight 2.0, while
    quality=0.5 gets weight ~1.41.
    """
    return [entry.quality ** alpha for entry in entries]
