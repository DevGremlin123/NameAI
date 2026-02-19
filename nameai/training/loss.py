"""Loss utilities for name generation training.

With HuggingFace Trainer, the standard cross-entropy loss is handled
automatically by T5ForConditionalGeneration. This module provides
optional quality-weighted sampling for the DataLoader.
"""

from __future__ import annotations

import random


def quality_weighted_sample(
    records: list[dict],
    target_size: int,
    alpha: float = 2.0,
    seed: int = 42,
) -> list[dict]:
    """Sample records with probability proportional to quality^alpha.

    Higher-quality examples (curated iconic brands) get sampled more often,
    effectively upweighting them in training.
    """
    rng = random.Random(seed)
    weights = [r.get("quality", 0.5) ** alpha for r in records]
    total_weight = sum(weights)
    probs = [w / total_weight for w in weights]

    sampled = rng.choices(records, weights=probs, k=target_size)
    return sampled
