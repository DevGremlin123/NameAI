"""Uniqueness scoring: how novel and distinctive is a generated name?

Checks against common English words, existing brand databases,
and measures character-level entropy and novelty.
"""

from __future__ import annotations

import math
import re
from collections import Counter

# Common English words that shouldn't be used as-is for brand names
# (Though they can appear as parts of creative names)
COMMON_WORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "day", "get", "has", "him", "his",
    "how", "its", "may", "new", "now", "old", "see", "way", "who", "boy",
    "did", "big", "let", "put", "say", "she", "too", "use", "man", "run",
    "good", "best", "free", "fast", "easy", "hard", "long", "just", "make",
    "like", "time", "very", "when", "come", "could", "than", "look", "only",
    "well", "back", "also", "work", "first", "even", "give", "most", "find",
    "here", "thing", "many", "some", "take", "want", "them", "same", "still",
    "company", "business", "service", "system", "group", "market", "world",
    "online", "digital", "global", "smart", "power", "energy", "health",
}

# Common word parts that signal generic names
GENERIC_PARTS = {
    "solutions", "services", "systems", "global", "group", "corp",
    "tech", "digital", "smart", "cyber", "cloud", "data", "info",
    "pro", "max", "plus", "hub", "lab", "net", "web", "app",
}


def character_entropy(word: str) -> float:
    """Shannon entropy of character distribution (normalized 0-1).

    Higher entropy = more diverse character usage = more unique.
    """
    word = word.lower()
    if len(word) <= 1:
        return 0.0

    counts = Counter(word)
    total = len(word)
    entropy = -sum(
        (c / total) * math.log2(c / total) for c in counts.values()
    )

    # Normalize by max possible entropy for this length
    max_entropy = math.log2(min(len(counts), 26))
    if max_entropy == 0:
        return 0.0
    return min(1.0, entropy / max_entropy)


def dictionary_novelty(word: str) -> float:
    """Score based on how far the name is from common dictionary words (0-1).

    1.0 = completely novel, 0.0 = exact common word match.
    """
    word_lower = word.lower()

    # Exact match with common words
    if word_lower in COMMON_WORDS:
        return 0.2  # Not zero — some brands are common words (Apple, Square)

    # Check if it contains generic parts
    for part in GENERIC_PARTS:
        if part in word_lower:
            return 0.4

    # Check if it's a simple concatenation of two common words
    for w in COMMON_WORDS:
        if len(w) >= 3 and word_lower.startswith(w):
            remainder = word_lower[len(w):]
            if remainder in COMMON_WORDS and len(remainder) >= 3:
                return 0.5

    # Novel word — high score
    return 0.9


def structural_novelty(word: str) -> float:
    """Score based on unusual letter patterns and combinations (0-1).

    Rewards creative letter combinations that feel fresh.
    """
    word = word.lower()
    if len(word) < 2:
        return 0.3

    score = 0.5

    # Reward unusual bigrams
    common_bigrams = {"th", "he", "in", "er", "an", "re", "on", "at", "en", "nd", "ti", "es", "or", "te", "of", "ed", "is", "it", "al", "ar", "st", "to", "nt", "ng"}
    bigrams = [word[i : i + 2] for i in range(len(word) - 1)]
    if bigrams:
        common_ratio = sum(1 for b in bigrams if b in common_bigrams) / len(bigrams)
        # Sweet spot: some familiar bigrams (pronounceable) but not all (generic)
        if 0.2 <= common_ratio <= 0.6:
            score += 0.25
        elif common_ratio < 0.2:
            score += 0.1  # Very unusual — might be hard to pronounce
        else:
            score += 0.05  # Too common — feels generic

    # Reward unusual first letter
    rare_starts = set("jkqxz")
    if word[0] in rare_starts:
        score += 0.1

    # Reward mixed character types if multi-word
    if any(c.isupper() for c in word[1:]):
        score += 0.05  # CamelCase adds distinctiveness

    return min(1.0, score)


def uniqueness_score(name: str) -> float:
    """Overall uniqueness score (0-1).

    Combines character entropy, dictionary novelty, and structural novelty.
    """
    alpha_name = re.sub(r"[^a-zA-Z]", "", name)
    if not alpha_name:
        return 0.0

    scores = {
        "entropy": character_entropy(alpha_name),
        "dictionary": dictionary_novelty(alpha_name),
        "structural": structural_novelty(alpha_name),
    }

    weights = {"entropy": 0.25, "dictionary": 0.4, "structural": 0.35}
    total = sum(scores[k] * weights[k] for k in scores)
    return round(total, 3)
