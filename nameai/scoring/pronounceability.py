"""Pronounceability scoring: can a human say this name?

Checks for valid English consonant clusters, vowel presence,
and penalizes unpronounceable sequences.
"""

from __future__ import annotations

import re

# Valid English onset consonant clusters (word/syllable beginnings)
VALID_ONSETS = {
    # Single consonants
    "b", "c", "d", "f", "g", "h", "j", "k", "l", "m",
    "n", "p", "q", "r", "s", "t", "v", "w", "x", "y", "z",
    # Common clusters
    "bl", "br", "cl", "cr", "dr", "dw", "fl", "fr", "gl", "gr",
    "pl", "pr", "sc", "sk", "sl", "sm", "sn", "sp", "st", "sw",
    "tr", "tw", "wr", "wh",
    # S-clusters
    "scr", "spl", "spr", "str", "squ",
    "sch", "shr", "thr",
    # Common borrowed
    "ch", "sh", "th", "ph", "kn", "gn",
}

# Valid English coda consonant clusters (word/syllable endings)
VALID_CODAS = {
    # Single
    "b", "c", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "x", "z",
    # Clusters
    "ct", "ft", "lb", "ld", "lf", "lk", "lm", "ln", "lp", "lt",
    "mb", "mp", "nd", "ng", "nk", "nt", "pt", "rb", "rc", "rd",
    "rf", "rg", "rk", "rl", "rm", "rn", "rp", "rs", "rt", "rv",
    "sk", "sp", "st", "ts", "xt",
    "nch", "nds", "ngs", "nks", "nts", "rds", "rks", "rms", "rns",
    "rts", "sts", "lts", "mps",
    "sh", "ch", "th", "ck",
}

VOWELS = set("aeiouyAEIOUY")


def _has_vowel(s: str) -> bool:
    return any(c in VOWELS for c in s)


def _consonant_cluster_score(word: str) -> float:
    """Score based on consonant cluster validity."""
    word = word.lower()

    # Find all consonant clusters
    clusters = re.findall(r"[^aeiouy]+", word)

    if not clusters:
        return 1.0

    penalties = 0
    total_clusters = 0

    for i, cluster in enumerate(clusters):
        if not cluster.isalpha():
            continue
        total_clusters += 1

        # Allow any valid onset or coda
        if cluster in VALID_ONSETS or cluster in VALID_CODAS:
            continue

        # Check if it can be split into valid onset + coda
        splittable = False
        for split_point in range(1, len(cluster)):
            left = cluster[:split_point]
            right = cluster[split_point:]
            if (left in VALID_CODAS or left in VALID_ONSETS) and (
                right in VALID_ONSETS or right in VALID_CODAS
            ):
                splittable = True
                break

        if not splittable:
            # Penalize based on cluster length
            penalties += min(len(cluster) - 1, 3) * 0.15

    if total_clusters == 0:
        return 1.0
    return max(0.0, 1.0 - penalties / total_clusters)


def _vowel_distribution_score(word: str) -> float:
    """Score based on vowel distribution — names need vowels to be speakable."""
    word = word.lower()
    alpha = [c for c in word if c.isalpha()]

    if not alpha:
        return 0.0

    if not _has_vowel(word):
        return 0.0

    # Check max consecutive consonants
    max_consec = 0
    current_consec = 0
    for c in alpha:
        if c not in VOWELS:
            current_consec += 1
            max_consec = max(max_consec, current_consec)
        else:
            current_consec = 0

    # Penalize long consonant runs
    if max_consec <= 2:
        return 1.0
    if max_consec == 3:
        return 0.8
    if max_consec == 4:
        return 0.5
    return max(0.1, 1.0 - max_consec * 0.15)


def _length_score(word: str) -> float:
    """Score based on length — too short or long names are hard to say/remember."""
    n = len(word)
    if n < 2:
        return 0.2
    if 3 <= n <= 8:
        return 1.0
    if n <= 12:
        return 0.8
    if n <= 16:
        return 0.6
    return max(0.2, 1.0 - (n - 12) * 0.05)


def pronounceability_score(name: str) -> float:
    """Overall pronounceability score (0-1).

    Combines consonant cluster validity, vowel distribution, and length.
    """
    alpha_name = re.sub(r"[^a-zA-Z]", "", name)
    if not alpha_name:
        return 0.0

    scores = {
        "clusters": _consonant_cluster_score(alpha_name),
        "vowels": _vowel_distribution_score(alpha_name),
        "length": _length_score(alpha_name),
    }

    # Weighted combination — clusters matter most
    weights = {"clusters": 0.45, "vowels": 0.35, "length": 0.2}
    total = sum(scores[k] * weights[k] for k in scores)
    return round(total, 3)
