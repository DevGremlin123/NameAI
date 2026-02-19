"""Phonaesthetic scoring: how pleasing a name sounds.

Evaluates vowel-consonant patterns, syllable structure, sonority,
and sound symbolism properties that make brand names memorable.
"""

from __future__ import annotations

import re

VOWELS = set("aeiouAEIOU")
CONSONANTS = set("bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ")

# Sonorous consonants (pleasant sounds) — nasals, liquids, glides
SONORANTS = set("mnlrwyMNLRWY")

# Plosives (harder, punchier sounds) — good for brand impact
PLOSIVES = set("bpdtgkBPDTGK")

# Fricatives (hissing sounds)
FRICATIVES = set("fvszshFVSZSH")


def count_syllables(word: str) -> int:
    """Estimate syllable count using vowel groups."""
    word = word.lower().strip()
    if not word:
        return 0
    # Count vowel groups
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
    # Adjust: silent e at end
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def vowel_consonant_balance(word: str) -> float:
    """Score the balance of vowels to consonants (0-1).

    Ideal ratio is around 40-50% vowels for English brand names.
    """
    letters = [c for c in word if c.isalpha()]
    if not letters:
        return 0.0
    vowel_count = sum(1 for c in letters if c in VOWELS)
    ratio = vowel_count / len(letters)
    # Bell curve centered at 0.42 (ideal vowel ratio)
    return max(0, 1 - abs(ratio - 0.42) * 4)


def syllable_score(word: str) -> float:
    """Score based on syllable count (0-1).

    Best brand names are 2-3 syllables (Netflix, Spotify, Discord).
    """
    n = count_syllables(word)
    if n <= 0:
        return 0.0
    if n == 2:
        return 1.0
    if n == 3:
        return 0.9
    if n == 1:
        return 0.7
    if n == 4:
        return 0.5
    return max(0.1, 1.0 - (n - 3) * 0.2)


def sonority_score(word: str) -> float:
    """Score the sonority profile (0-1).

    Good names often start with plosives (punchy) and have
    sonorants in the middle (pleasant flow).
    """
    word = word.lower()
    if not word:
        return 0.0

    score = 0.5  # Base score

    # Bonus for starting with a plosive or sonorant
    if word[0] in PLOSIVES:
        score += 0.2  # Punchy start (Brand, Tesla, Discord)
    elif word[0] in SONORANTS:
        score += 0.15  # Smooth start (Netflix, Loom, Roku)
    elif word[0] in VOWELS:
        score += 0.1  # Vowel start (Uber, Etsy, Arc)

    # Bonus for sonorant consonants in the body (pleasant flow)
    body = word[1:]
    if body:
        body_consonants = [c for c in body if c in CONSONANTS]
        if body_consonants:
            sonorant_ratio = sum(1 for c in body_consonants if c in SONORANTS) / len(body_consonants)
            score += sonorant_ratio * 0.15

    # Bonus for ending with a vowel or sonorant (open, flowing ending)
    if word[-1] in VOWELS:
        score += 0.1
    elif word[-1] in SONORANTS:
        score += 0.05

    return min(1.0, score)


def rhythm_score(word: str) -> float:
    """Score the rhythmic pattern of vowels and consonants (0-1).

    Alternating CV patterns score higher (e.g., "Roku" = CVCV = great).
    """
    pattern = []
    for c in word.lower():
        if c in VOWELS:
            pattern.append("V")
        elif c in CONSONANTS:
            pattern.append("C")

    if len(pattern) < 2:
        return 0.5

    # Count alternations
    alternations = sum(1 for i in range(1, len(pattern)) if pattern[i] != pattern[i - 1])
    max_alternations = len(pattern) - 1
    return alternations / max_alternations if max_alternations > 0 else 0.5


def phonaesthetic_score(name: str) -> float:
    """Overall phonaesthetic quality score (0-1).

    Combines vowel-consonant balance, syllable count, sonority, and rhythm.
    """
    # Only score alphabetic part
    alpha_name = re.sub(r"[^a-zA-Z]", "", name)
    if not alpha_name:
        return 0.0

    scores = {
        "vc_balance": vowel_consonant_balance(alpha_name),
        "syllables": syllable_score(alpha_name),
        "sonority": sonority_score(alpha_name),
        "rhythm": rhythm_score(alpha_name),
    }

    # Weighted combination
    weights = {"vc_balance": 0.2, "syllables": 0.35, "sonority": 0.25, "rhythm": 0.2}
    total = sum(scores[k] * weights[k] for k in scores)
    return round(total, 3)
