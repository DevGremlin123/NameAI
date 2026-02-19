"""Anti-slop filtering: blocklist, quality scoring, diversity enforcement."""

from __future__ import annotations

from pathlib import Path

from nameai.scoring.phonaesthetics import phonaesthetic_score
from nameai.scoring.pronounceability import pronounceability_score
from nameai.scoring.uniqueness import uniqueness_score


def load_blocklist(path: str | Path = "data/curated/slop_blocklist.txt") -> set[str]:
    """Load blocked words/patterns from file."""
    blocked = set()
    path = Path(path)
    if not path.exists():
        return blocked
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                blocked.add(line.lower())
    return blocked


def contains_blocked_word(name: str, blocklist: set[str]) -> bool:
    """Check if a name contains any blocked word/pattern.

    Short patterns (<=3 chars) only match as suffix to avoid false positives
    (e.g. "fy" shouldn't block "Spotify").
    Longer patterns match as substring anywhere in the name.
    """
    name_lower = name.lower()
    for blocked in blocklist:
        if len(blocked) <= 3:
            # Short patterns: only block if the name ends with it
            if name_lower.endswith(blocked):
                return True
        else:
            if blocked in name_lower:
                return True
    return False


def score_name(
    name: str,
    blocklist: set[str] | None = None,
    min_pronounceability: float = 0.4,
    min_phonaesthetic: float = 0.3,
    min_uniqueness: float = 0.5,
) -> dict:
    """Score a generated name on multiple quality dimensions.

    Returns:
        Dict with individual scores and whether the name passes all filters.
    """
    scores = {
        "name": name,
        "pronounceability": pronounceability_score(name),
        "phonaesthetics": phonaesthetic_score(name),
        "uniqueness": uniqueness_score(name),
        "blocked": False,
        "passes": True,
    }

    if blocklist and contains_blocked_word(name, blocklist):
        scores["blocked"] = True
        scores["passes"] = False

    if scores["pronounceability"] < min_pronounceability:
        scores["passes"] = False
    if scores["phonaesthetics"] < min_phonaesthetic:
        scores["passes"] = False
    if scores["uniqueness"] < min_uniqueness:
        scores["passes"] = False

    # Overall quality = geometric mean of scores
    scores["overall"] = (
        scores["pronounceability"] * scores["phonaesthetics"] * scores["uniqueness"]
    ) ** (1 / 3)

    return scores


def filter_and_rank(
    names: list[str],
    blocklist: set[str] | None = None,
    min_pronounceability: float = 0.4,
    min_phonaesthetic: float = 0.3,
    min_uniqueness: float = 0.5,
    diversity_threshold: float = 0.6,
) -> list[dict]:
    """Filter, deduplicate, enforce diversity, and rank generated names."""
    # Score all names
    scored = []
    for name in names:
        s = score_name(name, blocklist, min_pronounceability, min_phonaesthetic, min_uniqueness)
        if s["passes"]:
            scored.append(s)

    # Sort by overall quality
    scored.sort(key=lambda x: x["overall"], reverse=True)

    # Enforce diversity: remove names too similar to already-selected ones
    if diversity_threshold > 0:
        scored = _enforce_diversity(scored, diversity_threshold)

    return scored


def _enforce_diversity(scored: list[dict], threshold: float) -> list[dict]:
    """Remove names that are too similar to already-selected names."""
    selected = []
    for candidate in scored:
        if not selected:
            selected.append(candidate)
            continue
        # Check edit distance ratio against all selected names
        is_diverse = True
        for existing in selected:
            ratio = _edit_distance_ratio(candidate["name"].lower(), existing["name"].lower())
            if ratio < threshold:
                is_diverse = False
                break
        if is_diverse:
            selected.append(candidate)
    return selected


def _edit_distance_ratio(a: str, b: str) -> float:
    """Normalized edit distance: 1.0 = completely different, 0.0 = identical."""
    if a == b:
        return 0.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0

    # Levenshtein distance
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp

    return dp[n] / max_len
