"""Gopher/RefinedWeb-style quality heuristics for filtering raw web text.

Each check function returns True if a document PASSES (should be kept).
These are the same category of cheap, rule-based filters used ahead of
model-based filtering in RefinedWeb, Dolma, and FineWeb.
"""
import re

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "is", "are",
    "was", "were", "for", "on", "with", "as", "by", "at", "it", "this",
    "that", "be", "from", "has", "have", "not",
}


def has_min_words(text: str, min_words: int = 50) -> bool:
    return len(text.split()) >= min_words


def has_reasonable_word_length(text: str, low: float = 3.0, high: float = 10.0) -> bool:
    words = text.split()
    if not words:
        return False
    mean_len = sum(len(w) for w in words) / len(words)
    return low <= mean_len <= high


def has_enough_alpha_words(text: str, min_ratio: float = 0.7) -> bool:
    words = text.split()
    if not words:
        return False
    alpha = sum(1 for w in words if any(c.isalpha() for c in w))
    return (alpha / len(words)) >= min_ratio


def has_enough_stopwords(text: str, min_count: int = 2) -> bool:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return sum(1 for w in words if w in _STOPWORDS) >= min_count


def has_low_symbol_ratio(text: str, max_ratio: float = 0.1) -> bool:
    if not text:
        return False
    symbols = sum(1 for c in text if c in "#*…{}[]<>|~^")
    return (symbols / len(text)) <= max_ratio


QUALITY_CHECKS = [
    has_min_words,
    has_reasonable_word_length,
    has_enough_alpha_words,
    has_enough_stopwords,
    has_low_symbol_ratio,
]


def passes_quality_filters(text: str) -> tuple[bool, list[str]]:
    """Runs all quality checks. Returns (passed, names of failed checks)."""
    failed = [fn.__name__ for fn in QUALITY_CHECKS if not fn(text)]
    return (len(failed) == 0, failed)
