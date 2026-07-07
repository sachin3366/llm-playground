"""Exact paragraph-level deduplication.

Boilerplate (nav menus, footers, repeated disclaimers) tends to repeat
verbatim across documents even after HTML stripping. Hashing each paragraph
and dropping repeats catches this cheaply, without needing MinHash/SimHash
near-duplicate detection for a corpus this small.
"""
import hashlib
import re


def _normalize_for_hash(paragraph: str) -> str:
    return re.sub(r"\s+", " ", paragraph.strip().lower())


def dedup_paragraphs(text: str, seen_hashes: set[str]) -> str:
    """Drops paragraphs already present in `seen_hashes` (mutated in place)."""
    kept = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        digest = hashlib.sha256(_normalize_for_hash(para).encode("utf-8")).hexdigest()
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)
        kept.append(para)
    return "\n\n".join(kept)
