"""Step 2: clean raw crawled text into a training-ready corpus.

Pipeline: load raw .txt files -> unicode/whitespace normalization ->
English-language filter -> Gopher/RefinedWeb-style quality heuristics ->
paragraph-level dedup -> write data/processed/corpus.txt (+ a summary of
what was dropped and why).
"""
import argparse
import unicodedata
from pathlib import Path

from langdetect import LangDetectException, detect

from .dedup import dedup_paragraphs
from .filters import passes_quality_filters


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _is_english(text: str) -> bool:
    try:
        return detect(text[:2000]) == "en"
    except LangDetectException:
        return False


def clean_corpus(raw_dir: Path, out_path: Path) -> dict:
    stats = {
        "docs_seen": 0,
        "dropped_lang": 0,
        "dropped_quality": 0,
        "docs_kept": 0,
        "paragraphs_dropped_dup": 0,
    }
    seen_hashes: set[str] = set()
    kept_docs: list[str] = []

    for path in sorted(raw_dir.rglob("*.txt")):
        stats["docs_seen"] += 1
        text = _normalize(path.read_text(encoding="utf-8", errors="ignore"))

        if not _is_english(text):
            stats["dropped_lang"] += 1
            print(f"drop (non-English): {path}")
            continue

        passed, failed_checks = passes_quality_filters(text)
        if not passed:
            stats["dropped_quality"] += 1
            print(f"drop (quality: {', '.join(failed_checks)}): {path}")
            continue

        paragraphs_before = [p for p in text.split("\n\n") if p.strip()]
        deduped = dedup_paragraphs(text, seen_hashes)
        paragraphs_after = [p for p in deduped.split("\n\n") if p.strip()]
        stats["paragraphs_dropped_dup"] += len(paragraphs_before) - len(paragraphs_after)

        if not paragraphs_after:
            stats["dropped_quality"] += 1
            continue

        kept_docs.append(deduped)
        stats["docs_kept"] += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(kept_docs), encoding="utf-8")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Clean raw crawled text into a training corpus.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/corpus.txt"))
    args = parser.parse_args()

    stats = clean_corpus(args.raw_dir, args.out)
    print(f"\n{stats}")
    print(f"wrote cleaned corpus -> {args.out}")


if __name__ == "__main__":
    main()
