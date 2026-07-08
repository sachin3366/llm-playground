"""Train the from-scratch BPE tokenizer on the cleaned corpus and save it.

Usage:
    python -m module1_llm_playground.tokenizer.train_tokenizer \
        --corpus data/processed/corpus.txt \
        --vocab-size 1000 \
        --out data/tokenizer/merges.json
"""

import argparse
from pathlib import Path

from module1_llm_playground.tokenizer.bpe import BPETokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/processed/corpus.txt")
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--out", default="data/tokenizer/merges.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    text = Path(args.corpus).read_text(encoding="utf-8")
    print(f"Loaded corpus: {len(text):,} chars from {args.corpus}")

    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=args.vocab_size, verbose=args.verbose)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out_path)
    print(f"Saved {len(tokenizer.merges)} merges -> {out_path} (vocab size {tokenizer.vocab_size})")

    # Compression ratio: raw UTF-8 bytes vs. learned token ids, on the same text.
    raw_bytes = len(text.encode("utf-8"))
    token_ids = tokenizer.encode(text)
    ratio = raw_bytes / len(token_ids) if token_ids else float("nan")
    print(f"Compression: {raw_bytes:,} bytes -> {len(token_ids):,} tokens ({ratio:.2f}x)")

    # Roundtrip sanity check on a sample.
    sample = text[:200]
    ids = tokenizer.encode(sample)
    decoded = tokenizer.decode(ids)
    assert decoded == sample, "roundtrip mismatch: encode/decode did not reproduce the input"
    print("Roundtrip check passed on a 200-char sample.")
    print(f"Sample encoded to {len(ids)} tokens: {ids[:20]}{'...' if len(ids) > 20 else ''}")


if __name__ == "__main__":
    main()
