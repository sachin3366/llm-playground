"""Step 1 entrypoint: run manual crawling and/or Common Crawl sampling.

Usage:
    python -m module1_llm_playground.data_collection.collect --source both
"""
import argparse
from pathlib import Path

from .common_crawl import sample_common_crawl
from .manual_crawl import DEFAULT_SEED_URLS, crawl


def main():
    parser = argparse.ArgumentParser(description="Collect raw pretraining data.")
    parser.add_argument("--source", choices=["manual", "common_crawl", "both"], default="both")
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--cc-limit", type=int, default=20)
    args = parser.parse_args()

    if args.source in ("manual", "both"):
        crawl(DEFAULT_SEED_URLS, args.out_dir / "manual")
    if args.source in ("common_crawl", "both"):
        sample_common_crawl(args.out_dir / "common_crawl", args.cc_limit)


if __name__ == "__main__":
    main()
