"""Step 1b: sample pretraining text directly from a Common Crawl WET file.

Streams a single WET (WARC Encapsulated Text) file from the public Common
Crawl S3 bucket, extracts the first `limit` conversion records, and writes
each as a .txt file. This demonstrates the data source referenced in the
pretraining data-collection literature (RefinedWeb, FineWeb, and Dolma all
derive from Common Crawl) without downloading a full multi-GB crawl segment.
"""
import argparse
import gzip
from pathlib import Path

import requests
from warcio.archiveiterator import ArchiveIterator

CRAWL_INDEX_URL = "https://index.commoncrawl.org/collinfo.json"
CC_DATA_ROOT = "https://data.commoncrawl.org/"


def _latest_crawl_id() -> str:
    resp = requests.get(CRAWL_INDEX_URL, timeout=15)
    resp.raise_for_status()
    return resp.json()[0]["id"]  # most recent crawl is listed first


def _first_wet_path(crawl_id: str) -> str:
    paths_url = f"{CC_DATA_ROOT}crawl-data/{crawl_id}/wet.paths.gz"
    resp = requests.get(paths_url, timeout=30)
    resp.raise_for_status()
    first_line = gzip.decompress(resp.content).splitlines()[0]
    return first_line.decode("utf-8")


def sample_common_crawl(out_dir: Path, limit: int = 20, crawl_id: str | None = None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    crawl_id = crawl_id or _latest_crawl_id()
    wet_path = _first_wet_path(crawl_id)
    print(f"streaming {crawl_id} :: {wet_path}")

    written: list[Path] = []
    with requests.get(CC_DATA_ROOT + wet_path, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        for record in ArchiveIterator(resp.raw):
            if len(written) >= limit:
                break
            if record.rec_type != "conversion":
                continue
            text = record.content_stream().read().decode("utf-8", errors="ignore")
            if len(text) < 200:
                continue  # skip near-empty pages
            path = out_dir / f"record_{len(written):03d}.txt"
            path.write_text(text, encoding="utf-8")
            written.append(path)
    print(f"wrote {len(written)} records -> {out_dir}")
    return written


def main():
    parser = argparse.ArgumentParser(description="Sample plain text from a Common Crawl WET file.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw/common_crawl"))
    parser.add_argument("--limit", type=int, default=20, help="number of text records to keep")
    parser.add_argument("--crawl-id", default=None, help="e.g. CC-MAIN-2024-51 (default: latest)")
    args = parser.parse_args()
    sample_common_crawl(args.out_dir, args.limit, args.crawl_id)


if __name__ == "__main__":
    main()
