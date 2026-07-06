"""Step 1a: manual web crawling for pretraining data collection.

Fetches a small set of seed pages, respects robots.txt, strips HTML down to
visible paragraph text, and writes one .txt file per page under
data/raw/manual/.
"""
import argparse
import re
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

USER_AGENT = "llm-playground-crawler/0.1 (educational data collection)"

DEFAULT_SEED_URLS = [
    "https://en.wikipedia.org/wiki/Large_language_model",
    "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
    "https://en.wikipedia.org/wiki/Byte_pair_encoding",
    "https://en.wikipedia.org/wiki/Attention_(machine_learning)",
    "https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback",
]


def _robots_allow(url: str) -> bool:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser = RobotFileParser()
    parser.set_url(robots_url)
    try:
        # Fetch and decode ourselves: RobotFileParser.read() uses urllib and
        # chokes on a leading UTF-8 BOM (e.g. Wikipedia's robots.txt), which
        # silently makes it treat every rule as a disallow.
        resp = requests.get(robots_url, headers={"User-Agent": USER_AGENT}, timeout=10)
        resp.raise_for_status()
        parser.parse(resp.content.decode("utf-8-sig").splitlines())
    except requests.RequestException:
        return True  # robots.txt unreachable, assume allowed
    return parser.can_fetch(USER_AGENT, url)


def _extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "sup", "table", "nav", "footer"]):
        tag.decompose()
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = "\n\n".join(p for p in paragraphs if len(p) > 40)
    return re.sub(r"[ \t]+", " ", text)


def crawl(urls: list[str], out_dir: Path, delay: float = 1.0) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for url in urls:
        if not _robots_allow(url):
            print(f"skip (robots.txt disallows): {url}")
            continue
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)
        resp.raise_for_status()
        text = _extract_text(resp.text)
        slug = urlparse(url).path.strip("/").replace("/", "_") or "index"
        path = out_dir / f"{slug}.txt"
        path.write_text(text, encoding="utf-8")
        written.append(path)
        print(f"saved {len(text):,} chars -> {path}")
        time.sleep(delay)
    return written


def main():
    parser = argparse.ArgumentParser(description="Manually crawl a small set of seed pages.")
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw/manual"))
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    args = parser.parse_args()
    crawl(DEFAULT_SEED_URLS, args.out_dir, args.delay)


if __name__ == "__main__":
    main()
