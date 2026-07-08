# LLM Playground — Learn by Doing, Become an AI Engineer

A hands-on, 8-module journey building AI engineering projects from scratch: LLM pretraining/post-training, a fine-tuned chatbot, web-search and deep-research agents, an image generation service, RAG, and MCP/A2A protocols. Stack: **Python** + **Anthropic Claude API**.

Each module is broken into small steps. Every step gets its own commit and a checkbox update below.

## Roadmap

- [ ] **Module 1 — Build an LLM Playground**
  Pretraining (data collection/cleaning, tokenization, transformer architecture, text generation) + post-training (SFT, RLHF/PPO)
- [ ] **Module 2 — Build a Customer Support Chatbot**
  PEFT/LoRA fine-tuning, prompt engineering (few-shot, zero-shot, CoT, role prompting)
- [ ] **Module 3 — Build an "Ask-the-Web" Agent** (Perplexity-style)
  Agents vs. agentic systems, prompt chaining, routing, parallelization, reflection, orchestrator-worker
- [ ] **Module 4 — Build a "Deep Research" Capability**
  Reasoning models, inference-time scaling, CoT prompting, self-consistency, sequential revision, Tree of Thoughts
- [ ] **Module 5 — Build an Image Generation Service**
  VAEs, GANs, autoregressive models, diffusion (U-Net/DiT), text-to-image, diffusion training/sampling
- [ ] **Module 6 — Capstone Project**
  Choose-your-own project combining techniques from Modules 1–5
- [ ] **Module 7 — Learn Agent, RAG**
  Query decomposition, context engineering, memory, meta-agent patterns, single vs multi-agent RAG, hierarchical agentic RAG, multimodal RAG, context management
- [ ] **Module 8 — Learn MCP, A2A**
  MCP vs normal API tooling, clients/servers, tool/resource discovery, A2A definitions, AgentCards, capability discovery, latency/versioning

## Progress Log

Steps are logged here as they're completed.

### Module 1 — Build an LLM Playground

- [x] **Step 1 — Data collection.** `module1_llm_playground/data_collection/`
  - `manual_crawl.py`: crawls a small set of Wikipedia seed pages, checking
    `robots.txt` before each fetch, and extracts paragraph text via BeautifulSoup.
  - `common_crawl.py`: streams a real Common Crawl WET file (finds the latest
    crawl via the CC index, picks its first WET path, reads records with
    `warcio`) and saves the first N plain-text conversion records — the same
    raw source RefinedWeb/FineWeb/Dolma are built from.
  - Run both: `python -m module1_llm_playground.data_collection.collect --source both`
  - Bug found + fixed during testing: Python's `RobotFileParser.read()` silently
    treats *every* URL as disallowed on sites like Wikipedia because it doesn't
    strip the UTF-8 BOM at the start of `robots.txt`. Worked around by fetching
    robots.txt with `requests`, decoding as `utf-8-sig`, and feeding the lines
    to `parser.parse()` instead of `parser.read()`.

- [x] **Step 2 — Data cleaning.** `module1_llm_playground/data_cleaning/`
  - `filters.py`: Gopher/RefinedWeb-style heuristics (min word count, mean
    word length, alphabetic-word ratio, stopword presence, symbol ratio).
  - `dedup.py`: exact paragraph-level dedup via SHA-256 of normalized text
    (near-dup/MinHash skipped as overkill at this corpus size).
  - `clean.py`: orchestrates normalize → English-language filter
    (`langdetect`) → quality filters → dedup → `data/processed/corpus.txt`.
  - Run: `python -m module1_llm_playground.data_cleaning.clean`
  - Tested against real Step 1 output: 15 docs in, 9 dropped as non-English
    (mostly the Common Crawl sample), 6 kept, 0 duplicate paragraphs (corpus
    too small/diverse yet to trigger dedup).
  - **Known limitation found in testing:** a Common Crawl record that was
    just a generic "hosting provider placeholder" error page passed every
    quality heuristic — it's grammatically valid English with normal word
    length and stopword density, so nothing here flags it as boilerplate.
    Real pipelines catch this with classifier/perplexity-based filtering or
    template-dedup across domains; out of scope for this step's heuristics.

- [x] **Step 3 — BPE tokenizer from scratch.** `module1_llm_playground/tokenizer/`
  - `bpe.py`: byte-level BPE (GPT-2 style) implemented from scratch — text is
    pre-split with GPT-2's regex (so merges never cross a word/punctuation/
    whitespace boundary), converted to raw UTF-8 bytes (256 base tokens), then
    the most frequent adjacent byte-pair is merged into a new token id,
    repeated until `vocab_size` is reached. Supports `train`, `encode`,
    `decode`, and JSON `save`/`load` of the learned merges.
  - `train_tokenizer.py`: trains the tokenizer on `data/processed/corpus.txt`
    and saves the learned merges to `data/tokenizer/merges.json`.
  - Run: `python -m module1_llm_playground.tokenizer.train_tokenizer --verbose`
  - Tested against real Step 2 output: trained a 1000-token vocab (744
    merges) on the 159,791-byte corpus, compressing it to 57,497 tokens
    (2.78x). Verified exact `decode(encode(x)) == x` roundtrip on the full
    corpus plus edge cases (empty string, emoji/CJK unicode, whitespace-only
    input).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
