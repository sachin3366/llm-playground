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

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
