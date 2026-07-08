"""Byte-level BPE tokenizer, built from scratch (GPT-2 style).

Text is first split into chunks with a regex so merges never span a chunk
boundary (e.g. a letter run never merges with the following punctuation or
whitespace) — this is the same trick GPT-2's tokenizer uses. Each chunk is
then turned into raw UTF-8 bytes (0-255), and BPE repeatedly merges the most
frequent adjacent byte-pair into a new token id until `vocab_size` is reached.
"""

import json
from collections import Counter

import regex as re

# GPT-2's pre-tokenization regex: splits on contractions, then greedily grabs
# a leading-space + run of letters / digits / other-symbols / whitespace.
GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _get_pair_counts(ids_list, counts=None):
    counts = counts if counts is not None else Counter()
    for ids in ids_list:
        for a, b in zip(ids, ids[1:]):
            counts[(a, b)] += 1
    return counts


def _merge(ids, pair, new_id):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class BPETokenizer:
    def __init__(self):
        self.merges = {}  # (int, int) -> int, in creation order
        self.vocab = {i: bytes([i]) for i in range(256)}  # int -> bytes
        self._pattern = re.compile(GPT2_SPLIT_PATTERN)

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256, "vocab_size must cover the 256 base byte tokens"
        num_merges = vocab_size - 256

        chunks = self._pattern.findall(text)
        ids_list = [list(chunk.encode("utf-8")) for chunk in chunks]

        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}

        for i in range(num_merges):
            counts = _get_pair_counts(ids_list)
            if not counts:
                break  # no more pairs to merge (corpus exhausted)
            pair = max(counts, key=counts.get)
            new_id = 256 + i
            ids_list = [_merge(ids, pair, new_id) for ids in ids_list]
            merges[pair] = new_id
            vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(
                    f"merge {i + 1}/{num_merges}: {pair} -> {new_id} "
                    f"({vocab[new_id]!r}) had {counts[pair]} occurrences"
                )

        self.merges = merges
        self.vocab = vocab

    def encode(self, text):
        chunks = self._pattern.findall(text)
        ids = []
        for chunk in chunks:
            chunk_ids = list(chunk.encode("utf-8"))
            while len(chunk_ids) >= 2:
                counts = _get_pair_counts([chunk_ids])
                # merge whichever eligible pair was learned earliest in training
                pair = min(counts, key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                chunk_ids = _merge(chunk_ids, pair, self.merges[pair])
            ids.extend(chunk_ids)
        return ids

    def decode(self, ids):
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    def save(self, path):
        # Store merges as an ordered list of [a, b] pairs; ids 256, 257, ...
        # are implied by list position, so the file stays small and readable.
        ordered_pairs = sorted(self.merges, key=lambda p: self.merges[p])
        data = {"merges": [[a, b] for a, b in ordered_pairs]}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path):
        with open(path) as f:
            data = json.load(f)

        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(data["merges"]):
            new_id = 256 + i
            merges[(a, b)] = new_id
            vocab[new_id] = vocab[a] + vocab[b]

        self.merges = merges
        self.vocab = vocab

    @property
    def vocab_size(self):
        return len(self.vocab)
