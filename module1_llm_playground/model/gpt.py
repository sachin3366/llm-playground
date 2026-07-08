"""A small GPT-style transformer, built from scratch in PyTorch.

This mirrors the architecture from "Attention Is All You Need" / GPT-2, but
kept deliberately small and linear so each piece can be read top to bottom:

    tokens -> token embedding + position embedding
           -> N x [ self-attention -> feed-forward ]   (each with residual + layernorm)
           -> final layernorm
           -> linear projection to vocab logits

Every class below is one concept. Nothing here is optimized for speed or
production use — see nanoGPT / GPT-2 for that. The goal is to be able to
trace exactly what happens to a tensor at each step.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int  # how many distinct tokens the model can read/write (from Step 3's tokenizer)
    block_size: int = 128  # max sequence length (context window) the model can look at
    n_layer: int = 4  # how many transformer blocks are stacked
    n_head: int = 4  # how many attention heads per block
    n_embd: int = 128  # size of the embedding vector for each token
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """Lets each token look back at earlier tokens (and itself) to decide what
    information to pull forward, weighted by how relevant each earlier token is.

    "Causal" means a token is only allowed to attend to positions <= itself —
    it can't see the future. That's what makes this usable for next-token
    prediction: at training time every position predicts the *next* token
    using only what came before it.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must split evenly across heads"
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head

        # One linear layer produces all three projections at once (query, key, value),
        # each of size n_embd, concatenated together -> output size 3 * n_embd.
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        # After attention mixes information across tokens, this projects back
        # to n_embd so the result can be added into the residual stream.
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # A lower-triangular mask: causal_mask[i, j] is True when position j is
        # allowed to be attended to from position i (j <= i). Registered as a
        # buffer (not a parameter) since it's fixed, not learned.
        causal_mask = torch.tril(torch.ones(config.block_size, config.block_size)).bool()
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x):
        batch_size, seq_len, n_embd = x.shape

        # Project x into query/key/value, then split into separate heads.
        # Each head learns to attend to a different kind of relationship
        # (e.g. one head might track "the subject of this verb", another
        # "the previous occurrence of this word").
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3 * n_embd)
        q, k, v = qkv.split(n_embd, dim=2)  # each: (batch, seq_len, n_embd)

        def split_heads(t):
            # (batch, seq_len, n_embd) -> (batch, n_head, seq_len, head_size)
            t = t.view(batch_size, seq_len, self.n_head, self.head_size)
            return t.transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # Attention score: how much should position i attend to position j?
        # Dot product of query_i and key_j, scaled to keep gradients stable
        # (without scaling, larger head sizes push softmax into saturation).
        attn_scores = q @ k.transpose(-2, -1)  # (batch, n_head, seq_len, seq_len)
        attn_scores = attn_scores / (self.head_size ** 0.5)

        # Block attention to future positions by setting those scores to -inf
        # before the softmax, so they become exactly 0 probability.
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(~causal_mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of value vectors: each output position is a blend of
        # the value vectors of the positions it attends to.
        out = attn_weights @ v  # (batch, n_head, seq_len, head_size)

        # Merge heads back into one n_embd-sized vector per token.
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out


class FeedForward(nn.Module):
    """A per-token MLP applied after attention.

    Attention mixes information *across* tokens; this step processes each
    token's resulting vector *independently*, giving the model extra capacity
    to transform what attention gathered. The hidden layer is 4x wider than
    n_embd, following the original Transformer paper's convention.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden_size = 4 * config.n_embd
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """One transformer block: attention, then feed-forward, each wrapped in
    a residual connection and preceded by layernorm (the "pre-norm" layout
    used by GPT-2, which trains more stably than normalizing after the
    sublayer).

    The residual connections (`x = x + sublayer(x)`) are what let gradients
    flow cleanly through many stacked blocks instead of vanishing.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ff = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPT(nn.Module):
    """The full model: embed tokens + positions, run them through a stack of
    transformer blocks, then project back to vocabulary-sized logits.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_final = nn.LayerNorm(config.n_embd)

        # Projects the final hidden state at each position to a score for
        # every token in the vocabulary — the model's prediction for "what
        # comes next" at that position.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, token_ids, targets=None):
        batch_size, seq_len = token_ids.shape
        assert seq_len <= self.config.block_size, (
            f"sequence length {seq_len} exceeds block_size {self.config.block_size}"
        )

        positions = torch.arange(seq_len, device=token_ids.device)

        # Each token gets: (a) an embedding for *what* it is, plus (b) an
        # embedding for *where* it is in the sequence. Attention has no
        # built-in sense of order, so position embeddings are what tell the
        # model "this token comes before that one."
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)

        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)

        loss = None
        if targets is not None:
            # Next-token prediction loss: at every position, compare the
            # predicted distribution over the vocab to the token that
            # actually came next. Flatten batch and sequence dims together
            # since cross_entropy expects one prediction per row.
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
