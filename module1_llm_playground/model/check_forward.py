"""Sanity-check the GPT model against real data from Steps 2 and 3.

This isn't a training run (that's Step 5) — it just proves the architecture
is wired correctly before we build a training loop around it:

    1. Load the real trained tokenizer and encode the real corpus.
    2. Build small (input, target) batches by shifting the token stream by one
       (next-token prediction: predict token[i+1] from tokens[0..i]).
    3. Run a forward pass and confirm the loss is a finite, reasonable number.
    4. Run backward() and confirm every parameter actually receives a gradient
       (proves the whole graph is differentiable end to end).
    5. The most important correctness check: causal masking. If we change a
       *future* token in the input, the logits at earlier positions must not
       change at all. If they do, the model is illegally looking ahead.

Usage:
    python -m module1_llm_playground.model.check_forward
"""

from pathlib import Path

import torch

from module1_llm_playground.model.gpt import GPT, GPTConfig
from module1_llm_playground.tokenizer.bpe import BPETokenizer


def build_batches(token_ids, block_size, batch_size):
    """Slice a long list of token ids into (input, target) pairs for
    next-token prediction: target[i] is always input[i] shifted by one.
    """
    max_start = len(token_ids) - block_size - 1
    starts = torch.randint(0, max_start, (batch_size,))
    inputs = torch.stack([
        torch.tensor(token_ids[s : s + block_size]) for s in starts
    ])
    targets = torch.stack([
        torch.tensor(token_ids[s + 1 : s + block_size + 1]) for s in starts
    ])
    return inputs, targets


def check_causal_masking(model, config):
    """Prove positions can't see the future: change a late token and confirm
    only that position (and later ones) shift, never earlier positions.
    """
    torch.manual_seed(0)
    seq_len = min(16, config.block_size)
    tokens = torch.randint(0, config.vocab_size, (1, seq_len))

    model.eval()
    with torch.no_grad():
        logits_before, _ = model(tokens)

        edited = tokens.clone()
        edit_pos = seq_len - 1  # change only the last token
        edited[0, edit_pos] = (edited[0, edit_pos] + 1) % config.vocab_size
        logits_after, _ = model(edited)

    earlier_positions_changed = not torch.allclose(
        logits_before[:, :edit_pos], logits_after[:, :edit_pos]
    )
    if earlier_positions_changed:
        raise AssertionError(
            "Causal masking is broken: editing the last token changed logits "
            "at earlier positions, meaning the model can see the future."
        )
    print(f"Causal masking check passed: editing token at position {edit_pos} "
          f"left all {edit_pos} earlier positions' logits unchanged.")


def main():
    tokenizer = BPETokenizer()
    tokenizer.load("data/tokenizer/merges.json")

    text = Path("data/processed/corpus.txt").read_text(encoding="utf-8")
    token_ids = tokenizer.encode(text)
    print(f"Encoded corpus: {len(text):,} chars -> {len(token_ids):,} tokens "
          f"(vocab size {tokenizer.vocab_size})")

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=64,
        n_layer=4,
        n_head=4,
        n_embd=128,
    )
    model = GPT(config)
    print(f"Model: {config.n_layer} layers, {config.n_head} heads, "
          f"{config.n_embd}-dim embeddings -> {model.num_params():,} parameters")

    inputs, targets = build_batches(token_ids, config.block_size, batch_size=8)
    logits, loss = model(inputs, targets)

    print(f"Batch shapes: inputs {tuple(inputs.shape)}, "
          f"logits {tuple(logits.shape)}")
    print(f"Loss on an untrained model: {loss.item():.3f} "
          f"(random-guess baseline for this vocab is ~{torch.log(torch.tensor(float(config.vocab_size))).item():.3f})")

    loss.backward()
    missing_grads = [name for name, p in model.named_parameters() if p.grad is None]
    if missing_grads:
        raise AssertionError(f"These parameters never got a gradient: {missing_grads}")
    print(f"Backward pass OK: all {sum(1 for _ in model.parameters())} parameter "
          f"tensors received gradients.")

    check_causal_masking(model, config)


if __name__ == "__main__":
    main()
