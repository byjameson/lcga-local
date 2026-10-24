"""Data preparation: wikitext-2 (train/dev), wikitext-103 test, TinyStories validation."""
from __future__ import annotations

import os
import torch


def _tok(tokenizer, text):
    ids = tokenizer(text)["input_ids"]
    return torch.tensor(ids, dtype=torch.int32)


def get_wikitext(config="wikitext-2-raw-v1", split="train"):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", config, split=split)
    return "\n".join(ds["text"])


def get_tinystories(n_stories=6000):
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split="validation")
    n = min(n_stories, len(ds))
    return "\n\n".join(ds[i]["text"] for i in range(n))


def build_all(tokenizer, data_dir, force=False):
    os.makedirs(data_dir, exist_ok=True)
    jobs = {
        "wt2_train": lambda: get_wikitext("wikitext-2-raw-v1", "train"),
        "wt2_test": lambda: get_wikitext("wikitext-2-raw-v1", "test"),
        "wt103_test": lambda: get_wikitext("wikitext-103-raw-v1", "test"),
        "tinystories_valid": lambda: get_tinystories(6000),
    }
    out = {}
    for name, fn in jobs.items():
        path = os.path.join(data_dir, f"{name}.pt")
        if os.path.exists(path) and not force:
            out[name] = path
            continue
        text = fn()
        ids = _tok(tokenizer, text)
        torch.save(ids, path)
        out[name] = path
        print(f"[data] {name}: {len(ids)} tokens -> {path}", flush=True)
    return out


def load_tokens(path):
    return torch.load(path, weights_only=True)


def chunks_from_tokens(tokens, seq_len, max_chunks=None):
    """Non-overlapping chunks of seq_len tokens -> (n, seq_len) int64."""
    n = len(tokens) // seq_len
    if max_chunks is not None:
        n = min(n, max_chunks)
    return tokens[: n * seq_len].view(n, seq_len).long()


def samples_from_tokens(tokens, seq_len, max_samples=None, seed=0):
    """Random non-overlapping windows for training."""
    n = len(tokens) // seq_len
    if max_samples is not None:
        n = min(n, max_samples)
    return tokens[: n * seq_len].view(n, seq_len).long()


class TrainSampler:
    """Iterate random contiguous windows over a token stream (random offsets)."""

    def __init__(self, tokens, seq_len, generator=None):
        self.tokens = tokens
        self.seq_len = seq_len
        self.g = generator or torch.Generator().manual_seed(0)

    def batch(self, batch_size):
        max_start = len(self.tokens) - self.seq_len - 1
        starts = torch.randint(0, max_start, (batch_size,), generator=self.g).tolist()
        xs = torch.stack([self.tokens[s: s + self.seq_len].long() for s in starts])
        ys = torch.stack([self.tokens[s + 1: s + self.seq_len + 1].long() for s in starts])
        return xs, ys
