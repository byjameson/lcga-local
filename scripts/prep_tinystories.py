"""P3 veri hazirligi: TinyStories indir -> kendi BPE (8k) tokenizer -> tokenize -> kaydet.

Cikti dosyalari (data/):
  ts_bpe8k.json     kendi tokenizer'imiz (byte-level BPE, <|endoftext|> ozel token)
  ts_train.pt       uint16 tensor (train split)
  ts_valid.pt       uint16 tensor (validation split)

Kullanim:
  python scripts/prep_tinystories.py --smoke   # hizli boru hatti testi (kucuk)
  python scripts/prep_tinystories.py           # tam kosu (arka planda)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
EOS = "<|endoftext|>"


def train_tokenizer(texts, vocab):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab,
        special_tokens=[EOS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    print(f"[tok] egitim icin {len(texts)} hikaye", flush=True)
    t0 = time.time()
    tok.train_from_iterator(texts, trainer=trainer, length=len(texts))
    print(f"[tok] bitti ({time.time()-t0:.0f}s) vocab={tok.get_vocab_size()}", flush=True)
    return tok


def encode_split(tok, ds, eos_arr, desc):
    chunks = []
    n_bytes = 0
    t0 = time.time()
    B = 4000
    n = len(ds)
    for i in range(0, n, B):
        texts = ds[i:i + B]["text"]
        n_bytes += sum(len(t.encode("utf-8")) for t in texts)
        encs = tok.encode_batch(texts, add_special_tokens=False)
        parts = []
        for e in encs:
            parts.append(np.asarray(e.ids, dtype=np.uint16))
            parts.append(eos_arr)
        chunks.append(np.concatenate(parts))
        if (i // B) % 25 == 0:
            done = min(i + B, n)
            print(f"  [{desc}] {done}/{n} hikaye | {time.time()-t0:.0f}s", flush=True)
    arr = np.concatenate(chunks)
    print(f"  [{desc}] bitti: {len(arr)} token | {n_bytes/1e6:.1f}M byte | "
          f"{n_bytes/max(len(arr),1):.2f} byte/token | {time.time()-t0:.0f}s", flush=True)
    return torch.from_numpy(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--tok_stories", type=int, default=300000)
    args = ap.parse_args()

    if args.smoke:
        ds_t = load_dataset("roneneldan/TinyStories", split="validation[:2000]")
        ds_v = load_dataset("roneneldan/TinyStories", split="validation[:3000]")
        tok_texts = ds_t["text"]
        suffix = "_smoke"
    else:
        ds_t = load_dataset("roneneldan/TinyStories", split="train")
        ds_v = load_dataset("roneneldan/TinyStories", split="validation")
        n = min(args.tok_stories, len(ds_t))
        step = max(1, len(ds_t) // n)
        tok_texts = ds_t.select(range(0, n * step, step))["text"]
        suffix = ""
        print(f"[data] train={len(ds_t)} valid={len(ds_v)} hikaye", flush=True)

    tok_path = os.path.join(DATA, f"ts_bpe{args.vocab // 1000}k{suffix}.json")
    tok = train_tokenizer(tok_texts, args.vocab)
    tok.save(tok_path)
    print(f"[tok] kaydedildi: {tok_path}", flush=True)

    eos_arr = np.asarray([tok.token_to_id(EOS)], dtype=np.uint16)
    tr = encode_split(tok, ds_t, eos_arr, "train" + suffix)
    va = encode_split(tok, ds_v, eos_arr, "valid" + suffix)
    torch.save(tr, os.path.join(DATA, f"ts_train{suffix}.pt"))
    torch.save(va, os.path.join(DATA, f"ts_valid{suffix}.pt"))
    print(f"[ok] ts_train{suffix}.pt={len(tr)} tok | ts_valid{suffix}.pt={len(va)} tok", flush=True)

    ids = va[:400].tolist()
    txt = tok.decode(ids).replace("\n", " ")
    print("[roundtrip] " + txt[:200].encode("ascii", "replace").decode(), flush=True)


if __name__ == "__main__":
    main()
