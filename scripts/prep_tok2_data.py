"""P5 "Tokenizer 2" - Asama 2 veri hazirligi: secilen tokenizer'larla AYNI metni tokenize et.

Kaynak metin (tum kollar icin ayni):
  - train: TinyStories train ilk --stories hikaye
  - valid: validation[:--valid_n]  (arena asama-1 ile ayni dilim)

Kollar: ctrl8k (ts_bpe8k = klasik/kontrol), wb16k, sw8k, sw16k.

Cikti (data/): tok2_{name}_train.pt / tok2_{name}_valid.pt
  -> {"tokens": uint16 tensor, "n_bytes": int, "n_stories": int, "tok": relpath}
  n_bytes sayesinde egitim sirasinda tokenizer-bagimsiz bpb hesaplanabilir:
  bpb = (toplam CE nats) / ln(2) / n_bytes.

Ozet: results/tok2_prep_v1.json

Kullanim:
  python scripts/prep_tok2_data.py                 # tam kosu (arka planda)
  python scripts/prep_tok2_data.py --stories 5000 --valid_n 100   # hizli test
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from datasets import load_dataset
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
EOS = "<|endoftext|>"

ARMS = [
    ("ctrl8k", "ts_bpe8k.json"),
    ("wb16k", "tok2\\bpe_wb_16k.json"),
    ("sw8k", "tok2\\bpe_sw_8k.json"),
    ("sw16k", "tok2\\bpe_sw_16k.json"),
]


def encode_split(tok, ds, eos_arr, desc):
    """Hikaye hikaye tokenize; her hikayenin sonuna EOS. uint16 dizi + byte sayaci."""
    chunks = []
    n_bytes = 0
    t0 = time.time()
    B = 4000
    n = len(ds)
    for i in range(0, n, B):
        texts = ds[i: i + B]["text"]
        n_bytes += sum(len(t.encode("utf-8")) for t in texts)
        encs = tok.encode_batch(texts, add_special_tokens=False)
        parts = []
        for e in encs:
            parts.append(np.asarray(e.ids, dtype=np.uint16))
            parts.append(eos_arr)
        chunks.append(np.concatenate(parts))
        if (i // B) % 50 == 0:
            print(f"  [{desc}] {min(i + B, n)}/{n} hikaye | {time.time() - t0:.0f}s", flush=True)
    arr = np.concatenate(chunks)
    bpt = n_bytes / max(len(arr), 1)
    print(f"  [{desc}] bitti: {len(arr)} token | {n_bytes/1e6:.2f}M byte | "
          f"{bpt:.3f} byte/token | {time.time() - t0:.0f}s", flush=True)
    return torch.from_numpy(arr), n_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stories", type=int, default=1000000)
    ap.add_argument("--valid_n", type=int, default=3000)
    args = ap.parse_args()

    ds_t = load_dataset("roneneldan/TinyStories", split=f"train[:{args.stories}]")
    ds_v = load_dataset("roneneldan/TinyStories", split=f"validation[:{args.valid_n}]")
    print(f"[data] train={len(ds_t)} hikaye (ilk) | valid={len(ds_v)} hikaye", flush=True)

    rows = []
    for name, rel in ARMS:
        tok = Tokenizer.from_file(os.path.join(DATA, rel))
        eos_arr = np.asarray([tok.token_to_id(EOS)], dtype=np.uint16)
        print(f"[arm] {name} <- {rel} (vocab={tok.get_vocab_size()})", flush=True)
        tr, tr_b = encode_split(tok, ds_t, eos_arr, f"{name}_train")
        va, va_b = encode_split(tok, ds_v, eos_arr, f"{name}_valid")
        torch.save({"tokens": tr, "n_bytes": tr_b, "n_stories": len(ds_t), "tok": rel},
                   os.path.join(DATA, f"tok2_{name}_train.pt"))
        torch.save({"tokens": va, "n_bytes": va_b, "n_stories": len(ds_v), "tok": rel},
                   os.path.join(DATA, f"tok2_{name}_valid.pt"))
        rows.append({"name": name, "tok": rel, "vocab": tok.get_vocab_size(),
                     "train_tok": len(tr), "train_bytes": tr_b,
                     "train_bpt": round(tr_b / len(tr), 3),
                     "valid_tok": len(va), "valid_bytes": va_b,
                     "valid_bpt": round(va_b / len(va), 3),
                     "step_capacity": len(tr) // (512 * 48)})
        print(f"[ok] {name} kaydedildi", flush=True)

    out = {"stage": "tok2_data_v1", "stories": len(ds_t), "valid_n": len(ds_v),
           "ts": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "arms": rows}
    with open(os.path.join(RES, "tok2_prep_v1.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print("\n== TOK2 VERI OZETI ==", flush=True)
    print(f"{'kol':<8}{'vocab':>7}{'train_tok':>12}{'train B/t':>10}{'valid B/t':>10}{'step_kap':>9}")
    for r in rows:
        print(f"{r['name']:<8}{r['vocab']:>7}{r['train_tok']:>12}{r['train_bpt']:>10.3f}"
              f"{r['valid_bpt']:>10.3f}{r['step_capacity']:>9}")


if __name__ == "__main__":
    main()
