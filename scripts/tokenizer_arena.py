"""P5 "Tokenizer 2" - Asama 1: tokenizer adaylarini egit + sikistirma/hiz olc.

Adaylar:
  bpe_wb_{4k,8k,16k} : klasik byte-level BPE (kelime sinirli; GPT-2 tarzi regex split) - kontrol
  bpe_sw_{4k,8k,16k} : "super-kelime" BPE (use_regex=False; mergeler bosluk sinirini asabilir)
  + kontrol satiri: data/ts_bpe8k.json (P3'te 300k hikaye ile egitilen mevcut tokenizer)

Olcum (held-out TinyStories valid ilk 3000 hikaye):
  bytes/token (sikistirma), tokens/byte, encode hizi (MB/s).

Cikti: data/tok2/*.json (adaylar) + results/tokenizer_arena_v1.json (ozet tablo).

Asama 2 (GPU, sonra): her tokenizasyonla esit-adim mini-LM egit -> bpb arenasi.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
EOS = "<|endoftext|>"


def train_bpe(texts, vocab, superword):
    tok = Tokenizer(models.BPE())
    try:
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False,
                                                     use_regex=not superword)
    except TypeError:
        if superword:
            raise SystemExit("tokenizers surumu use_regex desteklemiyor")
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=vocab, special_tokens=[EOS],
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    t0 = time.time()
    tok.train_from_iterator(texts, trainer=tr, length=len(texts))
    return tok, time.time() - t0


def measure(tok, texts, desc):
    n_bytes = sum(len(t.encode("utf-8")) for t in texts)
    t0 = time.time()
    encs = tok.encode_batch(texts, add_special_tokens=False)
    dt = max(time.time() - t0, 1e-9)
    n_tok = sum(len(e.ids) for e in encs)
    # roundtrip dogrulama (ilk hikaye)
    rt = tok.decode(encs[0].ids) == texts[0]
    return {"desc": desc, "vocab": tok.get_vocab_size(), "n_bytes": n_bytes,
            "n_tok": n_tok, "bytes_per_tok": n_bytes / max(n_tok, 1),
            "tok_per_byte": n_tok / max(n_bytes, 1),
            "enc_mbps": (n_bytes / 1e6) / dt, "roundtrip_ok": rt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=300000)
    args = ap.parse_args()

    ds = load_dataset("roneneldan/TinyStories", split="train")
    n = min(args.sample, len(ds))
    step = max(1, len(ds) // n)
    train_texts = ds.select(range(0, n * step, step))["text"]
    valid_texts = load_dataset("roneneldan/TinyStories", split="validation[:3000]")["text"]
    print(f"[data] tok egitimi: {len(train_texts)} hikaye | valid olcum: {len(valid_texts)} hikaye", flush=True)

    os.makedirs(os.path.join(DATA, "tok2"), exist_ok=True)
    rows = []

    ctrl = os.path.join(DATA, "ts_bpe8k.json")
    if os.path.exists(ctrl):
        rows.append(measure(Tokenizer.from_file(ctrl), valid_texts, "p3_bpe8k_kontrol"))

    for superword in (False, True):
        for vocab in (4096, 8192, 16384):
            name = ("bpe_sw" if superword else "bpe_wb") + f"_{vocab // 1024}k"
            tok, dt = train_bpe(train_texts, vocab, superword)
            path = os.path.join(DATA, "tok2", name + ".json")
            tok.save(path)
            m = measure(tok, valid_texts, name)
            m["train_sec"] = round(dt, 1)
            m["path"] = os.path.relpath(path, ROOT)
            rows.append(m)
            print(f"[tok] {name}: egitim {dt:.0f}s | {m['bytes_per_tok']:.2f} byte/tok | "
                  f"enc {m['enc_mbps']:.1f} MB/s", flush=True)

    out = {"stage": "tokenizers_v1", "sample_stories": len(train_texts),
           "valid_stories": len(valid_texts),
           "ts": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "rows": rows}
    with open(os.path.join(RES, "tokenizer_arena_v1.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print("\n== TABLO (held-out valid, 3000 hikaye) ==")
    print(f"{'isim':<20}{'vocab':>7}{'byte/tok':>10}{'tok/byte':>10}{'enc MB/s':>10}{'rt':>4}")
    for r in rows:
        print(f"{r['desc']:<20}{r['vocab']:>7}{r['bytes_per_tok']:>10.3f}"
              f"{r['tok_per_byte']:>10.4f}{r['enc_mbps']:>10.1f}{'ok' if r['roundtrip_ok'] else 'HATA':>4}")


if __name__ == "__main__":
    main()
