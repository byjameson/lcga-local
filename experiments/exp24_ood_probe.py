"""exp24: out-of-domain probe for the WT2 fine-tune (base vs tuned 3B).

The checkpoint wins on held-out WT2 CE (exp23), but it was fine-tuned on
Wikipedia-like text. This probe checks whether that win costs quality on short
factual, code and Turkish continuations, using the same int8 paged engine.

Metric per probe: teacher-forced CE of the reference continuation and top-1
token accuracy; aggregated per domain.

Usage:
  python experiments/exp24_ood_probe.py
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lcsa.paged_qwen import PagedQwen  # noqa: E402

PROBES = [
    ("fact", "The capital of France is", " Paris"),
    ("fact", "The capital of Germany is", " Berlin"),
    ("fact", "The author of Hamlet was", " William Shakespeare"),
    ("fact", "Water boils at", " 100 degrees Celsius"),
    ("fact", "The largest planet in the solar system is", " Jupiter"),
    ("fact", "The chemical symbol for gold is", " Au"),
    ("fact", "Two plus three equals", " five"),
    ("fact", "The first president of the United States was", " George Washington"),
    ("code", "def add(a, b):\n    return", " a + b"),
    ("code", "for i in range(10):\n    print", "(i)"),
    ("code", "The HTML tag for a paragraph is", " <p>"),
    ("code", "SELECT * FROM users WHERE", " id = 1"),
    ("tr", "Fransa'nın başkenti", " Paris"),
    ("tr", "Türkiye'nin başkenti", " Ankara"),
    ("tr", "İki artı üç eşittir", " beş"),
    ("tr", "İstanbul'un en kalabalık ilçesi", " Esenyurt"),
    ("tr", "Ankara, Türkiye'nin", " başkentidir"),
    ("tr", "İstanbul'da bulunan ünlü cami", " Ayasofya"),
    ("tr", "Bir yıl kaç aydır?", " 12"),
    ("tr", "Türkiye'nin en yüksek dağı", " Ağrı Dağı"),
    ("tr", "Merhaba, nasılsın", "?"),
    ("tr", "Bugün hava çok", " güzel"),
    ("tr", "Kitap okumak", " güzeldir"),
    ("tr", "Python programlama dili", " popülerdir"),
]


def score_probe(eng, tok, prompt: str, ref: str):
    p = tok(prompt, add_special_tokens=False).input_ids
    full = tok(prompt + ref, add_special_tokens=False).input_ids
    aligned = full[:len(p)] == p
    start = len(p) if aligned else 1
    ids = torch.tensor([full], dtype=torch.long, device="cuda")
    eng.reset_cache()
    with torch.no_grad():
        logits = eng.forward(ids, start_pos=0, all_logits=True)
    lg = logits[0, start - 1:-1].float()
    tgt = ids[0, start:]
    ce = float(F.cross_entropy(lg, tgt, reduction="mean"))
    top1 = float((lg.argmax(-1) == tgt).float().mean())
    return ce, top1, int(tgt.numel()), aligned


def run_variant(label: str, model_path: str, ckpt: str | None,
                tok) -> list[dict]:
    sd = None
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    eng = PagedQwen(model_path, device="cuda", dtype=torch.bfloat16,
                    resident_k=None, max_seq=512, pin=False, quant="int8",
                    state_dict=sd, verbose=False)
    load_s = time.perf_counter() - t0
    del sd
    out = []
    for domain, prompt, ref in PROBES:
        ce, top1, n_tok, aligned = score_probe(eng, tok, prompt, ref)
        out.append({"variant": label, "domain": domain, "prompt": prompt,
                    "ref": ref, "ce": round(ce, 5),
                    "ppl": round(math.exp(min(ce, 20)), 3),
                    "top1": round(top1, 3), "ref_tokens": n_tok,
                    "aligned": aligned, "load_s": round(load_s, 2)})
        print(f"[{label}] {domain:4s} CE={ce:.4f} top1={top1:.2f} "
              f"({n_tok} tok) {prompt[:38]!r}", flush=True)
    peak = (round(torch.cuda.max_memory_allocated() / 2**20, 1)
            if torch.cuda.is_available() else None)
    for r in out:
        r["peak_vram_mb"] = peak
    del eng
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-3B")
    ap.add_argument("--ckpt", default="checkpoints/qwen3b_full_adaf_t300_best.pt")
    ap.add_argument("--tag", default="exp24")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    model_path = str(ROOT / args.model) if not Path(args.model).is_absolute() else args.model
    ckpt = str(ROOT / args.ckpt) if not Path(args.ckpt).is_absolute() else args.ckpt
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print(f"[exp24] {len(PROBES)} probes, model={Path(model_path).name}, "
          f"ckpt={Path(ckpt).name}", flush=True)
    base = run_variant("base", model_path, None, tok)
    tuned = run_variant("tuned", model_path, ckpt, tok)

    domains = sorted({p[0] for p in PROBES})
    agg = {}
    for d in domains:
        b = [r for r in base if r["domain"] == d]
        t = [r for r in tuned if r["domain"] == d]
        agg[d] = {
            "n": len(b),
            "ce_base": round(sum(r["ce"] for r in b) / len(b), 5),
            "ce_tuned": round(sum(r["ce"] for r in t) / len(t), 5),
            "ce_delta": round((sum(r["ce"] for r in t) / len(t)) -
                              (sum(r["ce"] for r in b) / len(b)), 5),
            "top1_base": round(sum(r["top1"] for r in b) / len(b), 3),
            "top1_tuned": round(sum(r["top1"] for r in t) / len(t), 3),
        }
    all_b = sum(r["ce"] for r in base) / len(base)
    all_t = sum(r["ce"] for r in tuned) / len(tuned)
    summary = {
        "exp": "exp24_ood",
        "tag": f"{args.tag}_summary",
        "model": Path(model_path).name,
        "checkpoint": Path(ckpt).name,
        "n_probes": len(PROBES),
        "ce_all_base": round(all_b, 5),
        "ce_all_tuned": round(all_t, 5),
        "ce_all_delta": round(all_t - all_b, 5),
        "domains": agg,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print("[summary] " + json.dumps(summary, ensure_ascii=True), flush=True)

    out = ROOT / "results" / f"{args.tag}_ood.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"rows": base + tuned, "summary": summary}, f,
                  indent=2, ensure_ascii=False)
    with open(ROOT / "results" / "results.jsonl", "a", encoding="utf-8") as f:
        for r in base + tuned:
            f.write(json.dumps({"exp": "exp24_ood",
                                "tag": f"{args.tag}_{r['variant']}",
                                **r}, ensure_ascii=False) + "\n")
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(f"[exp24] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
