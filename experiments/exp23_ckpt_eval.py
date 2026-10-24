"""exp23: held-out CE + generation for the base 3B vs our fine-tuned checkpoint.

Both variants run through the same int8 paged engine (the one the service uses),
so the comparison is "what the product would serve", not an fp32 reference.

Metrics:
  - held-out CE on the first N non-overlapping WT2 test chunks (teacher forcing)
  - greedy generation on a fixed prompt (text, tokens/s)
  - engine load time, peak VRAM, int8 quantization error

Usage:
  python experiments/exp23_ckpt_eval.py --chunks 4 --seq 256 --n_new 8
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lcsa.data import chunks_from_tokens, load_tokens  # noqa: E402
from lcsa.paged_qwen import PagedQwen  # noqa: E402

SEQ = 256
PROMPT = "The capital of Germany is"


def ascii_safe(s: str, n: int = 160) -> str:
    return s.encode("ascii", "backslashreplace").decode()[:n].replace("\n", " ")


def eval_ce(eng: PagedQwen, chunks: list[torch.Tensor], chunk: int = 64) -> float:
    total = 0.0
    n_tok = 0
    for ids in chunks:
        ids = ids.view(1, -1)
        eng.reset_cache()
        with torch.no_grad():
            logits = eng.forward(ids, start_pos=0, all_logits=True)
        T = ids.shape[1]
        for s in range(0, T - 1, chunk):
            e = min(s + chunk, T - 1)
            lg = logits[:, s:e].float().reshape(-1, logits.shape[-1])
            tgt = ids[:, s + 1:e + 1].reshape(-1).to(logits.device)
            total += float(F.cross_entropy(lg, tgt, reduction="sum"))
            n_tok += tgt.numel()
        del logits
    return total / max(n_tok, 1)


def run_variant(label: str, model_path: str, ckpt: str | None,
                chunks: list[torch.Tensor], n_new: int, seq: int) -> dict:
    from transformers import AutoTokenizer
    sd = None
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    eng = PagedQwen(model_path, device="cuda", dtype=torch.bfloat16,
                    resident_k=None, max_seq=max(seq, 512), pin=False, quant="int8",
                    state_dict=sd, verbose=True)
    load_s = time.perf_counter() - t0
    del sd
    t0 = time.perf_counter()
    ce = eval_ce(eng, chunks)
    ce_s = time.perf_counter() - t0
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    ids = tok(PROMPT, return_tensors="pt").input_ids
    g0 = time.perf_counter()
    out, stats = eng.generate(ids, max_new_tokens=n_new, temperature=0.0, seed=1234)
    gen_s = time.perf_counter() - g0
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    peak = (round(torch.cuda.max_memory_allocated() / 2**20, 1)
            if torch.cuda.is_available() else None)
    row = {
        "variant": label,
        "checkpoint": Path(ckpt).name if ckpt else None,
        "ce": round(ce, 5),
        "ce_chunks": len(chunks),
        "ce_seq": SEQ,
        "ce_s": round(ce_s, 2),
        "load_s": round(load_s, 2),
        "gen_s": round(gen_s, 3),
        "n_new": n_new,
        "tok_s": round(float(stats["tok_s"]), 3) if stats.get("tok_s") else None,
        "peak_vram_mb": peak,
        "quant_err_max": round(float(getattr(eng, "quant_err_max", 0.0)), 5),
        "text": ascii_safe(text, 200),
        "decoded": text,
    }
    print(f"[{label}] CE={row['ce']} ({ce_s:.1f}s) load={row['load_s']}s "
          f"gen={row['gen_s']}s peak={peak}MB quant_err={row['quant_err_max']}",
          flush=True)
    print(f"    text: {row['text']}", flush=True)
    del eng
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-3B")
    ap.add_argument("--ckpt", default="checkpoints/qwen3b_full_adaf_t300_best.pt")
    ap.add_argument("--data", default="data/wt2_test.pt")
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--seq", type=int, default=SEQ)
    ap.add_argument("--n_new", type=int, default=8)
    ap.add_argument("--tag", default="exp23")
    args = ap.parse_args()

    model_path = str(ROOT / args.model) if not Path(args.model).is_absolute() else args.model
    ckpt = str(ROOT / args.ckpt) if not Path(args.ckpt).is_absolute() else args.ckpt
    data_path = str(ROOT / args.data) if not Path(args.data).is_absolute() else args.data

    chunks = chunks_from_tokens(load_tokens(data_path), args.seq, args.chunks)
    print(f"[exp23] model={Path(model_path).name} chunks={len(chunks)} "
          f"seq={args.seq} ckpt={Path(ckpt).name}", flush=True)

    base = run_variant("base", model_path, None, chunks, args.n_new, args.seq)
    tuned = run_variant("tuned", model_path, ckpt, chunks, args.n_new, args.seq)
    delta = round(tuned["ce"] - base["ce"], 5)
    summary = {
        "exp": "exp23_ckpt_eval",
        "tag": f"{args.tag}_summary",
        "model": Path(model_path).name,
        "checkpoint": Path(ckpt).name,
        "ce_base": base["ce"],
        "ce_tuned": tuned["ce"],
        "ce_delta": delta,
        "ce_rel_pct": round(100.0 * delta / base["ce"], 2),
        "gen_base_s": base["gen_s"],
        "gen_tuned_s": tuned["gen_s"],
        "peak_base_mb": base["peak_vram_mb"],
        "peak_tuned_mb": tuned["peak_vram_mb"],
        "chunks": len(chunks),
        "seq": args.seq,
        "n_new": args.n_new,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print("[summary] " + json.dumps(summary, ensure_ascii=True), flush=True)

    out = ROOT / "results" / f"{args.tag}_ckpt_eval.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"rows": [base, tuned], "summary": summary}, f, indent=2,
                  ensure_ascii=False)
    with open(ROOT / "results" / "results.jsonl", "a", encoding="utf-8") as f:
        for label, r in (("base", base), ("tuned", tuned)):
            row = {"exp": "exp23_ckpt_eval", "tag": f"{args.tag}_{label}", **r}
            row.pop("decoded", None)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(f"[exp23] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
