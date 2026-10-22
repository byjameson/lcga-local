"""exp18: M2/M3 — expert paging + transition-based prefetch for MoE-lite (top-1 Switch).

Configs:
  paged    : one reusable expert slot per layer (capacity 0)
  cap1     : LRU cache of 1 expert per layer
  lr2      : LRU cache of 2 experts per layer (no prefetch ablation)
  pf2      : two slots per layer + P(next expert | current) prefetch (M3)
  resident : all experts on GPU (capacity=E, preload)

Metrics: tok/s, expert MB copied/token, hit rate, prefetch hit fraction, CE parity,
peak VRAM.
Usage:
  python experiments/exp18_moe_paging.py --n_new 24 --eval_chunks 4 --warm 128 --tag m3_prefetch
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lcsa.paged_moe import PagedMoE  # noqa: E402


def vram() -> dict:
    if not torch.cuda.is_available():
        return {}
    return {
        "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/moe4_best.pt")
    ap.add_argument("--tok", default="data/ts_bpe8k.json")
    ap.add_argument("--valid", default="data/tok2_ctrl8k_valid.pt")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--n_new", type=int, default=24)
    ap.add_argument("--eval_chunks", type=int, default=4)
    ap.add_argument("--eval_chunk_len", type=int, default=256)
    ap.add_argument("--warm", type=int, default=128,
                    help="prefetch transition warmup tokens (M3 pf2 mode)")
    ap.add_argument("--modes", default="",
                    help="comma list to run a subset, e.g. 'pf2,resident'")
    ap.add_argument("--tag", default="m2_moe_paged")
    args = ap.parse_args()

    ckpt = ROOT / args.ckpt
    if not ckpt.exists():
        raise SystemExit(f"missing checkpoint: {ckpt}")
    d = torch.load(str(ckpt), map_location="cpu")
    E = int(d["cfg"]["moe"])

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(ROOT / args.tok))
    prompt_ids = torch.tensor([tok.encode(args.prompt).ids], dtype=torch.long)
    valid = torch.load(str(ROOT / args.valid), map_location="cpu")["tokens"]
    n_eval = args.eval_chunks * args.eval_chunk_len
    eval_tokens = valid[:n_eval].long()

    print(f"[exp18] ckpt={ckpt.name} E={E} prompt={args.prompt!r} "
          f"prompt_tokens={prompt_ids.shape[1]} eval_tokens={eval_tokens.numel()}", flush=True)

    rows = []
    configs = [("paged", 0, False, False), ("cap1", 1, False, False),
               ("lr2", 2, False, False), ("pf2", 2, False, True),
               ("resident", E, True, False)]
    want = {m.strip() for m in args.modes.split(",") if m.strip()}
    if want:
        configs = [c for c in configs if c[0] in want]
    for name, capacity, preload, prefetch in configs:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        eng = PagedMoE(str(ckpt), device="cuda", capacity=capacity,
                       preload=preload, prefetch=prefetch, verbose=True)
        load_s = time.time() - t0
        _ = eng.forward(prompt_ids)  # warmup (kernels + buffers)
        warm_used = 0
        if prefetch and args.warm > 0:
            warm_used = min(args.warm, eval_tokens.numel())
            with torch.no_grad():
                _ = eng.forward(eval_tokens[:warm_used].unsqueeze(0))
        eng.reset_stats()
        torch.cuda.reset_peak_memory_stats()
        out, tok_s = eng.generate(prompt_ids, args.n_new, temp=0.0, seed=1234)
        gen_stats = eng.stats()
        gen_stat_vram = vram()
        sample = tok.decode(out[0, prompt_ids.shape[1]:].tolist(), skip_special_tokens=False)
        sample = sample.replace("\n", " ")[:300]

        eng.reset_stats()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        ce = eng.eval_ce(eval_tokens, chunk=args.eval_chunk_len)
        ce_s = time.time() - t0
        eval_stats = eng.stats()
        eval_vram = vram()

        passes = 1 + args.n_new
        row = {
            "exp": "exp18_moe_paging",
            "tag": f"{args.tag}_{name}",
            "mode": name,
            "capacity": capacity,
            "preload": preload,
            "prefetch": prefetch,
            "warm": warm_used,
            "ckpt": ckpt.name,
            "E": E,
            "prompt": args.prompt,
            "n_new": args.n_new,
            "tok_s": round(tok_s, 3),
            "sample": sample,
            "gen_expert_mb": gen_stats["expert_mb"],
            "gen_mb_per_pass": round(gen_stats["expert_mb"] / passes, 3),
            "gen_copies": gen_stats["expert_copies"],
            "gen_hits": gen_stats["expert_hits"],
            "gen_hit_rate": None if gen_stats["hit_rate"] is None else round(gen_stats["hit_rate"], 4),
            "gen_prefetch_copies": gen_stats["prefetch_copies"],
            "gen_prefetch_mb": gen_stats["prefetch_mb"],
            "gen_prefetch_used": gen_stats["prefetch_used"],
            "gen_prefetch_used_frac": (None if gen_stats["prefetch_used_frac"] is None
                                        else round(gen_stats["prefetch_used_frac"], 4)),
            "load_s": round(load_s, 2),
            "ce": round(ce, 5),
            "ce_s": round(ce_s, 2),
            "eval_expert_mb": eval_stats["expert_mb"],
            "eval_expert_copies": eval_stats["expert_copies"],
            "eval_hit_rate": None if eval_stats["hit_rate"] is None else round(eval_stats["hit_rate"], 4),
            "usage": eval_stats["usage"],
            **gen_stat_vram,
            "eval_peak_alloc_mb": eval_vram.get("peak_alloc_mb"),
            "eval_peak_reserved_mb": eval_vram.get("peak_reserved_mb"),
        }
        rows.append(row)
        print(json.dumps({k: row[k] for k in (
            "mode", "tok_s", "gen_mb_per_pass", "gen_hit_rate",
            "gen_prefetch_copies", "gen_prefetch_used", "gen_prefetch_used_frac",
            "ce", "peak_alloc_mb", "peak_reserved_mb")}, ensure_ascii=True), flush=True)
        print(f"[sample:{name}] {sample}", flush=True)
        del eng

    # parity summary
    base = next((r for r in rows if r["mode"] == "resident"), rows[0])
    for r in rows:
        r["ce_parity_vs_resident"] = round(r["ce"] - base["ce"], 6)
        r["sample_match_resident"] = (r["sample"] == base["sample"])
    if all(r["sample"] == base["sample"] for r in rows):
        print("[parity] all modes: sample == resident sample (greedy)", flush=True)
    else:
        print("[parity] WARNING: some samples differ from resident", flush=True)

    with open(ROOT / "results" / "results.jsonl", "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(ROOT / "results" / "exp18_moe_paging.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
