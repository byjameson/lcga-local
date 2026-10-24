"""Smoke-test the Qwen3-4B-Instruct-2507 model on the int8 paged engine.

Run: python scripts/smoke_qwen3_4b.py
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lcsa.paged_qwen import PagedQwen  # noqa: E402

MODEL = ROOT / "models" / "Qwen3-4B-Instruct-2507"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resident-quant", action="store_true",
                    help="keep all layers as int8 on the GPU (no CPU paging)")
    ap.add_argument("--quant-cache", type=Path, default=None,
                    help="path to the int8 resident cache (created on first run)")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    t0 = time.perf_counter()
    eng = PagedQwen(MODEL, device="cuda", dtype=torch.bfloat16,
                    resident_k=None, max_seq=512, pin=False, quant="int8",
                    resident_quant=args.resident_quant,
                    quant_cache=args.quant_cache, verbose=True)
    mode = "gpu-int8" if args.resident_quant else "paged"
    print(f"[smoke] {mode} load {time.perf_counter()-t0:.1f}s "
          f"resident={eng.resident_k}/{eng.n_layers}")

    prompts = [
        "Mısır'ın başkenti neresi? Kısa cevap ver.",
        "Üç meyve listele.",
        "What are the three primary colors? Answer briefly.",
    ]
    for q in prompts:
        msgs = [{"role": "user", "content": q}]
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").input_ids
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out, stats = eng.generate(ids, max_new_tokens=96, temperature=0.0,
                                  eos_id=tok.eos_token_id, seed=1234)
        dt = time.perf_counter() - t0
        answer = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        peak = torch.cuda.max_memory_allocated() / 2**20
        print(f"\nQ: {q}\nA: {answer!r}")
        print(f"   {dt:.2f}s prefill={stats['prefill_s']:.2f}s "
              f"decode={stats['tok_s']:.2f} tok/s peak={peak:.0f}MB "
              f"free={torch.cuda.mem_get_info()[0]/2**30:.2f}GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
