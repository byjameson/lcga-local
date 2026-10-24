"""exp25b: does the chat system prompt change quality? (EN vs TR vs TR+format)

Same 24 prompts and scoring as exp25; one model load, three system prompts.
This is a prompt ablation, not a new model claim: n=24, keyword-based.

Usage:
  python experiments/exp25b_system_prompt.py --tag exp25b
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
sys.path.insert(0, str(ROOT / "experiments"))

from exp25_chat_eval import MAX_NEW, PROMPTS, score, summarize  # noqa: E402

CHAT_MODEL = "models/Qwen2.5-1.5B-Instruct"

SYSTEMS = {
    "en": ("You are a helpful assistant. Answer concisely in the user's "
           "language."),
    "tr": ("Sen yardimci bir asistansin. Kullanicinin dilinde, kisa ve dogru "
           "cevap ver."),
    "tr_fmt": ("Sen yardimci bir asistansin. Kullanicinin dilinde, kisa ve "
               "dogru cevap ver. Liste istenirse numarali maddeler kullan."),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=CHAT_MODEL)
    ap.add_argument("--tag", default="exp25b")
    args = ap.parse_args()
    model_path = str(ROOT / args.model) if not Path(args.model).is_absolute() else args.model
    if not (Path(model_path) / "config.json").exists():
        raise SystemExit(f"chat model not found: {model_path}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()
    load_s = time.perf_counter() - t0
    print(f"[exp25b] load {load_s:.2f}s, {len(PROMPTS)} prompts x {len(SYSTEMS)} prompts",
          flush=True)

    variants = []
    for name, system in SYSTEMS.items():
        rows = []
        for cat, prompt, keywords, exact, needs_list in PROMPTS:
            text = tok.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt").input_ids.to("cuda")
            g0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            gen_s = time.perf_counter() - g0
            dec = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            n_new = int(out.shape[1] - ids.shape[1])
            sc = score((cat, prompt, keywords, exact, needs_list), dec)
            rows.append({
                "variant": name, "system": system, "category": cat,
                "prompt": prompt, "text": dec.strip(),
                "gen_s": round(gen_s, 3), "n_new": n_new,
                "tok_s": round(n_new / gen_s, 2) if gen_s else None, **sc,
            })
            print(f"[{name}] {cat:7s} -> {dec.strip()[:62].encode('ascii','backslashreplace').decode()!r}",
                  flush=True)
        summary = summarize({"label": name, "model": Path(model_path).name,
                             "load_s": round(load_s, 2),
                             "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                             "rows": rows})
        variants.append({"label": name, "system": system, "rows": rows,
                         "summary": summary})
        print("[summary] " + json.dumps(summary, ensure_ascii=True), flush=True)

    peak = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    summary = {
        "exp": "exp25b_system_prompt", "tag": f"{args.tag}_summary",
        "model": Path(model_path).name, "n": len(PROMPTS),
        "max_new_tokens": MAX_NEW, "load_s": round(load_s, 2),
        "peak_vram_mb": peak,
        "systems": {v["label"]: v["summary"] for v in variants},
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = ROOT / "results" / f"{args.tag}_system_prompt.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"variants": variants, "summary": summary}, f, indent=2,
                  ensure_ascii=False)
    with open(ROOT / "results" / "results.jsonl", "a", encoding="utf-8") as f:
        for v in variants:
            for r in v["rows"]:
                f.write(json.dumps({"exp": "exp25b_system_prompt",
                                    "tag": f"{args.tag}_{v['label']}", **r},
                                   ensure_ascii=False) + "\n")
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(f"[exp25b] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
