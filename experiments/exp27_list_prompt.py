"""exp27: can a targeted list-format hint fix the strict list score?

exp26 showed 5/8 list-format hits for the chat model. The model often writes
"Monday, Tuesday, Wednesday" on one line, which the strict three-line checker
counts as a miss even though the content is right. This probe tests two
low-risk hints on the same 8 list prompts:

  plain   : current product prompt
  suffix  : append "one item per line, starting with -" to the user turn
  system  : put the same rule in the system prompt

Only the chat model is loaded; 24 short generations.

Usage:
  python experiments/exp27_list_prompt.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from exp25_chat_eval import list_hit, norm  # noqa: E402
from exp26_quality60 import CHAT_MODEL, CHAT_SYSTEM, PROMPTS  # noqa: E402

MAX_NEW = 64
SUFFIX = "\nHer maddeyi ayri satirda tire (-) ile yaz."
SYS_HINT = " Liste istenirse her maddeyi ayri satirda tire (-) ile yaz."


def main() -> int:
    model_path = str(ROOT / CHAT_MODEL)
    if not (Path(model_path) / "config.json").exists():
        raise SystemExit(f"model not found: {model_path}")
    list_prompts = [p for p in PROMPTS if p[0] == "list"]
    print(f"[exp27] {len(list_prompts)} list prompts x 3 variants", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()
    load_s = time.perf_counter() - t0

    variants = []
    for label in ("plain", "suffix", "system"):
        rows = []
        for cat, prompt, keywords, exact, needs_list in list_prompts:
            user = prompt + (SUFFIX if label == "suffix" else "")
            system = CHAT_SYSTEM + (SYS_HINT if label == "system" else "")
            text = tok.apply_chat_template(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt").input_ids.to("cuda")
            g0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            gen_s = time.perf_counter() - g0
            dec = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            n_new = int(out.shape[1] - ids.shape[1])
            hit = list_hit(dec)
            rows.append({"variant": label, "category": cat, "prompt": prompt,
                         "text": dec.strip(), "gen_s": round(gen_s, 3),
                         "n_new": n_new,
                         "tok_s": round(n_new / gen_s, 2) if gen_s else None,
                         "list_hit": hit})
            safe = dec.strip().replace("\n", " / ")[:100].encode(
                "ascii", "backslashreplace").decode()
            print(f"[{label}] {hit} -> {safe!r}", flush=True)
        rate = sum(r["list_hit"] for r in rows) / len(rows)
        variants.append({"label": label, "model": Path(model_path).name,
                         "load_s": round(load_s, 2),
                         "peak_vram_mb": round(
                             torch.cuda.max_memory_allocated() / 2**20, 1),
                         "rows": rows,
                         "summary": {"n": len(rows),
                                     "list_hit_rate": round(rate, 3),
                                     "avg_gen_s": round(
                                         sum(r["gen_s"] for r in rows) / len(rows), 3)}})
        print(f"[summary] {label}: list={rate:.3f}", flush=True)

    out = {"variants": variants,
           "summary": {"exp": "exp27_list_prompt", "n": len(list_prompts),
                       "max_new_tokens": MAX_NEW,
                       "rates": {v["label"]: v["summary"]["list_hit_rate"]
                                 for v in variants},
                       "ts": time.strftime("%Y-%m-%d %H:%M:%S")}}
    path = ROOT / "results" / "exp27_list_prompt.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    with open(ROOT / "results" / "results.jsonl", "a", encoding="utf-8") as f:
        for v in variants:
            for r in v["rows"]:
                f.write(json.dumps({"exp": "exp27_list_prompt",
                                    "tag": f"exp27_{v['label']}", **r},
                                   ensure_ascii=False) + "\n")
        f.write(json.dumps(out["summary"], ensure_ascii=False) + "\n")
    print(f"[exp27] wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
