"""exp26: 60-prompt quality check (base 1.5B vs instruct 1.5B).

The 24 prompts from exp25 are kept and 36 are added: 12 Turkish facts,
12 English facts, 12 Turkish instructions, 12 English instructions, 6 math,
6 list-format prompts. The chat side uses the product's Turkish system prompt,
so this is a larger check of the current service, not a rerun of exp25.

Scoring is intentionally simple: keyword hit, exact match where defined, and a
strict "three bullet/numbered lines" list check. This is not a human eval.

Usage:
  python experiments/exp26_quality60.py --tag exp26
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from exp25_chat_eval import (  # noqa: E402
    MAX_NEW, PROMPTS as EXP25_PROMPTS, list_hit, norm, score, summarize,
)

BASE_MODEL = "models/Qwen2.5-1.5B"
CHAT_MODEL = "models/Qwen2.5-1.5B-Instruct"
MAX_NEW_60 = 64
CHAT_SYSTEM = ("Sen yardimci bir asistansin. Kullanicinin dilinde, kisa ve "
               "dogru cevap ver.")

# (category, prompt, keywords, exact_answer or None, needs_list)
EXTRA = [
    # 6 more Turkish facts -> 12 total
    ("fact_tr", "Ispanya'nin baskenti neresi?", ["madrid"], None, False),
    ("fact_tr", "Almanya'nin baskenti neresi?", ["berlin"], None, False),
    ("fact_tr", "Italya'nin baskenti neresi?", ["roma", "rome"], None, False),
    ("fact_tr", "Dunyadaki en buyuk okyanus hangisi?", ["pasifik"], None, False),
    ("fact_tr", "Insan vucudundaki en buyuk organ hangisi?", ["deri"], None, False),
    ("fact_tr", "Bir haftada kac gun vardir?", ["7"], None, False),
    # 6 more English facts -> 12 total
    ("fact_en", "What is the capital of Spain?", ["madrid"], None, False),
    ("fact_en", "What is the capital of Italy?", ["rome", "roma"], None, False),
    ("fact_en", "What is the largest ocean on Earth?", ["pacific"], None, False),
    ("fact_en", "What is the largest organ in the human body?", ["skin"], None, False),
    ("fact_en", "How many days are there in a week?", ["7"], None, False),
    ("fact_en", "At what temperature does water freeze (Celsius)?", ["0"], None, False),
    # 6 more Turkish instructions -> 12 total
    ("ins_tr", "Sadece 'kirmizi' yaz.", ["kirmizi"], "kirmizi", False),
    ("ins_tr", "3 + 5 kac? Sadece sayiyi yaz.", ["8"], "8", False),
    ("ins_tr", "Tesekkur etmek icin kisa bir cumle yaz.", ["tesekkur"], None, False),
    ("ins_tr", "Su cumleyi Ingilizceye cevir: Kitap masada.",
     ["book", "table"], None, False),
    ("ins_tr", "'Evet' kelimesini iceren kisa bir cumle yaz.", ["evet"], None, False),
    ("ins_tr", "5'ten 1'e kadar sayilari yaz.", ["5", "4", "3", "2", "1"], None, False),
    # 6 more English instructions -> 12 total
    ("ins_en", "Reply with exactly the word 'red'.", ["red"], "red", False),
    ("ins_en", "What is 3 + 5? Answer with just the number.", ["8"], "8", False),
    ("ins_en", "Write a short thank-you sentence.", ["thank"], None, False),
    ("ins_en", "Translate 'The book is on the table' into Turkish.",
     ["kitap", "masa"], None, False),
    ("ins_en", "Write one short sentence containing the word 'yes'.",
     ["yes"], None, False),
    ("ins_en", "Count down from 5 to 1.", ["5", "4", "3", "2", "1"], None, False),
    # 6 math prompts
    ("math", "12 * 7 = ? Sadece sayiyi yaz.", ["84"], "84", False),
    ("math", "100 - 37 = ? Sadece sayiyi yaz.", ["63"], "63", False),
    ("math", "9 + 6 * 2 = ? Islem onceligine dikkat et.", ["21"], "21", False),
    ("math", "Bir duzine kac eder? Sadece sayiyi yaz.", ["12"], "12", False),
    ("math", "15'in yarisi kac? Sadece sayiyi yaz.", ["7"], None, False),
    ("math", "2^5 = ? Sadece sayiyi yaz.", ["32"], "32", False),
    # 6 list-format prompts
    ("list", "Uc meyve listele.", [], None, True),
    ("list", "List three animals.", [], None, True),
    ("list", "Uc renk listele.", [], None, True),
    ("list", "List three programming languages.", [], None, True),
    ("list", "Uc ulke listele.", [], None, True),
    ("list", "List three days of the week.", [], None, True),
]

PROMPTS = EXP25_PROMPTS + EXTRA


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--chat", default=CHAT_MODEL)
    ap.add_argument("--tag", default="exp26")
    args = ap.parse_args()
    base_path = str(ROOT / args.base) if not Path(args.base).is_absolute() else args.base
    chat_path = str(ROOT / args.chat) if not Path(args.chat).is_absolute() else args.chat
    for p in (base_path, chat_path):
        if not (Path(p) / "config.json").exists():
            raise SystemExit(f"model not found: {p}")
    print(f"[exp26] {len(PROMPTS)} prompts, max_new={MAX_NEW_60}, "
          f"base={Path(base_path).name}, chat={Path(chat_path).name}", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    variants = []
    for label, model_path, chat in (("base", base_path, False),
                                    ("chat", chat_path, True)):
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
        model.eval()
        load_s = time.perf_counter() - t0
        rows = []
        for cat, prompt, keywords, exact, needs_list in PROMPTS:
            if chat:
                text = tok.apply_chat_template(
                    [{"role": "system", "content": CHAT_SYSTEM},
                     {"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True)
            else:
                text = prompt
            ids = tok(text, return_tensors="pt").input_ids.to("cuda")
            g0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=MAX_NEW_60, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            gen_s = time.perf_counter() - g0
            dec = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            n_new = int(out.shape[1] - ids.shape[1])
            sc = score((cat, prompt, keywords, exact, needs_list), dec)
            rows.append({
                "variant": label, "category": cat, "prompt": prompt,
                "text": dec.strip(), "gen_s": round(gen_s, 3),
                "n_new": n_new, "tok_s": round(n_new / gen_s, 2) if gen_s else None,
                **sc,
            })
            safe = dec.strip()[:64].encode("ascii", "backslashreplace").decode()
            print(f"[{label}] {cat:7s} -> {safe!r}", flush=True)
        peak = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        v = {"label": label, "model": Path(model_path).name,
             "load_s": round(load_s, 2), "peak_vram_mb": peak, "rows": rows}
        variants.append(v)
        print("[summary] " + json.dumps(summarize(v), ensure_ascii=True),
              flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    s_base, s_chat = summarize(variants[0]), summarize(variants[1])
    summary = {
        "exp": "exp26_quality60", "tag": f"{args.tag}_summary",
        "n": len(PROMPTS), "max_new_tokens": MAX_NEW_60,
        "chat_system": CHAT_SYSTEM,
        "base": s_base, "chat": s_chat,
        "delta_keyword_hit": round((s_chat["keyword_hit_rate"] or 0) -
                                   (s_base["keyword_hit_rate"] or 0), 3),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = ROOT / "results" / f"{args.tag}_quality60.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"variants": variants, "summary": summary}, f, indent=2,
                  ensure_ascii=False)
    with open(ROOT / "results" / "results.jsonl", "a", encoding="utf-8") as f:
        for v in variants:
            for r in v["rows"]:
                f.write(json.dumps({"exp": "exp26_quality60",
                                    "tag": f"{args.tag}_{v['label']}", **r},
                                   ensure_ascii=False) + "\n")
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(f"[exp26] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
