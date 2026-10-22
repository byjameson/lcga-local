"""exp25: small chat/instruction quality probe, base 1.5B vs instruct 1.5B.

The service has two 1.5B units: the base model (completion style) and, when the
model is downloaded, Qwen2.5-1.5B-Instruct. This probe measures what actually
changes, on 24 short prompts (6 Turkish instructions, 6 Turkish facts,
6 English instructions, 6 English facts).

Scores are deliberately simple and honest:
  - keyword hit: at least one expected keyword appears in the output
  - exact: normalized output equals the expected answer (only where defined)
  - list: at least 3 bullet/numbered lines (only for "list three ..." prompts)

This is not a human quality evaluation and the set is small; treat it as a
directional product check, not a benchmark claim.

Usage:
  python experiments/exp25_chat_eval.py
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import unicodedata
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE_MODEL = "models/Qwen2.5-1.5B"
CHAT_MODEL = "models/Qwen2.5-1.5B-Instruct"
MAX_NEW = 48

# (category, prompt, keywords, exact_answer or None, needs_list)
PROMPTS = [
    ("ins_tr", "Sadece 'mavi' yaz.", ["mavi"], "mavi", False),
    ("ins_tr", "2 + 2 kac? Sadece sayiyi yaz.", ["4"], "4", False),
    ("ins_tr", "Uc maddede su icmenin faydalarini yaz.", [], None, True),
    ("ins_tr", "Su cumleyi Turkceye cevir: The cat is on the table.",
     ["kedi", "masa"], None, False),
    ("ins_tr", "Istanbul mu daha kalabalik, Ankara mi? Kisa cevap ver.",
     ["istanbul"], None, False),
    ("ins_tr", "Merhaba, nasilsin?", ["merhaba", "iyi"], None, False),
    ("fact_tr", "Fransa'nin baskenti neresi?", ["paris"], None, False),
    ("fact_tr", "Turkiye'nin baskenti neresi?", ["ankara"], None, False),
    ("fact_tr", "Gunes sistemindeki en buyuk gezegen hangisi?",
     ["jupiter", "jüpiter"], None, False),
    ("fact_tr", "Altinin kimyasal sembolu nedir?", ["au"], None, False),
    ("fact_tr", "Su kac derecede kaynar?", ["100"], None, False),
    ("fact_tr", "Bir yilda kac ay vardir?", ["12"], None, False),
    ("fact_en", "What is the capital of France?", ["paris"], None, False),
    ("fact_en", "What is the capital of Germany?", ["berlin"], None, False),
    ("fact_en", "What is the largest planet in the solar system?",
     ["jupiter"], None, False),
    ("fact_en", "What is the chemical symbol for gold?", ["au"], None, False),
    ("fact_en", "At what temperature does water boil (Celsius)?",
     ["100"], None, False),
    ("fact_en", "How many months are there in a year?", ["12"], None, False),
    ("ins_en", "Reply with exactly the word 'blue'.", ["blue"], "blue", False),
    ("ins_en", "What is 7 * 8? Answer with just the number.", ["56"], "56", False),
    ("ins_en", "List three primary colors.", [], None, True),
    ("ins_en", "Translate 'good morning' into Turkish.",
     ["gunaydin", "günaydın"], None, False),
    ("ins_en", "Write one short sentence about the sea.", ["sea", "ocean"], None, False),
    ("ins_en", "Say hello in Turkish.", ["merhaba"], None, False),
]


def norm(s: str) -> str:
    """ASCII-fold Turkish text before keyword/exact matching.

    The checked-in prompts use ASCII Turkish, while the model often emits
    diacritics ("Teşekkür", "İstanbul", "ağzı"). Without folding, correct
    answers were scored as misses. NFKD + combining-mark removal handles the
    dotted/dotless i and the remaining Turkish letters explicitly.
    """
    s = s.translate(str.maketrans(
        "\u00e7\u011f\u0131\u00f6\u015f\u00fc\u00c7\u011e\u0130\u00d6\u015e\u00dc",
        "cgiosuCGIOSU"))
    s = s.casefold()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s)
    return " ".join(s.split())


def list_hit(text: str) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    n = 0
    for ln in lines:
        if ln[:2] in ("- ", "* ") or (ln[:1].isdigit() and "." in ln[:3]):
            n += 1
    return n >= 3


def score(prompt_row, text: str) -> dict:
    _cat, _prompt, keywords, exact, needs_list = prompt_row
    t = norm(text)
    hit = any(norm(k) in t for k in keywords) if keywords else None
    return {
        "keyword_hit": hit,
        "exact_hit": (t == norm(exact)) if exact else None,
        "list_hit": list_hit(text) if needs_list else None,
    }


def run_variant(label: str, model_path: str, chat: bool) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True).to("cuda")
    model.eval()
    load_s = time.perf_counter() - t0
    sys_msg = ("You are a helpful assistant. Answer concisely in the user's "
               "language.")
    rows = []
    for cat, prompt, keywords, exact, needs_list in PROMPTS:
        if chat:
            text = tok.apply_chat_template(
                [{"role": "system", "content": sys_msg},
                 {"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        else:
            text = prompt
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
            "variant": label, "category": cat, "prompt": prompt,
            "text": dec.strip(), "gen_s": round(gen_s, 3),
            "n_new": n_new, "tok_s": round(n_new / gen_s, 2) if gen_s else None,
            **sc,
        })
        safe = dec.strip()[:70].encode("ascii", "backslashreplace").decode()
        print(f"[{label}] {cat:7s} {prompt[:34]!r} -> {safe!r}", flush=True)
    peak = (round(torch.cuda.max_memory_allocated() / 2**20, 1)
            if torch.cuda.is_available() else None)
    for r in rows:
        r["peak_vram_mb"] = peak
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"label": label, "model": Path(model_path).name, "load_s": round(load_s, 2),
            "peak_vram_mb": peak, "rows": rows}


def summarize(v: dict) -> dict:
    rows = v["rows"]
    kw = [r["keyword_hit"] for r in rows if r["keyword_hit"] is not None]
    ex = [r["exact_hit"] for r in rows if r["exact_hit"] is not None]
    li = [r["list_hit"] for r in rows if r["list_hit"] is not None]
    by_cat = {}
    for cat in sorted({r["category"] for r in rows}):
        cr = [r for r in rows if r["category"] == cat]
        ckw = [r["keyword_hit"] for r in cr if r["keyword_hit"] is not None]
        cex = [r["exact_hit"] for r in cr if r["exact_hit"] is not None]
        by_cat[cat] = {
            "n": len(cr),
            "keyword_hit_rate": round(sum(ckw) / len(ckw), 3) if ckw else None,
            "exact_hit_rate": round(sum(cex) / len(cex), 3) if cex else None,
            "avg_tok_s": round(sum(r["tok_s"] for r in cr) / len(cr), 2),
        }
    return {
        "variant": v["label"], "model": v["model"],
        "n": len(rows),
        "keyword_hit_rate": round(sum(kw) / len(kw), 3) if kw else None,
        "exact_hit_rate": round(sum(ex) / len(ex), 3) if ex else None,
        "list_hit_rate": round(sum(li) / len(li), 3) if li else None,
        "avg_gen_s": round(sum(r["gen_s"] for r in rows) / len(rows), 3),
        "avg_tok_s": round(sum(r["tok_s"] for r in rows) / len(rows), 2),
        "load_s": v["load_s"], "peak_vram_mb": v["peak_vram_mb"],
        "by_category": by_cat,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--chat", default=CHAT_MODEL)
    ap.add_argument("--tag", default="exp25")
    args = ap.parse_args()
    base_path = str(ROOT / args.base) if not Path(args.base).is_absolute() else args.base
    chat_path = str(ROOT / args.chat) if not Path(args.chat).is_absolute() else args.chat
    if not (Path(chat_path) / "config.json").exists():
        raise SystemExit(f"chat model not found: {chat_path}")

    print(f"[exp25] {len(PROMPTS)} prompts, max_new={MAX_NEW}, "
          f"base={Path(base_path).name}, chat={Path(chat_path).name}", flush=True)
    base = run_variant("base", base_path, chat=False)
    chat = run_variant("chat", chat_path, chat=True)
    s_base = summarize(base)
    s_chat = summarize(chat)
    summary = {
        "exp": "exp25_chat_eval", "tag": f"{args.tag}_summary",
        "n": len(PROMPTS), "max_new_tokens": MAX_NEW,
        "base": s_base, "chat": s_chat,
        "delta_keyword_hit": round((s_chat["keyword_hit_rate"] or 0) -
                                   (s_base["keyword_hit_rate"] or 0), 3),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print("[summary] " + json.dumps(summary, ensure_ascii=True), flush=True)

    out = ROOT / "results" / f"{args.tag}_chat_eval.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"variants": [base, chat], "summary": summary}, f,
                  indent=2, ensure_ascii=False)
    with open(ROOT / "results" / "results.jsonl", "a", encoding="utf-8") as f:
        for v in (base, chat):
            for r in v["rows"]:
                f.write(json.dumps({"exp": "exp25_chat_eval",
                                    "tag": f"{args.tag}_{v['label']}",
                                    **r}, ensure_ascii=False) + "\n")
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(f"[exp25] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
