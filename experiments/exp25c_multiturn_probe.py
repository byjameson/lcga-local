"""exp25c: does the 1.5B instruct model use chat history (and does a memory
instruction help)?

Five tiny two-turn scenarios. The first user turn states a fact, the second
asks for it. We score whether the expected keyword appears in the second answer.
This is a small probe (n=5), not a conversation-quality benchmark.

Usage:
  python experiments/exp25c_multiturn_probe.py --tag exp25c
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

CHAT_MODEL = "models/Qwen2.5-1.5B-Instruct"
MAX_NEW = 32

SYSTEMS = {
    "base": ("Sen yardimci bir asistansin. Kullanicinin dilinde, kisa ve dogru "
             "cevap ver."),
    "memory": ("Sen yardimci bir asistansin. Kullanicinin dilinde, kisa ve dogru "
               "cevap ver. Kullanicinin onceki mesajlarindaki bilgileri hatirla "
               "ve soruldugunda dogru cevap ver."),
}

SCENARIOS = [
    ("isim", "Benim adim Ali.", "Benim adim ne?", ["ali"]),
    ("kedi", "Kedimin adi Pamuk.", "Kedimin adi ne?", ["pamuk"]),
    ("renk", "En sevdigim renk mavi.", "En sevdigim renk ne?", ["mavi"]),
    ("hava", "Bugun hava cok guzel.", "Bugun hava nasil?", ["guzel"]),
    ("toplanti", "Yarin sabah toplantim var.", "Yarin ne var?", ["toplanti"]),
]


def norm(s: str) -> str:
    s = s.lower().replace("\u0131", "i").replace("\u0130", "i")
    for a, b in (("\u00e7", "c"), ("\u011f", "g"), ("\u00f6", "o"),
                 ("\u015f", "s"), ("\u00fc", "u")):
        s = s.replace(a, b)
    return " ".join(s.split())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=CHAT_MODEL)
    ap.add_argument("--tag", default="exp25c")
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
    print(f"[exp25c] load {load_s:.2f}s, {len(SCENARIOS)} scenarios x "
          f"{len(SYSTEMS)} prompts", flush=True)

    def gen(msgs: list[dict]) -> str:
        text = tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    variants = []
    for name, system in SYSTEMS.items():
        rows = []
        for sc_name, fact, question, keys in SCENARIOS:
            msgs = [{"role": "system", "content": system},
                    {"role": "user", "content": fact}]
            first = gen(msgs)
            msgs += [{"role": "assistant", "content": first},
                     {"role": "user", "content": question}]
            g0 = time.perf_counter()
            second = gen(msgs)
            gen_s = time.perf_counter() - g0
            hit = any(k in norm(second) for k in keys)
            rows.append({"scenario": sc_name, "fact": fact, "question": question,
                         "first_answer": first, "second_answer": second,
                         "keyword_hit": hit, "gen_s": round(gen_s, 3)})
            print(f"[{name}] {sc_name:8s} hit={hit} -> "
                  f"{second[:60].encode('ascii','backslashreplace').decode()!r}",
                  flush=True)
        n = len(rows)
        summary = {"label": name, "n": n,
                   "keyword_hit_rate": round(sum(r["keyword_hit"] for r in rows) / n, 3),
                   "avg_gen_s": round(sum(r["gen_s"] for r in rows) / n, 3),
                   "model": Path(model_path).name}
        variants.append({"label": name, "system": system, "rows": rows,
                         "summary": summary})
        print("[summary] " + json.dumps(summary, ensure_ascii=True), flush=True)

    peak = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    summary = {
        "exp": "exp25c_multiturn_probe", "tag": f"{args.tag}_summary",
        "model": Path(model_path).name, "n": len(SCENARIOS),
        "max_new_tokens": MAX_NEW, "load_s": round(load_s, 2),
        "peak_vram_mb": peak,
        "variants": {v["label"]: v["summary"] for v in variants},
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = ROOT / "results" / f"{args.tag}_multiturn.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"variants": variants, "summary": summary}, f, indent=2,
                  ensure_ascii=False)
    with open(ROOT / "results" / "results.jsonl", "a", encoding="utf-8") as f:
        for v in variants:
            for r in v["rows"]:
                f.write(json.dumps({"exp": "exp25c_multiturn_probe",
                                    "tag": f"{args.tag}_{v['label']}", **r},
                                   ensure_ascii=False) + "\n")
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(f"[exp25c] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
