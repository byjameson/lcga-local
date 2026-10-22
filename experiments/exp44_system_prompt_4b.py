"""exp44: does a longer "anti-hallucination" system prompt help Qwen3-4B?

Small ablation, n=6 prompts x 3 system prompts. The product prompt added
"say when you are unsure, do not invent sources" and made the model *claim*
verification ("Verified from scientific sources") on two answers. The current
short prompt stayed concise and made no such claim. This is a negative result:
the longer prompt was not adopted.

Usage:
  python experiments/exp44_system_prompt_4b.py --tag exp44
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

from lcsa.paged_qwen import PagedQwen  # noqa: E402

MODEL = ROOT / "models" / "Qwen3-4B-Instruct-2507"

SYSTEMS = {
    "current": ("Sen yardimci bir asistansin. Kullanicinin dilinde, kisa ve "
                "dogru cevap ver."),
    "none": None,
    "product": ("Sen LCGA adli, kullanicinin cihazinda calisan yardimci bir "
                "yapay zeka asistanisin. Kullanicinin dilinde dogal ve dogru "
                "yanit ver. Emin olmadigin bilgilerde tahmin yuruttugunu "
                "soyle; kaynak uydurma."),
}

PROMPTS = [
    "Translate 'The dog is in the garden.' into Turkish.",
    "Write one short sentence containing the word 'ocean'.",
    "At what temperature does water boil (Celsius)?",
    "Misir'in baskenti neresi?",
    "Dunyanin en buyuk colu hangisi?",
    "Insan kalbinde kac odacik vardir?",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="exp44")
    args = ap.parse_args()
    if not (MODEL / "config.json").exists():
        raise SystemExit(f"model missing: {MODEL}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    eng = PagedQwen(MODEL, device="cuda", dtype=torch.bfloat16, resident_k=None,
                    max_seq=512, pin=False, quant="int8", resident_quant=True,
                    verbose=False)
    rows = []
    for prompt in PROMPTS:
        row = {"prompt": prompt}
        for name, system in SYSTEMS.items():
            msgs = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
            ids = tok(text, return_tensors="pt").input_ids
            out, _ = eng.generate(ids, max_new_tokens=80, temperature=0.0,
                                  eos_id=tok.eos_token_id, seed=1234)
            row[name] = tok.decode(out[0][ids.shape[1]:],
                                   skip_special_tokens=True)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=True), flush=True)
    out_path = ROOT / "results" / f"{args.tag}_system_prompt_4b.json"
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    summary = {
        "exp": "exp44_system_prompt_4b",
        "tag": args.tag,
        "n_prompts": len(PROMPTS),
        "systems": list(SYSTEMS),
        "finding": ("the longer product prompt added false verification claims "
                    "on 2/6 answers; current short prompt kept. Negative result."),
        "result_file": str(out_path.relative_to(ROOT)),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(ROOT / "results" / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print("[summary] " + json.dumps(summary, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
