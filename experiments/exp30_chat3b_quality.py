"""exp30: compare the 1.5B dense chat unit with the 3B paged quality unit.

Uses the same 24 prompts and simple scoring as exp25, but drives the real
Service: `chat` (dense, 1.5B) and `chat3b` (int8 paged, 3B Instruct). This
checks the product path and measures the latency/VRAM cost of the quality mode.

Usage:
  python experiments/exp30_chat3b_quality.py --tag exp30 --max-new 64
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from exp25_chat_eval import PROMPTS as PROMPTS_24  # noqa: E402
from exp25_chat_eval import score, summarize  # noqa: E402
from lcsa.service import Service  # noqa: E402


def run_variant(svc: Service, label: str, mode: str, max_new: int,
                prompts: list[tuple]) -> dict:
    from transformers import AutoTokenizer
    spec = svc.specs[mode]
    tok = AutoTokenizer.from_pretrained(
        str(ROOT / spec.model_path), trust_remote_code=True)
    rows = []
    load_s = 0.0
    for cat, prompt, keywords, exact, needs_list in prompts:
        out = svc.answer(prompt, mode=mode, max_new_tokens=max_new,
                         merge="best", budget_gb=5.0)
        text = str(out["per_unit"][mode]["text"]).strip()
        gen_s = float(out["per_unit"][mode]["gen_s"])
        load_s = max(load_s, float(out["timings"]["load_s"]))
        n_new = len(tok(text, add_special_tokens=False).input_ids)
        sc = score((cat, prompt, keywords, exact, needs_list), text)
        rows.append({
            "variant": label, "category": cat, "prompt": prompt,
            "text": text, "gen_s": round(gen_s, 3), "n_new": n_new,
            "tok_s": round(n_new / gen_s, 2) if gen_s else None,
            "peak_vram_mb": out.get("peak_vram_mb"),
            "score": out["per_unit"][mode]["score"],
            "unit": mode, **sc,
        })
        safe = text[:70].encode("ascii", "backslashreplace").decode()
        print(f"[{label}] {cat:7s} {prompt[:34]!r} -> {safe!r}", flush=True)
    peaks = [r["peak_vram_mb"] for r in rows if r["peak_vram_mb"] is not None]
    return {
        "label": label,
        "model": Path(spec.model_path).name,
        "unit": mode,
        "load_s": round(load_s, 2),
        "peak_vram_mb": max(peaks) if peaks else None,
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="exp30")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--set", choices=("exp25", "exp26", "holdout"),
                    default="exp25",
                    help="24-prompt exp25, 60-prompt exp26, or 35-prompt holdout")
    args = ap.parse_args()
    if not 1 <= args.max_new <= 128:
        raise SystemExit("--max-new must be 1..128")
    if args.set == "exp26":
        from exp26_quality60 import PROMPTS as prompts
    elif args.set == "holdout":
        from prompts_holdout import PROMPTS as prompts
    else:
        prompts = PROMPTS_24

    svc = Service(budget_gb=5.0, log_requests=False, cache_size=0,
                  enable_chat=True)
    for key in ("chat", "chat3b"):
        if key not in svc.specs:
            raise SystemExit(f"unit not available: {key}")
    variants = [
        run_variant(svc, "chat", "chat", args.max_new, prompts),
        run_variant(svc, "chat3b", "chat3b", args.max_new, prompts),
    ]
    s_chat = summarize(variants[0])
    s_3b = summarize(variants[1])
    summary = {
        "exp": "exp30_chat3b_quality",
        "tag": f"{args.tag}_summary",
        "n": len(prompts),
        "prompt_set": args.set,
        "max_new_tokens": args.max_new,
        "chat": s_chat,
        "chat3b": s_3b,
        "delta_keyword_hit": round((s_3b["keyword_hit_rate"] or 0) -
                                   (s_chat["keyword_hit_rate"] or 0), 3),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = ROOT / "results" / f"{args.tag}_chat3b_quality.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"variants": variants, "summary": summary}, f,
                  indent=2, ensure_ascii=False)
    with open(ROOT / "results" / "results.jsonl", "a", encoding="utf-8") as f:
        for v in variants:
            for r in v["rows"]:
                f.write(json.dumps({"exp": "exp30_chat3b_quality",
                                    "tag": f"{args.tag}_{v['label']}",
                                    **r}, ensure_ascii=False) + "\n")
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print("[summary] " + json.dumps(summary, ensure_ascii=True), flush=True)
    print(f"[exp30] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
