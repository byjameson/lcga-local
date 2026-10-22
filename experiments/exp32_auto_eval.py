"""exp32: measure the current rule-based `auto` router on the exp25 prompt set.

This is the baseline the confidence cascade has to beat. It records the chosen
units and the end-to-end latency, not just the answer text.

Usage:
  python experiments/exp32_auto_eval.py --tag exp32 --max-new 32
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

from exp25_chat_eval import PROMPTS, score  # noqa: E402
from lcsa.service import Service  # noqa: E402


def rate(values: list[bool | None]) -> float | None:
    x = [v for v in values if v is not None]
    return round(sum(x) / len(x), 3) if x else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="exp32")
    ap.add_argument("--max-new", type=int, default=32)
    args = ap.parse_args()
    svc = Service(budget_gb=5.0, log_requests=False, cache_size=0,
                  enable_chat=True)
    rows = []
    for cat, prompt, keywords, exact, needs_list in PROMPTS:
        t0 = time.perf_counter()
        out = svc.answer(prompt, mode="auto", max_new_tokens=args.max_new)
        wall = time.perf_counter() - t0
        text = str(out["answer"]).strip()
        sc = score((cat, prompt, keywords, exact, needs_list), text)
        rows.append({
            "variant": "auto", "category": cat, "prompt": prompt,
            "text": text, "units": out["route"]["units"],
            "reason": out["route"]["reason"],
            "gen_s": out["timings"]["gen_s"],
            "load_s": out["timings"]["load_s"],
            "elapsed_s": out["elapsed_s"],
            "wall_s": round(wall, 3),
            "peak_vram_mb": out["peak_vram_mb"],
            **sc,
        })
        print(f"[auto] {cat:7s} {prompt[:34]!r} -> "
              f"{text[:60].encode('ascii', 'backslashreplace').decode()!r} "
              f"units={out['route']['units']}", flush=True)
    summary = {
        "exp": "exp32_auto_eval",
        "tag": f"{args.tag}_summary",
        "n": len(rows),
        "max_new_tokens": args.max_new,
        "keyword_hit_rate": rate([r["keyword_hit"] for r in rows]),
        "exact_hit_rate": rate([r["exact_hit"] for r in rows]),
        "list_hit_rate": rate([r["list_hit"] for r in rows]),
        "avg_elapsed_s": round(sum(r["elapsed_s"] for r in rows) / len(rows), 3),
        "avg_gen_s": round(sum(r["gen_s"] for r in rows) / len(rows), 3),
        "peak_vram_mb": max(r["peak_vram_mb"] for r in rows),
        "unit_counts": {u: sum(1 for r in rows if u in r["units"])
                        for u in sorted({u for r in rows for u in r["units"]})},
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_path = ROOT / "results" / f"{args.tag}_auto_eval.json"
    out_path.write_text(json.dumps({"rows": rows, "summary": summary},
                                   indent=2, ensure_ascii=False),
                        encoding="utf-8")
    with open(ROOT / "results" / "results.jsonl", "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"exp": "exp32_auto_eval",
                                "tag": args.tag, **r}, ensure_ascii=False) + "\n")
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print("[summary] " + json.dumps(summary, ensure_ascii=True), flush=True)
    print(f"[exp32] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
