"""exp33: evaluate the cue-based `quality` router on an exp30 result file.

The router only sees the prompt text (not the exp25/exp26 category labels).
Rows are the already-generated `chat` and `chat3b` outputs, so this is an
offline policy evaluation, not a new model benchmark.

Usage:
  python experiments/exp33_quality_route_eval.py \
      --input results/exp34_holdout_chat3b_quality.json --tag exp33_holdout
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from exp31_router_eval import quality  # noqa: E402
from lcsa.service import route  # noqa: E402


def mean(xs: list[float]) -> float:
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--tag", default="exp33")
    args = ap.parse_args()
    data = json.loads((ROOT / args.input).read_text(encoding="utf-8"))
    variants = {v["label"]: v["rows"] for v in data["variants"]}
    chat_rows = variants["chat"]
    b_rows = variants["chat3b"]
    if len(chat_rows) != len(b_rows):
        raise SystemExit("variant row counts differ")

    picked = []
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for c_row, b_row in zip(chat_rows, b_rows):
        prompt = c_row["prompt"]
        keys, reason = route(prompt, "quality", chat=True, chat3b=True)
        key = keys[0] if keys and keys[0] in ("chat", "chat3b") else "chat"
        row = b_row if key == "chat3b" else c_row
        rec = {
            "category": c_row["category"], "prompt": prompt, "unit": key,
            "reason": reason, "quality": quality(row),
            "gen_s": row["gen_s"], "text": row["text"],
            "chat_score": c_row.get("score"),
        }
        picked.append(rec)
        by_cat[rec["category"]].append(rec)

    always_chat_q = mean([quality(r) for r in chat_rows])
    always_3b_q = mean([quality(r) for r in b_rows])
    oracle_q = mean([max(quality(c), quality(b))
                     for c, b in zip(chat_rows, b_rows)])
    summary = {
        "exp": "exp33_quality_route_eval",
        "tag": f"{args.tag}_summary",
        "input": args.input,
        "n": len(picked),
        "prompt_set": data.get("summary", {}).get("prompt_set", "unknown"),
        "quality": mean([r["quality"] for r in picked]),
        "always_chat_quality": always_chat_q,
        "always_chat3b_quality": always_3b_q,
        "oracle_quality": oracle_q,
        "avg_gen_s": mean([r["gen_s"] for r in picked]),
        "always_chat_gen_s": mean([r["gen_s"] for r in chat_rows]),
        "always_chat3b_gen_s": mean([r["gen_s"] for r in b_rows]),
        "units": dict(Counter(r["unit"] for r in picked)),
        "by_category": {
            cat: {
                "n": len(rs),
                "quality": mean([r["quality"] for r in rs]),
                "units": dict(Counter(r["unit"] for r in rs)),
                "avg_gen_s": mean([r["gen_s"] for r in rs]),
            }
            for cat, rs in sorted(by_cat.items())
        },
        "note": ("Cue router was designed on the 60-prompt exp26 set; "
                 "this report is only independent when --input is holdout."),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = ROOT / "results" / f"{args.tag}_quality_route.json"
    out.write_text(json.dumps({"rows": picked, "summary": summary},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    with open(ROOT / "results" / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)
    print(f"[exp33] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
