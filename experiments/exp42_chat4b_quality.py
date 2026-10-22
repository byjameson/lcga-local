"""exp42: compare Qwen2.5-3B-Instruct with Qwen3-4B-Instruct-2507.

Both are served by the same int8 paged engine and the same Service path. The
script uses the existing exp25/exp26 scoring so the numbers are directly
comparable with exp30/exp40, including the list-format metric.

Usage:
  python experiments/exp42_chat4b_quality.py --set exp26 --tag exp42_60
  python experiments/exp42_chat4b_quality.py --set holdout --tag exp42_holdout
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

from exp25_chat_eval import summarize  # noqa: E402
from exp30_chat3b_quality import run_variant  # noqa: E402
from lcsa.service import Service  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="exp42")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--set", choices=("exp26", "holdout"), default="exp26")
    ap.add_argument("--units", default="chat3b,chat4b")
    args = ap.parse_args()
    if not 1 <= args.max_new <= 128:
        raise SystemExit("--max-new must be 1..128")
    if args.set == "exp26":
        from exp26_quality60 import PROMPTS as prompts
    else:
        from prompts_holdout import PROMPTS as prompts

    keys = [k.strip() for k in args.units.split(",") if k.strip()]
    svc = Service(budget_gb=5.0, log_requests=False, cache_size=0,
                  enable_chat=True)
    for key in keys:
        if key not in svc.specs:
            raise SystemExit(f"unit not available: {key}")

    variants = []
    for key in keys:
        variants.append(run_variant(svc, key, key, args.max_new, prompts))

    summary = {
        "exp": "exp42_chat4b_quality",
        "tag": f"{args.tag}_summary",
        "n": len(prompts),
        "prompt_set": args.set,
        "max_new_tokens": args.max_new,
        "variants": {v["label"]: summarize(v) for v in variants},
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = ROOT / "results" / f"{args.tag}_chat4b_quality.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"variants": variants, "summary": summary}, f,
                  indent=2, ensure_ascii=False)
    with open(ROOT / "results" / "results.jsonl", "a", encoding="utf-8") as f:
        for v in variants:
            for r in v["rows"]:
                f.write(json.dumps({"exp": "exp42_chat4b_quality",
                                    "tag": f"{args.tag}_{v['label']}",
                                    **r}, ensure_ascii=False) + "\n")
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print("[summary] " + json.dumps(summary, ensure_ascii=True), flush=True)
    print(f"[exp42] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
