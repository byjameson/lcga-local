"""exp37: evaluate the optional non-Qwen dense model through the real Service.

This is an adapter/baseline experiment, not a claim that SmolLM2 is better.
It checks that the same route -> prompt -> pool -> generate path can serve a
model family other than Qwen. The result is compared descriptively with the
stored Qwen chat/chat3b rows for the same prompt set when available.

Usage:
  python experiments/exp37_alt_model_eval.py --set exp26 --tag exp37_60
  python experiments/exp37_alt_model_eval.py --set holdout --tag exp37_holdout
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

BASELINES = {
    "exp26": ROOT / "results" / "exp30_60_chat3b_quality.json",
    "holdout": ROOT / "results" / "exp34_holdout_chat3b_quality.json",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=("exp25", "exp26", "holdout"),
                    default="holdout")
    ap.add_argument("--tag", default="exp37")
    ap.add_argument("--max-new", type=int, default=64)
    args = ap.parse_args()
    if not 1 <= args.max_new <= 128:
        raise SystemExit("--max-new must be 1..128")
    if args.set == "exp26":
        from exp26_quality60 import PROMPTS as prompts
    elif args.set == "holdout":
        from prompts_holdout import PROMPTS as prompts
    else:
        from exp25_chat_eval import PROMPTS as prompts

    svc = Service(budget_gb=5.0, log_requests=False, cache_size=0,
                  enable_chat=True)
    if "alt" not in svc.specs:
        raise SystemExit("alt unit is not available (SmolLM2 model missing)")
    alt = run_variant(svc, "alt", "alt", args.max_new, prompts)
    s_alt = summarize(alt)

    baselines = {}
    base_path = BASELINES.get(args.set)
    if base_path and base_path.exists():
        data = json.loads(base_path.read_text(encoding="utf-8"))
        for v in data["variants"]:
            baselines[v["label"]] = summarize(v)

    summary = {
        "exp": "exp37_alt_model_eval",
        "tag": f"{args.tag}_summary",
        "prompt_set": args.set,
        "n": len(prompts),
        "max_new_tokens": args.max_new,
        "alt": s_alt,
        "baselines": baselines,
        "purpose": "non-Qwen dense adapter smoke/baseline; not a quality claim",
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = ROOT / "results" / f"{args.tag}_alt_quality.json"
    out.write_text(json.dumps({"variant": alt, "summary": summary},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    with open(ROOT / "results" / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)
    print(f"[exp37] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
