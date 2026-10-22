"""exp43: end-to-end `quality` measurement with Qwen3-4B as the main unit.

Unlike exp42 (raw unit output), this drives the public product path: the router
picks the unit, lists get the deterministic formatter, and the measured answer
is exactly what a user sees. The same exp25/exp26 scoring is used.

Usage:
  python experiments/exp43_quality4b_e2e.py --set exp26 --tag exp43_60
  python experiments/exp43_quality4b_e2e.py --set holdout --tag exp43_holdout
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

from exp25_chat_eval import score, summarize  # noqa: E402
from lcsa.service import Service  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="exp43")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--set", choices=("exp26", "holdout"), default="exp26")
    args = ap.parse_args()
    if not 1 <= args.max_new <= 128:
        raise SystemExit("--max-new must be 1..128")
    if args.set == "exp26":
        from exp26_quality60 import PROMPTS as prompts
    else:
        from prompts_holdout import PROMPTS as prompts

    svc = Service(budget_gb=5.0, log_requests=False, cache_size=0,
                  enable_chat=True)
    if "chat4b" not in svc.specs:
        raise SystemExit("chat4b unit is not available")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        str(ROOT / svc.specs["chat4b"].model_path), trust_remote_code=True)

    rows = []
    peak = 0
    load_s = 0.0
    routes = {"chat4b": 0, "chat": 0, "chat3b": 0, "deep": 0}
    for cat, prompt, keywords, exact, needs_list in prompts:
        out = svc.answer(prompt, mode="quality", max_new_tokens=args.max_new,
                         merge="best", budget_gb=5.0)
        unit = out["route"]["units"][0]
        routes[unit] = routes.get(unit, 0) + 1
        text = str(out["answer"]).strip()
        gen_s = float(out["timings"]["gen_s"])
        peak = max(peak, int(out.get("peak_vram_mb") or 0))
        load_s = max(load_s, float(out["timings"].get("load_s") or 0))
        n_new = len(tok(text, add_special_tokens=False).input_ids)
        sc = score((cat, prompt, keywords, exact, needs_list), text)
        rows.append({
            "variant": "quality4b", "category": cat, "prompt": prompt,
            "text": text, "unit": unit, "gen_s": round(gen_s, 3),
            "n_new": n_new,
            "tok_s": round(n_new / gen_s, 2) if gen_s else None,
            "peak_vram_mb": out.get("peak_vram_mb"),
            "elapsed_s": out.get("elapsed_s"),
            "route_reason": out["route"]["reason"],
            **sc,
        })
        safe = text[:70].encode("ascii", "backslashreplace").decode()
        print(f"[quality4b] {unit:7s} {cat:7s} {prompt[:34]!r} -> {safe!r}",
              flush=True)

    variant = {
        "label": "quality4b",
        "model": "Qwen3-4B-Instruct-2507 (+1.5B for social)",
        "load_s": round(load_s, 2),
        "peak_vram_mb": peak or None,
        "rows": rows,
    }
    summary = {
        "exp": "exp43_quality4b_e2e",
        "tag": f"{args.tag}_summary",
        "n": len(prompts),
        "prompt_set": args.set,
        "max_new_tokens": args.max_new,
        "quality4b": summarize(variant),
        "routes": routes,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = ROOT / "results" / f"{args.tag}_quality4b_e2e.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"variant": variant, "summary": summary}, f,
                  indent=2, ensure_ascii=False)
    with open(ROOT / "results" / "results.jsonl", "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"exp": "exp43_quality4b_e2e",
                                "tag": args.tag, **r},
                               ensure_ascii=False) + "\n")
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print("[summary] " + json.dumps(summary, ensure_ascii=True), flush=True)
    print(f"[exp43] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
