"""exp31: offline router evaluation from exp30's chat/chat3b rows.

This does not load a model. It compares deployable routing policies on the same
prompts:

  always_chat   - 1.5B dense, cheapest quality tier
  always_chat3b - 3B int8 paged, highest quality tier
  cascade       - start with chat; escalate to chat3b when chat's mean
                  continuation logprob is below a threshold
  oracle        - per-prompt best of the two (not deployable; upper bound)

Quality is the same simple pass/fail scoring as exp25/exp26 (exact, then
keyword, then strict list). Latency is generation seconds only; model load and
tier-switch cost are reported by exp30 and are not hidden here.

Thresholds are selected on a deterministic, category-stratified dev split and
reported on the held-out test split, plus the full-set Pareto curve. n is small;
treat this as a product signal, not a benchmark claim.

Usage:
  python experiments/exp31_router_eval.py --input results/exp30_chat3b_quality.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def quality(row: dict) -> float:
    if row.get("exact_hit") is not None:
        return 1.0 if row["exact_hit"] else 0.0
    if row.get("keyword_hit") is not None:
        return 1.0 if row["keyword_hit"] else 0.0
    if row.get("list_hit") is not None:
        return 1.0 if row["list_hit"] else 0.0
    return 0.0


def metrics(rows: list[dict], policy: str, threshold: float | None = None) -> dict:
    q = 0.0
    lat = 0.0
    esc = 0
    for r in rows:
        c, b = r["chat"], r["chat3b"]
        if policy == "always_chat":
            q += quality(c)
            lat += c["gen_s"]
        elif policy == "always_chat3b":
            q += quality(b)
            lat += b["gen_s"]
        elif policy == "oracle":
            if quality(b) > quality(c):
                q += quality(b)
                lat += b["gen_s"]
            else:
                q += quality(c)
                lat += c["gen_s"]
        elif policy == "cascade":
            take_b = (c.get("score") is not None and threshold is not None and
                      c["score"] < threshold)
            esc += int(take_b)
            if take_b:
                q += quality(b)
                lat += c["gen_s"] + b["gen_s"]
            else:
                q += quality(c)
                lat += c["gen_s"]
        else:
            raise ValueError(policy)
    n = max(1, len(rows))
    return {
        "policy": policy,
        "n": len(rows),
        "threshold": threshold,
        "quality": round(q / n, 4),
        "avg_gen_s": round(lat / n, 3),
        "escalation_rate": round(esc / n, 3) if policy == "cascade" else 0.0,
    }


def threshold_grid(rows: list[dict]) -> list[float]:
    scores = [r["chat"]["score"] for r in rows
              if r["chat"].get("score") is not None]
    if not scores:
        return [0.0]
    lo, hi = min(scores), max(scores)
    if abs(hi - lo) < 1e-9:
        return [lo]
    return [round(lo + (hi - lo) * i / 20, 4) for i in range(21)]


def choose_threshold(rows: list[dict], lam: float) -> tuple[float, dict]:
    best = None
    for t in threshold_grid(rows):
        m = metrics(rows, "cascade", t)
        # Quality is the objective; latency is a soft per-second penalty.
        utility = m["quality"] - lam * m["avg_gen_s"]
        cand = (utility, -m["avg_gen_s"], t, m)
        if best is None or cand[:2] > best[:2]:
            best = cand
    assert best is not None
    return best[2], best[3]


def stratified_split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Alternate prompts within each category; deterministic and balanced."""
    dev, test = [], []
    seen: dict[str, int] = {}
    for r in rows:
        cat = r["chat"]["category"]
        i = seen.get(cat, 0)
        seen[cat] = i + 1
        (dev if i % 2 == 0 else test).append(r)
    return dev, test


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/exp30_chat3b_quality.json")
    ap.add_argument("--tag", default="exp31")
    args = ap.parse_args()
    path = Path(args.input)
    if not path.is_absolute():
        path = ROOT / path
    data = json.loads(path.read_text(encoding="utf-8"))
    variants = {v["label"]: v["rows"] for v in data["variants"]}
    if "chat" not in variants or "chat3b" not in variants:
        raise SystemExit("input needs chat and chat3b variants")
    rows = [{"chat": c, "chat3b": b}
            for c, b in zip(variants["chat"], variants["chat3b"])]
    if any(r["chat"]["prompt"] != r["chat3b"]["prompt"] for r in rows):
        raise SystemExit("variant prompt order differs")

    always = [metrics(rows, "always_chat"),
              metrics(rows, "always_chat3b"),
              metrics(rows, "oracle")]
    dev, test = stratified_split(rows)
    lams = (0.02, 0.05, 0.10, 0.20, 0.50)
    cv = []
    for lam in lams:
        t, dev_m = choose_threshold(dev, lam)
        test_m = metrics(test, "cascade", t)
        cv.append({"lambda_per_s": lam, "threshold": t,
                   "dev": dev_m, "test": test_m})

    # Full-set Pareto curve (descriptive, not used for a held-out claim).
    curve = [metrics(rows, "cascade", t) for t in threshold_grid(rows)]
    curve.sort(key=lambda m: m["avg_gen_s"])
    summary = {
        "exp": "exp31_router_eval",
        "tag": f"{args.tag}_summary",
        "input": str(path.relative_to(ROOT)),
        "n": len(rows),
        "quality_definition": "exact if defined, else keyword, else strict list",
        "latency_definition": "generation seconds only; cascade adds both calls",
        "baselines": always,
        "dev_n": len(dev),
        "test_n": len(test),
        "cv": cv,
        "pareto_full": curve,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "limit": "n is small; threshold selection is CV on a small synthetic set",
    }
    out = ROOT / "results" / f"{args.tag}_router_eval.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    with open(ROOT / "results" / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(json.dumps({"baselines": always, "cv": cv}, ensure_ascii=True,
                     indent=2), flush=True)
    print(f"[exp31] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
