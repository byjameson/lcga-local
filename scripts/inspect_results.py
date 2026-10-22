"""Inspect results/results.jsonl: exp types, counts, NaN rows, one sample per type."""
import json
import math
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "results", "results.jsonl")


def has_nan(o):
    if isinstance(o, float):
        return math.isnan(o)
    if isinstance(o, dict):
        return any(has_nan(v) for v in o.values())
    if isinstance(o, list):
        return any(has_nan(v) for v in o)
    return False


def main():
    rows = []
    with open(P, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"total rows: {len(rows)}")
    c = Counter(r.get("exp") for r in rows)
    for k, v in sorted(c.items()):
        print(f"  {k:24s} {v}")
    bad = [(i, r.get("exp"), r.get("tag"), r.get("step")) for i, r in enumerate(rows)
           if has_nan(r)]
    print(f"rows containing NaN: {len(bad)}")
    for b in bad[:20]:
        print("   ", b)
    seen = set()
    for r in rows:
        e = r.get("exp")
        if e in seen:
            continue
        seen.add(e)
        s = json.dumps(r, ensure_ascii=False)
        print(f"--- sample [{e}]: {s[:400]}")


if __name__ == "__main__":
    sys.exit(main())
