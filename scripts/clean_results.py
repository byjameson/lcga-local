"""Move rows containing non-finite floats (NaN/Inf) from results.jsonl to
results_archive.jsonl. Run only when no experiment is appending to the file.

Usage: python scripts\\clean_results.py [--dry]
"""
import datetime
import json
import math
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "results", "results.jsonl")
ARCHIVE = os.path.join(ROOT, "results", "results_archive.jsonl")


def bad(o):
    if isinstance(o, float):
        return not math.isfinite(o)
    if isinstance(o, dict):
        return any(bad(v) for v in o.values())
    if isinstance(o, list):
        return any(bad(v) for v in o)
    return False


def main():
    dry = "--dry" in sys.argv
    rows = []
    with open(P, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(line)

    keep, drop = [], []
    for line in rows:
        (drop if bad(json.loads(line)) else keep).append(line)

    print(f"total={len(rows)} keep={len(keep)} drop={len(drop)}")
    for line in drop[:10]:
        r = json.loads(line)
        print("  drop:", r.get("exp"), r.get("tag"), r.get("step"))
    if dry or not drop:
        return
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copyfile(P, os.path.join(ROOT, "results", f"results_backup_{ts}.jsonl"))
    with open(ARCHIVE, "a", encoding="utf-8") as f:
        for line in drop:
            f.write(line + "\n")
    tmp = P + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for line in keep:
            f.write(line + "\n")
    os.replace(tmp, P)
    print(f"moved {len(drop)} rows to {ARCHIVE}; backup written")


if __name__ == "__main__":
    sys.exit(main())
