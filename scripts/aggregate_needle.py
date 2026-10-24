"""Aggregate several needle evaluation runs (seeds) into mean (min..max) per cell.

Usage:
  python scripts/aggregate_needle.py --pattern "top1_s*_e6"
  python scripts/aggregate_needle.py --pattern "top1_*_e6" --metric exact
"""
import argparse
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="top1_*_e6")
    ap.add_argument("--metric", default="exact", choices=("exact", "ce"))
    ap.add_argument("--policies", default="dense,topk,window")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(RESULTS, f"exp60_{args.pattern}.json")))
    if not files:
        print(f"no files for pattern {args.pattern}")
        return 1
    pols = [p.strip() for p in args.policies.split(",")]
    cells = {}
    for path in files:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        for cell, row in d["results"].items():
            for p in pols:
                if p in row:
                    cells.setdefault((cell, p), []).append(row[p][args.metric])

    print(f"{len(files)} runs: " + ", ".join(os.path.basename(f) for f in files))
    head = "cell".ljust(14) + "".join(p.rjust(22) for p in pols)
    print(head)
    for cell in sorted(cells_keys(cells)):
        line = cell.ljust(14)
        for p in pols:
            vals = cells.get((cell, p), [])
            if not vals:
                line += "-".rjust(22)
                continue
            mean = sum(vals) / len(vals)
            line += f"{mean:.3f} ({min(vals):.2f}..{max(vals):.2f})".rjust(22)
        print(line)
    return 0


def cells_keys(cells):
    return sorted({c for c, _p in cells},
                  key=lambda k: (str(k).split("_")[0], str(k)))


if __name__ == "__main__":
    raise SystemExit(main())
