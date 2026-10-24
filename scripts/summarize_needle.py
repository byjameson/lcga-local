"""Summarise every needle evaluation in results/ as one readable table.

Scans results/exp60_*.json (exp60_needle) and results/exp62_*.json (exp62_needle_gen)
and prints:
  1. a per-run matrix (CE / exact, plus hit rate for exp62),
  2. the "out-of-window" cells only -- the ones where a budget-matched static window
     cannot answer and content selection therefore matters.

Usage:
  python scripts/summarize_needle.py                 # all runs
  python scripts/summarize_needle.py --tag top1      # filter by tag substring
  python scripts/summarize_needle.py --markdown      # emit a markdown table
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def fmt(cell, with_hit=False):
    if cell is None:
        return "-"
    s = f"{cell['ce']:.3f}/{cell['exact']:.2f}"
    if with_hit and cell.get("hit") is not None:
        s += f"/h{cell['hit']:.2f}"
    return s


def load(tag_filter):
    runs = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "exp60_*.json")) +
                       glob.glob(os.path.join(RESULTS, "exp62_*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        tag = d.get("tag", "")
        if tag_filter and tag_filter not in tag:
            continue
        runs.append((os.path.basename(path), d))
    return runs


def policy_order(names):
    pref = ["dense", "topk", "window", "random", "local", "oracle"]
    return [p for p in pref if p in names] + [p for p in sorted(names) if p not in pref]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--out-of-window", action="store_true",
                    help="only cells where the window fails (exact == 0)")
    args = ap.parse_args()

    for fname, d in load(args.tag):
        res = d.get("results", {})
        if not res:
            continue
        pols = policy_order({p for cell in res.values() for p in cell})
        gen = d.get("exp") == "exp62_needle_gen"
        with_hit = gen
        print(f"\n=== {fname}  ({d.get('ckpt', '')}) ===")
        head = "cell".ljust(22) + "".join(p.rjust(18) for p in pols)
        print(head)
        for key in sorted(res, key=lambda k: (str(k).split("_")[0], str(k))):
            cell = res[key]
            if args.out_of_window and "window" in cell and cell["window"]["exact"] > 0:
                continue
            line = key.ljust(22) + "".join(fmt(cell.get(p), with_hit).rjust(18)
                                           for p in pols)
            print(line)

    if args.markdown:
        # one compact markdown table across all out-of-window cells
        print("\n| run | cell | " + " | ".join(["dense", "learned", "window"]) + " |")
        print("|---|---|---:|---:|---:|")
        for fname, d in load(args.tag):
            res = d.get("results", {})
            for key in sorted(res):
                cell = res[key]
                if "window" not in cell or cell["window"]["exact"] > 0:
                    continue
                if "topk" not in cell:
                    continue
                print(f"| {d.get('tag')} | {key} | {fmt(cell.get('dense'))} | "
                      f"{fmt(cell.get('topk'))} | {fmt(cell.get('window'))} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
