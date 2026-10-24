"""exp22c: fair parallel-vs-sequential comparison for the service.

The first service run compared a *cold* parallel request (both dense models
reloaded from host, +1.31s) with a *warm* sequential request (0s load), so the
wall-clock ratio conflated reload time with grouping. This script measures both
scenarios from the same warm state (both models resident), twice each, and
reports generation-only and wall times separately.

Also records the budget warning when a unit is larger than the declared budget.

Usage:
  python experiments/exp22_scen_fair.py --n_new 12
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lcsa.service import Service  # noqa: E402

QUERY = "The capital of Germany is"


def ascii_safe(s: str, n: int = 160) -> str:
    return s.encode("ascii", "backslashreplace").decode()[:n].replace("\n", " ")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_new", type=int, default=12)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--tag", default="exp22c")
    args = ap.parse_args()

    svc = Service(budget_gb=5.0, verbose=True)
    rows: list[dict] = []

    def warm() -> None:
        # both dense units resident before each measured scenario
        svc.answer("Warm up the model pool.", mode="ensemble",
                   max_new_tokens=4, merge="best", budget_gb=5.0)

    def run(label: str, budget: float, rep: int) -> dict:
        warm()
        t0 = time.perf_counter()
        r = svc.answer(QUERY, mode="ensemble", max_new_tokens=args.n_new,
                       merge="best", budget_gb=budget)
        wall = time.perf_counter() - t0
        row = {
            "exp": "exp22_service",
            "tag": f"{args.tag}_{label}_r{rep}",
            "query": QUERY,
            "mode": "ensemble",
            "n_new": args.n_new,
            "rep": rep,
            "budget_gb": budget,
            "n_groups": r["n_groups"],
            "parallel": r["parallel"],
            "wall_s": round(wall, 3),
            "load_s": r["timings"]["load_s"],
            "gen_s": r["timings"]["gen_s"],
            "peak_vram_mb": r["peak_vram_mb"],
            "warnings": r["warnings"],
            "plan": r["plan"],
            "per_unit": {k: v["gen_s"] for k, v in r["per_unit"].items()},
            "answer": ascii_safe(r["answer"], 200),
        }
        print(f"[{row['tag']}] wall={row['wall_s']}s load={row['load_s']}s "
              f"gen={row['gen_s']}s groups={row['n_groups']} "
              f"peak={row['peak_vram_mb']}MB warnings={len(row['warnings'])}",
              flush=True)
        return row

    for rep in range(1, args.reps + 1):
        rows.append(run("parallel", 5.0, rep))
        rows.append(run("sequential", 2.0, rep))

    par = [r for r in rows if r["tag"].startswith(f"{args.tag}_parallel")]
    seq = [r for r in rows if r["tag"].startswith(f"{args.tag}_sequential")]
    med = lambda xs: round(statistics.median(xs), 3) if xs else None  # noqa: E731
    summary = {
        "exp": "exp22_service",
        "tag": f"{args.tag}_summary",
        "n_new": args.n_new,
        "reps": args.reps,
        "parallel_wall_s": med([r["wall_s"] for r in par]),
        "sequential_wall_s": med([r["wall_s"] for r in seq]),
        "parallel_gen_s": med([r["gen_s"] for r in par]),
        "sequential_gen_s": med([r["gen_s"] for r in seq]),
        "wall_speedup": (round(med([r["wall_s"] for r in seq]) /
                               med([r["wall_s"] for r in par]), 3)),
        "gen_speedup": (round(med([r["gen_s"] for r in seq]) /
                              med([r["gen_s"] for r in par]), 3)),
        "parallel_peak_mb": med([r["peak_vram_mb"] for r in par]),
        "sequential_peak_mb": med([r["peak_vram_mb"] for r in seq]),
        "sequential_warnings": seq[0]["warnings"] if seq else [],
        "stats": svc.stats(),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print("[summary] " + json.dumps(summary, ensure_ascii=True), flush=True)

    out = ROOT / "results" / "exp22_scen_fair.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "summary": summary}, f, indent=2,
                  ensure_ascii=False)
    with open(ROOT / "results" / "results.jsonl", "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(f"[{args.tag}] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
