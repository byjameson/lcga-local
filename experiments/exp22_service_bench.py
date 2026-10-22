"""exp22: end-to-end service benchmark (route -> plan -> run -> merge).

Runs a set of real queries through `lcsa.service.Service` (same code path the
HTTP server uses), covering:
  - auto routing (fast / parallel fast+balanced / deep paged 3B)
  - cold vs warm (pool reload from host, paged engine cache)
  - parallel group vs sequential fallback (same units, smaller budget)

Writes one row per request plus a summary row to results/results.jsonl and a
full dump to results/exp22_service.json.

Usage:
  python experiments/exp22_service_bench.py --n_new 12 --tag exp22
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


def ascii_safe(s: str, n: int = 200) -> str:
    return s.encode("ascii", "backslashreplace").decode()[:n].replace("\n", " ")


QUERIES = [
    ("fact_en", "The capital of France is", "auto"),
    ("greeting_tr", "Merhaba, nasilsin?", "auto"),
    ("story_en", "Once upon a time, there was a little girl", "auto"),
    ("code_tr", "Python kodunda bir sayinin tersini yazan fonksiyon", "auto"),
    ("why_en", "Why is the sky blue?", "auto"),
    ("gen_en", "The quick brown fox jumps over the lazy dog", "auto"),
    ("math_tr", "Hesapla: 3 + 7 kactir?", "auto"),
    ("fact_en_warm", "The capital of France is", "auto"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_new", type=int, default=12)
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--tag", default="exp22_service")
    args = ap.parse_args()

    svc = Service(budget_gb=args.budget, verbose=True)
    rows: list[dict] = []

    print(f"[exp22] budget={args.budget}GB n_new={args.n_new} "
          f"queries={len(QUERIES)}", flush=True)

    for slug, query, mode in QUERIES:
        t0 = time.perf_counter()
        try:
            r = svc.answer(query, mode=mode, max_new_tokens=args.n_new,
                           merge="best")
            err = None
        except Exception as e:  # keep the benchmark going
            err = f"{type(e).__name__}: {e}"
            r = None
        wall = time.perf_counter() - t0
        row = {
            "exp": "exp22_service",
            "tag": f"{args.tag}_{slug}",
            "query": query,
            "mode": mode,
            "n_new": args.n_new,
            "wall_s": round(wall, 3),
            "error": err,
        }
        if r is not None:
            row.update({
                "route_units": r["route"]["units"],
                "route_reason": r["route"]["reason"],
                "n_groups": r["n_groups"],
                "parallel": r["parallel"],
                "plan": r["plan"],
                "load_s": r["timings"]["load_s"],
                "gen_s": r["timings"]["gen_s"],
                "peak_vram_mb": r["peak_vram_mb"],
                "free_vram_gb": r["free_vram_gb"],
                "answer": ascii_safe(r["answer"], 300),
                "per_unit": {k: {"gen_s": v["gen_s"], "score": v["score"],
                                 "text": ascii_safe(v["text"], 160)}
                             for k, v in r["per_unit"].items()},
            })
        rows.append(row)
        status = "ERR " + err if err else f"{wall:.2f}s peak={row.get('peak_vram_mb')}MB"
        print(f"[{slug}] {status} route={row.get('route_units')} "
              f"groups={row.get('n_groups')} par={row.get('parallel')}", flush=True)
        print(f"    answer: {row.get('answer', '')}", flush=True)

    # ---- scenario A: parallel vs sequential fallback (same units)
    scen = {}
    for label, budget in (("parallel", 5.0), ("sequential", 2.0)):
        t0 = time.perf_counter()
        r = svc.answer("The capital of Germany is", mode="ensemble",
                       max_new_tokens=args.n_new, merge="best", budget_gb=budget)
        wall = time.perf_counter() - t0
        scen[label] = {
            "budget_gb": budget,
            "wall_s": round(wall, 3),
            "n_groups": r["n_groups"],
            "parallel": r["parallel"],
            "load_s": r["timings"]["load_s"],
            "gen_s": r["timings"]["gen_s"],
            "peak_vram_mb": r["peak_vram_mb"],
            "answer": ascii_safe(r["answer"], 160),
        }
        print(f"[scen:{label}] wall={wall:.2f}s groups={r['n_groups']} "
              f"load={r['timings']['load_s']}s gen={r['timings']['gen_s']}s", flush=True)

    # ---- scenario B: paged engine cold vs cached (deep twice more)
    deep = {}
    for label in ("cached_1", "cached_2"):
        t0 = time.perf_counter()
        r = svc.answer("Python kodunda faktoriyel hesabi yaz", mode="deep",
                       max_new_tokens=8, merge="best")
        deep[label] = {"wall_s": round(time.perf_counter() - t0, 3),
                       "answer": ascii_safe(r["answer"], 120)}
        print(f"[deep:{label}] {deep[label]['wall_s']}s", flush=True)

    # ---- summary (medians over successful requests)
    ok = [r for r in rows if r["error"] is None]
    walls = [r["wall_s"] for r in ok]
    gens = [r["gen_s"] for r in ok]
    peak = max((r["peak_vram_mb"] or 0) for r in ok) if ok else None
    summary = {
        "exp": "exp22_service",
        "tag": f"{args.tag}_summary",
        "n_requests": len(rows),
        "n_ok": len(ok),
        "n_errors": len(rows) - len(ok),
        "median_wall_s": round(statistics.median(walls), 3) if walls else None,
        "max_wall_s": round(max(walls), 3) if walls else None,
        "median_gen_s": round(statistics.median(gens), 3) if gens else None,
        "peak_vram_mb_max": peak,
        "cold_first_s": rows[0]["wall_s"],
        "warm_repeat_s": rows[7]["wall_s"] if len(rows) > 7 else None,
        "parallel_s": scen["parallel"]["wall_s"],
        "sequential_s": scen["sequential"]["wall_s"],
        "speedup_parallel_vs_seq": (round(scen["sequential"]["wall_s"] /
                                          scen["parallel"]["wall_s"], 3)
                                    if scen["parallel"]["wall_s"] else None),
        "deep_cached_1_s": deep["cached_1"]["wall_s"],
        "deep_cached_2_s": deep["cached_2"]["wall_s"],
        "scenarios": scen,
        "stats": svc.stats(),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print("[summary] " + json.dumps(summary, ensure_ascii=True), flush=True)

    out = ROOT / "results" / "exp22_service.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "summary": summary}, f, indent=2, ensure_ascii=False)
    with open(ROOT / "results" / "results.jsonl", "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(f"[exp22] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
