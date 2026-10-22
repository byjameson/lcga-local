"""exp21: M4b - model pool vs load-per-request.

Three scenarios over request scripts with the same two units:
  nopool: load + generate + unload per request (scheduler baseline)
  pool:   ModelPool(5.0GB) keeps models resident across requests
  tight:  ModelPool(2.5GB) with requests [0.5B] -> [1.5B] -> [0.5B]
          to demonstrate LRU eviction and host->GPU reload (no disk)

Metrics: per-request load/gen/wall seconds, resident/host sets, evictions/reloads,
peak VRAM. Honest note: est_gb values are declared, not measured.
Usage:
  python experiments/exp21_pool.py --tag m4_pool
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lcsa.pool import ModelPool  # noqa: E402
from lcsa.scheduler import Unit, run_group  # noqa: E402


def vram() -> dict:
    return {
        "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m05", default="models/Qwen2.5-0.5B")
    ap.add_argument("--m15", default="models/Qwen2.5-1.5B")
    ap.add_argument("--est05", type=float, default=1.1)
    ap.add_argument("--est15", type=float, default=3.3)
    ap.add_argument("--n_new", type=int, default=16)
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--tight", type=float, default=2.5)
    ap.add_argument("--tag", default="m4_pool")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for this demo")

    def unit(name: str, path: str, est: float, prompt: str) -> Unit:
        return Unit(name, str(ROOT / path), est, prompt, args.n_new)

    def dual(p05: str, p15: str) -> list[Unit]:
        return [unit("qwen0.5b", args.m05, args.est05, p05),
                unit("qwen1.5b", args.m15, args.est15, p15)]

    requests = [
        dual("The capital of France is", "Once upon a time, there was a"),
        dual("The capital of Germany is", "In a distant galaxy,"),
        [unit("qwen1.5b", args.m15, args.est15, "The quick brown fox")],
    ]
    tight_requests = [
        [unit("qwen0.5b", args.m05, args.est05, "The capital of Italy is")],
        [unit("qwen1.5b", args.m15, args.est15, "Once upon a time, there was a")],
        [unit("qwen0.5b", args.m05, args.est05, "The capital of Spain is")],
    ]

    print(f"[exp21] free_vram={torch.cuda.mem_get_info()[0]/2**30:.2f}GB "
          f"budget={args.budget}GB tight={args.tight}GB n_new={args.n_new}", flush=True)

    rows: list[dict] = []

    def record(scen: str, i: int, units: list[Unit], r: dict, wall: float,
               extra: dict) -> None:
        row = {
            "exp": "exp21_pool",
            "tag": f"{args.tag}_{scen}",
            "scenario": scen,
            "req": i,
            "units": [u.name for u in units],
            "prompts": [u.prompt for u in units],
            "load_s": round(r["load_s"], 2),
            "gen_s": round(r["gen_s"], 2),
            "unload_s": round(r["unload_s"], 2),
            "wall_s": round(wall, 2),
            "texts": {u.name: u.text for u in units},
            "scores": {u.name: (round(u.score, 4) if u.score is not None else None)
                       for u in units},
            **extra,
            **vram(),
        }
        rows.append(row)
        txt = " | ".join(f"{u.name}:{u.text!r}" for u in units)
        safe = txt.encode("ascii", "backslashreplace").decode()[:200]
        print(f"[{scen} r{i}] load={r['load_s']:.2f}s gen={r['gen_s']:.2f}s "
              f"wall={wall:.2f}s {safe}", flush=True)
        if extra.get("stats"):
            print(f"  stats={extra['stats']}", flush=True)

    # ---- scenario 1: no pool (load/unload per request)
    for i, units in enumerate(requests, 1):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        r = run_group(units)
        record("nopool", i, units, r, time.perf_counter() - t0, {})

    # ---- scenario 2: pool with a 5GB budget
    pool = ModelPool(args.budget, verbose=True)
    for i, units in enumerate(requests, 1):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        r = pool.run(units)
        record("pool", i, units, r, time.perf_counter() - t0,
               {"stats": pool.stats()})
    pool.clear()

    # ---- scenario 3: tight budget -> LRU eviction + host reload
    pool = ModelPool(args.tight, verbose=True)
    for i, units in enumerate(tight_requests, 1):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        r = pool.run(units)
        record("tight", i, units, r, time.perf_counter() - t0,
               {"stats": pool.stats()})
    pool.clear()

    out = ROOT / "results" / "exp21_pool.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"requests": requests, "tight_requests": tight_requests,
                   "rows": rows}, f, indent=2, ensure_ascii=False, default=str)
    with open(ROOT / "results" / "results.jsonl", "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"[exp21] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
