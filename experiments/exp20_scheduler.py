"""exp20: M4 - multi-unit scheduler demo.

Units: Qwen2.5-0.5B and Qwen2.5-1.5B with declared VRAM estimates.
Reference: each unit run alone (load/gen/unload), summed.

Scenarios:
  par: budget from --budget_par (default 5.0GB) -> both fit -> planned together,
       units generate in parallel on separate CUDA streams.
  seq: budget from --budget_seq (default 2.5GB) -> planner splits them into
       separate groups -> load/generate/unload sequentially.

Metrics: load/gen/unload/total wall times, speedup vs reference, peak VRAM,
output match vs reference, plan decisions, merged output.
Usage:
  python experiments/exp20_scheduler.py --tag m4_scheduler
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

from lcsa.scheduler import Unit, free_vram_gb, merge_outputs, plan, run_group  # noqa: E402


def vram() -> dict:
    if not torch.cuda.is_available():
        return {}
    return {
        "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m05", default="models/Qwen2.5-0.5B")
    ap.add_argument("--m15", default="models/Qwen2.5-1.5B")
    ap.add_argument("--est05", type=float, default=1.1, help="declared GB for 0.5B unit")
    ap.add_argument("--est15", type=float, default=3.3, help="declared GB for 1.5B unit")
    ap.add_argument("--prompt05", default="The capital of France is")
    ap.add_argument("--prompt15", default="Once upon a time, there was a")
    ap.add_argument("--prompt_both", default="",
                    help="if set, both units receive this prompt (merge demo)")
    ap.add_argument("--n_new", type=int, default=24)
    ap.add_argument("--budget_par", type=float, default=5.0)
    ap.add_argument("--budget_seq", type=float, default=2.5)
    ap.add_argument("--merge", choices=["concat", "best", "both"], default="concat")
    ap.add_argument("--skip_ref", action="store_true")
    ap.add_argument("--tag", default="m4_scheduler")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for this demo")

    def make_units() -> list[Unit]:
        p05 = args.prompt_both or args.prompt05
        p15 = args.prompt_both or args.prompt15
        return [
            Unit("qwen0.5b", str(ROOT / args.m05), args.est05, p05, args.n_new),
            Unit("qwen1.5b", str(ROOT / args.m15), args.est15, p15, args.n_new),
        ]

    print(f"[exp20] free_vram={free_vram_gb():.2f}GB units=0.5B(est {args.est05}) "
          f"+1.5B(est {args.est15}) n_new={args.n_new}", flush=True)

    rows = []

    # ---- reference: each unit alone
    ref = {}
    if not args.skip_ref:
        for u in make_units():
            torch.cuda.empty_cache()
            r = run_group([u])
            ref[u.name] = {"gen_s": round(u.gen_s, 2), "text": u.text,
                           "load_s": round(r["load_s"], 2),
                           "unload_s": round(r["unload_s"], 2)}
            print(f"[ref:{u.name}] gen={u.gen_s:.2f}s text={u.text!r}", flush=True)
    ref_gen = sum(v["gen_s"] for v in ref.values())
    ref_total = sum(v["gen_s"] + v["load_s"] + v["unload_s"] for v in ref.values())

    # ---- scenarios
    for scen, budget in (("par", args.budget_par), ("seq", args.budget_seq)):
        units = make_units()
        groups = plan(units, budget)
        group_names = [[u.name for u in g] for g in groups]
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        free0 = free_vram_gb()
        t0 = time.perf_counter()
        agg = {"load_s": 0.0, "gen_s": 0.0, "unload_s": 0.0}
        for g in groups:
            r = run_group(g)
            for k in agg:
                agg[k] += r[k]
        total_s = time.perf_counter() - t0
        v = vram()
        texts = {u.name: u.text for u in units}
        scores = {u.name: (round(u.score, 4) if u.score is not None else None)
                  for u in units}
        match = None
        if ref:
            match = all(ref[u.name]["text"] == u.text for u in units)
        merged_concat = merge_outputs(units, "concat")
        merged_best = merge_outputs(units, "best")
        merged = merged_concat if args.merge == "concat" else merged_best
        best_pick = None
        scored = [u for u in units if u.score is not None]
        if scored:
            best_pick = max(scored, key=lambda u: u.score).name

        row = {
            "exp": "exp20_scheduler",
            "tag": f"{args.tag}_{scen}",
            "scenario": scen,
            "free_before_gb": round(free0, 2),
            "budget_gb": budget,
            "est_gb": {u.name: u.est_gb for u in units},
            "groups": group_names,
            "n_groups": len(groups),
            "load_s": round(agg["load_s"], 2),
            "gen_s": round(agg["gen_s"], 2),
            "unload_s": round(agg["unload_s"], 2),
            "total_s": round(total_s, 2),
            "ref_gen_s": round(ref_gen, 2),
            "ref_total_s": round(ref_total, 2),
            "speedup_gen": round(ref_gen / agg["gen_s"], 3) if agg["gen_s"] else None,
            "speedup_total": round(ref_total / total_s, 3) if total_s else None,
            "texts": texts,
            "scores": scores,
            "merge_policy": args.merge,
            "best_pick": best_pick,
            "merged_concat": merged_concat[:400],
            "merged_best": merged_best[:400],
            "outputs_match_ref": match,
            "merged": merged[:400],
            **v,
        }
        rows.append(row)
        print(f"[{scen}] plan={group_names} total={total_s:.2f}s gen={agg['gen_s']:.2f}s "
              f"match_ref={match} best_pick={best_pick}", flush=True)
        for name, txt in texts.items():
            safe = txt.encode("ascii", "backslashreplace").decode()[:160]
            print(f"[{scen}:{name}] {safe}", flush=True)
        if merged_best:
            safe = merged_best.encode("ascii", "backslashreplace").decode()[:200]
            print(f"[{scen}:merged_best] {safe}", flush=True)

    out_json = ROOT / "results" / "exp20_scheduler.json"
    with open(ROOT / "results" / "results.jsonl", "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"ref": ref, "rows": rows}, f, indent=2, ensure_ascii=False)
    print(f"[exp20] wrote {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
