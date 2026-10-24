"""Long-context step sweep for the disk-streamed QLoRA path (LONGCTX_PLAN E1).

Why not `--steps 1`: the first step includes a cold read of the whole quantized base
(19.2 GB for 32B at ~300 MB/s), which dwarfs the per-step time. Each point here runs
`--steps 3` and takes the median of the JSON summary, i.e. warm-cache steps.

Usage:
  python scripts/longctx_sweep.py --models 32B,14B --seqs 512,1024,2048,3072
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = {
    "32B": "models/Qwen2.5-32B-Instruct-bnb-4bit",
    "14B": "models/Qwen2.5-14B-Instruct",
    "7B": "models/Qwen2.5-7B-Instruct-bnb-4bit",
}


def run_point(model_dir, seq, steps):
    cmd = [sys.executable, os.path.join(ROOT, "experiments", "exp54_disk_qlora.py"),
           "--model-dir", model_dir, "--steps", str(steps), "--seq", str(seq),
           "--eval_chunks", "0", "--shared-dtype", "as-stored"]
    t0 = time.time()
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                         encoding="utf-8", errors="replace", timeout=3600)
    txt = out.stdout + out.stderr
    payload = None
    for line in txt.splitlines():
        line = line.strip()
        if line.startswith("{") and '"exp": "exp54_disk_qlora"' in line:
            try:
                payload = json.loads(line)
            except Exception:
                pass
    if payload is None:
        tail = "\n".join(txt.splitlines()[-6:])
        return {"error": "no json", "tail": tail, "wall_s": time.time() - t0}
    payload["wall_s"] = time.time() - t0
    payload["tok_per_s"] = seq / payload["step_s_med"] if payload.get("step_s_med") else None
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="32B,14B")
    ap.add_argument("--seqs", default="512,1024,2048,3072")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--out", default="results/longctx_sweep.json")
    args = ap.parse_args()

    rows = []
    for m in [x.strip() for x in args.models.split(",") if x.strip()]:
        model_dir = MODELS[m]
        for seq in [int(x) for x in args.seqs.split(",")]:
            print(f"[run] {m} seq={seq} steps={args.steps}", flush=True)
            r = run_point(model_dir, seq, args.steps)
            r.update({"model": m, "seq": seq})
            rows.append(r)
            if "error" in r:
                print(f"  FAILED: {r['tail']}", flush=True)
            else:
                print(f"  step={r['step_s_med']:.2f}s peak={r['peak_gb_max']:.2f}GB "
                      f"tok/s={r['tok_per_s']:.1f} rss={r['host_rss_gb']:.1f}GB "
                      f"blocks={r['blocks_on_gpu']}", flush=True)
            with open(os.path.join(ROOT, args.out), "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=1)

    print("\n| model | seq | step s | tok/s | peak GB | RSS GB |")
    print("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        if "error" in r:
            print(f"| {r['model']} | {r['seq']} | FAIL | - | - | - |")
        else:
            print(f"| {r['model']} | {r['seq']} | {r['step_s_med']:.2f} | "
                  f"{r['tok_per_s']:.1f} | {r['peak_gb_max']:.2f} | {r['host_rss_gb']:.1f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
