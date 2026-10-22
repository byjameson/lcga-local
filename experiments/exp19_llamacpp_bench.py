"""exp19: llama.cpp baseline benchmark (market comparison).

Runs llama-bench on a GGUF model and appends parsed rows to results.jsonl.
Examples:
  python experiments/exp19_llamacpp_bench.py --exe tools/llama.cpp/cpu/llama-bench.exe \
      --gguf models/gguf/Qwen2.5-3B-Q4_K_M.gguf --threads 8 --tag lcpp_cpu_q4
  python experiments/exp19_llamacpp_bench.py --exe tools/llama.cpp/cuda/llama-bench.exe \
      --gguf models/gguf/Qwen2.5-3B-Q4_K_M.gguf --ngl 99 --tag lcpp_cuda_q4
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ROW_RE = re.compile(r"^\|\s*(?P<model>.+?)\s*\|\s*(?P<size>[\d.]+)\s*(?P<unit>GiB|MiB|B)\s*\|"
                    r"\s*(?P<params>[\d.]+)\s*(?P<psize>[BM])\s*\|\s*(?P<backend>[A-Za-z0-9]+)\s*\|"
                    r"\s*(?:(?P<ngl>\d+)\s*\|\s*)?"
                    r"(?P<threads>\d+)\s*\|\s*(?P<test>\S+)\s*\|\s*(?P<ts>[\d.]+)\s*[^\d\s]*\s*(?P<err>[\d.]+)\s*\|")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default="tools/llama.cpp/cpu/llama-bench.exe")
    ap.add_argument("--gguf", default="models/gguf/Qwen2.5-3B-Q4_K_M.gguf")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--pp", type=int, default=64)
    ap.add_argument("--tg", type=int, default=32)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--ngl", type=int, default=None, help="GPU layers (CUDA build only)")
    ap.add_argument("--tag", default="lcpp_cpu_q4")
    args = ap.parse_args()

    exe = ROOT / args.exe
    gguf = ROOT / args.gguf
    if not exe.exists():
        raise SystemExit(f"missing exe: {exe}")
    if not gguf.exists():
        raise SystemExit(f"missing gguf: {gguf}")

    cmd = [str(exe), "-m", str(gguf), "-p", str(args.pp), "-n", str(args.tg),
           "-t", str(args.threads), "-r", str(args.reps)]
    if args.ngl is not None:
        cmd += ["-ngl", str(args.ngl)]
    print("[exp19] " + " ".join(cmd), flush=True)
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=str(exe.parent),
                         encoding="utf-8", errors="replace")
    print(out.stdout)
    if out.returncode != 0:
        print(out.stderr, file=sys.stderr)
        raise SystemExit(f"llama-bench failed rc={out.returncode}")

    rows = []
    for line in out.stdout.splitlines():
        m = ROW_RE.match(line.strip())
        if not m:
            continue
        g = m.groupdict()
        row = {
            "exp": "exp19_llamacpp",
            "tag": args.tag,
            "bench_model": g["model"],
            "gguf": gguf.name,
            "file_gib": float(g["size"]),
            "params_b": float(g["params"]),
            "backend": g["backend"],
            "threads": int(g["threads"]),
            "ngl": int(g["ngl"]) if g.get("ngl") else args.ngl,
            "test": g["test"],
            "t_s": float(g["ts"]),
            "err": float(g["err"]),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    if not rows:
        raise SystemExit("no benchmark rows parsed")
    with open(ROOT / "results" / "results.jsonl", "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {r["test"]: r["t_s"] for r in rows}
    print(f"[exp19] summary {args.tag}: {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
