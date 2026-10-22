"""Download the Qwen3-4B-Instruct-2507 weights used by the default chat model.

    python scripts/download_qwen3_4b.py

The download is ~8.0 GB and lands in models/Qwen3-4B-Instruct-2507. It is
resumable: re-running the command continues where it stopped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--dir", default="models/Qwen3-4B-Instruct-2507")
    args = ap.parse_args()
    dest = Path(args.dir)
    if not dest.is_absolute():
        dest = ROOT / dest
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub eksik: pip install huggingface_hub", file=sys.stderr)
        return 2
    print(f"[download] {args.repo} -> {dest} (~8.0 GB, resumable)", flush=True)
    path = snapshot_download(
        args.repo, local_dir=str(dest),
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.md"])
    total = sum(p.stat().st_size for p in Path(path).glob("*.safetensors"))
    print(f"[download] tamam: {path} ({total / 2**30:.2f} GiB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
