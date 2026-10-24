"""Download Qwen2.5-3B-Instruct for a paged-chat quality probe.

This is a separate optional model; it does not replace the 3B deep checkpoint.
Run in the background while the demo server is idle:

  python scripts/download_3b_instruct.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "models" / "Qwen2.5-3B-Instruct"
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

print(f"[download] {TARGET}", flush=True)
path = snapshot_download(
    "Qwen/Qwen2.5-3B-Instruct",
    local_dir=str(TARGET),
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
)
print(f"[download] done: {path}", flush=True)
