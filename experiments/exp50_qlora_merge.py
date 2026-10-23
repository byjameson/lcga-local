"""exp50: merge a QLoRA adapter into the base model and save a servable checkpoint.

The QLoRA PoC (`exp49`) trains small LoRA adapters on a frozen NF4 base. This
script merges the adapter back into a bf16 base so the result can be served by
the paged inference engine (`exp16`) and compared with the base model.

Memory note: the 3B base in bf16 is ~6.2 GB, which fits this machine's free
host RAM; 7B bf16 would not.

Usage:
  python experiments/exp50_qlora_merge.py \
      --base models/Qwen2.5-3B-Instruct \
      --adapter checkpoints/qlora_qwen3b_r16_t10 \
      --save checkpoints/qwen3b_qlora_merged
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

from lcsa.metrics import append_jsonl, now  # noqa: E402

try:
    import psutil
    _proc = psutil.Process()

    def host_rss_gb():
        return round(_proc.memory_info().rss / 1e9, 2)
except Exception:
    def host_rss_gb():
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="models/Qwen2.5-3B-Instruct")
    ap.add_argument("--adapter", default="checkpoints/qlora_qwen3b_r16_t10")
    ap.add_argument("--save", default="checkpoints/qwen3b_qlora_merged")
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--tag", default="qlora_merge")
    args = ap.parse_args()

    base_path = str(ROOT / args.base)
    adapter_path = str(ROOT / args.adapter)
    save_path = ROOT / args.save

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    print(f"[exp50] base={Path(base_path).name} adapter={Path(adapter_path).name} "
          f"dtype={args.dtype}", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    base = AutoModelForCausalLM.from_pretrained(
        base_path,
        dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    print(f"[load] {time.time() - t0:.1f}s host_rss={host_rss_gb()}GB", flush=True)

    t1 = time.time()
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    merged = model.merge_and_unload()
    print(f"[merge] {time.time() - t1:.1f}s host_rss={host_rss_gb()}GB", flush=True)

    t2 = time.time()
    save_path.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(save_path), safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(str(save_path))
    save_s = time.time() - t2
    n_params = sum(p.numel() for p in merged.parameters())
    size_gb = sum(f.stat().st_size for f in save_path.glob("*.safetensors")) / 1e9
    print(f"[save] {save_path} {size_gb:.2f}GB {save_s:.1f}s host_rss={host_rss_gb()}GB",
          flush=True)

    row = {
        "exp": "exp50_qlora_merge",
        "tag": args.tag,
        "base": Path(base_path).name,
        "adapter": Path(adapter_path).name,
        "dtype": args.dtype,
        "params": n_params,
        "save_dir": args.save,
        "save_gb": round(size_gb, 3),
        "load_s": round(t1 - t0, 2),
        "merge_s": round(time.time() - t2, 2),
        "save_s": round(save_s, 2),
        "host_rss_gb": host_rss_gb(),
        "ts": now(),
    }
    print(json.dumps(row, ensure_ascii=True), flush=True)
    append_jsonl(str(ROOT / "results" / "results.jsonl"), row)
    with open(ROOT / "results" / "exp50_qlora_merge.log", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
