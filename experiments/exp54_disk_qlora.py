"""exp54: disk-backed (mmap) streaming QLoRA -- weights live on the SSD.

Companion to `exp52`. Instead of keeping the 4-bit base in host RAM, the
pre-quantized bnb-4bit safetensors are memory-mapped and each decoder block is
materialized on the GPU on demand and dropped afterwards. Host RAM becomes a
free, OS-managed page cache for the hot blocks, so the model size is bounded by
disk space (and step time), not by RAM.

Usage:
  # 1) turn a local bf16 model into a bnb-4bit checkpoint (once)
  python experiments/exp54_disk_qlora.py --prepare --model models/Qwen2.5-3B \
      --save-quant checkpoints/qwen3b_bnb4bit
  # 2) train LoRA on it straight from disk
  python experiments/exp54_disk_qlora.py --model-dir checkpoints/qwen3b_bnb4bit \
      --steps 5 --seq 128 --eval_chunks 1 --tag disk_qlora3b
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

from lcsa.data import TrainSampler, chunks_from_tokens, load_tokens  # noqa: E402
from lcsa.disk_quant_store import (  # noqa: E402
    DiskQuantStore,
    build_skeleton,
    load_layer,
    materialize_shared,
    release_layer,
)
from lcsa.lora import count_lora_params, lora_state_dict  # noqa: E402
from lcsa.metrics import append_jsonl, now  # noqa: E402
from lcsa.stream_quant_lora import StreamingQuantLoRATrainer  # noqa: E402
from lcsa.stream_train import stream_eval_ce  # noqa: E402

try:
    import psutil
    _proc = psutil.Process()

    def host_rss_gb():
        return round(_proc.memory_info().rss / 1e9, 2)

    def host_private_gb():
        # private (non-shared) bytes: excludes the mmap'd weight pages, which the
        # OS can evict -- that is the number that matters for "does it fit in RAM"
        try:
            return round(_proc.memory_info().private / 1e9, 2)
        except Exception:
            return None
except Exception:
    def host_rss_gb():
        return None

    def host_private_gb():
        return None


def prepare(model: str, save_quant: str) -> int:
    """Quantize a local bf16 model into a bnb-4bit checkpoint (uint8 + qstate)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    src = str(ROOT / model)
    dst = ROOT / save_quant
    print(f"[prepare] {model} -> {save_quant}", flush=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(
        src, quantization_config=bnb, device_map="cpu", low_cpu_mem_usage=True)
    print(f"[prepare] loaded {time.time()-t0:.1f}s rss={host_rss_gb()}GB", flush=True)
    dst.mkdir(parents=True, exist_ok=True)
    m.save_pretrained(str(dst), safe_serialization=True, max_shard_size="2GB")
    AutoTokenizer.from_pretrained(src).save_pretrained(str(dst))
    print(f"[prepare] saved {dst} rss={host_rss_gb()}GB", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--model", default="models/Qwen2.5-3B")
    ap.add_argument("--save-quant", default="checkpoints/qwen3b_bnb4bit")
    ap.add_argument("--model-dir", default="checkpoints/qwen3b_bnb4bit")
    ap.add_argument("--data", default="data/wt2_train.pt")
    ap.add_argument("--eval_data", default="data/wt2_test.pt")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--targets", choices=["attn", "all"], default="attn")
    ap.add_argument("--head", choices=["bf16", "nf4"], default="bf16")
    ap.add_argument("--shared-dtype", choices=["fp32", "as-stored"], default="fp32",
                    help="fp32 matches prepare_model_for_kbit_training; 'as-stored' "
                         "keeps embed/head in the checkpoint dtype (big models)")
    ap.add_argument("--eval_chunks", type=int, default=0)
    ap.add_argument("--save", default="")
    ap.add_argument("--tag", default="disk_qlora")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.prepare:
        return prepare(args.model, args.save_quant)

    model_dir = str(ROOT / args.model_dir)
    print(f"[exp54] model_dir={Path(model_dir).name} steps={args.steps} seq={args.seq} "
          f"lora_r={args.lora_r} targets={args.targets} head={args.head}", flush=True)

    t0 = time.time()
    model, cfg = build_skeleton(
        model_dir, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        targets=args.targets, head=args.head,
        low_precision_shared=(args.shared_dtype == "as-stored"),
    )
    store = DiskQuantStore(model_dir, device="cuda")
    base = materialize_shared(model, store, cfg, device="cuda")
    print(f"[load] skeleton+shared {time.time()-t0:.1f}s keys={len(store.weight_map)} "
          f"rss={host_rss_gb()}GB", flush=True)

    def stage(layer, i):
        load_layer(layer, i, store, device="cuda")

    def release(layer, i):
        release_layer(layer, i, store)

    tr = StreamingQuantLoRATrainer(
        model, device="cuda", loss_chunk=args.chunk, lr=args.lr, clip=1.0,
        verbose=args.verbose, stage=stage, release=release,
    )
    print(f"[trainer] layers={len(tr.layers)} lora_params={tr.n_lora/1e6:.2f}M "
          f"tied={tr.tied} rss={host_rss_gb()}GB params={count_lora_params(model)/1e6:.2f}M",
          flush=True)

    tokens = load_tokens(str(ROOT / args.data))
    sampler = TrainSampler(tokens, args.seq, generator=torch.Generator().manual_seed(1234))
    batches = [sampler.batch(1) for _ in range(args.steps)]

    ce0 = ce1 = None
    if args.eval_chunks > 0:
        ev = chunks_from_tokens(load_tokens(str(ROOT / args.eval_data)), args.seq, args.eval_chunks)
        ce0 = sum(stream_eval_ce(tr, ev[i].unsqueeze(0), args.chunk)
                  for i in range(len(ev))) / len(ev)
        print(f"[eval] before CE={ce0:.4f}", flush=True)

    losses, times, peaks = [], [], []
    for i, (xs, ys) in enumerate(batches):
        loss, dt, peak = tr.step(xs, ys)
        losses.append(round(loss, 6))
        times.append(round(dt, 3))
        peaks.append(round(peak, 3))
        print(f"[step] {i+1}/{len(batches)} loss={loss:.4f} {dt:.2f}s peak={peak:.2f}GB "
              f"rss={host_rss_gb()}GB", flush=True)

    if args.eval_chunks > 0:
        ce1 = sum(stream_eval_ce(tr, ev[i].unsqueeze(0), args.chunk)
                  for i in range(len(ev))) / len(ev)
        print(f"[eval] after CE={ce1:.4f} (delta {ce1-ce0:+.4f})", flush=True)

    saved = None
    if args.save:
        save_path = ROOT / args.save
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(lora_state_dict(model), str(save_path / "adapter.pt"))
        (save_path / "adapter_config.json").write_text(json.dumps({
            "r": args.lora_r, "alpha": args.lora_alpha, "targets": args.targets,
            "base": Path(model_dir).name,
        }, indent=2), encoding="utf-8")
        saved = args.save
        print(f"[save] {save_path}", flush=True)

    row = {
        "exp": "exp54_disk_qlora",
        "tag": args.tag,
        "model": Path(model_dir).name,
        "weights": "mmap from disk (no RAM-resident base)",
        "base_quant": "nf4",
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "targets": args.targets,
        "head": args.head,
        "shared_dtype": args.shared_dtype,
        "trainable_params": tr.n_lora,
        "steps": len(losses),
        "seq": args.seq,
        "lr": args.lr,
        "losses": losses,
        "step_s_med": sorted(times)[len(times) // 2] if times else None,
        "peak_gb_max": max(peaks) if peaks else None,
        "host_rss_gb": host_rss_gb(),
        "host_private_gb": host_private_gb(),
        "ce_before": round(ce0, 5) if ce0 is not None else None,
        "ce_after": round(ce1, 5) if ce1 is not None else None,
        "ce_delta": round(ce1 - ce0, 6) if (ce0 is not None and ce1 is not None) else None,
        "saved": saved,
        "ts": now(),
    }
    print(json.dumps(row, ensure_ascii=True), flush=True)
    append_jsonl(str(ROOT / "results" / "results.jsonl"), row)
    with open(ROOT / "results" / "exp54_disk_qlora.log", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
