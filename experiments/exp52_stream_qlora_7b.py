"""exp52: streaming QLoRA on Qwen2.5-7B (4-bit CPU base, one block on GPU).

This is our own answer to the 7B-on-6GB question. The HF in-memory QLoRA path
spilled (peak 8.4-8.7 GB) because the whole 4-bit model + compute temporaries
were on the GPU. Here the base stays in 4-bit on CPU and only one decoder block
is streamed to the GPU at a time; LoRA gradients are accumulated on CPU.

Usage:
  python experiments/exp52_stream_qlora_7b.py --steps 1 --seq 128 --tag qlora7b_stream
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
from lcsa.metrics import append_jsonl, now  # noqa: E402
from lcsa.stream_quant_lora import StreamingQuantLoRATrainer  # noqa: E402
from lcsa.stream_train import stream_eval_ce  # noqa: E402

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
    ap.add_argument("--model", default="models/Qwen2.5-7B-Instruct")
    ap.add_argument("--data", default="data/wt2_train.pt")
    ap.add_argument("--eval_data", default="data/wt2_test.pt")
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--eval_chunks", type=int, default=0)
    ap.add_argument("--save", default="checkpoints/stream_qlora_7b")
    ap.add_argument("--tag", default="qlora7b_stream")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    model_path = str(ROOT / args.model)
    data_path = str(ROOT / args.data)
    eval_path = str(ROOT / args.eval_data)
    print(f"[exp52] model={Path(model_path).name} steps={args.steps} seq={args.seq} "
          f"lora_r={args.lora_r}", flush=True)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    print(f"[load] {time.time()-t0:.1f}s rss={host_rss_gb()}GB", flush=True)

    tr = StreamingQuantLoRATrainer(
        model, device="cuda", loss_chunk=args.chunk, lr=args.lr,
        clip=1.0, verbose=args.verbose,
    )
    print(f"[trainer] layers={len(tr.layers)} lora_params={tr.n_lora/1e6:.2f}M "
          f"tied={tr.tied} rss={host_rss_gb()}GB", flush=True)

    tokens = load_tokens(data_path)
    sampler = TrainSampler(tokens, args.seq, generator=torch.Generator().manual_seed(1234))
    batches = [sampler.batch(1) for _ in range(args.steps)]

    ce0 = ce1 = None
    if args.eval_chunks > 0:
        ev = chunks_from_tokens(load_tokens(eval_path), args.seq, args.eval_chunks)
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

    save_path = ROOT / args.save
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_path))
    print(f"[save] {save_path}", flush=True)

    row = {
        "exp": "exp52_stream_qlora",
        "tag": args.tag,
        "model": Path(model_path).name,
        "base_quant": "nf4",
        "base_device": "cpu",
        "streamed": "one block on GPU",
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "trainable_params": tr.n_lora,
        "trainable_pct": round(100.0 * tr.n_lora / sum(p.numel() for p in model.parameters()), 4),
        "steps": len(losses),
        "seq": args.seq,
        "lr": args.lr,
        "losses": losses,
        "step_s_med": sorted(times)[len(times) // 2] if times else None,
        "peak_gb_max": max(peaks) if peaks else None,
        "host_rss_gb": host_rss_gb(),
        "ce_before": round(ce0, 5) if ce0 is not None else None,
        "ce_after": round(ce1, 5) if ce1 is not None else None,
        "ce_delta": round(ce1 - ce0, 6) if (ce0 is not None and ce1 is not None) else None,
        "saved": args.save,
        "ts": now(),
    }
    print(json.dumps(row, ensure_ascii=True), flush=True)
    append_jsonl(str(ROOT / "results" / "results.jsonl"), row)
    with open(ROOT / "results" / "exp52_stream_qlora.log", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
