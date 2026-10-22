"""exp49: QLoRA (NF4 4-bit base + LoRA) proof-of-concept on Qwen2.5-3B.

Goal: show that a 3B model can be *fine-tuned* on a 6 GB laptop GPU when the
frozen base is quantized to NF4 and only small LoRA adapters are trained. This
is the practical alternative to full-parameter streaming when RAM/VRAM are the
binding constraints.

It is a short, honest PoC:
  * 3B only (7B bf16 does not fit this machine's free host RAM);
  * one seed, a few steps, WikiText-2 next-token loss;
  * no claim of instruction quality or long-run convergence.

Usage:
  python experiments/exp49_qlora_3b.py --steps 10 --seq 256 --tag qlora3b
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lcsa.data import TrainSampler, chunks_from_tokens, load_tokens  # noqa: E402
from lcsa.metrics import append_jsonl, now  # noqa: E402

try:
    import psutil
    _proc = psutil.Process()

    def host_rss_gb():
        return round(_proc.memory_info().rss / 1e9, 2)
except Exception:
    def host_rss_gb():
        return None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-3B-Instruct")
    ap.add_argument("--data", default="data/wt2_train.pt")
    ap.add_argument("--eval_data", default="data/wt2_test.pt")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--eval_chunks", type=int, default=4)
    ap.add_argument("--save", default="checkpoints/qlora_qwen3b_r16")
    ap.add_argument("--tag", default="qlora3b")
    return ap.parse_args()


@torch.no_grad()
def eval_ce(model, chunks, device: str) -> float:
    model.eval()
    total, n = 0.0, 0
    for chunk in chunks:
        ids = chunk.unsqueeze(0).to(device)
        out = model(input_ids=ids, labels=ids)
        ntok = ids.shape[1] - 1
        total += float(out.loss) * ntok
        n += ntok
    model.train()
    return total / max(n, 1)


def main() -> int:
    args = parse_args()
    model_path = str(ROOT / args.model)
    data_path = str(ROOT / args.data)
    eval_path = str(ROOT / args.eval_data)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    print(f"[exp49] model={Path(model_path).name} steps={args.steps} seq={args.seq} "
          f"lr={args.lr} lora_r={args.lora_r}", flush=True)

    t0 = time.time()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    load_s = time.time() - t0
    print(f"[load] {load_s:.1f}s host_rss={host_rss_gb()}GB", flush=True)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[lora] trainable={n_train/1e6:.2f}M / total={n_total/1e9:.3f}B "
          f"({100.0 * n_train / max(n_total, 1):.2f}%)", flush=True)

    optimizer = None
    opt_name = "adamw"
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdamW8bit(trainable, lr=args.lr, betas=(0.9, 0.999))
        opt_name = "paged_adamw8bit"
    except Exception as exc:  # pragma: no cover - platform fallback
        print(f"[opt] paged_adamw8bit unavailable ({exc}); using AdamW", flush=True)
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999))

    tokens = load_tokens(data_path)
    sampler = TrainSampler(tokens, args.seq, generator=torch.Generator().manual_seed(1234))
    batches = [sampler.batch(args.batch_size) for _ in range(args.steps)]

    ev_chunks = chunks_from_tokens(load_tokens(eval_path), args.seq, args.eval_chunks)
    ce0 = eval_ce(model, ev_chunks, "cuda")
    print(f"[eval] before CE={ce0:.4f} ({len(ev_chunks)} chunks)", flush=True)

    losses, times, peaks = [], [], []
    torch.cuda.reset_peak_memory_stats()
    t_all = time.time()
    for i, (xs, ys) in enumerate(batches):
        t = time.perf_counter()
        xs = xs.to("cuda")
        out = model(input_ids=xs, labels=xs)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        peak = torch.cuda.max_memory_allocated() / 1e9
        losses.append(round(float(loss), 6))
        times.append(round(dt, 3))
        peaks.append(round(peak, 3))
        print(f"[step] {i+1}/{len(batches)} loss={float(loss):.4f} {dt:.2f}s "
              f"peak={peak:.2f}GB rss={host_rss_gb()}GB", flush=True)

    wall = time.time() - t_all
    ce1 = eval_ce(model, ev_chunks, "cuda")
    print(f"[eval] after CE={ce1:.4f} (delta {ce1 - ce0:+.4f})", flush=True)

    save_path = ROOT / args.save
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"[save] {save_path}", flush=True)

    row = {
        "exp": "exp49_qlora",
        "tag": args.tag,
        "model": Path(model_path).name,
        "base_quant": "nf4",
        "double_quant": True,
        "compute_dtype": "bfloat16",
        "params": n_total,
        "trainable_params": n_train,
        "trainable_pct": round(100.0 * n_train / max(n_total, 1), 4),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "optimizer": opt_name,
        "steps": len(losses),
        "seq": args.seq,
        "lr": args.lr,
        "losses": losses,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "step_s_med": sorted(times)[len(times) // 2] if times else None,
        "wall_s": round(wall, 1),
        "load_s": round(load_s, 1),
        "peak_gb_max": max(peaks) if peaks else None,
        "host_rss_gb": host_rss_gb(),
        "ce_before": round(ce0, 5),
        "ce_after": round(ce1, 5),
        "ce_delta": round(ce1 - ce0, 6),
        "saved": args.save,
        "ts": now(),
    }
    print(json.dumps(row, ensure_ascii=True), flush=True)
    append_jsonl(str(ROOT / "results" / "results.jsonl"), row)
    with open(ROOT / "results" / "exp49_qlora.log", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
