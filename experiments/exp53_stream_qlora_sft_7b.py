"""exp53: streaming QLoRA SFT on Qwen2.5-7B (4-bit CPU base, one block on GPU).

This is the instruction-tuning counterpart to `exp52`: the same streaming QLoRA
engine, but trained on Dolly-15k with the Qwen chat template. It shows that a
7B model can be *instruction-fine-tuned* on a 6 GB card when the base stays in
4-bit on CPU and only one decoder block is on the GPU at a time.

Usage:
  python experiments/exp53_stream_qlora_sft_7b.py --steps 100 --seq 256 --tag stream_sft7b
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lcsa.metrics import append_jsonl, now  # noqa: E402
from lcsa.stream_quant_lora import StreamingQuantLoRATrainer  # noqa: E402

try:
    import psutil
    _proc = psutil.Process()

    def host_rss_gb():
        return round(_proc.memory_info().rss / 1e9, 2)
except Exception:
    def host_rss_gb():
        return None


def format_messages(row: dict) -> list[dict]:
    user = row["instruction"]
    if row.get("context"):
        user = f"{user}\n\nContext:\n{row['context']}"
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": row["response"]},
    ]


def encode_chat(tokenizer, row: dict, seq: int):
    messages = format_messages(row)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=False)
    full = tokenizer(full_text, add_special_tokens=False).input_ids
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False,
                                                 add_generation_prompt=True)
    prompt = tokenizer(prompt_text, add_special_tokens=False).input_ids
    full = full[:seq]
    labels = list(full)
    if 0 < len(prompt) < len(full) and full[:len(prompt)] == prompt[:len(full)]:
        labels = [-100] * min(len(prompt), len(full)) + labels[min(len(prompt), len(full)):]
    if len(labels) < 2:
        return None
    return torch.tensor([full], dtype=torch.long), torch.tensor([labels], dtype=torch.long)


@torch.no_grad()
def eval_loss(trainer, rows, tokenizer, seq: int) -> float:
    """Masked SFT loss using the streaming forward (base is on CPU, 4-bit)."""
    trainer.model.eval()
    total, n = 0.0, 0
    for row in rows:
        enc = encode_chat(tokenizer, row, seq)
        if enc is None:
            continue
        ids, labels = enc
        ys = torch.full_like(ids, -100)
        ys[:, :-1] = labels[:, 1:]
        h, _cache, _pos = trainer._forward_cached(ids)
        z = trainer.norm(h)
        logits = trainer.lm_head(z).float()
        tgt = ys.reshape(-1)
        valid = tgt != -100
        v = int(valid.sum())
        if v == 0:
            continue
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1])[valid],
            tgt[valid].to(logits.device),
            reduction="sum",
        )
        total += float(ce)
        n += v
    trainer.model.train()
    return total / max(n, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-7B-Instruct")
    ap.add_argument("--dataset", default="databricks/databricks-dolly-15k")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--train_size", type=int, default=512)
    ap.add_argument("--eval_size", type=int, default=32)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--targets", choices=["attn", "all"], default="attn",
                    help="attn: q/k/v/o only; all: attention + mlp projections")
    ap.add_argument("--save", default="checkpoints/stream_qlora_7b_sft")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tag", default="stream_sft7b")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model_path = str(ROOT / args.model)
    print(f"[exp53] model={Path(model_path).name} steps={args.steps} seq={args.seq} "
          f"lora_r={args.lora_r}", flush=True)

    ds = load_dataset(args.dataset, split="train").shuffle(seed=args.seed)
    train_rows = [ds[i] for i in range(min(args.train_size, len(ds)))]
    eval_rows = [ds[len(train_rows) + i] for i in range(min(args.eval_size, len(ds) - len(train_rows)))]

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    targets = (["q_proj", "k_proj", "v_proj", "o_proj"]
               if args.targets == "attn"
               else ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"])
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, lora)
    print(f"[load] {time.time()-t0:.1f}s rss={host_rss_gb()}GB", flush=True)

    tr = StreamingQuantLoRATrainer(
        model, device="cuda", loss_chunk=64, lr=args.lr, clip=1.0, verbose=args.verbose
    )
    print(f"[trainer] layers={len(tr.layers)} lora_params={tr.n_lora/1e6:.2f}M "
          f"rss={host_rss_gb()}GB", flush=True)

    ev_before = eval_loss(tr, eval_rows, tokenizer, args.seq)
    print(f"[eval] before loss={ev_before:.4f} ({len(eval_rows)} rows)", flush=True)

    losses, times, peaks = [], [], []
    t_all = time.time()
    for i in range(args.steps):
        row = train_rows[i % len(train_rows)]
        enc = encode_chat(tokenizer, row, args.seq)
        if enc is None:
            continue
        ids, labels = enc
        # Streaming step expects targets aligned with positions; use -100 masks.
        ys = torch.full_like(ids, -100)
        ys[:, :-1] = labels[:, 1:]
        loss, dt, peak = tr.step(ids, ys, ignore_index=-100)
        losses.append(round(loss, 6))
        times.append(round(dt, 3))
        peaks.append(round(peak, 3))
        if (i + 1) % 10 == 0:
            print(f"[step] {i+1}/{args.steps} loss={loss:.4f} {dt:.2f}s peak={peak:.2f}GB "
                  f"rss={host_rss_gb()}GB", flush=True)

    wall = time.time() - t_all
    ev_after = eval_loss(tr, eval_rows, tokenizer, args.seq)
    print(f"[eval] after loss={ev_after:.4f} (delta {ev_after-ev_before:+.4f})", flush=True)

    save_path = ROOT / args.save
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"[save] {save_path}", flush=True)

    row = {
        "exp": "exp53_stream_qlora_sft",
        "tag": args.tag,
        "model": Path(model_path).name,
        "dataset": args.dataset,
        "base_quant": "nf4",
        "base_device": "cpu",
        "streamed": "one block on GPU",
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "targets": args.targets,
        "trainable_params": tr.n_lora,
        "steps": len(losses),
        "seq": args.seq,
        "lr": args.lr,
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "losses": losses,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "step_s_med": sorted(times)[len(times) // 2] if times else None,
        "wall_s": round(wall, 1),
        "peak_gb_max": max(peaks) if peaks else None,
        "host_rss_gb": host_rss_gb(),
        "eval_loss_before": round(ev_before, 5),
        "eval_loss_after": round(ev_after, 5),
        "eval_loss_delta": round(ev_after - ev_before, 6),
        "saved": args.save,
        "ts": now(),
    }
    print(json.dumps(row, ensure_ascii=True), flush=True)
    append_jsonl(str(ROOT / "results" / "results.jsonl"), row)
    with open(ROOT / "results" / "exp53_stream_qlora_sft.log", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
