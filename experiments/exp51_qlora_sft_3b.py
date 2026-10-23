"""exp51: short QLoRA SFT on Qwen2.5-3B-Instruct with Dolly-15k.

This is the instruction-tuning counterpart to `exp49` (which used WikiText-2
next-token loss). It is still a PoC:
  * a few hundred training rows and a handful of steps;
  * one seed, no hyperparameter search;
  * eval loss + a few generated samples, not a benchmark.

Usage:
  python experiments/exp51_qlora_sft_3b.py --steps 20 --seq 384 --tag qlora_sft3b
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


def encode_chat(tokenizer, row: dict, seq: int) -> tuple[torch.Tensor, torch.Tensor]:
    messages = format_messages(row)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=False)
    full = tokenizer(full_text, add_special_tokens=False).input_ids
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False,
                                                 add_generation_prompt=True)
    prompt = tokenizer(prompt_text, add_special_tokens=False).input_ids
    full = full[:seq]
    labels = list(full)
    # Mask the user/prompt prefix when the tokenizer keeps a clean prefix.
    if 0 < len(prompt) < len(full) and full[:len(prompt)] == prompt[:len(full)]:
        labels = [-100] * min(len(prompt), len(full)) + labels[min(len(prompt), len(full)):]
    if len(labels) < 2:
        return None
    return torch.tensor([full], dtype=torch.long), torch.tensor([labels], dtype=torch.long)


@torch.no_grad()
def eval_loss(model, rows, tokenizer, seq: int) -> float:
    model.eval()
    total, n = 0.0, 0
    for row in rows:
        enc = encode_chat(tokenizer, row, seq)
        if enc is None:
            continue
        ids, labels = enc
        ids = ids.to("cuda")
        labels = labels.to("cuda")
        out = model(input_ids=ids, labels=labels)
        ntok = int((labels[:, 1:] != -100).sum())
        total += float(out.loss) * max(ntok, 1)
        n += max(ntok, 1)
    model.train()
    return total / max(n, 1)


@torch.no_grad()
def sample_answers(model, tokenizer, prompts, max_new=48):
    model.eval()
    out = []
    for p in prompts:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        gen = model.generate(input_ids=ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
        ans = tokenizer.decode(gen[0][ids.shape[1]:], skip_special_tokens=True).strip()
        out.append({"prompt": p, "answer": ans[:400]})
    model.train()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-3B-Instruct")
    ap.add_argument("--dataset", default="databricks/databricks-dolly-15k")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seq", type=int, default=384)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--train_size", type=int, default=256)
    ap.add_argument("--eval_size", type=int, default=16)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--save", default="checkpoints/qlora_qwen3b_sft")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tag", default="qlora_sft3b")
    args = ap.parse_args()

    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model_path = str(ROOT / args.model)
    print(f"[exp51] model={Path(model_path).name} steps={args.steps} seq={args.seq} "
          f"lr={args.lr} train={args.train_size}", flush=True)

    ds = load_dataset(args.dataset, split="train").shuffle(seed=args.seed)
    train_rows = [ds[i] for i in range(min(args.train_size, len(ds)))]
    eval_rows = [ds[len(train_rows) + i] for i in range(min(args.eval_size, len(ds) - len(train_rows)))]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb_config, device_map={"": 0}
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    print(f"[load] {time.time()-t0:.1f}s host_rss={host_rss_gb()}GB", flush=True)

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
          f"({100.0*n_train/max(n_total,1):.2f}%)", flush=True)

    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdamW8bit(trainable, lr=args.lr, betas=(0.9, 0.999))
        opt_name = "paged_adamw8bit"
    except Exception as exc:
        print(f"[opt] paged_adamw8bit unavailable ({exc}); using AdamW", flush=True)
        optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999))
        opt_name = "adamw"

    ev_before = eval_loss(model, eval_rows, tokenizer, args.seq)
    print(f"[eval] before loss={ev_before:.4f} ({len(eval_rows)} rows)", flush=True)

    losses, times, peaks = [], [], []
    torch.cuda.reset_peak_memory_stats()
    t_all = time.time()
    for i in range(args.steps):
        row = train_rows[i % len(train_rows)]
        enc = encode_chat(tokenizer, row, args.seq)
        if enc is None:
            continue
        ids, labels = enc
        ids = ids.to("cuda")
        labels = labels.to("cuda")
        t = time.perf_counter()
        out = model(input_ids=ids, labels=labels)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        peak = torch.cuda.max_memory_allocated() / 1e9
        losses.append(round(float(loss.detach()), 6))
        times.append(round(dt, 3))
        peaks.append(round(peak, 3))
        print(f"[step] {i+1}/{args.steps} loss={float(loss):.4f} {dt:.2f}s "
              f"peak={peak:.2f}GB rss={host_rss_gb()}GB", flush=True)

    wall = time.time() - t_all
    ev_after = eval_loss(model, eval_rows, tokenizer, args.seq)
    print(f"[eval] after loss={ev_after:.4f} (delta {ev_after-ev_before:+.4f})", flush=True)

    samples = sample_answers(model, tokenizer, [
        "Give three tips for learning a new language.",
        "What is the capital of France?",
        "Summarize the benefits of exercise in one sentence.",
    ])
    for s in samples:
        print(f"[sample] Q: {s['prompt']}\n         A: {s['answer'][:200]}", flush=True)

    save_path = ROOT / args.save
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"[save] {save_path}", flush=True)

    row = {
        "exp": "exp51_qlora_sft",
        "tag": args.tag,
        "model": Path(model_path).name,
        "dataset": args.dataset,
        "base_quant": "nf4",
        "params": n_total,
        "trainable_params": n_train,
        "trainable_pct": round(100.0 * n_train / max(n_total, 1), 4),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "optimizer": opt_name,
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
        "samples": samples,
        "saved": args.save,
        "ts": now(),
    }
    print(json.dumps(row, ensure_ascii=True), flush=True)
    append_jsonl(str(ROOT / "results" / "results.jsonl"), row)
    with open(ROOT / "results" / "exp51_qlora_sft.log", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
