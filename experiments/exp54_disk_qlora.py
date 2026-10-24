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
    LayerCache,
    build_skeleton,
    load_layer,
    materialize_shared,
    release_layer,
)
from lcsa.autotune import plan_for  # noqa: E402
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


def _format_messages(row: dict) -> list[dict]:
    user = row["instruction"]
    if row.get("context"):
        user = f"{user}\n\nContext:\n{row['context']}"
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": row["response"]},
    ]


def _encode_chat(tokenizer, row: dict, seq: int):
    """Chat-template encode with the prompt tokens masked out of the loss."""
    messages = _format_messages(row)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=False)
    full = tokenizer(full_text, add_special_tokens=False).input_ids
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False,
                                                add_generation_prompt=True)
    prompt = tokenizer(prompt_text, add_special_tokens=False).input_ids
    full = full[:seq]
    labels = list(full)
    if 0 < len(prompt) < len(full) and full[:len(prompt)] == prompt[:len(full)]:
        k = min(len(prompt), len(full))
        labels = [-100] * k + labels[k:]
    if len(labels) < 2:
        return None
    return (torch.tensor([full], dtype=torch.long),
            torch.tensor([labels], dtype=torch.long))


@torch.no_grad()
def _masked_eval(trainer, rows, tokenizer, seq: int) -> float:
    """Masked SFT loss through the streaming (disk-backed) forward."""
    trainer.model.eval()
    total, n = 0.0, 0
    for row in rows:
        enc = _encode_chat(tokenizer, row, seq)
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
        total += float(torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1])[valid],
            tgt[valid].to(logits.device), reduction="sum"))
        n += v
    trainer.model.train()
    return total / max(n, 1)


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
    ap.add_argument("--blocks-on-gpu", default="adaptive",
                    help="'adaptive' = start at 1 block and grow from the measured "
                         "peak; 'auto' = static plan from free VRAM; or an integer")
    ap.add_argument("--adapt-every", type=int, default=10,
                    help="re-measure and resize the block window every N steps "
                         "(0 = only after the first step)")
    ap.add_argument("--vram-soft-gb", type=float, default=0.0,
                    help="grow towards this VRAM budget (0 = 80%% of the device)")
    ap.add_argument("--vram-hard-gb", type=float, default=0.0,
                    help="never exceed this VRAM budget (0 = 88%% of the device); "
                         "useful to leave room for the display/other apps")
    ap.add_argument("--eval_chunks", type=int, default=0)
    ap.add_argument("--eval_seq", type=int, default=0,
                    help="evaluation sequence length (default = --seq); set it to a "
                         "common value to compare runs trained at different lengths")
    ap.add_argument("--activation-cache", choices=["gpu", "cpu"], default="gpu",
                    help="where the per-layer inputs for the recompute-backward live; "
                         "'cpu' frees VRAM at a PCIe round-trip per layer and is what "
                         "lets context beyond ~1k fit on a 6 GB card")
    ap.add_argument("--sft", action="store_true",
                    help="train on instruction data (Dolly-15k) with prompt masking")
    ap.add_argument("--dataset", default="databricks/databricks-dolly-15k")
    ap.add_argument("--train_size", type=int, default=256)
    ap.add_argument("--eval_size", type=int, default=8)
    ap.add_argument("--save", default="")
    ap.add_argument("--tag", default="disk_qlora")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    if args.eval_seq <= 0:
        args.eval_seq = args.seq

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

    # Plan before anything is materialized: torch.cuda.mem_get_info() then sees
    # the real free VRAM (the LM head is loaded *after* this, and plan_for
    # accounts for its size itself).
    mode_blocks = str(args.blocks_on_gpu).lower()
    if mode_blocks in ("auto", "adaptive"):
        plan = plan_for(model_dir, seq=args.seq, lora_r=args.lora_r,
                        head=args.head, loss_chunk=args.chunk)
        print(plan.summary(), flush=True)
        n_blocks = plan.blocks_on_gpu
        block_gb, n_layers = plan.block_gb, plan.n_layers
    else:
        plan = None
        n_blocks = int(args.blocks_on_gpu)
        block_gb, n_layers = 0.0, 0

    # measured autotune: always start from the safest window (1 block) and let
    # the first real step tell us how much room is actually left
    start_blocks = 1 if mode_blocks == "adaptive" else n_blocks

    base = materialize_shared(model, store, cfg, device="cuda")
    print(f"[load] skeleton+shared {time.time()-t0:.1f}s keys={len(store.weight_map)} "
          f"rss={host_rss_gb()}GB", flush=True)

    layers = list(base.model.layers)
    cache = LayerCache(store, layers, device="cuda", capacity=start_blocks)

    # --- measured autotune -------------------------------------------------
    adapt = None
    adapt_hist: list = []
    if mode_blocks == "adaptive" and plan is not None:
        dev_total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        soft_gb = args.vram_soft_gb or dev_total_gb * 0.80   # grow towards this
        hard_gb = args.vram_hard_gb or dev_total_gb * 0.88   # never exceed this
        every = max(0, int(args.adapt_every))
        state = {"step": 0, "hist": []}
        adapt_hist = state["hist"]

        def _marginal_gb() -> float:
            """Measured VRAM cost of one extra resident block (fallback: nominal).

            Scan back for the most recent pair of measurements taken at *different*
            capacities; consecutive steps usually share the same window.
            """
            h = state["hist"]
            for k in range(len(h) - 1, 0, -1):
                c1, p1 = h[k]
                c0, p0 = h[k - 1]
                if c1 != c0:
                    return max(1e-6, (p1 - p0) / (c1 - c0))
            return block_gb or 0.25

        def adapt(peak_gb: float, i: int) -> None:
            state["step"] += 1
            n = state["step"]
            state["hist"].append((cache.capacity, round(peak_gb, 3)))
            if not (n == 1 or (every > 0 and n % every == 0)):
                return
            margin = _marginal_gb()
            if peak_gb > hard_gb:
                shrink = int((peak_gb - hard_gb) // max(margin, 1e-9)) + 1
                cache.set_capacity(max(1, cache.capacity - shrink))
                torch.cuda.empty_cache()  # release what the allocator still holds
                print(f"[autotune] step {n}: peak {peak_gb:.2f}GB > hard "
                      f"{hard_gb:.2f}GB -> {cache.capacity} blocks", flush=True)
                return
            head = soft_gb - peak_gb
            extra = int((head * 0.9) // max(margin, 1e-9)) if head > 0 else 0
            # never *grow* past the hard limit either: otherwise soft > hard
            # configurations oscillate between grow and shrink
            if extra > 0 and peak_gb + extra * margin > hard_gb:
                extra = max(0, int((hard_gb - peak_gb) // max(margin, 1e-9)))
            if extra >= 1 and cache.capacity < n_layers:
                before = cache.capacity
                cache.set_capacity(min(n_layers, before + extra))
                print(f"[autotune] step {n}: peak {peak_gb:.2f}GB, marginal "
                      f"{margin * 1024:.0f}MB/block -> {before}->{cache.capacity} blocks",
                      flush=True)

    def stage(layer, i):
        cache.stage(layer, i)

    def release(layer, i):
        cache.release(layer, i)

    print(f"[cache] keeping up to {cache.capacity} block(s) resident on the GPU",
          flush=True)

    tr = StreamingQuantLoRATrainer(
        model, device="cuda", loss_chunk=args.chunk, lr=args.lr, clip=1.0,
        verbose=args.verbose, stage=stage, release=release,
        cache_device=args.activation_cache,
    )
    print(f"[trainer] layers={len(tr.layers)} lora_params={tr.n_lora/1e6:.2f}M "
          f"tied={tr.tied} rss={host_rss_gb()}GB params={count_lora_params(model)/1e6:.2f}M",
          flush=True)

    losses, times, peaks = [], [], []
    ce0 = ce1 = None
    wall = 0.0
    t_all = time.time()

    if args.sft:
        from datasets import load_dataset
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_dir)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        ds = load_dataset(args.dataset, split="train").shuffle(seed=1234)
        train_rows = [ds[i] for i in range(min(args.train_size, len(ds)))]
        eval_rows = [ds[len(train_rows) + i]
                     for i in range(min(args.eval_size, len(ds) - len(train_rows)))]
        print(f"[sft] {Path(args.dataset).name} train={len(train_rows)} "
              f"eval={len(eval_rows)}", flush=True)
        ce0 = _masked_eval(tr, eval_rows, tok, args.seq)
        print(f"[eval] before loss={ce0:.4f}", flush=True)
        for i in range(args.steps):
            enc = _encode_chat(tok, train_rows[i % len(train_rows)], args.seq)
            if enc is None:
                continue
            ids, labels = enc
            ys = torch.full_like(ids, -100)
            ys[:, :-1] = labels[:, 1:]
            loss, dt, peak = tr.step(ids, ys, ignore_index=-100)
            losses.append(round(loss, 6))
            times.append(round(dt, 3))
            peaks.append(round(peak, 3))
            if adapt is not None:
                adapt(peak, i)
            if (i + 1) % 10 == 0:
                print(f"[step] {i+1}/{args.steps} loss={loss:.4f} {dt:.2f}s "
                      f"peak={peak:.2f}GB rss={host_rss_gb()}GB", flush=True)
        ce1 = _masked_eval(tr, eval_rows, tok, args.seq)
        print(f"[eval] after loss={ce1:.4f} (delta {ce1-ce0:+.4f})", flush=True)
        wall = time.time() - t_all
    else:
        tokens = load_tokens(str(ROOT / args.data))
        sampler = TrainSampler(tokens, args.seq, generator=torch.Generator().manual_seed(1234))
        batches = [sampler.batch(1) for _ in range(args.steps)]

        if args.eval_chunks > 0:
            ev = chunks_from_tokens(load_tokens(str(ROOT / args.eval_data)), args.eval_seq,
                                    args.eval_chunks)
            ce0 = sum(stream_eval_ce(tr, ev[i].unsqueeze(0), args.chunk)
                      for i in range(len(ev))) / len(ev)
            print(f"[eval] before CE={ce0:.4f} on {args.eval_chunks} chunks of "
                  f"{args.eval_seq}", flush=True)

        for i, (xs, ys) in enumerate(batches):
            loss, dt, peak = tr.step(xs, ys)
            losses.append(round(loss, 6))
            times.append(round(dt, 3))
            peaks.append(round(peak, 3))
            if adapt is not None:
                adapt(peak, i)
            print(f"[step] {i+1}/{len(batches)} loss={loss:.4f} {dt:.2f}s peak={peak:.2f}GB "
                  f"rss={host_rss_gb()}GB", flush=True)

        if args.eval_chunks > 0:
            ce1 = sum(stream_eval_ce(tr, ev[i].unsqueeze(0), args.chunk)
                      for i in range(len(ev))) / len(ev)
            print(f"[eval] after CE={ce1:.4f} (delta {ce1-ce0:+.4f})", flush=True)

        wall = time.time() - t_all

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
        "mode": "sft" if args.sft else "lm",
        "dataset": args.dataset if args.sft else None,
        "model": Path(model_dir).name,
        "weights": "mmap from disk (no RAM-resident base)",
        "base_quant": "nf4",
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "targets": args.targets,
        "head": args.head,
        "shared_dtype": args.shared_dtype,
        "blocks_mode": args.blocks_on_gpu,
        "blocks_on_gpu": cache.capacity,
        "blocks_on_gpu_initial": start_blocks,
        "blocks_history": adapt_hist,
        "trainable_params": tr.n_lora,
        "steps": len(losses),
        "seq": args.seq,
        "lr": args.lr,
        "losses": losses,
        "step_s_med": sorted(times)[len(times) // 2] if times else None,
        "wall_s": round(wall, 1),
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
