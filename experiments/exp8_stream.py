"""exp8: P1 "parça parça" (block-streaming) training — verification + demo.

Phase A (parity, --mem_fraction 0): same model, same batches, same LoRA init.
  naive : whole model + full-graph autograd on GPU (reference)
  stream: base blocks live in CPU RAM, stream to GPU one block at a time,
          recomputed in reverse for backward (lcsa/stream_train.py)
  Expect: matching loss trajectories (tiny fp differences from chunked CE).

Phase B (the point, --mem_fraction / --vram_gb): a hard per-process VRAM cap.
  naive  : expected to OOM (e.g. 1.5B fp16 = 3GB weights > cap)
  stream : keeps training with bounded VRAM (one block + activations + head)

Usage:
  python experiments/exp8_stream.py --mode both --steps 5 --seq 512 --tag smoke
  python experiments/exp8_stream.py --model models/Qwen2.5-1.5B --mode both \
      --steps 10 --seq 1024 --vram_gb 2.1 --tag 1p5b_cap21
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM

from lcsa.data import TrainSampler, chunks_from_tokens, load_tokens
from lcsa.lora import lora_state_dict
from lcsa.metrics import append_jsonl, now
from lcsa.stream_train import NaiveLoRATrainer, StreamingLoRATrainer, stream_eval_ce


try:
    import psutil
    _proc = psutil.Process()

    def host_rss_gb():
        return round(_proc.memory_info().rss / 1e9, 2)
except Exception:  # psutil optional
    def host_rss_gb():
        return None


def set_vram_cap(fraction: float | None, gb: float | None) -> float | None:
    """Cap this process's caching allocator (before any big allocation)."""
    if not fraction and not gb:
        return None
    total = torch.cuda.get_device_properties(0).total_memory
    f = fraction if fraction else (gb * 1e9 / total)
    torch.cuda.set_per_process_memory_fraction(f, 0)
    return f


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-0.5B")
    ap.add_argument("--data", default="data/wt2_train.pt")
    ap.add_argument("--mode", default="both", choices=["naive", "stream", "both"])
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--mem_fraction", type=float, default=0.0)
    ap.add_argument("--vram_gb", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clip", type=float, default=0.0)
    ap.add_argument("--save_lora", default="")
    ap.add_argument("--eval_chunks", type=int, default=0)
    ap.add_argument("--eval_data", default="data/wt2_test.pt")
    ap.add_argument("--tag", default="exp8_smoke")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cap = set_vram_cap(args.mem_fraction or None, args.vram_gb or None)
    dev = "cuda"
    print(f"[exp8] model={args.model} mode={args.mode} steps={args.steps} seq={args.seq} "
          f"lora_r={args.r} lr={args.lr} cap_fraction={cap}", flush=True)

    tokens = load_tokens(args.data)
    sampler = TrainSampler(tokens, args.seq, generator=torch.Generator().manual_seed(1234))
    batches = [sampler.batch(1) for _ in range(args.steps)]

    def load_fresh():
        torch.manual_seed(args.seed)          # identical LoRA init for both trainers
        m = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
        m.eval()
        return m

    def run_naive():
        m = load_fresh()
        tr = NaiveLoRATrainer(m, device=dev, lr=args.lr, r_attn=args.r, r_mlp=0, clip=args.clip)
        print(f"[naive] lora params: {tr.n_lora:,}", flush=True)
        losses, times, peaks = [], [], []
        for i, (xs, ys) in enumerate(batches):
            loss, dt, peak = tr.step(xs, ys)
            losses.append(round(loss, 6)); times.append(round(dt, 3)); peaks.append(round(peak, 3))
            print(f"[naive] step {i+1}/{len(batches)} loss={loss:.4f} {dt:.2f}s peak={peak:.2f}GB",
                  flush=True)
        res = {"losses": losses, "times_s": times, "peak_gb": peaks, "n_lora": tr.n_lora,
               "host_rss_gb": host_rss_gb()}
        return res, m

    def run_stream():
        m = load_fresh()
        tr = StreamingLoRATrainer(m, device=dev, lr=args.lr, r_attn=args.r, r_mlp=0,
                                  loss_chunk=args.chunk, clip=args.clip, verbose=args.verbose)
        print(f"[stream] lora params: {tr.n_lora:,} host_rss={host_rss_gb()}GB", flush=True)
        ce0 = ce1 = None
        if args.eval_chunks > 0:
            ev_chunks = chunks_from_tokens(load_tokens(args.eval_data), args.seq,
                                           args.eval_chunks)
            ce0 = sum(stream_eval_ce(tr, ev_chunks[i].unsqueeze(0), args.chunk)
                      for i in range(len(ev_chunks))) / len(ev_chunks)
            print(f"[stream] dev CE before: {ce0:.4f} ({len(ev_chunks)}x{args.seq} chunks)",
                  flush=True)
        losses, times, peaks = [], [], []
        for i, (xs, ys) in enumerate(batches):
            loss, dt, peak = tr.step(xs, ys)
            losses.append(round(loss, 6)); times.append(round(dt, 3)); peaks.append(round(peak, 3))
            print(f"[stream] step {i+1}/{len(batches)} loss={loss:.4f} {dt:.2f}s peak={peak:.2f}GB",
                  flush=True)
        res = {"losses": losses, "times_s": times, "peak_gb": peaks, "n_lora": tr.n_lora,
               "host_rss_gb": host_rss_gb()}
        if args.eval_chunks > 0:
            ce1 = sum(stream_eval_ce(tr, ev_chunks[i].unsqueeze(0), args.chunk)
                      for i in range(len(ev_chunks))) / len(ev_chunks)
            print(f"[stream] dev CE after:  {ce1:.4f} (delta {ce1 - ce0:+.4f})", flush=True)
            res.update({"ce_before": round(ce0, 6), "ce_after": round(ce1, 6),
                        "eval_chunks": len(ev_chunks)})
        return res, m

    out = {"tag": args.tag, "model": args.model, "mode": args.mode, "steps": args.steps,
           "seq": args.seq, "lr": args.lr, "r": args.r, "chunk": args.chunk,
           "cap_fraction": cap, "seed": args.seed,
           "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)}

    naive_model = stream_model = None
    if args.mode in ("naive", "both"):
        try:
            out["naive"], naive_model = run_naive()
        except torch.cuda.OutOfMemoryError as e:
            msg = (str(e).splitlines() or ["OOM"])[0]
            out["naive"] = {"oom": True, "error": msg}
            print(f"[naive] OOM (expected under the cap): {msg}", flush=True)
            torch.cuda.empty_cache()
        except RuntimeError as e:  # some Windows builds surface OOM as plain RuntimeError
            if "out of memory" in str(e).lower():
                out["naive"] = {"oom": True, "error": str(e).splitlines()[0]}
                print(f"[naive] OOM (expected under the cap): {e}", flush=True)
                torch.cuda.empty_cache()
            else:
                raise

    if args.mode in ("stream", "both"):
        torch.cuda.empty_cache()  # drop the naive phase's cached blocks: true peak for stream
        out["stream"], stream_model = run_stream()

    trained = stream_model if stream_model is not None else naive_model
    if args.save_lora and trained is not None:
        sd = lora_state_dict(trained)
        torch.save({"lora": sd,
                    "meta": {"model": args.model, "r": args.r, "steps": args.steps,
                             "lr": args.lr, "seq": args.seq, "tag": args.tag,
                             "mode": args.mode, "clip": args.clip, "ts": now()}},
                   args.save_lora)
        out["saved_lora"] = args.save_lora
        print(f"[exp8] saved LoRA adapter -> {args.save_lora} ({len(sd)} tensors)", flush=True)

    na, st = out.get("naive", {}), out.get("stream", {})
    if "losses" in na and "losses" in st:
        diffs = [abs(a - b) for a, b in zip(na["losses"], st["losses"])]
        out["loss_absdiff_max"] = max(diffs)
        print(f"[exp8] max |naive-stream| loss diff: {max(diffs):.2e}", flush=True)

    os.makedirs("results", exist_ok=True)
    fname = f"results/exp8_{args.tag}.json"
    with open(fname, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"[exp8] wrote {fname}", flush=True)

    def compact(entry):
        if "losses" not in entry:
            return {"oom": True}
        n = len(entry["losses"])
        out = {"steps": n, "loss_last": entry["losses"][-1], "loss_tail": entry["losses"][-5:],
               "s_per_step": round(sum(entry["times_s"]) / n, 3),
               "peak_gb": max(entry["peak_gb"]), "n_lora": entry["n_lora"],
               "host_rss_gb": entry.get("host_rss_gb")}
        for k in ("ce_before", "ce_after"):
            if k in entry:
                out[k] = entry[k]
        return out

    row = {"exp": "exp8_stream", "tag": args.tag, "model": args.model, "mode": args.mode,
           "seq": args.seq, "lr": args.lr, "cap_fraction": cap}
    if na:
        row["naive"] = compact(na)
    if st:
        row["stream"] = compact(st)
    if "loss_absdiff_max" in out:
        row["loss_absdiff_max"] = out["loss_absdiff_max"]
    if "saved_lora" in out:
        row["saved_lora"] = out["saved_lora"]
    append_jsonl("results/results.jsonl", row)
    print("[exp8] appended results.jsonl", flush=True)


if __name__ == "__main__":
    main()
