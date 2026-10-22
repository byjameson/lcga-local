"""exp17: full-parameter block-streaming training on a small GPU.

Usage (smoke):
  python experiments/exp17_full_train.py --model models/Qwen2.5-0.5B \
      --data data/wt2_train.pt --steps 5 --seq 256 --lr 1e-4 --tag full_smoke

3B demo (100 steps, WikiText-2):
  python experiments/exp17_full_train.py --model models/Qwen2.5-3B \
      --data data/wt2_train.pt --eval_data data/wt2_test.pt \
      --steps 100 --seq 512 --lr 2e-5 --eval_chunks 16 --tag full3b_t100
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
from lcsa.stream_full import StreamingSGDTrainer  # noqa: E402
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
    ap.add_argument("--model", default="models/Qwen2.5-0.5B")
    ap.add_argument("--data", default="data/wt2_train.pt")
    ap.add_argument("--eval_data", default="data/wt2_test.pt")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--opt", choices=["sgd", "adafactor"], default="sgd")
    ap.add_argument("--adaf_beta2", type=float, default=0.999)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--master", action="store_true",
                    help="keep fp32 master weights on CPU (exact fp32 updates)")
    ap.add_argument("--decay", choices=["none", "cos"], default="none")
    ap.add_argument("--min_lr", type=float, default=0.0)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--eval_chunks", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=0,
                    help="run eval every N steps and track the best checkpoint")
    ap.add_argument("--save_best", default="", help="path for the best-eval checkpoint")
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--profile", action="store_true",
                    help="time H2D/D2H transfers vs compute per step")
    ap.add_argument("--prefetch", action="store_true",
                    help="async next-layer H2D on a side stream (overlap with compute)")
    ap.add_argument("--buffers", action="store_true",
                    help="per-block GPU buffers; forward pass skips the copy-back")
    ap.add_argument("--pinned", action="store_true",
                    help="pinned host staging for block transfers (implies --buffers)")
    ap.add_argument("--double-buffer", action="store_true", dest="double_buffer",
                    help="two pinned host/GPU pools + copy stream; overlap next-block "
                         "H2D with current-block compute (implies --buffers)")
    ap.add_argument("--save", default="")
    ap.add_argument("--tag", default="full_smoke")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    model_path = str(ROOT / args.model) if not Path(args.model).is_absolute() else args.model
    data_path = str(ROOT / args.data) if not Path(args.data).is_absolute() else args.data
    eval_path = str(ROOT / args.eval_data) if not Path(args.eval_data).is_absolute() else args.eval_data

    print(f"[exp17] model={Path(model_path).name} steps={args.steps} seq={args.seq} "
          f"lr={args.lr} clip={args.clip} opt={args.opt}", flush=True)
    tokens = load_tokens(data_path)
    sampler = TrainSampler(tokens, args.seq, generator=torch.Generator().manual_seed(1234))
    batches = [sampler.batch(1) for _ in range(args.steps)]

    from transformers import AutoModelForCausalLM
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
    model.eval()
    print(f"[load] {time.time()-t0:.1f}s host_rss={host_rss_gb()}GB", flush=True)
    tr = StreamingSGDTrainer(model, device="cuda", loss_chunk=args.chunk, lr=args.lr,
                             clip=args.clip, weight_decay=args.wd, opt=args.opt,
                             adaf_beta2=args.adaf_beta2, warmup=args.warmup,
                             master=args.master, verbose=args.verbose,
                             profile=args.profile, prefetch=args.prefetch,
                             buffered=args.buffers, pinned=args.pinned,
                             double_buffer=args.double_buffer)
    if args.decay != "none":
        tr.set_schedule(args.steps, args.decay, args.min_lr)
    print(f"[trainer] params={tr.n_params/1e9:.4f}B tied={tr.tied} "
          f"host_rss={host_rss_gb()}GB", flush=True)

    ce0 = ce1 = None
    if args.eval_every and args.eval_chunks <= 0:
        raise SystemExit("--eval_every requires --eval_chunks > 0")
    if args.eval_chunks > 0:
        ev = chunks_from_tokens(load_tokens(eval_path), args.seq, args.eval_chunks)
        t0 = time.time()
        ce0 = sum(stream_eval_ce(tr, ev[i].unsqueeze(0), args.chunk)
                  for i in range(len(ev))) / len(ev)
        print(f"[eval] before CE={ce0:.4f} ({len(ev)} chunks, {time.time()-t0:.1f}s)", flush=True)

    evals = []
    best_ce, best_step = None, None

    def record_eval(step: int) -> None:
        nonlocal best_ce, best_step
        t_ev = time.time()
        ce = sum(stream_eval_ce(tr, ev[i].unsqueeze(0), args.chunk)
                 for i in range(len(ev))) / len(ev)
        evals.append({"step": step, "ce": round(ce, 5)})
        print(f"[eval] step={step} CE={ce:.4f} ({time.time()-t_ev:.1f}s)", flush=True)
        if best_ce is None or ce < best_ce:
            best_ce, best_step = ce, step
            if args.save_best:
                bp = ROOT / args.save_best
                bp.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), bp)
                print(f"[save_best] {bp} (CE={ce:.4f} @ step {step})", flush=True)

    losses, times, peaks = [], [], []
    prof_prev: dict = {}
    prof_last = None
    t_all = time.time()
    for i, (xs, ys) in enumerate(batches):
        loss, dt, peak = tr.step(xs, ys)
        losses.append(round(loss, 6))
        times.append(round(dt, 3))
        peaks.append(round(peak, 3))
        prof_msg = ""
        if args.profile:
            snap = dict(tr.prof)
            prof_last = {k: round(v - prof_prev.get(k, 0.0), 3) for k, v in snap.items()}
            prof_prev = snap
            prof_msg = " prof[" + " ".join(f"{k}={v:.2f}" for k, v in prof_last.items()) + "]"
        if (i + 1) % args.log_every == 0:
            print(f"[step] {i+1}/{len(batches)} loss={loss:.4f} {dt:.2f}s "
                  f"peak={peak:.2f}GB rss={host_rss_gb()}GB{prof_msg}", flush=True)
        if args.eval_every and (i + 1) % args.eval_every == 0:
            record_eval(i + 1)
    wall = time.time() - t_all

    if args.eval_chunks > 0:
        if not evals or evals[-1]["step"] != len(losses):
            record_eval(len(losses))
        ce1 = evals[-1]["ce"]
        print(f"[eval] after CE={ce1:.4f} (delta {ce1-ce0:+.4f})", flush=True)

    if args.save:
        save_path = ROOT / args.save
        save_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        torch.save(model.state_dict(), save_path)
        print(f"[save] {save_path} ({save_path.stat().st_size/1e9:.2f}GB, {time.time()-t0:.1f}s)",
              flush=True)

    row = {
        "exp": "exp17_full_train",
        "tag": args.tag,
        "model": Path(model_path).name,
        "params": tr.n_params,
        "steps": len(losses),
        "seq": args.seq,
        "lr": args.lr,
        "clip": args.clip,
        "wd": args.wd,
        "opt": args.opt,
        "adaf_beta2": args.adaf_beta2 if args.opt == "adafactor" else None,
        "warmup": args.warmup,
        "master": args.master,
        "decay": args.decay,
        "min_lr": args.min_lr if args.decay != "none" else None,
        "losses": losses,
        "loss_last": losses[-1] if losses else None,
        "loss_first": losses[0] if losses else None,
        "step_s_med": sorted(times)[len(times) // 2] if times else None,
        "wall_s": round(wall, 1),
        "peak_gb_max": max(peaks) if peaks else None,
        "host_rss_gb": host_rss_gb(),
        "ce_before": ce0,
        "ce_after": ce1,
        "eval_every": args.eval_every or None,
        "evals": evals,
        "best_ce": round(best_ce, 5) if best_ce is not None else None,
        "best_step": best_step,
        "save_best": args.save_best or None,
        "saved": args.save or None,
        "prof_last": prof_last,
        "prof_total": ({k: round(v, 3) for k, v in tr.prof.items()}
                       if args.profile else None),
        "prefetch": bool(args.prefetch),
        "buffered": bool(args.buffers or args.pinned or args.double_buffer),
        "pinned": bool(args.pinned),
        "double_buffer": bool(args.double_buffer),
        "ts": now(),
    }
    if ce0 is not None and ce1 is not None:
        row["ce_delta"] = round(ce1 - ce0, 6)
    print(json.dumps(row, ensure_ascii=True), flush=True)
    append_jsonl(str(ROOT / "results" / "results.jsonl"), row)
    with open(ROOT / "results" / "exp17_full.log", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
