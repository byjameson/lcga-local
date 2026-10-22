"""P2 (exp10): tokenizer'sız byte-level prototip eğitimi/değerlendirmesi.

Aynı bütçeyle (adım × batch × pencere) iki model eğitir:
  patch : iki-katmanlı sabit-patch BytePatchLM (P byte / global adım)
  flat  : düz nedensel byte transformer (baseline)
Metrik: wikitext-2 valid/test **bit/byte** (bpb); ayrıca örnek üretim, eğitim
duvar süresi ve byte/s. Sonuçlar results/results.jsonl'e tek satır/model olarak
yazılır; en iyi valid anındaki ağırlık checkpoints/{tag}_{model}.pt'ye kaydedilir.

Kullanım:
  python experiments/exp10_byte.py --tag p2 --steps 12000 --batch 32
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from lcsa.byte_lm import (BytePatchLM, FlatByteLM, bpb, eval_bpb, generate_flat,
                          generate_patch, load_bytes, sample_windows)
from lcsa.metrics import append_jsonl, now

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
RESULTS = os.path.join(ROOT, "results", "results.jsonl")


def build_model(name, args, device):
    if name == "patch":
        m = BytePatchLM(P=args.P, W=args.W, byte_dim=64, dim=256, layers=6,
                        heads=8, lo_layers=2, lo_heads=4)
    elif name == "flat":
        m = FlatByteLM(W=args.W, dim=256, layers=6, heads=8)
    else:
        raise ValueError(name)
    return m.to(device)


def train_one(name, args, device, train_bytes, valid_bytes, test_bytes):
    torch.manual_seed(args.seed)
    model = build_model(name, args, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.tag}/{name}] params={n_params/1e6:.2f}M cfg={model.cfg()}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.0)

    def lr_at(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog))

    gen = torch.Generator().manual_seed(args.seed)
    evals = []
    best_bpb = float("inf")
    best_state = None
    train_secs = 0.0
    step_ms = []

    def do_eval(step):
        nonlocal best_bpb, best_state
        v = eval_bpb(model, valid_bytes, args.W, args.eval_windows, args.batch, device)
        evals.append({"step": step, "valid_bpb": round(v, 4)})
        print(f"  [{name} @{step}] valid_bpb={v:.4f}", flush=True)
        if v < best_bpb:
            best_bpb = v
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}

    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            do_eval(step)
        if step == args.steps:
            break
        for pg in opt.param_groups:
            pg["lr"] = args.lr * lr_at(step)
        xb = sample_windows(train_bytes, args.W, args.batch, gen).to(device)
        if device == "cuda":
            torch.cuda.synchronize()   # WDDM: sync'siz ölçüm enqueue-artefaktı verir
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        nll, cnt = model.byte_loss(xb)
        (nll / cnt).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        train_secs += dt
        if step % 20 == 0:
            step_ms.append(dt * 1000)

    # --- en iyi ağırlıkla final değerlendirme --------------------------------
    model.load_state_dict(best_state)
    test_b = eval_bpb(model, test_bytes, args.W, args.test_windows, args.batch, device)
    med_ms = sorted(step_ms)[len(step_ms) // 2] if step_ms else 0.0
    print(f"  [{name}] best_valid_bpb={best_bpb:.4f} test_bpb={test_b:.4f} "
          f"step_med={med_ms:.1f}ms", flush=True)

    # --- örnek üretim (naif; demo) -------------------------------------------
    t0 = time.time()
    if name == "patch":
        sample = generate_patch(model, 240, device, temp=0.8, top_k=50, seed=args.seed)
    else:
        sample = generate_flat(model, 240, device, temp=0.8, top_k=50, seed=args.seed)
    gen_secs = time.time() - t0
    text = sample.decode("utf-8", errors="replace")
    safe = text[:240].replace("\n", "\\n")
    safe = safe.encode("ascii", "replace").decode("ascii")
    print(f"  [{name}] sample ({gen_secs:.1f}s for 240B):\n    {safe}", flush=True)

    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"{args.tag}_{name}.pt")
    torch.save({"state": best_state, "cfg": model.cfg(),
                "meta": {"tag": args.tag, "model": name, "steps": args.steps,
                         "lr": args.lr, "batch": args.batch, "W": args.W,
                         "P": args.P if name == "patch" else None,
                         "seed": args.seed, "best_valid_bpb": best_bpb,
                         "test_bpb": test_b, "ts": now()}}, ckpt_path)
    print(f"  [{name}] ckpt -> {ckpt_path}", flush=True)

    row = {"exp": "exp10_byte", "tag": args.tag, "model": name, "params": n_params,
           "cfg": model.cfg(), "steps": args.steps, "lr": args.lr,
           "warmup": args.warmup, "batch": args.batch, "W": args.W,
           "P": args.P if name == "patch" else None, "seed": args.seed,
           "evals": evals, "best_valid_bpb": round(best_bpb, 4),
           "test_bpb": round(test_b, 4), "step_med_ms": round(med_ms, 2),
           "train_min": round(train_secs / 60, 2),
           "bytes_per_s": round(args.steps * args.batch * args.W / max(train_secs, 1e-9)),
           "gen_sec_240": round(gen_secs, 2), "sample": text,
           "ts": now()}
    append_jsonl(RESULTS, row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p2")
    ap.add_argument("--which", default="patch,flat")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--W", type=int, default=512)
    ap.add_argument("--P", type=int, default=4)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--eval_windows", type=int, default=64)
    ap.add_argument("--test_windows", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gen_bytes", type=int, default=240, help="(bilgi) demo üretim boyu sabit 240")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_bytes = load_bytes(os.path.join(DATA_DIR, "bytes_wt2_train.pt"))
    valid_bytes = load_bytes(os.path.join(DATA_DIR, "bytes_wt2_valid.pt"))
    test_bytes = load_bytes(os.path.join(DATA_DIR, "bytes_wt2_test.pt"))
    print(f"[data] train={len(train_bytes)} valid={len(valid_bytes)} "
          f"test={len(test_bytes)} bytes", flush=True)

    for name in args.which.split(","):
        name = name.strip()
        if name:
            t0 = time.time()
            train_one(name, args, device, train_bytes, valid_bytes, test_bytes)
            print(f"[{args.tag}/{name}] done in {(time.time()-t0)/60:.1f} min\n", flush=True)


if __name__ == "__main__":
    main()
