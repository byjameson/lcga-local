"""EXP2: train LCGA routers with local distillation losses (dense block-mass KL + attn-output MSE)."""
import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from lcsa.data import load_tokens, TrainSampler, chunks_from_tokens
from lcsa.harness import load_model, quick_ppl
from lcsa.metrics import append_jsonl, now, routing_diagnostics
from lcsa.model import (patch_model, router_parameters, set_sparse_enabled,
                        begin_capture, end_capture, count_router_params)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "Qwen2.5-0.5B")
DATA_DIR = os.path.join(ROOT, "data")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
RESULTS = os.path.join(ROOT, "results", "results.jsonl")
DEV = "cuda"


def save_ckpt(model, path, meta):
    base = model.model
    state = {}
    for i, layer in enumerate(base.layers):
        w = layer.self_attn
        sd = {n: p.detach().cpu().clone() for n, p in w.named_parameters() if not n.startswith("orig.")}
        state[i] = sd
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state": state, "meta": meta}, path)


def load_ckpt(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    base = model.model
    for i, layer in enumerate(base.layers):
        w = layer.self_attn
        sd = ckpt["state"][i]
        own = dict(w.named_parameters())
        for n, t in sd.items():
            own[n].data.copy_(t.to(own[n].device))
    return ckpt.get("meta", {})


@torch.no_grad()
def quick_dev_eval(model, chunks, device, max_chunks=32):
    set_sparse_enabled(False)
    dense = quick_ppl(model, chunks, device=device, max_chunks=max_chunks)
    set_sparse_enabled(True)
    sparse = quick_ppl(model, chunks, device=device, max_chunks=max_chunks)
    set_sparse_enabled(False)
    return dense, sparse


@torch.no_grad()
def quick_recall(model, chunks, device, max_chunks=4):
    set_sparse_enabled(True)
    acc = []
    for i in range(min(max_chunks, len(chunks))):
        begin_capture()
        model(input_ids=chunks[i].to(device).unsqueeze(0))
        cap = end_capture()
        diag = routing_diagnostics(cap)
        acc.append([d["recall_mean"] for d in diag])
    set_sparse_enabled(False)
    if not acc:
        return []
    t = torch.tensor(acc)  # (n_chunks, n_layers)
    return t.mean(0).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--lam_kl", type=float, default=0.5)
    ap.add_argument("--budget", type=float, default=0.25)
    ap.add_argument("--n_concepts", type=int, default=64)
    ap.add_argument("--graph_rank", type=int, default=16)
    ap.add_argument("--dynamic", action="store_true")
    ap.add_argument("--no_rel_bias", action="store_true")
    ap.add_argument("--content_rank", type=int, default=0)
    ap.add_argument("--tag", type=str, default="v1")
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--init_ckpt", type=str, default="")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tok_meta = {"args": vars(args)}
    print(f"[{now()}] exp2 tag={args.tag} args={vars(args)}", flush=True)

    model = load_model(MODEL_DIR, device=DEV)
    wrappers = patch_model(model, budget=args.budget, block=64, n_concepts=args.n_concepts,
                           graph_rank=args.graph_rank, dynamic=args.dynamic,
                           content_rank=args.content_rank, capture_dense=True,
                           rel_bias=not args.no_rel_bias)
    params = list(router_parameters(model))
    n_params = count_router_params(model)
    print(f"[model] patched {len(wrappers)} layers, router params: {n_params/1e6:.3f}M", flush=True)
    if args.init_ckpt:
        load_ckpt(model, args.init_ckpt)
        print(f"[model] warm-started from {args.init_ckpt}", flush=True)
    torch_meta = {"router_params": n_params}
    torch_meta.update(tok_meta)

    tokens = load_tokens(os.path.join(DATA_DIR, "wt2_train.pt"))
    sampler = TrainSampler(tokens, args.seq)
    dev_tokens = load_tokens(os.path.join(DATA_DIR, "wt2_test.pt"))
    dev_chunks = chunks_from_tokens(dev_tokens, 1024, max_chunks=128)

    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    def lr_at(step):
        # LambdaLR expects a MULTIPLIER of base lr (returning an absolute lr here
        # would give an effective peak lr of args.lr**2 ~ 2.25e-6).
        if step < args.warmup:
            return (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    model.train()
    best_ppl = float("inf")
    t_start = time.time()
    run = {"mse": 0.0, "kl": 0.0, "n": 0, "t": time.time()}

    for step in range(1, args.steps + 1):
        x, y = sampler.batch(args.batch)
        x = x.to(DEV)
        set_sparse_enabled(True)          # training forward must run the LCGA path
        begin_capture()
        _ = model.model(input_ids=x, use_cache=False)   # skip lm_head (memory); sparse+grad
        captured = end_capture()
        loss = torch.zeros((), device=DEV)
        mse_acc, kl_acc = 0.0, 0.0
        for i, cap in enumerate(captured):
            mse = F.mse_loss(cap["student"], cap["dense"])
            kl = wrappers[i].router.kl_mass_loss(cap["mass"], cap["probs"])
            loss = loss + mse + args.lam_kl * kl
            mse_acc += mse.item()
            kl_acc += kl.item()
        loss = loss / len(captured)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        del captured

        run["mse"] += mse_acc / len(wrappers)
        run["kl"] += kl_acc / len(wrappers)
        run["n"] += 1

        if step % args.log_every == 0:
            dt = time.time() - run["t"]
            msg = (f"[step {step}/{args.steps}] loss={loss.item():.4f} "
                   f"mse={run['mse']/run['n']:.5f} kl={run['kl']/run['n']:.4f} "
                   f"gnorm={float(gnorm):.2f} lr={sched.get_last_lr()[0]:.2e} "
                   f"{dt/run['n']*1000:.0f}ms/step vram={torch.cuda.max_memory_allocated()/1e9:.2f}GB")
            print(msg, flush=True)
            append_jsonl(RESULTS, {"exp": "exp2_train", "tag": args.tag, "step": step,
                                   "loss": loss.item(), "mse": run["mse"]/run["n"],
                                   "kl": run["kl"]/run["n"], "gnorm": float(gnorm),
                                   "lr": sched.get_last_lr()[0], "ts": now()})
            run = {"mse": 0.0, "kl": 0.0, "n": 0, "t": time.time()}

        if step % args.eval_every == 0 or step == args.steps:
            dense, sparse = quick_dev_eval(model, dev_chunks, DEV, max_chunks=32)
            recall = quick_recall(model, dev_chunks, DEV, max_chunks=3)
            g = [round(float(torch.sigmoid(w.gate).detach().mean()), 3) for w in wrappers.values()]
            print(f"  [eval @{step}] dense_ppl={dense['ppl']:.3f} sparse_ppl={sparse['ppl']:.3f} "
                  f"(delta={sparse['ppl']-dense['ppl']:+.3f}) recall0={recall[0]:.3f} "
                  f"recall_last={recall[-1]:.3f} gate_mean={sum(g)/len(g):.3f}", flush=True)
            append_jsonl(RESULTS, {"exp": "exp2_eval", "tag": args.tag, "step": step,
                                   "dense_ppl": dense["ppl"], "sparse_ppl": sparse["ppl"],
                                   "recall": recall, "gate": g, "ts": now()})
            if sparse["ppl"] < best_ppl:
                best_ppl = sparse["ppl"]
                save_ckpt(model, os.path.join(CKPT_DIR, f"{args.tag}_best.pt"),
                          {"args": vars(args), "router_params": n_params, "step": step,
                           "dev_ppl": sparse["ppl"]})
                print(f"  [ckpt] saved best (ppl={best_ppl:.3f})", flush=True)
            model.train()

    save_ckpt(model, os.path.join(CKPT_DIR, f"{args.tag}_final.pt"),
              {"args": vars(args), "router_params": n_params, "dev_ppl": best_ppl})
    total = time.time() - t_start
    print(f"[{now()}] exp2 done in {total/60:.1f} min. best dev sparse ppl={best_ppl:.3f}", flush=True)
    append_jsonl(RESULTS, {"exp": "exp2_done", "tag": args.tag, "best_dev_ppl": best_ppl,
                           "minutes": total / 60, "ts": now()})


if __name__ == "__main__":
    main()
