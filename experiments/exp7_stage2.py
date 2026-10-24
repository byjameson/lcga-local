"""EXP7: stage-2 adaptation — LoRA + LM loss with the sparse attention active.

Per step (batch=1, T=1024):
  pass 1 (no_grad, dense_only): dense teacher block-mass + output for every layer
  pass 2 (grad, stage2): exact-only sparse forward via the differentiable gather
      reference, output recall-gain scaling, losses:
        loss = CE + lam_kl * KL(dense_mass || router probs) + lam_mse * MSE(student, dense_out)

Trainable: LoRA (attn q/k/v/o, optionally MLP) + router + gain parameters.
The base model is frozen; use lcsa.lora.merge_lora() after training so inference
runs at zero extra cost.

--dense_control runs the same LoRA training with the ORIGINAL dense attention
(no sparsity) to separate "LoRA domain adaptation" from "sparse compensation".
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from lcsa.data import load_tokens, chunks_from_tokens, TrainSampler
from lcsa.harness import load_model, quick_ppl
from lcsa.lora import apply_lora, count_lora_params, lora_parameters
from lcsa.metrics import append_jsonl, now, routing_diagnostics
from lcsa.model import (begin_capture, count_router_params, end_capture, get_aux_losses,
                        patch_model, router_parameters, set_gather_checkpoint,
                        set_sparse_enabled, set_stage2_mode)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "Qwen2.5-0.5B")
DATA_DIR = os.path.join(ROOT, "data")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
RESULTS = os.path.join(ROOT, "results", "results.jsonl")
DEV = "cuda"


def save_ckpt(model, path, meta):
    state = {}
    for i, layer in enumerate(model.model.layers):
        w = layer.self_attn
        state[i] = {n: p.detach().cpu().clone()
                    for n, p in w.named_parameters() if p.requires_grad}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"state": state, "meta": meta}, path)


def load_ckpt(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for i, layer in enumerate(model.model.layers):
        w = layer.self_attn
        own = dict(w.named_parameters())
        for n, t in ckpt["state"][i].items():
            if n in own:
                own[n].data.copy_(t.to(own[n].device))
    return ckpt.get("meta", {})


@torch.no_grad()
def dev_eval(model, dev_chunks, device, max_chunks=16, recall_chunks=2):
    res = {}
    set_sparse_enabled(False)
    res["dense_ppl"] = quick_ppl(model, dev_chunks, device=device,
                                 max_chunks=max_chunks)["ppl"]
    set_sparse_enabled(True)
    res["sparse_ppl"] = quick_ppl(model, dev_chunks, device=device,
                                  max_chunks=max_chunks)["ppl"]
    acc = []
    for i in range(min(recall_chunks, len(dev_chunks))):
        begin_capture()
        model(input_ids=dev_chunks[i].to(device).unsqueeze(0))
        cap = end_capture()
        diag = routing_diagnostics(cap)
        acc.append([d["recall_mean"] for d in diag])
    set_sparse_enabled(False)
    if acc:
        t = torch.tensor(acc)
        res["recall0"] = float(t[:, 0].mean())
        res["recall_last"] = float(t[:, -1].mean())
        res["recall_mean"] = float(t.mean())
    return res


def _ce_chunk(logits_chunk, lab):
    return F.cross_entropy(logits_chunk.float().transpose(1, 2), lab, reduction="sum")


def ce_loss_chunked(logits, labels, chunk=256):
    T = labels.shape[1]
    total = None
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        term = torch.utils.checkpoint.checkpoint(
            _ce_chunk, logits[:, s:e], labels[:, s:e], use_reentrant=False)
        total = term if total is None else total + term
    return total / T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr_router", type=float, default=3e-4)
    ap.add_argument("--lr_lora", type=float, default=1e-3)
    ap.add_argument("--lam_kl", type=float, default=0.5)
    ap.add_argument("--lam_mse", type=float, default=0.3)
    ap.add_argument("--budget", type=float, default=0.25)
    ap.add_argument("--lora_attn", type=int, default=16)
    ap.add_argument("--lora_mlp", type=int, default=0)
    ap.add_argument("--tag", type=str, default="v2")
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--init_ckpt", type=str, default="checkpoints/v1_best.pt")
    ap.add_argument("--dense_control", action="store_true")
    ap.add_argument("--ckpt_gather", action="store_true",
                    help="recompute the gather attention in backward (saves ~1.4GB)")
    ap.add_argument("--tf32", action="store_true",
                    help="allow TF32 fp32 matmuls (faster reference paths on Ampere)")
    ap.add_argument("--mem_profile", action="store_true")
    args = ap.parse_args()

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    print(f"[{now()}] exp7 tag={args.tag} args={vars(args)}", flush=True)

    model = load_model(MODEL_DIR, device=DEV)
    wrappers = patch_model(model, budget=args.budget, block=64, n_concepts=64,
                           graph_rank=16, capture_dense=True, use_coarse=False,
                           recall_gain=True, null_init=-1e9)
    set_gather_checkpoint(args.ckpt_gather)
    if args.init_ckpt:
        load_ckpt(model, args.init_ckpt)
        print(f"[model] warm-started from {args.init_ckpt}", flush=True)
    if args.lora_attn > 0 or args.lora_mlp > 0:
        apply_lora(model, r_attn=args.lora_attn, r_mlp=args.lora_mlp)
    for w in wrappers.values():
        w.gate.requires_grad_(False)
        # v1 checkpoints carry null_logit=-3 (the old buggy scaling); force it OFF.
        w.null_logit.data.fill_(-1e9)
        w.null_logit.requires_grad_(False)
        w.coarse_null.requires_grad_(False)
        w.lambda_router.requires_grad_(False)

    router_ps = [p for p in router_parameters(model) if p.requires_grad]
    lora_ps = list(lora_parameters(model))
    n_router = sum(p.numel() for p in router_ps)
    n_lora = count_lora_params(model)
    print(f"[model] patched {len(wrappers)} layers | trainable router/gain "
          f"{n_router/1e6:.3f}M + lora {n_lora/1e6:.3f}M", flush=True)

    tokens = load_tokens(os.path.join(DATA_DIR, "wt2_train.pt"))
    sampler = TrainSampler(tokens, args.seq)
    dev_tokens = load_tokens(os.path.join(DATA_DIR, "wt2_test.pt"))
    dev_chunks = chunks_from_tokens(dev_tokens, 1024, max_chunks=16)

    opt = torch.optim.AdamW([
        {"params": router_ps, "lr": args.lr_router},
        {"params": lora_ps, "lr": args.lr_lora},
    ], betas=(0.9, 0.95), weight_decay=0.0)

    def lr_at(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    best = float("inf")
    t_start = time.time()
    run = {"ce": 0.0, "kl": 0.0, "mse": 0.0, "n": 0, "t": time.time()}

    def do_eval(step):
        nonlocal best
        model.eval()
        r = dev_eval(model, dev_chunks, DEV)
        gains = [float(torch.exp(w.gain_logit).detach().mean()) for w in wrappers.values()]
        alphas = [float(w.gain_alpha.detach().mean()) for w in wrappers.values()]
        r["gain_mean"] = sum(gains) / len(gains)
        r["alpha_mean"] = sum(alphas) / len(alphas)
        r["step"] = step
        print(f"  [eval @{step}] dense={r['dense_ppl']:.3f} sparse={r['sparse_ppl']:.3f} "
              f"(delta={r['sparse_ppl']-r['dense_ppl']:+.3f}) "
              f"recall0={r.get('recall0', 0):.3f} recall_last={r.get('recall_last', 0):.3f} "
              f"gain={r['gain_mean']:.3f} alpha={r['alpha_mean']:.3f}", flush=True)
        append_jsonl(RESULTS, {"exp": "exp7_eval", "tag": args.tag, **r, "ts": now()})
        if not args.dense_control and r["sparse_ppl"] < best:
            best = r["sparse_ppl"]
            save_ckpt(model, os.path.join(CKPT_DIR, f"{args.tag}_best.pt"),
                      {"args": vars(args), "step": step, "dev_sparse_ppl": r["sparse_ppl"],
                       "dev_dense_ppl": r["dense_ppl"]})
            print(f"  [ckpt] saved best (sparse ppl={best:.3f})", flush=True)
        if args.dense_control and r["dense_ppl"] < best:
            best = r["dense_ppl"]
            save_ckpt(model, os.path.join(CKPT_DIR, f"{args.tag}_best.pt"),
                      {"args": vars(args), "step": step, "dev_dense_ppl": r["dense_ppl"]})
            print(f"  [ckpt] saved best (dense ppl={best:.3f})", flush=True)
        model.train()

    model.train()
    do_eval(0)

    for step in range(1, args.steps + 1):
        prof = {}
        t0 = time.time()
        x, y = sampler.batch(args.batch)
        x, y = x.to(DEV), y.to(DEV)
        kl_v = mse_v = 0.0
        if args.dense_control:
            set_sparse_enabled(False)
            out = model(input_ids=x, use_cache=False)
            ce = ce_loss_chunked(out.logits, y)
            loss = ce
        else:
            set_stage2_mode(False, dense_only=True)      # pass 1: dense targets
            set_sparse_enabled(True)
            with torch.no_grad():
                model.model(input_ids=x, use_cache=False)
            prof["pass1_s"] = time.time() - t0
            prof["pass1_gb"] = torch.cuda.memory_allocated() / 1e9
            t0 = time.time()
            set_stage2_mode(True, keep_targets=True)     # pass 2: sparse student
            out = model(input_ids=x, use_cache=False)
            prof["p2fwd_s"] = time.time() - t0
            prof["p2fwd_gb"] = torch.cuda.memory_allocated() / 1e9
            t0 = time.time()
            ce = ce_loss_chunked(out.logits, y)
            aux = get_aux_losses()
            n_aux = max(1, len(aux))
            kl = sum(a[0] for a in aux) / n_aux
            mse = sum(a[1] for a in aux) / n_aux
            kl_v = float(kl.detach())
            mse_v = float(mse.detach())
            loss = ce + args.lam_kl * kl + args.lam_mse * mse
            prof["ce_s"] = time.time() - t0
            prof["ce_gb"] = torch.cuda.memory_allocated() / 1e9
            t0 = time.time()
        loss.backward()
        prof["bwd_s"] = time.time() - t0
        prof["bwd_gb"] = torch.cuda.memory_allocated() / 1e9
        t0 = time.time()
        gnorm = torch.nn.utils.clip_grad_norm_(router_ps + lora_ps, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        set_stage2_mode(False)
        set_sparse_enabled(False)
        prof["opt_s"] = time.time() - t0
        del out
        if args.mem_profile:
            prof["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
            parts = " ".join(f"{k}={v:.2f}" for k, v in prof.items())
            print(f"  [prof step {step}] {parts}", flush=True)
            torch.cuda.reset_peak_memory_stats()
        run["ce"] += float(ce)
        run["kl"] += kl_v
        run["mse"] += mse_v
        run["n"] += 1

        if step % args.log_every == 0:
            dt = time.time() - run["t"]
            lrs = sched.get_last_lr()
            print(f"[step {step}/{args.steps}] ce={run['ce']/run['n']:.4f} "
                  f"kl={run['kl']/run['n']:.4f} mse={run['mse']/run['n']:.5f} "
                  f"gnorm={float(gnorm):.2f} lr={lrs[0]:.2e}/{lrs[-1]:.2e} "
                  f"{dt/run['n']*1000:.0f}ms/step vram={torch.cuda.max_memory_allocated()/1e9:.2f}GB",
                  flush=True)
            append_jsonl(RESULTS, {"exp": "exp7_train", "tag": args.tag, "step": step,
                                   "ce": run["ce"]/run["n"], "kl": run["kl"]/run["n"],
                                   "mse": run["mse"]/run["n"], "gnorm": float(gnorm),
                                   "lr_router": lrs[0], "lr_lora": lrs[-1], "ts": now()})
            run = {"ce": 0.0, "kl": 0.0, "mse": 0.0, "n": 0, "t": time.time()}

        if (args.eval_every > 0 and step % args.eval_every == 0) or step == args.steps:
            do_eval(step)

    save_ckpt(model, os.path.join(CKPT_DIR, f"{args.tag}_final.pt"),
              {"args": vars(args), "step": args.steps, "best_dev": best})
    total = time.time() - t_start
    print(f"[{now()}] exp7 done in {total/60:.1f} min. best dev ppl={best:.3f}", flush=True)
    append_jsonl(RESULTS, {"exp": "exp7_done", "tag": args.tag, "best_dev_ppl": best,
                           "minutes": total / 60, "ts": now()})


if __name__ == "__main__":
    main()
