"""Check what a trained router ACTUALLY selects at eval time.

Training logs a high p_fact, yet exp60 shows no retrieval. This script loads the
checkpoint exactly like exp60 does, builds both a trainer-style and an exp60-style
needle example, and prints -- per layer, at the answer query row -- the probability
on the fact block and whether the *selected* idx contains it. It separates
"probs are good but selection differs" from "probs are bad at eval".

Usage:
  python experiments/exp61_eval_check.py --ckpt checkpoints/needle_ret_c64_final.pt --seq 1024 --depth 0.5
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from lcsa.data import load_tokens
from lcsa.harness import load_model
from lcsa.model import (begin_capture, end_capture, patch_model, set_selection_mode,
                        set_sparse_enabled)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "Qwen2.5-0.5B")
DATA_DIR = os.path.join(ROOT, "data")
DEV = "cuda"
QUESTION = " Question: What is the secret code? Answer:"


def load_ckpt(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for i, layer in enumerate(model.model.layers):
        own = dict(layer.self_attn.named_parameters())
        for n, t in ckpt["state"][i].items():
            if n in own:
                own[n].data.copy_(t.to(own[n].device))
    return ckpt.get("meta", {})


def build(tok, filler, T, depth, style, block=64):
    if style == "exp60":                       # body_len = T - fact - q - ans
        code = "7311"
        fact = tok(f" The secret code is {code}.", add_special_tokens=False).input_ids
        ans = tok(f" {code}", add_special_tokens=False).input_ids
        q = tok(QUESTION, add_special_tokens=False).input_ids
        body_len = T - len(fact) - len(q) - len(ans)
        pos = int(body_len * depth)
        body = filler[:pos] + fact + filler[pos:body_len]
        ids = body + q + ans
    else:                                      # trainer style (same arithmetic)
        code = "7311"
        fact = tok(f" The secret code is {code}.", add_special_tokens=False).input_ids
        ans = tok(f" {code}", add_special_tokens=False).input_ids
        q = tok(QUESTION, add_special_tokens=False).input_ids
        body_len = max(8, T - len(fact) - len(q) - len(ans))
        pos = int(body_len * depth)
        body = filler[:pos] + fact + filler[pos:body_len]
        ids = body + q + ans
    assert len(ids) == T, (len(ids), T)
    return torch.tensor([ids], dtype=torch.long, device=DEV), (pos + len(fact) - 1) // block, len(ans)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--depth", type=float, default=0.5)
    ap.add_argument("--style", default="exp60", choices=("exp60", "train"))
    args = ap.parse_args()

    from transformers import AutoTokenizer

    model = load_model(MODEL_DIR, device=DEV)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    meta = torch.load(args.ckpt, map_location="cpu", weights_only=False).get("meta", {})
    margs = meta.get("args", {})
    print(f"[meta] { {k: margs[k] for k in ('loss', 'content_rank', 'budget', 'seq') if k in margs} }", flush=True)
    patch_model(model, budget=margs.get("budget", 0.25),
                n_concepts=margs.get("n_concepts", 64),
                graph_rank=margs.get("graph_rank", 16),
                dynamic=margs.get("dynamic", False),
                content_rank=margs.get("content_rank", 0),
                rel_bias=not margs.get("no_rel_bias", False),
                capture_dense=False, use_coarse=False, null_init=-1e9)
    load_ckpt(model, args.ckpt)   # AFTER patch_model

    filler = load_tokens(os.path.join(DATA_DIR, "wt2_train.pt"))[:60000].tolist()
    x, fact_block, ans_len = build(tok, filler, args.seq, args.depth, args.style)
    NB = args.seq // 64
    rows = [(args.seq - 1) // 64]

    # dense CE first
    set_sparse_enabled(False)
    with torch.no_grad():
        logits = model(input_ids=x, logits_to_keep=ans_len + 1).logits[0, :-1].float()
    tgt = x[0, -ans_len:]
    print(f"dense CE={F.cross_entropy(logits, tgt).item():.3f} "
          f"exact={int((logits.argmax(-1) == tgt).all())} fact_block={fact_block}/{NB}",
          flush=True)

    set_sparse_enabled(True)
    set_selection_mode("topk")
    begin_capture()
    with torch.no_grad():
        logits = model(input_ids=x, logits_to_keep=ans_len + 1).logits[0, :-1].float()
    caps = end_capture()
    print(f"sparse CE={F.cross_entropy(logits, tgt).item():.3f} "
          f"exact={int((logits.argmax(-1) == tgt).all())}", flush=True)
    print(" layer  p_fact  in_idx  idx(answer row)")
    for L, cap in enumerate(caps):
        pr = cap["probs"][0, rows[0]]
        idx = [int(v) for v in cap["idx"][0, rows[0]] if int(v) >= 0]
        hit = fact_block in idx
        if hit or L in (0, 6, 11, 17, 23):
            print(f"  L{L:<3d}  {float(pr[fact_block]):.3f}   {int(hit)}     {idx}", flush=True)
    n_hit = sum(1 for L, cap in enumerate(caps)
                if fact_block in [int(v) for v in cap["idx"][0, rows[0]] if int(v) >= 0])
    print(f"[summary] layers selecting the fact block at the answer row: {n_hit}/{len(caps)}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
