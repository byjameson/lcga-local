"""EXP60: does content-based block selection ever pay? A needle-in-a-haystack probe.

Plain language-modelling perplexity cannot reward looking far away: the next token
almost never depends on a distant block, so a loss-minimising router correctly
ignores distant content (see LCGA_FINDINGS.md). This probe creates the one situation
where distant content *does* matter: a unique fact is planted at a known depth and
the model is asked to recall it.

For each (T, depth, policy) we measure teacher-forced CE on the answer tokens and
exact-match accuracy. Policies: dense, learned top-k, budget-matched static window,
random, and the 2-block local prior.

Usage:
  python experiments/exp60_needle.py --ckpt checkpoints/abl_diag_best.pt \
      --Ts 2048,8192 --depths 0.1,0.5,0.9 --n_prompts 4
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from lcsa.data import load_tokens
from lcsa.harness import load_model
from lcsa.lora import apply_lora, merge_lora
from lcsa.metrics import append_jsonl, now
from lcsa.model import patch_model, set_selection_mode, set_sparse_enabled

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "Qwen2.5-0.5B")
DATA_DIR = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results", "results.jsonl")
DEV = "cuda"


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
def score_prompt(model, tok, ids, answer_len):
    """CE over the answer tokens, using only the last (answer_len+1) logits."""
    keep = answer_len + 1
    logits = model(input_ids=ids, logits_to_keep=keep).logits[0, :-1].float()
    tgt = ids[0, -answer_len:]
    ce = F.cross_entropy(logits, tgt, reduction="mean").item()
    exact = int((logits.argmax(-1) == tgt).all())
    return ce, exact


def build_ids(tok, filler, T, depth, code, question):
    """filler + fact at `depth` + question + answer, truncated to T + answer."""
    fact = tok(f" The secret code is {code}.", add_special_tokens=False).input_ids
    ans = tok(f" {code}", add_special_tokens=False).input_ids
    q = tok(question, add_special_tokens=False).input_ids
    budget = max(8, T - len(fact) - len(q) - len(ans))
    pos = int(budget * depth)
    body = filler[:pos] + fact + filler[pos:budget]
    ids = body + q + ans
    return torch.tensor([ids], dtype=torch.long, device=DEV), len(ans)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--Ts", default="2048,8192")
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--n_prompts", type=int, default=4)
    ap.add_argument("--tag", default="needle")
    ap.add_argument("--policies", default="dense,topk,window,random,local,oracle",
                    help="oracle = top-m by TRUE dense block mass (ceiling for any router)")
    ap.add_argument("--v2", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    model = load_model(MODEL_DIR, device=DEV)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    meta = torch.load(args.ckpt, map_location="cpu", weights_only=False).get("meta", {})
    margs = meta.get("args", {})
    wrappers = patch_model(model, budget=margs.get("budget", 0.25),
                           n_concepts=margs.get("n_concepts", 64),
                           graph_rank=margs.get("graph_rank", 16),
                           dynamic=margs.get("dynamic", False),
                           content_rank=margs.get("content_rank", 0),
                           rel_bias=not margs.get("no_rel_bias", False),
                           capture_dense=False, use_coarse=False, null_init=-1e9,
                           recall_gain=args.v2)
    if args.v2:
        apply_lora(model, r_attn=margs.get("lora_attn", 16), r_mlp=margs.get("lora_mlp", 0))
    load_ckpt(model, args.ckpt)   # AFTER patch_model, otherwise nothing is loaded
    for p in model.parameters():
        p.requires_grad_(False)
    for w in wrappers.values():
        w.null_logit.data.fill_(-1e9)
    if args.merge:
        merge_lora(model)

    toks = load_tokens(os.path.join(DATA_DIR, "wt2_train.pt"))
    filler = toks[: max(40000, 20000)].tolist()
    question = " Question: What is the secret code? Answer:"

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    out = {"exp": "exp60_needle", "tag": args.tag, "ckpt": args.ckpt,
           "ts": now(), "results": {}}
    rng = random.Random(0)

    for T in [int(t) for t in args.Ts.split(",")]:
        for depth in [float(d) for d in args.depths.split(",")]:
            ce_acc = {p: [] for p in policies}
            ex_acc = {p: [] for p in policies}
            for _ in range(args.n_prompts):
                code = str(rng.randrange(1000, 9999))
                ids, ans_len = build_ids(tok, filler, T, depth, code, question)
                for p in policies:
                    if p == "dense":
                        set_sparse_enabled(False)
                    else:
                        set_sparse_enabled(True)
                        set_selection_mode("topk" if p == "topk" else p)
                    try:
                        ce, ex = score_prompt(model, tok, ids, ans_len)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        ce, ex = float("nan"), 0
                    ce_acc[p].append(ce)
                    ex_acc[p].append(ex)
                del ids
                torch.cuda.empty_cache()
            set_sparse_enabled(False)
            set_selection_mode("topk")
            row = {p: {"ce": sum(ce_acc[p]) / len(ce_acc[p]),
                       "exact": sum(ex_acc[p]) / len(ex_acc[p])} for p in policies}
            out["results"][f"T{T}_d{depth}"] = row
            line = "  ".join(f"{p}:{row[p]['ce']:.3f}/{row[p]['exact']:.2f}" for p in policies)
            print(f"[T={T} depth={depth:.1f}] {line}", flush=True)

    with open(os.path.join(ROOT, "results", f"exp60_{args.tag}.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    append_jsonl(RESULTS, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
