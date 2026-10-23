"""EXP62: does a retrieval-trained router generalise, or does it match a phrase?

Three conditions, same evaluator for all policies:
  seen      one fact, the phrasing used in training
  unseen    one fact, a phrasing/question never seen in training
  two_facts two planted facts (secret + backup) at different depths; the question
            asks for the backup one, so the router must pick the *queried* block,
            not merely "a fact-looking block"

Policies: dense, learned top-k, budget-matched window, oracle (top-m by true dense
block mass). For top-k/oracle we also report the fraction of layers selecting the
*target* fact block at the query row, read from the capture (no extra cost).

Usage:
  python experiments/exp62_needle_gen.py --ckpt checkpoints/needle_ret_c64_final.pt \
      --seq 2048 --depths 0.1,0.5,0.9 --n_prompts 4
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
from lcsa.metrics import append_jsonl, now
from lcsa.model import (begin_capture, end_capture, patch_model, set_selection_mode,
                        set_sparse_enabled)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "Qwen2.5-0.5B")
DATA_DIR = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results", "results.jsonl")
DEV = "cuda"

SEEN = (" The secret code is {c}.", " Question: What is the secret code? Answer:")
UNSEEN = (" Account recovery token: {c}.",
          " Question: What is the account recovery token? Answer:")
BACKUP = (" The backup key is {c}.", " Question: What is the backup key? Answer:")


def load_ckpt(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for i, layer in enumerate(model.model.layers):
        own = dict(layer.self_attn.named_parameters())
        for n, t in ckpt["state"][i].items():
            if n in own:
                own[n].data.copy_(t.to(own[n].device))
    return ckpt.get("meta", {})


def build(tok, filler, T, facts, question, block=64):
    """facts: list of (template, code, depth in [0,1]).

    Returns ids, target block (the LAST fact in the list), answer length. Facts are
    spliced into a single body of length T - len(q) - len(ans); they are placed in
    ascending depth order so earlier splices do not shift later positions.
    """
    fact_tok = [tok(t.format(c=c), add_special_tokens=False).input_ids
                for t, c, _ in facts]
    ans = tok(f" {facts[-1][1]}", add_special_tokens=False).input_ids
    q = tok(question, add_special_tokens=False).input_ids
    body_total = T - len(q) - len(ans)
    order = sorted(range(len(facts)), key=lambda i: facts[i][2])
    out, cur, target_pos, target = [], 0, None, order[-1]
    for i in order:
        depth = facts[i][2]
        p = min(int(body_total * depth), body_total - len(fact_tok[i]) - 8)
        p = max(p, cur + 1)
        out.extend(filler[cur:p])
        out.extend(fact_tok[i])
        cur = p
        if i == target:
            target_pos = len(out) - 1
    out.extend(filler[cur:body_total])
    body = out[:body_total]
    assert target_pos is not None and target_pos < len(body), (target_pos, len(body))
    ids = body + q + ans
    assert len(ids) == T, (len(ids), T)
    return (torch.tensor([ids], dtype=torch.long, device=DEV),
            min(target_pos // block, (T - len(ans) - 1) // block), len(ans))


@torch.no_grad()
def score(model, ids, ans_len):
    logits = model(input_ids=ids, logits_to_keep=ans_len + 1).logits[0, :-1].float()
    tgt = ids[0, -ans_len:]
    return F.cross_entropy(logits, tgt).item(), int((logits.argmax(-1) == tgt).all())


@torch.no_grad()
def hit_rate(model, ids, ans_len, fact_block, block=64):
    """Fraction of layers whose selection at the query row contains fact_block."""
    begin_capture()
    model(input_ids=ids, logits_to_keep=ans_len + 1)
    caps = end_capture()
    row = (ids.shape[1] - 1) // block
    hits = 0
    for cap in caps:
        idx = [int(v) for v in cap["idx"][0, row] if int(v) >= 0]
        hits += int(fact_block in idx)
    return hits / max(1, len(caps))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--n_prompts", type=int, default=4)
    ap.add_argument("--tag", default="gen")
    ap.add_argument("--v2", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    model = load_model(MODEL_DIR, device=DEV)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    meta = torch.load(args.ckpt, map_location="cpu", weights_only=False).get("meta", {})
    margs = meta.get("args", {})
    patch_model(model, budget=margs.get("budget", 0.25),
                n_concepts=margs.get("n_concepts", 64),
                graph_rank=margs.get("graph_rank", 16),
                dynamic=margs.get("dynamic", False), content_rank=margs.get("content_rank", 0),
                rel_bias=not margs.get("no_rel_bias", False),
                capture_dense=False, use_coarse=False, null_init=-1e9,
                recall_gain=args.v2)
    load_ckpt(model, args.ckpt)              # AFTER patch_model
    for p in model.parameters():
        p.requires_grad_(False)

    filler = load_tokens(os.path.join(DATA_DIR, "wt2_train.pt"))[:60000].tolist()
    policies = ["dense", "topk", "window", "oracle"]
    out = {"exp": "exp62_needle_gen", "tag": args.tag, "ckpt": args.ckpt,
           "seq": args.seq, "ts": now(), "results": {}}
    rng = random.Random(0)

    for depth in [float(d) for d in args.depths.split(",")]:
        d_secret = 1.0 - depth
        if abs(d_secret - depth) < 0.2:      # keep the two facts apart
            d_secret = 0.2 if depth > 0.5 else 0.8
        conds = {
            "seen": [(SEEN[0], str(rng.randrange(1000, 9999)), depth)],
            "unseen": [(UNSEEN[0], str(rng.randrange(1000, 9999)), depth)],
            "two_facts": [(SEEN[0], str(rng.randrange(1000, 9999)), d_secret),
                          (BACKUP[0], str(rng.randrange(1000, 9999)), depth)],
        }
        for name, facts in conds.items():
            question = SEEN[1] if name == "seen" else (
                UNSEEN[1] if name == "unseen" else BACKUP[1])
            acc = {p: {"ce": [], "exact": [], "hit": []} for p in policies}
            for k in range(args.n_prompts):
                code = str(rng.randrange(1000, 9999))
                facts_k = [(t, code if i == len(facts) - 1 else c, d)
                           for i, (t, c, d) in enumerate(facts)]
                ids, fact_block, ans_len = build(tok, filler, args.seq, facts_k, question)
                for p in policies:
                    if p == "dense":
                        set_sparse_enabled(False)
                    else:
                        set_sparse_enabled(True)
                        set_selection_mode(p)
                    try:
                        ce, ex = score(model, ids, ans_len)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        ce, ex = float("nan"), 0
                    acc[p]["ce"].append(ce)
                    acc[p]["exact"].append(ex)
                    if p in ("topk", "oracle"):
                        acc[p]["hit"].append(hit_rate(model, ids, ans_len, fact_block))
                del ids
                torch.cuda.empty_cache()
            set_sparse_enabled(False)
            set_selection_mode("topk")
            row = {p: {"ce": sum(acc[p]["ce"]) / len(acc[p]["ce"]),
                       "exact": sum(acc[p]["exact"]) / len(acc[p]["exact"]),
                       "hit": (sum(acc[p]["hit"]) / len(acc[p]["hit"])) if acc[p]["hit"] else None}
                   for p in policies}
            out["results"][f"d{depth}_{name}"] = row
            line = "  ".join(
                f"{p}:{row[p]['ce']:.3f}/{row[p]['exact']:.2f}"
                + (f"/h{row[p]['hit']:.2f}" if row[p]["hit"] is not None else "")
                for p in policies)
            print(f"[d={depth:.1f} {name:9s}] {line}", flush=True)

    with open(os.path.join(ROOT, "results", f"exp62_{args.tag}.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    append_jsonl(RESULTS, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
