"""EXP64: is the router's selection driven by the QUESTION, or just by "a fact"?

The same context contains two facts with distinct phrasings at known depths:
    The secret code is A.   ...   The backup key is B.
We then ask two different questions about the *same* context and record which fact
block the router selects at the query row. A genuine content router must follow the
question (diagonal); a "find anything fact-like" router would pick the same block
both times (constant column), and a position prior would ignore both.

Metrics per question: CE / exact on the answer, plus the hit rate for each fact
block (fraction of layers selecting it).

Usage:
  python experiments/exp64_query_swap.py --ckpt checkpoints/needle_para900_final.pt \
      --seq 2048 --depths 0.2,0.5,0.8 --n_prompts 4
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
RESULTS = os.path.join(ROOT, "results", "results.jsonl")
DEV = "cuda"

FACTS = [
    (" The secret code is {c}.", " Question: What is the secret code? Answer:", "secret"),
    (" The backup key is {c}.", " Question: What is the backup key? Answer:", "backup"),
]


def load_ckpt(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for i, layer in enumerate(model.model.layers):
        own = dict(layer.self_attn.named_parameters())
        for n, t in ckpt["state"][i].items():
            if n in own:
                own[n].data.copy_(t.to(own[n].device))
    return ckpt.get("meta", {})


def build(tok, filler, T, facts, q_idx, block=64):
    """facts: list of (template, code, depth, name). Ask facts[q_idx].

    Returns ids, block of each fact (same order), answer length.
    """
    assert all(len(f) == 4 for f in facts), facts
    ans = tok(f" {facts[q_idx][1]}", add_special_tokens=False).input_ids
    q = tok(FACTS[q_idx][1], add_special_tokens=False).input_ids
    body_total = T - len(q) - len(ans)
    order = sorted(range(len(facts)), key=lambda i: facts[i][2])
    out, cur, blocks = [], 0, {}
    for i in order:
        f = tok(facts[i][0].format(c=facts[i][1]), add_special_tokens=False).input_ids
        p = min(int(body_total * facts[i][2]), body_total - len(f) - 8)
        p = max(p, cur + 24)
        out.extend(filler[cur:p])
        out.extend(f)
        cur = p
        blocks[i] = min((len(out) - 2) // block,      # -2: the last token is ".", the
                                                      # code digits are just before it
                        (body_total + len(q) - 2) // block)
    out.extend(filler[cur:body_total])
    ids = out[:body_total] + q + ans
    assert len(ids) == T, (len(ids), T)
    return (torch.tensor([ids], dtype=torch.long, device=DEV),
            [blocks[i] for i in range(len(facts))], len(ans))


@torch.no_grad()
def score(model, ids, ans_len):
    logits = model(input_ids=ids, logits_to_keep=ans_len + 1).logits[0, :-1].float()
    tgt = ids[0, -ans_len:]
    return (F.cross_entropy(logits, tgt).item(),
            int((logits.argmax(-1) == tgt).all()))


@torch.no_grad()
def hits(model, ids, ans_len, blocks, block=64):
    begin_capture()
    model(input_ids=ids, logits_to_keep=ans_len + 1)
    caps = end_capture()
    row = (ids.shape[1] - 1) // block
    out = []
    for b in blocks:
        n = sum(1 for cap in caps
                if b in [int(v) for v in cap["idx"][0, row] if int(v) >= 0])
        out.append(n / max(1, len(caps)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--depths", default="0.2,0.5,0.8")
    ap.add_argument("--n_prompts", type=int, default=4)
    ap.add_argument("--tag", default="swap")
    ap.add_argument("--model", default="", help="base model dir (default: from ckpt meta)")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    meta = torch.load(args.ckpt, map_location="cpu", weights_only=False).get("meta", {})
    margs = meta.get("args", {})
    model_dir = args.model or margs.get("model", "models/Qwen2.5-0.5B")
    if not os.path.isabs(model_dir):
        model_dir = os.path.join(ROOT, model_dir)
    model = load_model(model_dir, device=DEV)
    tok = AutoTokenizer.from_pretrained(model_dir)
    patch_model(model, budget=margs.get("budget", 0.25),
                n_concepts=margs.get("n_concepts", 64),
                graph_rank=margs.get("graph_rank", 16),
                dynamic=margs.get("dynamic", False),
                content_rank=margs.get("content_rank", 0),
                rel_bias=not margs.get("no_rel_bias", False),
                capture_dense=False, use_coarse=False, null_init=-1e9)
    load_ckpt(model, args.ckpt)
    for p in model.parameters():
        p.requires_grad_(False)
    set_sparse_enabled(True)      # the whole point is the learned sparse selection

    filler = load_tokens(os.path.join(ROOT, "data", "wt2_train.pt"))[:60000].tolist()
    rng = random.Random(0)
    out = {"exp": "exp64_query_swap", "tag": args.tag, "ckpt": args.ckpt,
           "ts": now(), "results": {}}

    for d in [float(x) for x in args.depths.split(",")]:
        # fact 0 (secret) shallow, fact 1 (backup) deep; keep them apart
        facts = [(FACTS[0][0], str(rng.randrange(1000, 9999)), 0.2, FACTS[0][2]),
                 (FACTS[1][0], str(rng.randrange(1000, 9999)), min(0.85, 0.2 + d),
                  FACTS[1][2])]
        acc = {name: {"ce": [], "exact": [], "hit": []} for name in ("secret", "backup")}
        for _ in range(args.n_prompts):
            for q_idx, name in enumerate(("secret", "backup")):
                ids, blocks, ans_len = build(tok, filler, args.seq, facts, q_idx)
                ce, ex = score(model, ids, ans_len)
                h = hits(model, ids, ans_len, blocks)
                acc[name]["ce"].append(ce)
                acc[name]["exact"].append(ex)
                acc[name]["hit"].append(h)
                del ids
                torch.cuda.empty_cache()
        row = {}
        for name in ("secret", "backup"):
            n = len(acc[name]["ce"])
            h0 = sum(a[0] for a in acc[name]["hit"]) / n
            h1 = sum(a[1] for a in acc[name]["hit"]) / n
            row[name] = {"ce": sum(acc[name]["ce"]) / n,
                         "exact": sum(acc[name]["exact"]) / n,
                         "hit_secret": h0, "hit_backup": h1}
        out["results"][f"d{d}"] = row
        print(f"[d={d:.2f}] ask=secret: CE {row['secret']['ce']:.3f} "
              f"exact {row['secret']['exact']:.2f} hit(secret)={row['secret']['hit_secret']:.2f} "
              f"hit(backup)={row['secret']['hit_backup']:.2f}", flush=True)
        print(f"[d={d:.2f}] ask=backup: CE {row['backup']['ce']:.3f} "
              f"exact {row['backup']['exact']:.2f} hit(secret)={row['backup']['hit_secret']:.2f} "
              f"hit(backup)={row['backup']['hit_backup']:.2f}", flush=True)

    with open(os.path.join(ROOT, "results", f"exp64_{args.tag}.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    append_jsonl(RESULTS, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
