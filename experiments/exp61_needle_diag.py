"""Diagnose the needle teacher: is dense attention actually pointing at the fact?

Prints, for one needle example: dense CE/exact on the answer, and per-layer the
attention mass the answer tokens put on the block that holds the fact, plus the
top-5 key blocks with their decoded text. Answers whether the *teacher* carries a
usable retrieval signal before we blame the router.

Usage:
  python experiments/exp61_needle_diag.py --seq 1024 --depth 0.5 --layer 12
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from lcsa.attention import repeat_kv
from lcsa.data import load_tokens
from lcsa.harness import load_model
from lcsa.model import (begin_capture, end_capture, patch_model, set_capture_qk,
                        set_sparse_enabled)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "Qwen2.5-0.5B")
DATA_DIR = os.path.join(ROOT, "data")
DEV = "cuda"
QUESTION = " Question: What is the secret code? Answer:"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--depth", type=float, default=0.5)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--teacher", default="127")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model = load_model(MODEL_DIR, device=DEV)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    wrappers = patch_model(model, budget=0.25, block=64, n_concepts=64, graph_rank=0,
                           dynamic=False, content_rank=0, rel_bias=True,
                           capture_dense=True, use_coarse=False, null_init=-1e9)

    filler_all = load_tokens(os.path.join(DATA_DIR, "wt2_train.pt"))[:60000].tolist()
    code = str(rng.randrange(1000, 9999))
    fact = tok(f" The secret code is {code}.", add_special_tokens=False).input_ids
    ans = tok(f" {code}", add_special_tokens=False).input_ids
    q = tok(QUESTION, add_special_tokens=False).input_ids
    body_len = args.seq - len(fact) - len(q) - len(ans)
    pos = int(body_len * args.depth)
    start = 0
    body = (filler_all[start:start + pos] + fact
            + filler_all[start + pos:start + body_len])
    ids = body + q + ans
    assert len(ids) == args.seq, (len(ids), args.seq)
    x = torch.tensor([ids], dtype=torch.long, device=DEV)
    fact_block = (pos + len(fact) - 1) // 64
    print(f"seq={args.seq} depth={args.depth:.2f} code={code} ans_len={len(ans)} "
          f"fact_block={fact_block} fact_span_tokens=[{pos},{pos+len(fact)}) "
          f"ans_positions=[{args.seq-len(ans)},{args.seq})", flush=True)

    # 1) dense student quality on this example (sparse disabled -> original attention)
    set_sparse_enabled(False)
    with torch.no_grad():
        out = model(input_ids=x, logits_to_keep=len(ans) + 1)
    logits = out.logits[0, :-1].float()
    tgt = x[0, -len(ans):]
    ce = F.cross_entropy(logits, tgt).item()
    exact = int((logits.argmax(-1) == tgt).all())
    print(f"[dense] answer CE={ce:.4f} exact={exact} "
          f"pred={[tok.decode([i]) for i in logits.argmax(-1).tolist()]}", flush=True)

    # 2) teacher attention of the answer tokens, from the captured q/k
    set_sparse_enabled(True)
    set_capture_qk(True)
    begin_capture()
    with torch.no_grad():
        model.model(input_ids=x, use_cache=False)
    cap_list = end_capture()
    NB = args.seq // 64
    ar = list(range(args.seq - len(ans), args.seq))
    keypos = torch.arange(args.seq, device=DEV)
    print("\n layer  mass@fact  rank  top5 blocks (mass)")
    for L in (0, 6, 11, 17, 23):
        cap = cap_list[L]
        qq, kk = cap["q"], cap["k"]
        H = qq.shape[1]
        if kk.shape[1] != H:
            kk = repeat_kv(kk, H // kk.shape[1])
        logits_a = torch.einsum("bhrd,bhtd->bhrt", qq[:, :, ar, :].float(), kk.float()) \
            * wrappers[L].scale
        mask = keypos.view(1, 1, 1, -1) > torch.tensor(ar, device=DEV).view(1, 1, -1, 1)
        att = torch.softmax(logits_a.masked_fill(mask, float("-inf")), -1)
        att_b = att.view(1, H, len(ar), NB, 64).sum(-1).sum(2).mean(1)[0]  # (NB,)
        att_b = att_b / att_b.sum()
        v, i = att_b.topk(5)
        rank = int((att_b > att_b[fact_block]).sum()) + 1
        pairs = ", ".join(f"{int(j)}:{float(val):.3f}" for j, val in zip(i, v))
        print(f"  L{L:<3d} {float(att_b[fact_block]):.3f}      {rank}   {pairs}", flush=True)
        if L == args.layer:
            # token-level view at the last answer token: what does it read?
            a1 = att[0, :, -1, :].mean(0)                      # (T,)
            v2, i2 = a1.topk(8)
            print("   last-answer-token top tokens:")
            for val, tp in zip(v2, i2):
                tp = int(tp)
                print(f"     {float(val):.3f}  pos={tp:5d} blk={tp//64:3d} "
                      f"{tok.decode(x[0, max(0, tp-6):tp+1].tolist())!r}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
