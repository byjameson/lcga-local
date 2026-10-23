"""EXP58: quality evaluation at long context without materialising full logits.

`exp3` evaluates by taking `model(...).logits`, which is `T x vocab` — at T=16384
that tensor alone is ~5 GB and OOMs a 6 GB card. Here we run the backbone to get
hidden states and apply the LM head in position chunks, which is numerically the
same thing with bounded memory.

Reports dense/sparse PPL (+ top-1 agreement, KL) and the random/local selection
controls at a given context length.

Usage:
  python experiments/exp58_eval_long.py --ckpt checkpoints/abl_diag_best.pt \
      --seq 16384 --max_chunks 2 --controls
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from lcsa.data import chunks_from_tokens, load_tokens
from lcsa.harness import load_model
from lcsa.lora import apply_lora, merge_lora
from lcsa.metrics import append_jsonl, now, routing_diagnostics
from lcsa.model import (begin_capture, end_capture, get_profile, patch_model,
                        set_profiling, set_selection_mode, set_sparse_enabled)

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
def hidden(model, ids):
    return model.model(input_ids=ids).last_hidden_state


@torch.no_grad()
def ce_from_hidden(model, h, ids, pos_chunk=128, want_logits=False):
    """Mean CE over positions 0..T-2; optionally also returns per-chunk info."""
    T = ids.shape[1]
    total, n = 0.0, 0
    top1_num, kl_sum = 0, 0.0
    for a in range(0, T - 1, pos_chunk):
        b = min(a + pos_chunk, T - 1)
        logits = model.lm_head(h[:, a:b]).float()
        tgt = ids[0, a + 1:b + 1]
        total += float(F.cross_entropy(logits[0], tgt, reduction="sum"))
        n += b - a
    return total / max(n, 1), n


@torch.no_grad()
def pair_stats(model, ids, pos_chunk=128):
    """Dense vs sparse: PPL, top-1 agreement and KL(teacher||student), chunked."""
    set_sparse_enabled(False)
    h_d = hidden(model, ids)
    set_sparse_enabled(True)
    h_s = hidden(model, ids)
    set_sparse_enabled(False)

    T = ids.shape[1]
    ce_d = ce_s = 0.0
    agree = 0
    kl = 0.0
    n = 0
    for a in range(0, T - 1, pos_chunk):
        b = min(a + pos_chunk, T - 1)
        ld = model.lm_head(h_d[:, a:b]).float()
        ls = model.lm_head(h_s[:, a:b]).float()
        tgt = ids[0, a + 1:b + 1]
        ce_d += float(F.cross_entropy(ld[0], tgt, reduction="sum"))
        ce_s += float(F.cross_entropy(ls[0], tgt, reduction="sum"))
        agree += int((ld.argmax(-1) == ls.argmax(-1)).sum())
        logp_d = F.log_softmax(ld[0], dim=-1)
        logp_s = F.log_softmax(ls[0], dim=-1)
        kl += float((logp_d.exp() * (logp_d - logp_s)).sum())
        n += b - a
    del h_d, h_s
    torch.cuda.empty_cache()
    return {"ce_dense": ce_d / n, "ce_sparse": ce_s / n, "n_tok": n,
            "top1": agree / n, "kl": kl / n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seq", type=int, default=16384)
    ap.add_argument("--max_chunks", type=int, default=2)
    ap.add_argument("--pos_chunk", type=int, default=128)
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--tag", default="long")
    ap.add_argument("--v2", action="store_true")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    model = load_model(MODEL_DIR, device=DEV)
    meta = load_ckpt(model, args.ckpt)
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
        load_ckpt(model, args.ckpt)
    for p in model.parameters():
        p.requires_grad_(False)
    for w in wrappers.values():
        w.null_logit.data.fill_(-1e9)
    if args.merge:
        merge_lora(model)

    toks = load_tokens(os.path.join(DATA_DIR, "wt2_test.pt"))
    chunks = chunks_from_tokens(toks, args.seq, max_chunks=args.max_chunks)
    print(f"[exp58] T={args.seq} chunks={len(chunks)} ckpt={os.path.basename(args.ckpt)}",
          flush=True)

    t0 = time.time()
    ppl_d, ppl_s, agree, kl = [], [], [], []
    for i, ch in enumerate(chunks):
        ids = ch.to(DEV).unsqueeze(0)
        st = pair_stats(model, ids, args.pos_chunk)
        ppl_d.append(st["ce_dense"])
        ppl_s.append(st["ce_sparse"])
        agree.append(st["top1"])
        kl.append(st["kl"])
        print(f"  chunk {i}: dense={st['ce_dense']:.4f} sparse={st['ce_sparse']:.4f} "
              f"top1={st['top1']:.4f} kl={st['kl']:.4f}", flush=True)
        del ids
        torch.cuda.empty_cache()

    import math
    dense_ce = sum(ppl_d) / len(ppl_d)
    sparse_ce = sum(ppl_s) / len(ppl_s)
    rec = {
        "exp": "exp58_eval_long", "tag": args.tag, "ckpt": args.ckpt,
        "seq": args.seq, "chunks": len(chunks),
        "dense_ce": dense_ce, "sparse_ce": sparse_ce,
        "dense_ppl": math.exp(dense_ce), "sparse_ppl": math.exp(sparse_ce),
        "rel_loss": (math.exp(sparse_ce) / math.exp(dense_ce)) - 1.0,
        "top1": sum(agree) / len(agree), "kl": sum(kl) / len(kl),
        "sec": round(time.time() - t0, 1), "ts": now(),
    }
    print(f"[main] T={args.seq}: dense_ppl={rec['dense_ppl']:.3f} "
          f"sparse_ppl={rec['sparse_ppl']:.3f} rel={rec['rel_loss']:+.1%} "
          f"top1={rec['top1']:.4f} ({rec['sec']}s)", flush=True)

    if args.controls:
        for mode in ("random", "local"):
            set_selection_mode(mode)
            ces = []
            for ch in chunks:
                ids = ch.to(DEV).unsqueeze(0)
                set_sparse_enabled(True)
                h = hidden(model, ids)
                set_sparse_enabled(False)
                ce, _ = ce_from_hidden(model, h, ids, args.pos_chunk)
                ces.append(ce)
                del ids, h
                torch.cuda.empty_cache()
            ce = sum(ces) / len(ces)
            rec[f"{mode}_ce"] = ce
            rec[f"{mode}_ppl"] = math.exp(ce)
            print(f"[{mode}] ppl={math.exp(ce):.3f}", flush=True)
        set_selection_mode("topk")

    out_path = os.path.join(ROOT, "results", f"exp58_{args.tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    append_jsonl(RESULTS, rec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
