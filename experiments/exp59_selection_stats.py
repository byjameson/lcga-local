"""EXP59: what does the router actually select?

For each layer and query block, classify every selected key block as

  * sink      : block 0 (always mandatory),
  * window    : inside the contiguous recency window of the same size,
  * content   : anything else — i.e. the blocks the router adds *beyond* a plain
                sliding window of the same budget.

This is the direct measurement of "is content routing doing anything", and it
explains the long-context quality trend in LCGA_FINDINGS.md.

Usage:
  python experiments/exp59_selection_stats.py --ckpt checkpoints/abl_diag_best.pt \
      --Ts 1024,4096,8192
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from lcsa.data import chunks_from_tokens, load_tokens
from lcsa.harness import load_model
from lcsa.lora import apply_lora, merge_lora
from lcsa.metrics import append_jsonl, now
from lcsa.model import (begin_capture, end_capture, patch_model, set_selection_mode,
                        set_sparse_enabled)

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
def composition(idx, budget):
    """Fraction of selected blocks that are sink / window / content."""
    # idx: (B, NB, MMAX) int32, -1 = empty
    B, NB, MMAX = idx.shape
    dev = idx.device
    I = torch.arange(NB, device=dev).view(1, NB, 1)
    valid = idx >= 0
    n_valid = valid.sum()
    if n_valid == 0:
        return None
    j = idx.clamp_min(0).long()
    is_sink = (j == 0) & valid
    # window of the same size: the (m-1) blocks before I, plus I itself
    m = valid.sum(-1).view(B, NB, 1)               # (B,NB,1) actual count
    lo = (I - (m - 2)).clamp_min(0)
    is_window = (j <= I) & (j >= lo) & valid & (~is_sink)
    is_content = valid & (~is_sink) & (~is_window)
    n = float(n_valid)
    return {"sink": float(is_sink.sum()) / n,
            "window": float(is_window.sum()) / n,
            "content": float(is_content.sum()) / n,
            "mean_m": float(m.float().mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--Ts", default="1024,4096,8192")
    ap.add_argument("--max_chunks", type=int, default=2)
    ap.add_argument("--tag", default="selstats")
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
    budget = margs.get("budget", 0.25)

    toks = load_tokens(os.path.join(DATA_DIR, "wt2_test.pt"))
    out = {"exp": "exp59_selection_stats", "tag": args.tag, "ckpt": args.ckpt,
           "budget": budget, "ts": now(), "by_T": {}}

    for T in [int(t) for t in args.Ts.split(",")]:
        chunks = chunks_from_tokens(toks, T, max_chunks=args.max_chunks)
        agg = []
        per_layer = []
        set_sparse_enabled(True)
        for ch in chunks:
            ids = ch.to(DEV).unsqueeze(0)
            begin_capture()
            model(input_ids=ids)
            cap = end_capture()
            for entry in cap:
                if "idx" not in entry:
                    continue
                c = composition(entry["idx"], budget)
                if c:
                    agg.append(c)
            del ids
            torch.cuda.empty_cache()
        set_sparse_enabled(False)

        n = max(len(agg), 1)
        rec = {k: sum(c[k] for c in agg) / n for k in ("sink", "window", "content")}
        rec["mean_m"] = sum(c["mean_m"] for c in agg) / n
        out["by_T"][f"T{T}"] = rec
        print(f"[T={T}] sink={rec['sink']:.1%} window={rec['window']:.1%} "
              f"content={rec['content']:.1%} (mean m={rec['mean_m']:.1f})", flush=True)

    with open(os.path.join(ROOT, "results", f"exp59_{args.tag}.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    append_jsonl(RESULTS, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
