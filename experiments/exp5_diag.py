"""EXP5: decompose the sparse-attention quality gap.

Variants (all on dev chunks, T=1024):
  - dense (fast SDPA) reference
  - trained router selection, full LCGA mixture      rho in {0.125,0.25,0.5,1.0}
  - trained router selection, exact path only        rho in {0.125,0.25,0.5}
      (use_coarse=False: isolates the coarse-mixture contribution)
  - oracle selection (top-m by TRUE dense mass), full mixture
  - oracle selection, exact only
  - local (mandatory sink+diagonal only) reference
Recall = fraction of dense attention block mass covered by selected blocks.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from lcsa.data import load_tokens, chunks_from_tokens
from lcsa.harness import load_model, quick_ppl
from lcsa.metrics import append_jsonl, now, routing_diagnostics
from lcsa.model import (patch_model, set_sparse_enabled, begin_capture, end_capture,
                        set_selection_mode)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "Qwen2.5-0.5B")
DATA_DIR = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results", "results.jsonl")
DEV = "cuda"


def load_ckpt(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for i, layer in enumerate(model.model.layers):
        own = dict(layer.self_attn.named_parameters())
        for n, t in ckpt["state"][i].items():
            if n in own:
                own[n].data.copy_(t.to(own[n].device))
    return ckpt.get("meta", {})


@torch.no_grad()
def recall_stats(model, chunks, device, max_chunks=4):
    acc = []
    for i in range(min(max_chunks, len(chunks))):
        begin_capture()
        model(input_ids=chunks[i].to(device).unsqueeze(0))
        cap = end_capture()
        diag = routing_diagnostics(cap)
        acc.append([d["recall_mean"] for d in diag])
    t = torch.tensor(acc)
    return {"mean": float(t.mean()), "first": float(t[:, 0].mean()),
            "last": float(t[:, -1].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--tag", type=str, default="diag")
    ap.add_argument("--max_chunks", type=int, default=32)
    ap.add_argument("--recall_chunks", type=int, default=4)
    args = ap.parse_args()

    model = load_model(MODEL_DIR, device=DEV)
    meta = load_ckpt(model, args.ckpt)
    margs = meta.get("args", {})
    wrappers = patch_model(model, budget=margs.get("budget", 0.25), block=64,
                           n_concepts=margs.get("n_concepts", 64),
                           graph_rank=margs.get("graph_rank", 16),
                           dynamic=margs.get("dynamic", False),
                           content_rank=margs.get("content_rank", 0),
                           rel_bias=not margs.get("no_rel_bias", False),
                           capture_dense=True)
    load_ckpt(model, args.ckpt)
    print(f"[{now()}] loaded {args.ckpt} budget={margs.get('budget', 0.25)}", flush=True)

    toks = load_tokens(os.path.join(DATA_DIR, "wt2_test.pt"))
    chunks = chunks_from_tokens(toks, 1024, max_chunks=args.max_chunks)

    out = {"exp": "exp5_diag", "ckpt": args.ckpt, "tag": args.tag, "ts": now(),
           "budget": margs.get("budget", 0.25), "variants": {}}

    set_sparse_enabled(False)
    dense = quick_ppl(model, chunks, device=DEV)
    out["dense_ppl"] = dense["ppl"]
    print(f"[dense] ppl={dense['ppl']:.3f}", flush=True)

    def run_variant(name, rho, coarse=True, mode="topk"):
        t0 = time.time()
        for w in wrappers.values():
            w.budget = rho
            w.use_coarse = coarse
        set_selection_mode(mode)
        set_sparse_enabled(True)
        sp = quick_ppl(model, chunks, device=DEV)
        rec = recall_stats(model, chunks, DEV, max_chunks=args.recall_chunks)
        set_sparse_enabled(False)
        rec_pct = {k: round(v, 3) for k, v in rec.items()}
        out["variants"][name] = {"ppl": sp["ppl"], "delta": sp["ppl"] - dense["ppl"],
                                 "recall": rec_pct, "sec": round(time.time() - t0, 1)}
        print(f"[{name}] ppl={sp['ppl']:.3f} (delta={sp['ppl']-dense['ppl']:+.3f}) "
              f"recall_mean={rec['mean']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        set_selection_mode("topk")

    # A) trained router, full mixture
    for rho in [0.125, 0.25, 0.5, 1.0]:
        run_variant(f"trained_rho{rho}", rho, coarse=True)
    # B) trained router, exact path only
    for rho in [0.125, 0.25, 0.5]:
        run_variant(f"exactonly_rho{rho}", rho, coarse=False)
    # C) oracle selection
    for rho in [0.125, 0.25, 0.5]:
        run_variant(f"oracle_rho{rho}", rho, coarse=True, mode="oracle")
    for rho in [0.125, 0.25]:
        run_variant(f"oracle_exactonly_rho{rho}", rho, coarse=False, mode="oracle")
    # D) mandatory only
    run_variant("local_mandatory", margs.get("budget", 0.25), coarse=True, mode="local")

    # restore
    for w in wrappers.values():
        w.budget = margs.get("budget", 0.25)
        w.use_coarse = True

    gates = [round(float(torch.sigmoid(w.gate).detach().mean()), 3) for w in wrappers.values()]
    out["gate_per_layer"] = gates
    out["gate_mean"] = sum(gates) / len(gates)
    print(f"[gate] mean={out['gate_mean']:.3f} min={min(gates)} max={max(gates)}", flush=True)

    with open(os.path.join(ROOT, "results", f"exp5_{args.tag}.json"), "w") as f:
        json.dump(out, f, indent=1)
    append_jsonl(RESULTS, out)
    print(f"[{now()}] exp5 done", flush=True)


if __name__ == "__main__":
    main()
