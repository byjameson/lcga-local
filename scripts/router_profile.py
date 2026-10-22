"""A-phase router profiling: where does the LCGA router cost go?

Part 1 (micro): synthetic hidden states -> each ConceptGraphRouter sub-op timed
  back-to-back on GPU (x24 to emulate all layers).
Part 2 (real): v2 model forward with set_profiling -> router_ms / kernel_ms /
  coarse_ms buckets (ground truth incl. launch gaps) at given T.

Usage:
  python scripts\router_profile.py --tag baseline --Ts 1024,4096,8192 [--ckpt checkpoints\v2_best.pt]
Writes: results\router_profile_<tag>.json  (diagnostic only, not results.jsonl)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from lcsa.concepts import ConceptGraphRouter
from lcsa.harness import load_model
from lcsa.lora import apply_lora
from lcsa.metrics import now
from lcsa.model import (get_profile, patch_model, set_profiling,
                        set_selection_mode, set_sparse_enabled)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = "cuda"
MODEL_DIR = os.path.join(ROOT, "models", "Qwen2.5-0.5B")


def load_ckpt(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for i, layer in enumerate(model.model.layers):
        w = layer.self_attn
        own = dict(w.named_parameters())
        for n, t in ckpt["state"][i].items():
            if n in own:
                own[n].data.copy_(t.to(own[n].device))
    return ckpt.get("meta", {})


def bench_ms(fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True)
    e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


@torch.no_grad()
def part1_micro(d_model, Ts, budget=0.25):
    router = ConceptGraphRouter(d_model, 64, 16, rel_bias=True).to(DEV).eval()
    for p in router.parameters():
        p.requires_grad_(False)
    out = {}
    for T in Ts:
        h16 = torch.randn(1, T, d_model, device=DEV).half()
        h32 = h16.float()
        ql, kl = router.concept_logits(h32)
        scores = router.block_scores(ql, kl, h32, block=64)

        rec = {}
        rec["cast_f16_f32"] = bench_ms(lambda: h16.float())
        rec["concept_logits"] = bench_ms(lambda: router.concept_logits(h32))
        rec["block_scores"] = bench_ms(lambda: router.block_scores(ql, kl, h32, block=64))
        rec["select_blocks"] = bench_ms(
            lambda: router.select_blocks(scores, block=64, budget=budget))
        rec["masked_probs"] = bench_ms(lambda: router.masked_probs(scores, block=64))

        def pipeline():
            hh = h16.float()
            q, k = router.concept_logits(hh)
            sc = router.block_scores(q, k, hh, block=64)
            idx, _ = router.select_blocks(sc, block=64, budget=budget)
            pr = router.masked_probs(sc, block=64)
            return idx, pr

        rec["pipeline"] = bench_ms(pipeline)
        rec["pipeline_x24"] = rec["pipeline"] * 24
        out[str(T)] = {k: round(v, 3) for k, v in rec.items()}
        print(f"[micro] T={T}: " + " ".join(
            f"{k}={v:.2f}" for k, v in rec.items() if k != "pipeline_x24"), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", type=str, default="baseline")
    ap.add_argument("--Ts", type=str, default="1024,4096,8192")
    ap.add_argument("--ckpt", type=str, default=os.path.join("checkpoints", "v2_best.pt"))
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    Ts = [int(t) for t in args.Ts.split(",")]

    model = load_model(MODEL_DIR, device=DEV)
    d_model = model.config.hidden_size

    out = {"tag": args.tag, "ts": now(), "d_model": d_model, "ckpt": args.ckpt,
           "micro_per_layer": {}, "real_buckets": {}}

    print(f"[{now()}] part 1: micro sub-op timings (x24 emulation)", flush=True)
    out["micro_per_layer"] = part1_micro(d_model, Ts)

    print(f"[{now()}] part 2: real-model profile buckets", flush=True)
    meta = load_ckpt(model, args.ckpt)
    margs = meta.get("args", {})
    wrappers = patch_model(model, budget=margs.get("budget", 0.25),
                           n_concepts=margs.get("n_concepts", 64),
                           graph_rank=margs.get("graph_rank", 16),
                           dynamic=margs.get("dynamic", False),
                           content_rank=margs.get("content_rank", 0),
                           rel_bias=not margs.get("no_rel_bias", False),
                           capture_dense=False, use_coarse=False, null_init=-1e9,
                           recall_gain=True)
    apply_lora(model, r_attn=margs.get("lora_attn", 16), r_mlp=margs.get("lora_mlp", 0))
    load_ckpt(model, args.ckpt)
    for p in model.parameters():
        p.requires_grad_(False)
    for w in wrappers.values():
        w.null_logit.data.fill_(-1e9)
    set_selection_mode("topk")

    for T in Ts:
        ids = torch.randint(0, 1000, (1, T), device=DEV)
        torch.cuda.empty_cache()
        set_sparse_enabled(True)
        with torch.no_grad():
            model(input_ids=ids, logits_to_keep=1)  # warm (profiling off)
        set_profiling(True)
        with torch.no_grad():
            for _ in range(3):
                model(input_ids=ids, logits_to_keep=1)
        prof = {k: round(v / 3, 3) for k, v in get_profile().items()}
        set_profiling(False)
        set_sparse_enabled(False)
        out["real_buckets"][str(T)] = prof
        print(f"[real] T={T}: {prof} reserved={torch.cuda.memory_reserved()/1e9:.2f}GB",
              flush=True)

    path = os.path.join(ROOT, "results", f"router_profile_{args.tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[{now()}] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
