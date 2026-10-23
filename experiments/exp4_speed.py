"""EXP4: end-to-end prefill speed (dense vs sparse), component breakdown, decode speed."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from lcsa.decode import prefill_and_record, decode_step
from lcsa.harness import load_model
from lcsa.lora import apply_lora, merge_lora
from lcsa.metrics import append_jsonl, now, cuda_time_ms
from lcsa.model import (patch_model, set_sparse_enabled, set_profiling, get_profile,
                        set_router_graph)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "Qwen2.5-0.5B")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--tag", type=str, default="speed")
    ap.add_argument("--Ts", type=str, default="1024,2048,4096,8192")
    ap.add_argument("--n_layers_sparse", type=int, default=24)
    ap.add_argument("--v2", action="store_true", help="stage-2 LoRA checkpoint (clean config)")
    ap.add_argument("--merge", action="store_true", help="merge LoRA into base weights")
    ap.add_argument("--pin_dense", action="store_true",
                    help="pin dense timing to FLASH/EFFICIENT sdpa backends (deterministic reference)")
    ap.add_argument("--router-graph", action="store_true", default=True,
                    help="replay the router pipeline as a CUDA graph (A2 path)")
    ap.add_argument("--no-router-graph", action="store_false", dest="router_graph")
    args = ap.parse_args()
    Ts = [int(t) for t in args.Ts.split(",")]

    model = load_model(MODEL_DIR, device=DEV)
    meta = load_ckpt(model, args.ckpt) if os.path.exists(args.ckpt) else {}
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
    if os.path.exists(args.ckpt):
        load_ckpt(model, args.ckpt)
    for p in model.parameters():
        p.requires_grad_(False)   # inference-only: no autograd graph (would spill VRAM at large T)
    for w in wrappers.values():
        w.null_logit.data.fill_(-1e9)   # v1 ckpts carry the buggy -3; clean config forces off
    if args.merge:
        merge_lora(model)
        print("[lora] merged into base weights", flush=True)
    print(f"[{now()}] loaded {args.ckpt}, budget={wrappers[0].budget}", flush=True)

    out = {"exp": "exp4_speed", "ts": now(), "ckpt": args.ckpt, "tag": args.tag,
           "meta": meta, "prefill": {}, "decode": {}}

    # ---- prefill core forward (logits_to_keep=1) ------------------------------
    for T in Ts:
        ids = torch.randint(0, 1000, (1, T), device=DEV)
        try:
            set_sparse_enabled(False)
            with torch.no_grad():
                if args.pin_dense:
                    from torch.nn.attention import SDPBackend, sdpa_kernel
                    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                        ms_dense = cuda_time_ms(lambda: model(input_ids=ids, logits_to_keep=1),
                                                warmup=3, iters=10)
                else:
                    ms_dense = cuda_time_ms(lambda: model(input_ids=ids, logits_to_keep=1),
                                            warmup=3, iters=10)
            set_sparse_enabled(True)
            ms_sparse = cuda_time_ms(lambda: model(input_ids=ids, logits_to_keep=1), warmup=3, iters=10)
            set_sparse_enabled(False)
            rec = {"dense_ms": ms_dense["median"], "sparse_ms": ms_sparse["median"],
                   "speedup": round(ms_dense["median"] / ms_sparse["median"], 3)}
            print(f"[prefill] T={T}: dense={ms_dense['median']:.1f}ms sparse={ms_sparse['median']:.1f}ms "
                  f"speedup={rec['speedup']}x", flush=True)

            # component breakdown (per forward, summed over layers)
            set_sparse_enabled(True)
            set_profiling(True)
            with torch.no_grad():
                for _ in range(3):
                    model(input_ids=ids, logits_to_keep=1)
            prof = get_profile()
            set_profiling(False)
            set_sparse_enabled(False)
            rec["breakdown_ms_per_fwd"] = {k: round(v / 3, 3) for k, v in prof.items()}
            rec["breakdown_ratio_of_sparse"] = {k: round(v / 3 / ms_sparse["median"], 4)
                                                for k, v in prof.items()}
            print(f"          breakdown: {rec['breakdown_ms_per_fwd']}", flush=True)
            out["prefill"][f"T{T}"] = rec
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            out["prefill"][f"T{T}"] = "OOM"
            print(f"[prefill] T={T}: OOM", flush=True)
        del ids
        torch.cuda.empty_cache()

    # ---- decode speed ----------------------------------------------------------
    for T0 in [1024, 2048, 4096]:
        n_steps = 64
        ids = torch.randint(0, 1000, (1, T0), device=DEV)
        try:
            set_sparse_enabled(False)
            logits, cache = prefill_and_record(model, ids, block=64, max_new_tokens=n_steps + 8)
            tok = torch.tensor([[100]], device=DEV)

            # warmup
            for i in range(3):
                decode_step(model, tok, cache, T0 + i, sparse=False)
            torch.cuda.synchronize()
            t0 = time.time()
            for i in range(n_steps):
                decode_step(model, tok, cache, T0 + 3 + i, sparse=False)
            torch.cuda.synchronize()
            dense_step_ms = (time.time() - t0) / n_steps * 1000

            # sparse (fresh cache) — clean config: exact-only (no coarse path)
            set_sparse_enabled(True)
            logits2, cache2 = prefill_and_record(model, ids, block=64, max_new_tokens=n_steps + 8)
            for i in range(3):
                decode_step(model, tok, cache2, T0 + i, sparse=True, budget=wrappers[0].budget,
                            use_ksum=False)
            torch.cuda.synchronize()
            t0 = time.time()
            for i in range(n_steps):
                decode_step(model, tok, cache2, T0 + 3 + i, sparse=True, budget=wrappers[0].budget,
                            use_ksum=False)
            torch.cuda.synchronize()
            sparse_step_ms = (time.time() - t0) / n_steps * 1000
            set_sparse_enabled(False)

            rec = {"dense_ms_per_step": round(dense_step_ms, 3),
                   "sparse_ms_per_step": round(sparse_step_ms, 3),
                   "speedup": round(dense_step_ms / sparse_step_ms, 3)}
            out["decode"][f"ctx{T0}"] = rec
            print(f"[decode] ctx={T0}: dense={dense_step_ms:.2f}ms/tok sparse={sparse_step_ms:.2f}ms/tok "
                  f"speedup={rec['speedup']}x", flush=True)
            del cache, cache2
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            out["decode"][f"ctx{T0}"] = "OOM"
            print(f"[decode] ctx={T0}: OOM", flush=True)
        del ids
        torch.cuda.empty_cache()

    with open(os.path.join(ROOT, "results", f"exp4_{args.tag}.json"), "w") as f:
        json.dump(out, f, indent=1)
    append_jsonl(RESULTS, out)
    print(f"[{now()}] exp4 done", flush=True)


if __name__ == "__main__":
    main()
