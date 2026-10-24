"""Clean speed reference: dense FIRST (pinned FLASH/EFFICIENT backends, fresh process),
then sparse. Avoids the allocator-pollution artefact of interleaved exp4 runs on 6GB."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from lcsa.harness import load_model
from lcsa.lora import apply_lora, merge_lora
from lcsa.metrics import append_jsonl, cuda_time_ms, now
from lcsa.model import (get_profile, patch_model, set_profiling, set_router_graph,
                        set_sparse_enabled)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = "cuda"
MODEL_DIR = os.path.join(ROOT, "models", "Qwen2.5-0.5B")
RESULTS = os.path.join(ROOT, "results", "results.jsonl")


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
    ap.add_argument("--tag", type=str, default="ref")
    ap.add_argument("--Ts", type=str, default="1024,2048,4096,8192,12288,16384")
    ap.add_argument("--v2", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--graph", action="store_true",
                    help="A2: CUDA-graph router replay (enable AFTER weights are final)")
    args = ap.parse_args()
    Ts = [int(t) for t in args.Ts.split(",")]

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
        p.requires_grad_(False)   # inference-only: no autograd graph (would spill VRAM at large T)
    for w in wrappers.values():
        w.null_logit.data.fill_(-1e9)
    if args.merge:
        merge_lora(model)
    if args.graph:
        set_router_graph(True)
    print(f"[{now()}] ready, tag={args.tag} graph={args.graph}", flush=True)

    from torch.nn.attention import SDPBackend, sdpa_kernel

    out = {"exp": "exp4_speed", "ts": now(), "ckpt": args.ckpt, "tag": args.tag,
           "meta": meta, "prefill": {}, "decode": {}, "note": "clean dense-first, backends pinned"}
    for T in Ts:
        ids = torch.randint(0, 1000, (1, T), device=DEV)
        torch.cuda.empty_cache()
        rec = {"dense_pinned": True}
        try:
            set_sparse_enabled(False)
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                ms_dense = cuda_time_ms(lambda: model(input_ids=ids, logits_to_keep=1),
                                        warmup=3, iters=10)
            rec["dense_ms"] = ms_dense["median"]
            rec["dense_min"] = ms_dense["min"]
            print(f"[dense] T={T}: {ms_dense['median']:.1f}ms (min {ms_dense['min']:.1f}) "
                  f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)

            set_sparse_enabled(True)
            ms_sparse = cuda_time_ms(lambda: model(input_ids=ids, logits_to_keep=1),
                                     warmup=3, iters=10)
            rec["sparse_ms"] = ms_sparse["median"]
            rec["sparse_min"] = ms_sparse["min"]
            rec["speedup"] = round(ms_dense["median"] / ms_sparse["median"], 3)
            print(f"[sparse] T={T}: {ms_sparse['median']:.1f}ms speedup={rec['speedup']}x "
                  f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)

            set_profiling(True)
            with torch.no_grad():
                for _ in range(3):
                    model(input_ids=ids, logits_to_keep=1)
            prof = get_profile()
            set_profiling(False)
            rec["breakdown_ms_per_fwd"] = {k: round(v / 3, 3) for k, v in prof.items()}
            print(f"          breakdown: {rec['breakdown_ms_per_fwd']}", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            rec["error"] = "OOM"
            print(f"[T={T}] OOM", flush=True)
        set_sparse_enabled(False)
        out["prefill"][f"T{T}"] = rec
        del ids
        torch.cuda.empty_cache()

    with open(os.path.join(ROOT, "results", f"exp4_{args.tag}.json"), "w") as f:
        json.dump(out, f, indent=1)
    append_jsonl(RESULTS, out)
    print(f"[{now()}] speed_ref done", flush=True)


if __name__ == "__main__":
    main()
