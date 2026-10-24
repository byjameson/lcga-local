"""v2 context curve: dense vs sparse@0.25/0.5 PPL at T=1024..8192 (wt2_test).

Clean config (exact-only, null off) + v2 LoRA/gain. Complements exp6 (v1) so the
stage-2 quality gain can be checked at longer contexts too.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from lcsa.data import chunks_from_tokens, load_tokens
from lcsa.harness import load_model, quick_ppl
from lcsa.lora import apply_lora
from lcsa.metrics import append_jsonl, now, routing_diagnostics
from lcsa.model import (begin_capture, end_capture, patch_model, set_selection_mode,
                        set_sparse_enabled)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--tag", type=str, default="v2")
    ap.add_argument("--Ts", type=str, default="1024,2048,4096,8192")
    ap.add_argument("--chunks", type=str, default="32,16,16,8")
    args = ap.parse_args()
    Ts = [int(t) for t in args.Ts.split(",")]
    nch = [int(c) for c in args.chunks.split(",")]

    model = load_model(os.path.join(ROOT, "models", "Qwen2.5-0.5B"), device=DEV)
    meta = load_ckpt(model, args.ckpt)
    margs = meta.get("args", {})
    wrappers = patch_model(model, budget=margs.get("budget", 0.25), block=64,
                           n_concepts=margs.get("n_concepts", 64),
                           graph_rank=margs.get("graph_rank", 16),
                           dynamic=margs.get("dynamic", False),
                           content_rank=margs.get("content_rank", 0),
                           rel_bias=not margs.get("no_rel_bias", False),
                           capture_dense=True, use_coarse=False, null_init=-1e9,
                           recall_gain=True)
    apply_lora(model, r_attn=margs.get("lora_attn", 16), r_mlp=margs.get("lora_mlp", 0))
    load_ckpt(model, args.ckpt)
    for p in model.parameters():
        p.requires_grad_(False)
    for w in wrappers.values():
        w.null_logit.data.fill_(-1e9)
        w.use_coarse = False
    print(f"[{now()}] loaded {args.ckpt}", flush=True)

    toks = load_tokens(os.path.join(ROOT, "data", "wt2_test.pt"))
    out = {"exp": "v2_context", "ckpt": args.ckpt, "tag": args.tag, "ts": now(), "curve": {}}

    for T, n in zip(Ts, nch):
        chunks = chunks_from_tokens(toks, T, max_chunks=n)
        rec = {"n_chunks": len(chunks)}
        set_sparse_enabled(False)
        r = quick_ppl(model, chunks, device=DEV)
        rec["dense_ppl"] = r["ppl"]
        print(f"[T={T}] dense ppl={r['ppl']:.3f} ({len(chunks)} chunks)", flush=True)

        for rho in (0.25, 0.5):
            for w in wrappers.values():
                w.budget = rho
            set_sparse_enabled(True)
            r = quick_ppl(model, chunks, device=DEV)
            set_sparse_enabled(False)
            rec[f"sparse_{rho}"] = r["ppl"]
            rec[f"delta_{rho}"] = r["ppl"] - rec["dense_ppl"]
            print(f"[T={T}] sparse rho={rho}: ppl={r['ppl']:.3f} "
                  f"(delta={rec[f'delta_{rho}']:+.3f})", flush=True)

        # recall on 3 chunks at rho=0.25
        for w in wrappers.values():
            w.budget = 0.25
        set_sparse_enabled(True)
        acc = []
        for i in range(min(3, len(chunks))):
            begin_capture()
            model(input_ids=chunks[i].to(DEV).unsqueeze(0))
            acc.append([d["recall_mean"] for d in routing_diagnostics(end_capture())])
        set_sparse_enabled(False)
        t = torch.tensor(acc)
        rec["recall_mean_r25"] = float(t.mean())
        rec["recall_first"] = float(t[:, 0].mean())
        rec["recall_last"] = float(t[:, -1].mean())
        out["curve"][f"T{T}"] = rec
        print(f"[T={T}] recall@0.25 mean={rec['recall_mean_r25']:.3f}", flush=True)

    append_jsonl(os.path.join(ROOT, "results", "results.jsonl"), out)
    print(f"[{now()}] v2_context done", flush=True)


if __name__ == "__main__":
    main()
