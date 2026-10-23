"""EXP3: full evaluation of a trained LCGA checkpoint (PPL, KL, agreement, diagnostics, controls)."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from lcsa.data import load_tokens, chunks_from_tokens
from lcsa.harness import load_model, quick_ppl
from lcsa.metrics import append_jsonl, now, eval_pair, routing_diagnostics, cuda_time_ms
from lcsa.lora import apply_lora, merge_lora
from lcsa.model import (patch_model, set_sparse_enabled, begin_capture, end_capture,
                        set_selection_mode, set_profiling, get_profile)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "Qwen2.5-0.5B")
DATA_DIR = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results", "results.jsonl")
DEV = "cuda"


def load_ckpt(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    base = model.model
    for i, layer in enumerate(base.layers):
        w = layer.self_attn
        sd = ckpt["state"][i]
        own = dict(w.named_parameters())
        for n, t in sd.items():
            if n in own:
                own[n].data.copy_(t.to(own[n].device))
    return ckpt.get("meta", {})


@torch.no_grad()
def recall_stats(model, chunks, device, max_chunks=8):
    set_sparse_enabled(True)
    acc = []
    for i in range(min(max_chunks, len(chunks))):
        begin_capture()
        model(input_ids=chunks[i].to(device).unsqueeze(0))
        cap = end_capture()
        diag = routing_diagnostics(cap)
        acc.append([d["recall_mean"] for d in diag])
    set_sparse_enabled(False)
    t = torch.tensor(acc)
    return {"mean": float(t.mean()), "per_layer": t.mean(0).tolist(),
            "first": float(t[:, 0].mean()), "last": float(t[:, -1].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--tag", type=str, default="eval")
    ap.add_argument("--max_chunks", type=int, default=128)
    ap.add_argument("--seq", type=int, default=1024)    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--v2", action="store_true", help="stage-2 LoRA checkpoint (clean config)")
    ap.add_argument("--merge", action="store_true", help="merge LoRA into base weights")
    args = ap.parse_args()

    model = load_model(MODEL_DIR, device=DEV)
    meta = load_ckpt(model, args.ckpt)
    margs = meta.get("args", {})
    print(f"[{now()}] loaded ckpt {args.ckpt} meta={meta}", flush=True)
    wrappers = patch_model(model, budget=margs.get("budget", 0.25), block=64,
                           n_concepts=margs.get("n_concepts", 64),
                           graph_rank=margs.get("graph_rank", 16),
                           dynamic=margs.get("dynamic", False),
                           content_rank=margs.get("content_rank", 0),
                           rel_bias=not margs.get("no_rel_bias", False),
                           use_coarse=False, null_init=-1e9, recall_gain=args.v2)
    if args.v2:
        apply_lora(model, r_attn=margs.get("lora_attn", 16), r_mlp=margs.get("lora_mlp", 0))
    load_ckpt(model, args.ckpt)   # second pass now that modules exist with right config
    for w in wrappers.values():
        w.null_logit.data.fill_(-1e9)   # v1 ckpts carry the buggy -3; clean config forces off
    if args.merge:
        merge_lora(model)
        print("[lora] merged into base weights", flush=True)

    tokens = {k: load_tokens(os.path.join(DATA_DIR, f"{k}.pt")) for k in
              ["wt2_train", "wt2_test", "wt103_test", "tinystories_valid"]}
    maxc = 32 if args.quick else args.max_chunks
    T = args.seq
    sets = {
        f"dev_wt2_test_{T}": (tokens["wt2_test"], T, maxc),
        f"test_wt103_{T}": (tokens["wt103_test"], T, maxc),
        f"test_tinystories_{T}": (tokens["tinystories_valid"], T, maxc),
    }

    def dense_fn(ids):
        set_sparse_enabled(False)
        return model(input_ids=ids).logits

    def sparse_fn(ids):
        set_sparse_enabled(True)
        return model(input_ids=ids).logits

    out = {"exp": "exp3_eval", "ckpt": args.ckpt, "tag": args.tag, "ts": now(),
           "meta": meta, "budget": margs.get("budget", 0.25), "sets": {}}

    for name, (toks, T, mc) in sets.items():
        chunks = chunks_from_tokens(toks, T, max_chunks=mc)
        t0 = time.time()
        res = eval_pair(dense_fn, sparse_fn, chunks, DEV)
        res["sec"] = round(time.time() - t0, 1)
        set_sparse_enabled(True)
        rec = recall_stats(model, chunks, DEV, max_chunks=4)
        set_sparse_enabled(False)
        res["recall"] = rec
        out["sets"][name] = res
        print(f"[{name}] teacher_ppl={res['ppl_teacher']:.3f} student_ppl={res['ppl_student']:.3f} "
              f"kl(t||s)={res['kl_teacher_student']:.4f} top1={res['top1_agreement']:.4f} "
              f"recall={rec['mean']:.3f} ({res['sec']}s)", flush=True)

    # ---- budget sweep on dev (sparse PPL only, dense computed once) -----------
    dev_chunks = chunks_from_tokens(tokens["wt2_test"], T, max_chunks=maxc)
    set_sparse_enabled(False)
    dense_dev = quick_ppl(model, dev_chunks, device=DEV)
    sweep = {"dense_ppl": dense_dev["ppl"]}
    for rho in [0.125, 0.25, 0.5, 1.0]:
        for w in wrappers.values():
            w.budget = rho
        set_sparse_enabled(True)
        sp = quick_ppl(model, dev_chunks, device=DEV)
        set_sparse_enabled(False)
        rec = recall_stats(model, dev_chunks, DEV, max_chunks=3)
        sweep[f"rho_{rho}"] = {"sparse_ppl": sp["ppl"], "delta": sp["ppl"] - dense_dev["ppl"],
                               "recall": rec["mean"]}
        print(f"[sweep] rho={rho}: sparse_ppl={sp['ppl']:.3f} (delta={sp['ppl']-dense_dev['ppl']:+.3f}) "
              f"recall={rec['mean']:.3f}", flush=True)
    for w in wrappers.values():
        w.budget = margs.get("budget", 0.25)
    out["budget_sweep"] = sweep

    # ---- controls: random & local selection at trained budget -----------------
    for mode in ["random", "local"]:
        set_selection_mode(mode)
        set_sparse_enabled(True)
        sp = quick_ppl(model, dev_chunks, device=DEV)
        rec = recall_stats(model, dev_chunks, DEV, max_chunks=3)
        set_sparse_enabled(False)
        out[f"control_{mode}"] = {"ppl": sp["ppl"], "delta": sp["ppl"] - dense_dev["ppl"],
                                  "recall": rec["mean"]}
        print(f"[control] {mode}: ppl={sp['ppl']:.3f} (delta={sp['ppl']-dense_dev['ppl']:+.3f}) "
              f"recall={rec['mean']:.3f}", flush=True)
    set_selection_mode("topk")

    with open(os.path.join(ROOT, "results", f"exp3_{args.tag}.json"), "w") as f:
        json.dump(out, f, indent=1)
    append_jsonl(RESULTS, out)
    print(f"[{now()}] exp3 done", flush=True)


if __name__ == "__main__":
    main()
