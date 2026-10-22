"""C phase: decode benchmark — naive harness (lcsa/decode.py) vs optimized (lcsa/decode_fast.py).

Per context T0: dense-naive, dense-fast (GQA SDPA), dense-fast (expanded cache),
sparse-naive, sparse-fast — each on a fresh identical prefill; 3 warmup + N timed
steps; logits parity checked on the first step vs the naive path of the same kind.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from lcsa.decode import decode_step, prefill_and_record
from lcsa.decode_fast import decode_step_fast, prefill_fast
from lcsa.decode_graph import GraphDecoder
from lcsa.harness import load_model
from lcsa.lora import apply_lora, merge_lora
from lcsa.metrics import append_jsonl, now
from lcsa.model import LCAGAttention, patch_model, set_sparse_enabled

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


def parity(a, b):
    """(max abs logit diff, top-1 agreement) between two (1,1,V) logit tensors."""
    return [round(float((a - b).abs().max()), 7),
            round(float((a.argmax(-1) == b.argmax(-1)).float().mean()), 5)]


def timed_steps(step_fn, start_pos, n_steps, warmup=3):
    for i in range(warmup):
        step_fn(start_pos + i)
    torch.cuda.synchronize()
    t0 = time.time()
    for i in range(n_steps):
        step_fn(start_pos + warmup + i)
    torch.cuda.synchronize()
    return (time.time() - t0) / n_steps * 1000


def gqa_supported():
    try:
        q = torch.randn(1, 4, 1, 16, device=DEV, dtype=torch.float16)
        k = torch.randn(1, 2, 8, 16, device=DEV, dtype=torch.float16)
        v = torch.randn(1, 2, 8, 16, device=DEV, dtype=torch.float16)
        torch.nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        return True
    except Exception as e:  # unsupported backend combination
        print(f"[warn] enable_gqa not usable here: {type(e).__name__}: {e}", flush=True)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints", "v2_best.pt"))
    ap.add_argument("--tag", default="c_phase")
    ap.add_argument("--Ts", default="1024,2048,4096,8192")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--graph", action="store_true", help="also time the CUDA-graph decoder")
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
                           recall_gain=True)
    apply_lora(model, r_attn=margs.get("lora_attn", 16), r_mlp=margs.get("lora_mlp", 0))
    load_ckpt(model, args.ckpt)
    for p in model.parameters():
        p.requires_grad_(False)
    for w in wrappers.values():
        w.null_logit.data.fill_(-1e9)
    merge_lora(model)
    budget = wrappers[0].budget
    print(f"[{now()}] ready tag={args.tag} budget={budget}", flush=True)

    H = model.config.num_attention_heads
    has_gqa = gqa_supported()
    out = {"exp": "exp9_decode", "ts": now(), "tag": args.tag, "ckpt": args.ckpt,
           "steps": args.steps, "budget": budget, "gqa_sdpa": has_gqa, "contexts": {}}

    for T0 in Ts:
        ids = torch.randint(0, 1000, (1, T0), device=DEV)
        tok = torch.tensor([[100]], device=DEV)
        n = args.steps
        rec = {}
        try:
            # --- naive dense
            set_sparse_enabled(False)
            _, cache_nd = prefill_and_record(model, ids, block=64, max_new_tokens=n + 8)
            rec["dense_naive_ms"] = round(timed_steps(
                lambda p: decode_step(model, tok, cache_nd, p, sparse=False), T0, n), 3)
            # --- fast dense (GQA SDPA)
            if has_gqa:
                _, cache_fg = prefill_fast(model, ids, block=64, max_new_tokens=n + 8)
                rec["dense_fast_gqa_ms"] = round(timed_steps(
                    lambda p: decode_step_fast(model, tok, cache_fg, p, mode="dense",
                                               use_expanded=False), T0, n), 3)
            else:
                rec["dense_fast_gqa_ms"] = "unsupported"
            # --- fast dense (expanded cache)
            _, cache_fe = prefill_fast(model, ids, block=64, max_new_tokens=n + 8)
            cache_fe.expand_kv(H)
            rec["dense_fast_expanded_ms"] = round(timed_steps(
                lambda p: decode_step_fast(model, tok, cache_fe, p, mode="dense",
                                           use_expanded=True), T0, n), 3)
            # --- naive sparse
            set_sparse_enabled(True)
            _, cache_ns = prefill_and_record(model, ids, block=64, max_new_tokens=n + 8)
            rec["sparse_naive_ms"] = round(timed_steps(
                lambda p: decode_step(model, tok, cache_ns, p, sparse=True, budget=budget,
                                      use_ksum=False), T0, n), 3)
            set_sparse_enabled(False)
            # --- fast sparse
            _, cache_fs = prefill_fast(model, ids, block=64, max_new_tokens=n + 8)
            rec["sparse_fast_ms"] = round(timed_steps(
                lambda p: decode_step_fast(model, tok, cache_fs, p, mode="sparse",
                                           budget=budget), T0, n), 3)

            # --- parity on the first decode step (fresh caches, same pos/token)
            set_sparse_enabled(False)
            _, c1 = prefill_and_record(model, ids, block=64, max_new_tokens=n + 8)
            lg_nd = decode_step(model, tok, c1, T0, sparse=False)
            if has_gqa:
                _, c2 = prefill_fast(model, ids, block=64, max_new_tokens=n + 8)
                lg_fg = decode_step_fast(model, tok, c2, T0, mode="dense", use_expanded=False)
                rec["parity_dense_gqa"] = parity(lg_nd, lg_fg)
            _, c3 = prefill_fast(model, ids, block=64, max_new_tokens=n + 8)
            c3.expand_kv(H)
            lg_fe = decode_step_fast(model, tok, c3, T0, mode="dense", use_expanded=True)
            set_sparse_enabled(True)
            _, c4 = prefill_and_record(model, ids, block=64, max_new_tokens=n + 8)
            lg_ns = decode_step(model, tok, c4, T0, sparse=True, budget=budget, use_ksum=False)
            set_sparse_enabled(False)
            _, c5 = prefill_fast(model, ids, block=64, max_new_tokens=n + 8)
            lg_fs = decode_step_fast(model, tok, c5, T0, mode="sparse", budget=budget)
            rec["parity_dense_expanded"] = parity(lg_nd, lg_fe)
            rec["parity_sparse"] = parity(lg_ns, lg_fs)

            # --- CUDA-graph decode -------------------------------------------
            if args.graph:
                max_len = ((T0 + n + 8 + 63) // 64) * 64
                cache_p = c5  # fresh fast prefill (identical for all paths)
                try:
                    gd = GraphDecoder(model, max_len, mode="dense", block=64)
                    gd.load_prefill(cache_p, T0)
                    lg_dg = gd.step(100).clone()
                    torch.cuda.synchronize()
                    rec["parity_dense_graph"] = parity(lg_nd, lg_dg)
                    gd.load_prefill(cache_p, T0)
                    for _ in range(5):
                        gd.step(100)
                    torch.cuda.synchronize()
                    t_g = time.time()
                    for _ in range(n):
                        gd.step(100)
                    torch.cuda.synchronize()
                    rec["dense_graph_ms"] = round((time.time() - t_g) / n * 1000, 3)
                    del gd
                    torch.cuda.empty_cache()
                except Exception as e:
                    rec["dense_graph_ms"] = f"error: {type(e).__name__}: {str(e)[:150]}"
                    torch.cuda.empty_cache()
                try:
                    gs = GraphDecoder(model, max_len, mode="sparse", block=64, budget=budget)
                    gs.load_prefill(cache_p, T0)
                    lg_sg = gs.step(100).clone()
                    torch.cuda.synchronize()
                    # birincil referans naive sparse (gerçek anahtar semantiği;
                    # eager fast yol kısmi bloğun sıfır-slotlarını softmax'a katıyor)
                    rec["parity_sparse_graph"] = parity(lg_ns, lg_sg)
                    rec["parity_sparse_graph_vsf"] = parity(lg_fs, lg_sg)
                    gs.load_prefill(cache_p, T0)
                    for _ in range(5):
                        gs.step(100)
                    torch.cuda.synchronize()
                    t_g = time.time()
                    for _ in range(n):
                        gs.step(100)
                    torch.cuda.synchronize()
                    rec["sparse_graph_ms"] = round((time.time() - t_g) / n * 1000, 3)
                    del gs
                    torch.cuda.empty_cache()
                except Exception as e:
                    rec["sparse_graph_ms"] = f"error: {type(e).__name__}: {str(e)[:150]}"
                    torch.cuda.empty_cache()

            sp = {"sparse_fast": round(rec["dense_naive_ms"] / rec["sparse_fast_ms"], 3)}
            if has_gqa:
                sp["dense_gqa"] = round(rec["dense_naive_ms"] / rec["dense_fast_gqa_ms"], 3)
            rec["speedup_vs_dense_naive"] = sp
            if has_gqa:
                rec["sparse_vs_dense_fast"] = round(
                    rec["dense_fast_gqa_ms"] / rec["sparse_fast_ms"], 3)
            print(f"[ctx {T0}] dense naive={rec['dense_naive_ms']}ms gqa={rec.get('dense_fast_gqa_ms')}ms "
                  f"exp={rec['dense_fast_expanded_ms']}ms | sparse naive={rec['sparse_naive_ms']}ms "
                  f"fast={rec['sparse_fast_ms']}ms | graph dense={rec.get('dense_graph_ms')}ms "
                  f"sparse={rec.get('sparse_graph_ms')}ms | parity sparse(top1)="
                  f"{rec['parity_sparse'][1]} sgraph(top1)={rec.get('parity_sparse_graph', ['?', '?'])[1]}",
                  flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            rec["error"] = "OOM"
            print(f"[ctx {T0}] OOM", flush=True)
        except RuntimeError as e:
            torch.cuda.empty_cache()
            rec["error"] = f"{type(e).__name__}: {e}"[:300]
            print(f"[ctx {T0}] ERROR: {rec['error']}", flush=True)
        out["contexts"][f"T{T0}"] = rec
        del ids, tok
        torch.cuda.empty_cache()

    with open(os.path.join(ROOT, "results", f"exp9_{args.tag}.json"), "w") as f:
        json.dump(out, f, indent=1)
    append_jsonl(RESULTS, out)
    print(f"[{now()}] decode_ref done", flush=True)


if __name__ == "__main__":
    main()
