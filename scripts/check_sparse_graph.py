"""Sparse-graph parity check: CUDA-graph decoder vs fixed eager paths, T0=1024.

Background: the eager sparse decode harnesses (lcsa/decode.py, lcsa/decode_fast.py)
used to include the unwritten future slots of the partial current block in the
softmax (zero keys -> score 0 -> softmax weight exp(0)=1). The training-time
semantics (gather_sparse_attention / triton kernel: kpos <= qpos) masks those
slots, and so does the CUDA-graph decoder — which is why graph-vs-eager parity
failed. After fixing both eager paths, this script verifies on identical
prefills that

  * per-layer block selection matches (graph idx_dbg vs eager last_sel), and
  * first-step logits parity is tight (naive is the primary reference).

T0=1024 is a block boundary: worst case (63 future slots in the current block).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from lcsa.decode import decode_step, prefill_and_record
from lcsa.decode_fast import decode_step_fast, prefill_fast
from lcsa.decode_graph import GraphDecoder
from lcsa.harness import load_model
from lcsa.lora import apply_lora, merge_lora
from lcsa.model import patch_model, set_sparse_enabled

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


def parity(a, b):
    return [round(float((a - b).abs().max()), 7),
            round(float((a.argmax(-1) == b.argmax(-1)).float().mean()), 5)]


def main():
    torch.manual_seed(0)
    model = load_model(os.path.join(ROOT, "models", "Qwen2.5-0.5B"), device=DEV)
    ckpt_path = os.path.join(ROOT, "checkpoints", "v2_best.pt")
    meta = load_ckpt(model, ckpt_path)
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
    load_ckpt(model, ckpt_path)
    for p in model.parameters():
        p.requires_grad_(False)
    for w in wrappers.values():
        w.null_logit.data.fill_(-1e9)
    merge_lora(model)
    budget = wrappers[0].budget

    T0, n = 1024, 64
    ids = torch.randint(0, 1000, (1, T0), device=DEV)
    tok = torch.tensor([[100]], device=DEV)
    max_new = n + 8

    # --- naive sparse (primary historical reference harness)
    set_sparse_enabled(True)
    _, c4 = prefill_and_record(model, ids, block=64, max_new_tokens=max_new)
    lg_ns = decode_step(model, tok, c4, T0, sparse=True, budget=budget, use_ksum=False)
    sel_ns = [cl["last_sel"].tolist() for cl in c4.layers]
    set_sparse_enabled(False)

    # --- fast sparse
    _, c5 = prefill_fast(model, ids, block=64, max_new_tokens=max_new)
    lg_fs = decode_step_fast(model, tok, c5, T0, mode="sparse", budget=budget)
    sel_fs = [cl["last_sel"].tolist() for cl in c5.layers]

    # --- CUDA-graph sparse on a fresh identical prefill
    _, c6 = prefill_fast(model, ids, block=64, max_new_tokens=max_new)
    max_len = ((T0 + max_new + 63) // 64) * 64
    gs = GraphDecoder(model, max_len, mode="sparse", block=64, budget=budget)
    gs.load_prefill(c6, T0)
    lg_sg = gs.step(100).clone()
    torch.cuda.synchronize()
    sel_sg = [t.tolist() for t in gs.idx_dbg]

    n_layers = len(sel_ns)
    mism_nf = [i for i in range(n_layers) if sorted(sel_ns[i]) != sorted(sel_fs[i])]
    mism_ng = [i for i in range(n_layers) if sorted(sel_ns[i]) != sorted(sel_sg[i])]
    print(f"[T0={T0}] parity naive-vs-fast : {parity(lg_ns, lg_fs)}", flush=True)
    print(f"[T0={T0}] parity naive-vs-graph: {parity(lg_ns, lg_sg)}", flush=True)
    print(f"[T0={T0}] selection naive==fast : {n_layers - len(mism_nf)}/{n_layers} layers",
          flush=True)
    print(f"[T0={T0}] selection naive==graph: {n_layers - len(mism_ng)}/{n_layers} layers",
          flush=True)
    if mism_ng:
        i = mism_ng[0]
        print(f"  first mismatch layer {i}: naive={sel_ns[i]} graph={sel_sg[i]}", flush=True)


if __name__ == "__main__":
    main()
