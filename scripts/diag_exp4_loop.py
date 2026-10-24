"""Replicate exp4's prefill loop exactly, print attention implementation + timings."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from lcsa.harness import load_model
from lcsa.lora import apply_lora, merge_lora
from lcsa.metrics import cuda_time_ms
from lcsa.model import get_profile, patch_model, set_profiling, set_sparse_enabled

DEV = "cuda"
MODEL = "models/Qwen2.5-0.5B"
CKPT = "checkpoints/v2_best.pt"


def load_ckpt(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for i, layer in enumerate(model.model.layers):
        w = layer.self_attn
        own = dict(w.named_parameters())
        for n, t in ckpt["state"][i].items():
            if n in own:
                own[n].data.copy_(t.to(own[n].device))


model = load_model(MODEL, device=DEV)
print("impl after load:", model.model.layers[0].self_attn.config._attn_implementation, flush=True)
wrappers = patch_model(model, budget=0.25, n_concepts=64, graph_rank=16,
                       use_coarse=False, null_init=-1e9, recall_gain=True)
print("impl after patch:", model.model.layers[0].self_attn.orig.config._attn_implementation, flush=True)
load_ckpt(model, CKPT)
apply_lora(model, r_attn=16, r_mlp=0)
load_ckpt(model, CKPT)
merge_lora(model)
for w in wrappers.values():
    w.null_logit.data.fill_(-1e9)
print("impl after lora/merge:", model.model.layers[0].self_attn.orig.config._attn_implementation, flush=True)

for T in [1024, 2048, 4096]:
    ids = torch.randint(0, 1000, (1, T), device=DEV)
    set_sparse_enabled(False)
    ms_dense = cuda_time_ms(lambda: model(input_ids=ids, logits_to_keep=1), warmup=3, iters=10)
    set_sparse_enabled(True)
    ms_sparse = cuda_time_ms(lambda: model(input_ids=ids, logits_to_keep=1), warmup=3, iters=10)
    set_sparse_enabled(False)
    set_sparse_enabled(True)
    set_profiling(True)
    with torch.no_grad():
        for _ in range(3):
            model(input_ids=ids, logits_to_keep=1)
    prof = get_profile()
    set_profiling(False)
    set_sparse_enabled(False)
    print("T=%d dense=%.1f sparse=%.1f impl=%s reserved=%.2fGB" % (
        T, ms_dense["median"], ms_sparse["median"],
        model.model.layers[0].self_attn.orig.config._attn_implementation,
        torch.cuda.memory_reserved() / 1e9), flush=True)
    del ids
    torch.cuda.empty_cache()
