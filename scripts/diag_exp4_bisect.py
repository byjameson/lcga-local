"""Bisect: does running the sparse (Triton) path corrupt the subsequent dense sdpa timing?"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from lcsa.harness import load_model
from lcsa.lora import apply_lora, merge_lora
from lcsa.model import patch_model, set_sparse_enabled

DEV = "cuda"
MODEL = "models/Qwen2.5-0.5B"
CKPT = "checkpoints/v2_best.pt"


def bench(fn, warmup=3, iters=7):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000


def load_ckpt(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for i, layer in enumerate(model.model.layers):
        w = layer.self_attn
        own = dict(w.named_parameters())
        for n, t in ckpt["state"][i].items():
            if n in own:
                own[n].data.copy_(t.to(own[n].device))


print("flash_sdp", torch.backends.cuda.flash_sdp_enabled(),
      "mem_eff", torch.backends.cuda.mem_efficient_sdp_enabled(),
      "math", torch.backends.cuda.math_sdp_enabled(), flush=True)

model = load_model(MODEL, device=DEV)
wrappers = patch_model(model, budget=0.25, n_concepts=64, graph_rank=16,
                       use_coarse=False, null_init=-1e9, recall_gain=True)
load_ckpt(model, CKPT)
apply_lora(model, r_attn=16, r_mlp=0)
load_ckpt(model, CKPT)
merge_lora(model)
for w in wrappers.values():
    w.null_logit.data.fill_(-1e9)

ids4096 = torch.randint(0, 1000, (1, 4096), device=DEV)
dense = lambda: model(input_ids=ids4096, logits_to_keep=1)

set_sparse_enabled(False)
print("dense@4096 before sparse: %.1f ms  (alloc %.2fGB reserved %.2fGB)" % (
    bench(dense), torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9), flush=True)

# run the sparse path once at 2048 (Triton kernel compiles/runs)
ids2048 = torch.randint(0, 1000, (1, 2048), device=DEV)
set_sparse_enabled(True)
for _ in range(3):
    model(input_ids=ids2048, logits_to_keep=1)
torch.cuda.synchronize()
set_sparse_enabled(False)
torch.cuda.empty_cache()
print("after sparse@2048: alloc %.2fGB reserved %.2fGB" % (
    torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9), flush=True)

print("dense@4096 after sparse: %.1f ms" % bench(dense), flush=True)
torch.cuda.empty_cache()
print("dense@4096 after empty_cache: %.1f ms" % bench(dense), flush=True)

# also try a fresh allocation of the ids tensor before measuring
del ids4096
torch.cuda.empty_cache()
ids4096b = torch.randint(0, 1000, (1, 4096), device=DEV)
dense2 = lambda: model(input_ids=ids4096b, logits_to_keep=1)
print("dense@4096 new ids: %.1f ms" % bench(dense2), flush=True)
