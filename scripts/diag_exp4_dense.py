"""Diagnose exp4 dense slowness: isolate LoRA fp32 path + autograd + merge effects."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from lcsa.harness import load_model
from lcsa.lora import apply_lora, lora_parameters, merge_lora
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


model = load_model(MODEL, device=DEV)
wrappers = patch_model(model, budget=0.25, n_concepts=64, graph_rank=16,
                       use_coarse=False, null_init=-1e9, recall_gain=True)
load_ckpt(model, CKPT)
apply_lora(model, r_attn=16, r_mlp=0)
load_ckpt(model, CKPT)
for w in wrappers.values():
    w.null_logit.data.fill_(-1e9)
print("lora params requires_grad:", [p.requires_grad for p in list(lora_parameters(model))[:2]], flush=True)
set_sparse_enabled(False)

for T in [1024, 2048, 4096]:
    ids = torch.randint(0, 1000, (1, T), device=DEV)
    fn = lambda: model(input_ids=ids, logits_to_keep=1)

    a = bench(fn)  # as exp4: no no_grad, lora grads on
    with torch.no_grad():
        b = bench(fn)
    for p in lora_parameters(model):
        p.requires_grad_(False)
    c = bench(fn)
    merge_lora(model)
    d = bench(fn)  # merged, lora branch still computed but zeroed
    print(f"T={T}: as-exp4={a:.1f}ms  no_grad={b:.1f}ms  grads_off={c:.1f}ms  merged={d:.1f}ms", flush=True)
    del ids
    torch.cuda.empty_cache()
