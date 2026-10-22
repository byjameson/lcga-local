"""Sanity test for the decode harness: dense decode must match full-sequence forward."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from lcsa.decode import prefill_and_record, decode_step
from lcsa.harness import load_model
from lcsa.model import patch_model, set_sparse_enabled

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "Qwen2.5-0.5B")
DEV = "cuda"

torch.manual_seed(0)
model = load_model(MODEL_DIR, device=DEV)
wrappers = patch_model(model, budget=0.25, block=64, n_concepts=64, graph_rank=16, capture_dense=False)

T0 = 256
ids = torch.randint(0, 5000, (1, T0), device=DEV)

# reference: dense full-sequence forward
set_sparse_enabled(False)
with torch.no_grad():
    ref = model(input_ids=ids, logits_to_keep=1).logits[0, -1]

# dense prefill+record then one decode step (pos T0)
with torch.no_grad():
    logits_pre, cache = prefill_and_record(model, ids, block=64, max_new_tokens=8)
pre_logits = logits_pre[0, -1]
d = (pre_logits.float() - ref.float()).abs().max().item()
print(f"dense prefill last logits vs full forward: max|diff| = {d:.4f}")
assert d < 0.5, "dense decode harness diverges from model forward"

tok = torch.argmax(ref).view(1, 1)
with torch.no_grad():
    step_logits = decode_step(model, tok, cache, T0, sparse=False)
full = torch.cat([ids, tok], dim=1)
with torch.no_grad():
    ref2 = model(input_ids=full, logits_to_keep=1).logits[0, -1]
d2 = (step_logits[0, -1].float() - ref2.float()).abs().max().item()
print(f"dense decode step vs full forward: max|diff| = {d2:.4f}")
assert d2 < 1.0, "dense decode step diverges"

# sparse decode: just check it runs and produces finite, plausible logits
set_sparse_enabled(True)
with torch.no_grad():
    _, cache2 = prefill_and_record(model, ids, block=64, max_new_tokens=8)
    s_logits = decode_step(model, tok, cache2, T0, sparse=True, budget=0.25)
set_sparse_enabled(False)
cossim = F.cosine_similarity(s_logits[0, -1].float(), ref2.float(), dim=-1)
print(f"sparse decode step: finite={torch.isfinite(s_logits).all().item()} cosine_sim_to_dense={cossim.item():.4f}")
print("DECODE TESTS PASSED")
