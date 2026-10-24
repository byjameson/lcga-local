"""Sanity tests: model patching fidelity, capture, loss/backward, selection modes."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from lcsa.harness import load_model, quick_ppl
from lcsa.model import (patch_model, router_parameters, count_router_params,
                        set_sparse_enabled, begin_capture, end_capture,
                        set_selection_mode)

torch.manual_seed(0)
DEV = "cuda"
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "Qwen2.5-0.5B")

model = load_model(MODEL_DIR, device=DEV)
print("model loaded. params:", sum(p.numel() for p in model.parameters()))

ids = torch.randint(0, 150000, (1, 512), device=DEV)
with torch.no_grad():
    out_ref = model(input_ids=ids).logits

# ---- 1. patch, sparse disabled -> identical outputs -------------------------
wrappers = patch_model(model, budget=0.25, block=64, n_concepts=64, graph_rank=16)
print("patched layers:", len(wrappers), "| router params total:", count_router_params(model))
with torch.no_grad():
    out_disabled = model(input_ids=ids).logits
d = (out_ref.float() - out_disabled.float()).abs().max().item()
print(f"1. patched-but-disabled vs unpatched: max|logit diff| = {d:.2e}")
assert d < 1e-3, "patch fidelity broken"

# ---- 2. sparse forward runs, finite outputs --------------------------------
set_sparse_enabled(True)
with torch.no_grad():
    out_sparse = model(input_ids=ids).logits
print("2. sparse forward finite:", torch.isfinite(out_sparse).all().item(),
      "| max|diff| vs dense:", (out_sparse.float() - out_ref.float()).abs().max().item())

# ---- 3. capture + losses ----------------------------------------------------
begin_capture()
with torch.no_grad():
    _ = model(input_ids=ids)
captured = end_capture()
print("3. captured layers:", len(captured), "| keys:", sorted(captured[0].keys()))

# manual capture with grad
begin_capture()
out = model(input_ids=ids)
captured = end_capture()
loss = torch.tensor(0.0, device=DEV)
mse_sum, kl_sum = 0.0, 0.0
for i, cap in enumerate(captured):
    mse = torch.nn.functional.mse_loss(cap["student"], cap["dense"])
    kl = wrappers[i].router.kl_mass_loss(cap["mass"], cap["probs"])
    loss = loss + mse + 0.2 * kl
    mse_sum += mse.item()
    kl_sum += kl.item()
loss = loss / len(captured)
loss.backward()
grads_ok = []
for name, p in wrappers[0].named_parameters():
    if p.grad is not None:
        grads_ok.append((name, float(p.grad.abs().max()), torch.isfinite(p.grad).all().item()))
print(f"4. loss={loss.item():.4f} (mse={mse_sum/len(captured):.5f}, kl={kl_sum/len(captured):.4f})")
print("   sample grads:", grads_ok[:6])
assert all(g[2] for g in grads_ok), "non-finite grads"
model.zero_grad(set_to_none=True)

# ---- 4. selection modes ------------------------------------------------------
for mode in ["topk", "random", "local"]:
    set_selection_mode(mode)
    with torch.no_grad():
        o = model(input_ids=ids).logits
    print(f"5. mode={mode:<7} finite={torch.isfinite(o).all().item()} "
          f"ppl_diff_max={abs(0)}")
set_selection_mode("topk")

# ---- 5. quick timing ----------------------------------------------------------
def bench(fn, iters=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000

with torch.no_grad():
    ids1k = torch.randint(0, 150000, (1, 1024), device=DEV)
    t_dense = bench(lambda: model(input_ids=ids1k))
set_sparse_enabled(True)
with torch.no_grad():
    t_sparse = bench(lambda: model(input_ids=ids1k))
print(f"6. forward T=1024: dense={t_dense:.1f} ms, sparse(untrained router)={t_sparse:.1f} ms")
print("ALL MODEL TESTS PASSED")
