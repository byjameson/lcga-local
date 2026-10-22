"""Correctness + speed tests for the LCGA Triton sparse attention kernel."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import torch.nn.functional as F

from lcsa.attention import (dense_attention_ref, gather_sparse_attention,
                            sparse_attention_triton, build_simple_idx, repeat_kv)

torch.manual_seed(0)
dev = 'cuda'
B, H, G, T, D = 2, 14, 2, 512, 64
scale = D ** -0.5
q = torch.randn(B, H, T, D, device=dev, dtype=torch.float16) * 0.5
k = torch.randn(B, G, T, D, device=dev, dtype=torch.float16) * 0.5
v = torch.randn(B, G, T, D, device=dev, dtype=torch.float16)

NB = T // 64

# ---- test 1: full causal idx vs dense reference -----------------------------
idx_full = torch.full((B, NB, NB), -1, dtype=torch.int32, device=dev)
for I in range(NB):
    idx_full[:, I, :I + 1] = torch.arange(I + 1, device=dev, dtype=torch.int32)
null = torch.full((H,), -1e9, device=dev)   # effectively "no null key"
out_k = sparse_attention_triton(q, k, v, idx_full, scale, null_logit=null)
out_d, _ = dense_attention_ref(q, k, v, scale)
d1 = (out_k.float() - out_d.float()).abs().max().item()
print(f"test1 full-causal kernel vs dense-ref: max|diff| = {d1:.2e}")

# ---- test 2: partial selection vs gather reference --------------------------
idx_p = torch.full((B, NB, NB), -1, dtype=torch.int32, device=dev)
for I in range(NB):
    m = max(2, int(round(0.25 * (I + 1))))
    sel = {0, I} | set(torch.randint(0, I + 1, (m,)).tolist())
    sel = sorted(sel)[:m]
    idx_p[:, I, :len(sel)] = torch.tensor(sel, dtype=torch.int32, device=dev)
    # blank the rest
out_k2 = sparse_attention_triton(q, k, v, idx_p, scale, null_logit=null)
out_g2 = gather_sparse_attention(q, k, v, idx_p, scale, null_logit=null)
d2 = (out_k2.float() - out_g2.float()).abs().max().item()
print(f"test2 partial kernel vs gather-ref  : max|diff| = {d2:.2e}")

# ---- test 3: null logit ------------------------------------------------------
null3 = torch.randn(H, device=dev) * 1.5
out_k3 = sparse_attention_triton(q, k, v, idx_p, scale, null_logit=null3)
out_g3 = gather_sparse_attention(q, k, v, idx_p, scale, null_logit=null3)
d3 = (out_k3.float() - out_g3.float()).abs().max().item()
print(f"test3 null-logit kernel vs gather   : max|diff| = {d3:.2e}")

assert d1 < 5e-2 and d2 < 5e-2 and d3 < 5e-2, "kernel correctness failed"

# ---- test 4: speed ------------------------------------------------------------
def timeit(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]

print("\n--- speed (B=1, H=14, G=2, D=64, fp16) ---")
for T2 in [1024, 2048, 4096]:
    NB2 = T2 // 64
    q2 = torch.randn(1, H, T2, D, device=dev, dtype=torch.float16) * 0.5
    k2 = torch.randn(1, G, T2, D, device=dev, dtype=torch.float16) * 0.5
    v2 = torch.randn(1, G, T2, D, device=dev, dtype=torch.float16)
    ke = repeat_kv(k2, H // G)
    ve = repeat_kv(v2, H // G)
    ms_sdpa = timeit(lambda: F.scaled_dot_product_attention(q2, ke, ve, is_causal=True))
    for budget in [0.125, 0.25, 0.5]:
        MMAX = max(2, int(round(budget * NB2)))
        idx2 = torch.full((1, NB2, MMAX), -1, dtype=torch.int32, device=dev)
        for I in range(NB2):
            m = max(2, min(I + 1, MMAX))
            sel = sorted(({0, I} | set(torch.randint(0, I + 1, (m,)).tolist())))[:m]
            idx2[0, I, :len(sel)] = torch.tensor(sel, dtype=torch.int32, device=dev)
        ms_k = timeit(lambda: sparse_attention_triton(q2, k2, v2, idx2, scale, null_logit=null), warmup=5, iters=30)
        print(f"T={T2:5d} budget={budget:<5} kernel={ms_k:7.3f} ms   (sdpa dense={ms_sdpa:7.3f} ms, ratio={ms_sdpa/ms_k:.2f}x)")
        del idx2
    print()
print("ALL TESTS PASSED")
