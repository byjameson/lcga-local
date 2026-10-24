import torch, time
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    B, H, S, D = 2, 4, 256, 64
    q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)

    def mask_mod(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    bm = create_block_mask(mask_mod, B, H, S, S, device='cuda')
    out = flex_attention(q, k, v, block_mask=bm)
    torch.cuda.synchronize()
    print("flex_attention WORKS, out shape:", tuple(out.shape))
    t0 = time.time()
    for _ in range(10):
        out = flex_attention(q, k, v, block_mask=bm)
    torch.cuda.synchronize()
    print("flex ok, 10 iters sec:", round(time.time() - t0, 3))
except Exception as e:
    print("flex_attention FAILED:", type(e).__name__, str(e)[:300])

# Also test compiling / simple sdpa perf baseline
import torch.nn.functional as F
q = torch.randn(1, 14, 2048, 64, device='cuda', dtype=torch.float16)
k = torch.randn(1, 14, 2048, 64, device='cuda', dtype=torch.float16)
v = torch.randn(1, 14, 2048, 64, device='cuda', dtype=torch.float16)
for _ in range(3):
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(20):
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
torch.cuda.synchronize()
print("sdpa N=2048 per-iter ms:", round((time.time() - t0) / 20 * 1000, 3))
