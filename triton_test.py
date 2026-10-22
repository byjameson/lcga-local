import torch, time
import torch.nn.functional as F

print("torch", torch.__version__)
try:
    import triton
    import triton.language as tl
    print("triton", triton.__version__)

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    x = torch.randn(1024, device='cuda')
    y = torch.randn(1024, device='cuda')
    out = torch.empty(1024, device='cuda')
    add_kernel[(4,)](x, y, out, 1024, BLOCK=256)
    torch.cuda.synchronize()
    print("triton custom kernel WORKS, err:", (out - (x + y)).abs().max().item())
except Exception as e:
    print("triton FAILED:", type(e).__name__, str(e)[:400])

# flex attention compiled test
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    B, H, S, D = 1, 14, 2048, 64
    q = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, S, D, device='cuda', dtype=torch.float16)

    def mask_mod(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    bm = create_block_mask(mask_mod, B, H, S, S, device='cuda')
    compiled_flex = torch.compile(flex_attention, fullgraph=True)
    out = compiled_flex(q, k, v, block_mask=bm)
    torch.cuda.synchronize()
    print("compiled flex WORKS")
    for _ in range(3):
        out = compiled_flex(q, k, v, block_mask=bm)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        out = compiled_flex(q, k, v, block_mask=bm)
    torch.cuda.synchronize()
    print("compiled flex N=2048 ms:", round((time.time() - t0) / 20 * 1000, 3))
except Exception as e:
    print("compiled flex FAILED:", type(e).__name__, str(e)[:500])

# sdpa counterpart
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
print("sdpa N=2048 ms:", round((time.time() - t0) / 20 * 1000, 3))
