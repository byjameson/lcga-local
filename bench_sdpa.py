import torch, torch.nn.functional as F

torch.manual_seed(0)
print("torch", torch.__version__, "cuda", torch.cuda.get_device_name(0))

def bench(N, H=14, D=64, B=1, warmup=15, iters=60):
    q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
    for _ in range(warmup):
        F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record(); F.scaled_dot_product_attention(q, k, v, is_causal=True); e.record()
        torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    del q, k, v
    torch.cuda.empty_cache()
    ts = sorted(ts)
    return {"median": round(ts[len(ts)//2], 3), "min": round(ts[0], 3), "p90": round(ts[int(len(ts)*0.9)], 3)}

for N, B in [(1024, 1), (2048, 1), (4096, 1), (1024, 8)]:
    for rep in range(3):
        print(f"N={N} B={B} rep{rep}", bench(N, B=B), flush=True)
