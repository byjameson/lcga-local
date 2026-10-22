import torch, time, statistics, inspect
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask

torch.manual_seed(0)
dev = 'cuda'
print("BlockMask public API:", [m for m in dir(BlockMask) if not m.startswith('_')][:40])
try:
    print("from_kv_blocks sig:", inspect.signature(BlockMask.from_kv_blocks))
except Exception as e:
    print("no from_kv_blocks:", e)

def timeit(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = torch.cuda.Event(True); t1 = torch.cuda.Event(True)
        t0.record(); fn(); t1.record(); torch.cuda.synchronize()
        times.append(t0.elapsed_time(t1))
    return statistics.median(times)

H, D = 14, 64
cf = torch.compile(flex_attention, fullgraph=True)

def sparse_gather_attn(q, k, v, kidx, block_size, causal=True):
    # kidx: (nb, nsel) long tensor, sorted, all < nb
    B, H_, N, D_ = q.shape
    nb, nsel = kidx.shape
    bs = block_size
    qb = q.view(B, H_, nb, bs, D_)
    kb = k.view(B, H_, nb, bs, D_)[:, :, kidx]        # (B,H,nb,nsel,bs,D)
    vb = v.view(B, H_, nb, bs, D_)[:, :, kidx]        # (B,H,nb,nsel,bs,D)
    scores = torch.einsum("bhmid,bhmnjd->bhmnij", qb, kb) * (D_ ** -0.5)
    if causal:
        qpos = torch.arange(nb, device=q.device).view(nb, 1, 1, bs).expand(nb, 1, bs, 1) * bs
        i_pos = qpos + torch.arange(bs, device=q.device).view(1, 1, bs, 1)
        kpos = kidx.view(nb, nsel, 1, 1) * bs + torch.arange(bs, device=q.device).view(1, 1, 1, bs)
        mask = kpos <= i_pos  # (nb,nsel,bs,bs)
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhmnij,bhmnjd->bhmnid", probs, vb)
    return out.sum(dim=3).reshape(B, H_, N, D_)

for N in [1024, 2048, 4096]:
    q = torch.randn(1, H, N, D, device=dev, dtype=torch.float16)
    k = torch.randn(1, H, N, D, device=dev, dtype=torch.float16)
    v = torch.randn(1, H, N, D, device=dev, dtype=torch.float16)

    ms_sdpa = timeit(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
    print(f"N={N}  sdpa causal       : {ms_sdpa:7.3f} ms")

    def mask_mod(b, h, qi, kv):
        return qi >= kv
    bm = create_block_mask(mask_mod, 1, H, N, N, device=dev)
    ms_flex = timeit(lambda: cf(q, k, v, block_mask=bm), warmup=5, iters=30)
    print(f"N={N}  flex causal cmp   : {ms_flex:7.3f} ms")

    # block-sparse 25% budget flex
    bs = 64
    nb = N // bs
    nsel = max(2, int(round(nb * 0.25)))
    kidx = []
    for I in range(nb):
        idx = {0, I}
        while len(idx) < nsel:
            idx.add(int(torch.randint(0, nb, (1,))))
        kidx.append(sorted(idx)[:nsel])
    kidx = torch.tensor(kidx, device=dev)
    kv_num_blocks = torch.full((1, H, nb), nsel, dtype=torch.int32, device=dev)
    kv_indices = kidx.view(1, 1, nb, nsel).expand(1, H, nb, nsel).to(torch.int32).contiguous()
    try:
        bm_sparse = BlockMask.from_kv_blocks(kv_num_blocks, kv_indices, BLOCK_SIZE=bs, mask_mod=None)
        ms_flex_s = timeit(lambda: cf(q, k, v, block_mask=bm_sparse), warmup=5, iters=30)
        print(f"N={N}  flex sparse25%    : {ms_flex_s:7.3f} ms")
    except Exception as e:
        print(f"N={N}  flex sparse FAILED: {type(e).__name__}: {str(e)[:200]}")

    try:
        ms_g = timeit(lambda: sparse_gather_attn(q, k, v, kidx, bs), warmup=5, iters=30)
        print(f"N={N}  gather sparse25%  : {ms_g:7.3f} ms")
    except Exception as e:
        print(f"N={N}  gather sparse FAILED: {type(e).__name__}: {str(e)[:200]}")
    print()

# reference: what does full dense attendance cost in theory at N=4096
print("done")
