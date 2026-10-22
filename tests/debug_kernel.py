import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from lcsa.attention import dense_attention_ref, sparse_attention_triton, gather_sparse_attention

torch.manual_seed(0)
dev = 'cuda'
B, H, G, T, D = 1, 2, 1, 128, 64
scale = D ** -0.5
q = torch.randn(B, H, T, D, device=dev, dtype=torch.float16) * 0.5
k = torch.randn(B, G, T, D, device=dev, dtype=torch.float16) * 0.5
v = torch.randn(B, G, T, D, device=dev, dtype=torch.float16)
NB = T // 64

idx = torch.full((B, NB, NB), -1, dtype=torch.int32, device=dev)
for I in range(NB):
    idx[:, I, :I + 1] = torch.arange(I + 1, device=dev, dtype=torch.int32)

out_k = sparse_attention_triton(q, k, v, idx, scale, null_logit=torch.zeros(H, device=dev))
out_d, _ = dense_attention_ref(q, k, v, scale)
out_g = gather_sparse_attention(q, k, v, idx, scale)
# sdpa reference
ke = k.repeat_interleave(H // G, dim=1)
ve = v.repeat_interleave(H // G, dim=1)
out_s = F.scaled_dot_product_attention(q, ke, ve, is_causal=True)

print("kernel vs gather  :", (out_k.float() - out_g.float()).abs().max().item())
print("kernel vs dense   :", (out_k.float() - out_d.float()).abs().max().item())
print("gather vs dense   :", (out_g.float() - out_d.float()).abs().max().item())
print("dense  vs sdpa    :", (out_d.float() - out_s.float()).abs().max().item())

diff = (out_k.float() - out_d.float()).abs()
am = torch.argmax(diff)
print("argmax of diff:", torch.unravel_index(am, diff.shape))
dg = (out_k.float() - out_g.float()).abs()
print("argmax kernel-gather:", torch.unravel_index(torch.argmax(dg), dg.shape))
for iblk in range(NB):
    print(f"  qblock {iblk}: k-d={diff[:, :, iblk*64:(iblk+1)*64].max().item():.4f}  k-g={dg[:, :, iblk*64:(iblk+1)*64].max().item():.4f}")
print("kernel[0,0,:3]:", out_k[0, 0, :3, :3].tolist())
print("gather[0,0,:3]:", out_g[0, 0, :3, :3].tolist())
print("dense [0,0,:3]:", out_d[0, 0, :3, :3].tolist())
