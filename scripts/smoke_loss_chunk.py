"""Chunk'li lm_head loss dogrulamasi: yeni MiniLM.loss == eski formul (fp tolerans)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from lcsa.mini_lm import MiniLM

torch.manual_seed(0)
m = MiniLM(vocab=512, dim=64, layers=2, heads=4, ctx=64, mlp_hidden=128)
x = torch.randint(0, 512, (3, 65))
chunk = 64

# eski formul (referans): tam logits + chunk'li CE
with torch.no_grad():
    logits = m(x[:, :-1])
    tgt = x[:, 1:].reshape(-1)
    lg = logits.reshape(-1, logits.size(-1)).float()
    tot = None
    for i in range(0, tgt.numel(), chunk):
        c = F.cross_entropy(lg[i: i + chunk], tgt[i: i + chunk], reduction="sum")
        tot = c if tot is None else tot + c
    ref = (tot / tgt.numel()).item()

new = m.loss(x, chunk=chunk)
d = abs(new.item() - ref)
print("eski=%.6f yeni=%.6f fark=%.2e" % (ref, new.item(), d))
assert d < 1e-4, "FARK BUYUK: %s" % d

new.backward()
g = m.tok.weight.grad
assert g is not None and torch.isfinite(g).all(), "grad problem"
hg = None
for p in m.blocks[0].parameters():
    if p.grad is not None:
        hg = p.grad
        break
assert hg is not None and torch.isfinite(hg).all(), "blok gradyani yok/bozuk"
print("grad OK, smoke PASS")
