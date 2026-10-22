"""P4.1 CPU smoke: dongulu derinlik (loops) + geriye uyumluluk hizli kontrol.

Kontroller:
 1) loops, parametre EKLEMEZ (flat ile ayni param sayisi)
 2) cfg()'de loops alani var; varsayilan 1
 3) loops>1 forward'i etkiler (loss L'ye gore degisir)
 4) forward/backward + 40 adim mini egitimde loss duser
CPU'da kosar; GPU'ya dokunmaz.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from lcsa.mini_lm import MiniLM

torch.manual_seed(0)

m1 = MiniLM(vocab=512, dim=64, layers=4, heads=4, ctx=128, mlp_hidden=128, loops=2)
m2 = MiniLM(vocab=512, dim=64, layers=4, heads=4, ctx=128, mlp_hidden=128)  # loops=1 varsayilan
p1 = sum(p.numel() for p in m1.parameters())
p2 = sum(p.numel() for p in m2.parameters())
assert p1 == p2, f"param esit degil: {p1} vs {p2}"
assert m1.cfg()["loops"] == 2 and m2.cfg()["loops"] == 1
print(f"[ok] param esit: {p1} | cfg loops ok")

idx = torch.randint(0, 512, (2, 129))
with torch.no_grad():
    m1.loops = 1
    l1 = float(m1.loss(idx))
    m1.loops = 2
    l2 = float(m1.loss(idx))
    m1.loops = 4
    l4 = float(m1.loss(idx))
assert abs(l1 - l2) > 1e-6 and abs(l2 - l4) > 1e-6, "loops ciktiyi etkilemiyor!"
print(f"[ok] loops etkili: L1={l1:.3f} L2={l2:.3f} L4={l4:.3f}")

m1.train()
m1.loops = 2
opt = torch.optim.AdamW(m1.parameters(), lr=3e-3)
g = torch.Generator().manual_seed(1)
xb = torch.randint(0, 512, (4, 129), generator=g)   # SABIT batch: ezberlenebilir olmali
losses = []
for i in range(40):
    opt.zero_grad(set_to_none=True)
    loss = m1.loss(xb)
    loss.backward()
    opt.step()
    losses.append(float(loss.detach()))
first = sum(losses[:5]) / 5
last = sum(losses[-5:]) / 5
print(f"[ok] egitim (sabit batch): ilk5={first:.3f} son5={last:.3f}")
assert last < first, "loss dusmuyor!"
print("[SMOKE OK]")
