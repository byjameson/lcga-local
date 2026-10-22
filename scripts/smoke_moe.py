"""MoE-lite (exp11 --moe_experts) smoke: model matematigi + uctan uca mini kosu."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch

from lcsa.mini_lm import MiniLM

torch.manual_seed(0)

# 1) model: esit-param kontrolu + aux + gradyan akislari + usage
dense = MiniLM(vocab=512, dim=64, layers=2, heads=4, ctx=64, mlp_hidden=64,
               loops=1, moe_experts=0)
moe = MiniLM(vocab=512, dim=64, layers=2, heads=4, ctx=64, mlp_hidden=16,
             loops=1, moe_experts=4)
nd = sum(p.numel() for p in dense.parameters())
nm = sum(p.numel() for p in moe.parameters())
print(f"dense params={nd} | moe params={nm} (fark={nm - nd})")
assert abs(nm - nd) < nd * 0.02 + 500, "MoE param sayisi dense'e yakin olmali"

x = torch.randint(0, 512, (3, 65))
moe.train()
loss = moe.loss(x, add_aux=True)
ce = moe.loss(x)
print(f"loss(add_aux)={float(loss):.4f} ce={float(ce):.4f} fark={float(loss - ce):.6f}")
assert float(loss - ce) >= 0, "aux loss negatif olamaz"
loss.backward()
r = moe.blocks[0].mlp.router
assert r.weight.grad is not None and torch.isfinite(r.weight.grad).all(), "router gradyani yok"
assert float(r.weight.grad.abs().sum()) > 0, "router gradyani sifir (p_top akisi kirik)"
eg = moe.blocks[0].mlp.w3[0].weight.grad
assert eg is not None and torch.isfinite(eg).all(), "uzman gradyani yok"
u = moe.blocks[0].mlp.last_usage
assert abs(float(u.sum()) - 1.0) < 1e-5, "kullanim toplami 1 olmali"
print("moe model smoke OK; usage L0 =", [round(float(v), 3) for v in u.tolist()])

# 2) uctan uca: exp11 mini kosu (CPU/cuda farketmez; ts_smoke verisi)
PY = sys.executable
EXP = os.path.join(ROOT, "experiments", "exp11_own_lm.py")
cmd = (f'"{PY}" "{EXP}" --tag moesmoke --steps 40 --batch 4 --ctx 64 --dim 64 '
       f'--layers 2 --heads 4 --mlp_hidden 64 --moe_experts 4 --vocab 8192 '
       f'--data_prefix ts_smoke --warmup 10 --eval_every 20 --n_eval 4 --gen_every 1000')
p = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=ROOT, timeout=900)
out = (p.stdout or "") + (p.stderr or "")
print(out[-1400:])
assert p.returncode == 0, f"e2e FAIL rc={p.returncode}"
assert "[moe @" in out, "moe usage logu yok"
assert "[done]" in out, "done yok"
print("MOE SMOKE PASS")
