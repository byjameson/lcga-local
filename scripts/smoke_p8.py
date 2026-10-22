"""P8 (exp14) CPU smoke: kucuk modelle corr + related/random train uctan uca."""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch

from lcsa.mini_lm import MiniLM

PY = sys.executable
EXP = os.path.join(ROOT, "experiments", "exp14_p8_clusters.py")
CKPT = os.path.join(ROOT, "checkpoints", "p8smoke_ckpt.pt")
NPZ = os.path.join(ROOT, "results", "p8smoke_clusters.npz")

CFG = "--vocab 8192 --dim 64 --layers 2 --heads 4 --ctx 64 --mlp_hidden 128"


def run(args, tag):
    cmd = f'"{PY}" "{EXP}" {args}'
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       cwd=ROOT, timeout=900)
    out = (p.stdout or "") + (p.stderr or "")
    print(f"--- {tag} rc={p.returncode} ---")
    print(out[-1800:])
    assert p.returncode == 0, f"{tag} FAIL rc={p.returncode}"
    return out


# 1) kucuk ckpt
torch.manual_seed(0)
m = MiniLM(vocab=8192, dim=64, layers=2, heads=4, ctx=64, mlp_hidden=128)
os.makedirs(os.path.join(ROOT, "checkpoints"), exist_ok=True)
torch.save({"model": m.state_dict(), "step": 0, "best_val": 99.0, "evals": [],
            "args": {"vocab": 8192, "dim": 64, "layers": 2, "heads": 4,
                     "ctx": 64, "mlp_hidden": 128, "loops": 1}}, CKPT)
print("tiny ckpt yazildi:", CKPT)

# 2) corr
out = run(f"--mode corr --resume \"{CKPT}\" {CFG} --data_prefix ts_smoke "
          f"--corr_batches 2 --corr_batch 4 --out \"{NPZ}\" --seed 3", "corr")
assert "kesim skoru" in out and "C0=" in out
z = np.load(NPZ)
names = [str(s) for s in z["names"]]
rel = z["rel"]
rnd = z["rnd"]
assert len(names) == 6, names
assert set(rel.tolist()) == {0, 1} and set(rnd.tolist()) == {0, 1}
print("corr OK: related", dict(zip(names, rel.tolist())),
      "| rnd", dict(zip(names, rnd.tolist())),
      "| cut rel=%.4f rnd=%.4f" % (float(z["cut_rel"]), float(z["cut_rnd"])))

# 3) train related + random (CPU, kisa)
for sched, tag in (("related", "p8smoke_rel"), ("random", "p8smoke_rnd")):
    out = run(f"--mode train --schedule {sched} --cluster_file \"{NPZ}\" "
              f"--resume \"{CKPT}\" {CFG} --data_prefix ts_smoke --tag {tag} "
              f"--steps 40 --batch 4 --eval_every 20 --n_eval 4 --ckpt_every 40 "
              f"--steps_total 40 --warmup 10 --check_invariants", f"train-{sched}")
    assert "[p8 done]" in out, f"{tag}: done yok"
    assert "[p8 eval" in out, f"{tag}: eval yok"

print("P8 SMOKE PASS")
