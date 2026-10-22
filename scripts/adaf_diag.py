"""Adafactor update-magnitude diagnostics (M5).

Runs a few streaming steps with a subclass that measures, for a sample of
parameters, the raw grad RMS, weight RMS and the applied update RMS/max.
Usage: python scripts/adaf_diag.py [--steps 2] [--lr 1e-4]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lcsa.data import TrainSampler, load_tokens  # noqa: E402
from lcsa.stream_full import StreamingSGDTrainer  # noqa: E402


class Diag(StreamingSGDTrainer):
    def _adafactor(self, ps, t, lr):
        diag = [(n, p) for n, p in ps if n == "L0.self_attn.q_proj.weight"
                or n == "L0.mlp.gate_proj.weight" or n == "L0.input_layernorm.weight"]
        before = {n: p.detach().clone() for n, p in diag}
        grms = {n: float(p.grad.float().pow(2).mean().sqrt()) for n, p in diag}
        super()._adafactor(ps, t, lr)
        for n, p in diag:
            d = p.float() - before[n].float()
            wr = float(before[n].float().pow(2).mean().sqrt())
            print(f"  t={t} {n:34s} g_rms={grms[n]:.3e} w_rms={wr:.3e} "
                  f"upd_rms={float(d.pow(2).mean().sqrt()):.3e} upd_max={float(d.abs().max()):.3e} "
                  f"upd_rms/w_rms={float(d.pow(2).mean().sqrt())/max(wr,1e-12):.3e}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-0.5B")
    ap.add_argument("--data", default="data/wt2_train.pt")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--opt", default="adafactor")
    ap.add_argument("--clip", type=float, default=1.0)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(str(ROOT / args.model), dtype=torch.bfloat16)
    model.eval()
    tr = Diag(model, device="cuda", lr=args.lr, clip=args.clip, opt=args.opt)
    sampler = TrainSampler(load_tokens(str(ROOT / args.data)), args.seq,
                           generator=torch.Generator().manual_seed(1234))
    for i in range(args.steps):
        xs, ys = sampler.batch(1)
        loss, dt, peak = tr.step(xs, ys)
        print(f"step {i+1}: loss={loss:.4f} {dt:.2f}s peak={peak:.2f}GB", flush=True)
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
