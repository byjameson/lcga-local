"""Step-effect diagnostics: no-update vs sgd vs adafactor (M5).

For each optimizer: run the same batches, and for a few watched parameters
report the applied weight delta (RMS, max, fraction of elements changed).
The fraction changed reveals bf16 quantization: a step smaller than half a ULP
rounds to zero for most elements.
Usage: python scripts/step_diag.py [--steps 2]
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

WATCH = ["L0.self_attn.q_proj.weight", "L0.mlp.gate_proj.weight",
         "embed.weight", "norm.weight"]


def watched(tr):
    yield "L0.q_proj.weight", tr.layers[0].self_attn.q_proj.weight
    yield "L0.gate_proj.weight", tr.layers[0].mlp.gate_proj.weight
    yield "embed.weight", tr.embed.weight
    yield "norm.weight", tr.norm.weight


def run(tag: str, opt: str, lr: float, steps: int, update: bool) -> None:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        str(ROOT / "models/Qwen2.5-0.5B"), dtype=torch.bfloat16)
    model.eval()
    tr = StreamingSGDTrainer(model, device="cuda", lr=lr, clip=1.0, opt=opt)
    sampler = TrainSampler(load_tokens(str(ROOT / "data/wt2_train.pt")), 256,
                           generator=torch.Generator().manual_seed(1234))
    snap = {n: p.detach().float().cpu().clone() for n, p in watched(tr)}
    print(f"=== {tag} (opt={opt} lr={lr:g} update={update}) ===", flush=True)
    losses = []
    for i in range(steps):
        xs, ys = sampler.batch(1)
        loss, dt, peak = tr.step(xs, ys, update=update)
        losses.append(round(loss, 4))
        print(f"  step {i+1}: loss={loss:.4f} {dt:.2f}s peak={peak:.2f}GB", flush=True)
        if update:
            for n, p in watched(tr):
                d = p.detach().float().cpu() - snap[n]
                changed = (d != 0).float().mean().item()
                print(f"    {n:24s} drms={d.pow(2).mean().sqrt():.3e} "
                      f"dmax={d.abs().max():.3e} changed={changed*100:.1f}%", flush=True)
        torch.cuda.empty_cache()
    print(f"  losses={losses}", flush=True)
    del model, tr
    torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2)
    args = ap.parse_args()
    run("no-update baseline", "sgd", 0.0, args.steps, update=False)
    run("SGD", "sgd", 1e-3, args.steps, update=True)
    run("Adafactor", "adafactor", 1e-4, args.steps, update=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
