"""exp17b: gradient parity -- streaming full-parameter backward vs naive full-GPU.

Runs one batch through both paths (no weight update), compares gradients for
selected tensors. A small model (0.5B) fits fully on the GPU, so it is the
reference. Expected: gradients match to bf16 noise (max rel diff ~1e-2 or less).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lcsa.data import TrainSampler, load_tokens  # noqa: E402
from lcsa.stream_full import StreamingSGDTrainer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-0.5B")
    ap.add_argument("--data", default="data/wt2_train.pt")
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--tag", default="full_parity")
    args = ap.parse_args()

    model_path = str(ROOT / args.model) if not Path(args.model).is_absolute() else args.model
    tokens = load_tokens(str(ROOT / args.data))
    sampler = TrainSampler(tokens, args.seq, generator=torch.Generator().manual_seed(1234))
    xs, ys = sampler.batch(1)

    from transformers import AutoModelForCausalLM

    # ---- streaming path (no update; grads remain on parameters) ----
    m1 = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
    m1.eval()
    tr = StreamingSGDTrainer(m1, device="cuda", loss_chunk=args.chunk, lr=0.0, clip=0.0)
    loss_s, dt_s, peak_s = tr.step(xs, ys, update=False)
    print(f"[stream] loss={loss_s:.6f} {dt_s:.2f}s peak={peak_s:.2f}GB", flush=True)

    # ---- naive path (whole model on GPU) ----
    m2 = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
    m2.to("cuda")
    m2.eval()
    xs2, ys2 = xs.to("cuda"), ys.to("cuda")
    # NOTE: sampler.ys is already next-token aligned, so compute CE directly
    # (passing labels=ys would shift a second time and inflate the loss).
    logits = m2(input_ids=xs2).logits
    loss_n = float(torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(), ys2.reshape(-1)).detach())
    loss_t = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), ys2.reshape(-1))
    loss_t.backward()
    print(f"[naive]  loss={loss_n:.6f}", flush=True)

    names = [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        f"model.layers.{len(m1.model.layers)-1}.mlp.down_proj.weight",
        "model.norm.weight",
    ]
    g1 = dict(m1.named_parameters())
    g2 = dict(m2.named_parameters())
    results = []
    for name in names:
        a = g1[name].grad
        b = g2[name].grad
        if a is None or b is None:
            results.append({"name": name, "grad_a": a is not None, "grad_b": b is not None})
            continue
        a = a.detach().float().cpu().flatten()
        b = b.detach().float().cpu().flatten()
        diff = (a - b).abs()
        rmse = float(((a - b) ** 2).mean().sqrt())
        rms_b = float((b ** 2).mean().sqrt())
        cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        results.append({
            "name": name,
            "max_abs_diff": float(diff.max()),
            "rel_rmse": rmse / max(rms_b, 1e-12),
            "cosine": cos,
        })
        print(f"[grad] {name}: maxdiff={float(diff.max()):.3e} "
              f"rel_rmse={rmse/max(rms_b,1e-12):.3e} cos={cos:.5f}", flush=True)

    row = {
        "exp": "exp17_full_parity",
        "tag": args.tag,
        "model": Path(model_path).name,
        "seq": args.seq,
        "loss_stream": round(loss_s, 6),
        "loss_naive": round(loss_n, 6),
        "loss_abs_diff": abs(loss_s - loss_n),
        "stream_peak_gb": round(peak_s, 3),
        "grads": results,
    }
    print(json.dumps({k: v for k, v in row.items() if k != "grads"}, ensure_ascii=True))
    with open(ROOT / "results" / "exp17_parity.json", "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2)
    with open(ROOT / "results" / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
