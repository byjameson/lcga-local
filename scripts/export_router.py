"""Export a router-only checkpoint for publication (drops the frozen base weights).

`exp61`/`exp2` save `dict(layer.self_attn.named_parameters())`, which includes the
frozen Qwen projections (~105 MB per checkpoint). The publishable artifact is the
router itself (~11 MB): patch the base model, then copy the keys whose names match.

Usage:
  python scripts/export_router.py --ckpt checkpoints/mix7_final.pt \
      --out checkpoints/router_needle_mix7.pt
"""
import argparse
import os

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(ROOT, args.ckpt)
    dst = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    ck = torch.load(src, map_location="cpu", weights_only=False)
    state = []
    for layer in ck["state"]:
        state.append({k: v for k, v in layer.items() if not k.startswith("orig.")})
    meta = dict(ck.get("meta", {}))
    meta["export"] = ("router-only: load with patch_model(...) using meta.args "
                      "(content_rank, graph_rank, n_concepts, budget) and copy these "
                      "keys into layer.self_attn; frozen base weights are not included")
    out = {"state": state, "meta": meta}
    torch.save(out, dst)
    mb = os.path.getsize(dst) / 1e6
    print(f"[ok] {dst} ({mb:.1f} MB, {len(state)} layers, router-only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
