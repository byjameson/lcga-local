"""Does a base model retrieve a needle at all? (no checkpoint, no router training)

The LCGA harness needs the full fp16 model + the teacher capture, so before planning a
scale test we check the prerequisite: can this model's *dense* attention find a planted
fact, and does a budget-matched window fail (i.e. is the task informative at this scale)?

Usage:
  python scripts/probe_dense.py --model models/Qwen2.5-3B --Ts 1024,8192 --depths 0.1,0.5,0.9
"""
import argparse
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from lcsa.data import load_tokens
from lcsa.harness import load_model
from lcsa.model import patch_model, set_selection_mode, set_sparse_enabled

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = "cuda"
QUESTION = " Question: What is the secret code? Answer:"


def load_exp60():
    path = os.path.join(ROOT, "experiments", "exp60_needle.py")
    spec = importlib.util.spec_from_file_location("e60", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@torch.no_grad()
def score(model, ids, ans_len):
    logits = model(input_ids=ids, logits_to_keep=ans_len + 1).logits[0, :-1].float()
    tgt = ids[0, -ans_len:]
    return (F.cross_entropy(logits, tgt).item(),
            int((logits.argmax(-1) == tgt).all()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--Ts", default="1024,8192")
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    ap.add_argument("--budget", type=float, default=0.25)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    model_dir = args.model if os.path.isabs(args.model) else os.path.join(ROOT, args.model)
    model = load_model(model_dir, device=DEV)
    tok = AutoTokenizer.from_pretrained(model_dir)
    n_blocks = None
    wrappers = patch_model(model, budget=args.budget, n_concepts=64, graph_rank=0,
                           dynamic=False, content_rank=0, rel_bias=True,
                           capture_dense=False, use_coarse=False, null_init=-1e9)
    for p in model.parameters():
        p.requires_grad_(False)

    mod = load_exp60()
    filler = load_tokens(os.path.join(ROOT, "data", "wt2_train.pt"))[:40000].tolist()

    for T in [int(x) for x in args.Ts.split(",")]:
        for depth in [float(x) for x in args.depths.split(",")]:
            ids, ans_len = mod.build_ids(tok, filler, T, depth, "7311", QUESTION)
            set_sparse_enabled(False)
            ce_d, ex_d = score(model, ids, ans_len)
            set_selection_mode("window")
            set_sparse_enabled(True)
            ce_w, ex_w = score(model, ids, ans_len)
            set_sparse_enabled(False)
            print(f"[T={T} d={depth:.1f}] dense {ce_d:.3f}/{ex_d}  "
                  f"window {ce_w:.3f}/{ex_w}", flush=True)
            del ids
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
