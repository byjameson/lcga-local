"""EXP65: how good is the teacher itself?

The label-free objective distils the dense teacher's content argmax at the answer
rows, and it plateaus at p(fact) ~ 0.5. Before calling that a failure, measure the
ceiling: how often is the teacher's own content argmax the planted fact block?

If the teacher points at the fact only ~55% of the time (per layer), then a perfect
student converges to ~0.55 and the remaining gap to the labelled run (0.91) is
teacher noise, not a weak objective.

Reports, per layer and overall:
  acc_content_argmax : P(argmax over content blocks == fact block)
  acc_all_argmax     : P(argmax over all blocks == fact block)
  mass_fact          : teacher mass on the fact block (mean)
  content_mass       : teacher mass outside {sink, self, last m-1 blocks}

Usage:
  python experiments/exp65_teacher_accuracy.py --seq 1024 --batches 8 --depth_min 0.1
"""
import argparse
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from lcsa.attention import repeat_kv
from lcsa.data import load_tokens
from lcsa.harness import load_model
from lcsa.model import begin_capture, end_capture, patch_model, set_capture_qk, set_sparse_enabled

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV = "cuda"


def load_sampler_module():
    path = os.path.join(ROOT, "experiments", "exp61_train_router_needle.py")
    spec = importlib.util.spec_from_file_location("e61", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-0.5B")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--budget", type=float, default=0.25)
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--paraphrase", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth_min", type=float, default=0.0,
                    help="only count samples whose fact block is farther than the recency "
                         "window from the answer row (i.e. content must be selected)")
    ap.add_argument("--ckpt", default="",
                    help="optional trained router: the teacher is computed from ITS hidden "
                         "states (layer-wise distillation), so an untrained router "
                         "underestimates the teacher")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    mod = load_sampler_module()
    model_dir = os.path.join(ROOT, args.model) if not os.path.isabs(args.model) else args.model
    model = load_model(model_dir, device=DEV)
    tok = AutoTokenizer.from_pretrained(model_dir)
    wrappers = patch_model(model, budget=args.budget, block=args.block, n_concepts=64,
                           graph_rank=0, dynamic=False, content_rank=0, rel_bias=True,
                           capture_dense=True, use_coarse=False, null_init=-1e9)
    meta = {}
    if args.ckpt:
        path = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(ROOT, args.ckpt)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        meta = ck.get("meta", {}).get("args", {})
        print(f"[ckpt] {args.ckpt} (loss={meta.get('loss')}, "
              f"content_rank={meta.get('content_rank')})", flush=True)
    else:
        ck = None
    wrappers = patch_model(model, budget=args.budget, block=args.block,
                           n_concepts=meta.get("n_concepts", 64),
                           graph_rank=meta.get("graph_rank", 0), dynamic=False,
                           content_rank=meta.get("content_rank", 0),
                           rel_bias=not meta.get("no_rel_bias", False),
                           capture_dense=True, use_coarse=False, null_init=-1e9)
    if ck is not None:
        for i, layer in enumerate(model.model.layers):
            own = dict(layer.self_attn.named_parameters())
            for n, t in ck["state"][i].items():
                if n in own:
                    own[n].data.copy_(t.to(own[n].device))
    for p in model.parameters():
        p.requires_grad_(False)

    filler = load_tokens(os.path.join(ROOT, "data", "wt2_train.pt"))[:60000].tolist()
    sampler = mod.NeedleSampler(tok, filler, args.seq, block=args.block, seed=args.seed,
                                paraphrase=bool(args.paraphrase))
    n_blocks = args.seq // args.block
    m_sel = max(1, int(round(args.budget * n_blocks)))
    keys_all = torch.arange(args.seq, device=DEV)

    per_layer = {}
    n_used = 0
    for b in range(args.batches):
        x, starts, facts = sampler.batch(args.batch)
        x = x.to(DEV)
        facts = facts.to(DEV)
        set_sparse_enabled(True)
        set_capture_qk(True)
        begin_capture()
        with torch.no_grad():
            model.model(input_ids=x, use_cache=False)
        caps = end_capture()
        for L, cap in enumerate(caps):
            q, k = cap["q"], cap["k"]
            H = q.shape[1]
            if k.shape[1] != H:
                k = repeat_kv(k, H // k.shape[1])
            ar = list(range(args.seq - sampler.ans_len, args.seq))
            logits = torch.einsum("bhrd,bhtd->bhrt", q[:, :, ar, :].float(), k.float()) \
                * wrappers[L].scale
            mask = keys_all.view(1, 1, 1, -1) > torch.tensor(ar, device=DEV).view(1, 1, -1, 1)
            att = torch.softmax(logits.masked_fill(mask, float("-inf")), -1)
            teacher = att.view(x.shape[0], H, len(ar), n_blocks, args.block).sum(-1).sum(2)
            teacher = teacher / teacher.sum(-1, keepdim=True).clamp_min(1e-9)  # (B,H,NB)
            nb = teacher.shape[-1]
            jj = torch.arange(nb, device=DEV).view(1, 1, -1)
            rr = torch.tensor([(args.seq - 1) // args.block], device=DEV).view(1, 1, 1)
            cmask = ((jj != 0) & (jj != rr) & (jj <= rr - m_sel)).float()
            masked = teacher.masked_fill(cmask.view(1, 1, -1) == 0, -1.0)
            a_content = (masked.argmax(-1) == facts[:, None]).float().mean().item()
            a_all = (teacher.argmax(-1) == facts[:, None]).float().mean().item()
            mf = teacher.gather(-1, facts[:, None, None].expand(-1, H, 1)).mean().item()
            cm = (teacher * cmask.view(1, 1, -1)).sum(-1).mean().item()
            row = (args.seq - 1) // args.block
            ps = cap["probs"][:, row:row + 1, :].gather(
                -1, facts.view(-1, 1, 1)).mean().item()
            d = per_layer.setdefault(L, {"a_content": 0.0, "a_all": 0.0, "mf": 0.0,
                                         "cm": 0.0, "ps": 0.0, "n": 0})
            d["a_content"] += a_content
            d["a_all"] += a_all
            d["mf"] += mf
            d["cm"] += cm
            d["ps"] += ps
            d["n"] += 1
        n_used += 1

    print(f"teacher accuracy over {n_used} batches, seq={args.seq}, "
          f"budget={args.budget}, m={m_sel}, paraphrase={args.paraphrase}")
    print(" layer  acc_content  acc_all  mass_fact  content_mass  p_student")
    agg = {"a_content": 0.0, "a_all": 0.0, "mf": 0.0, "cm": 0.0, "ps": 0.0, "n": 0}
    for L in sorted(per_layer):
        d = per_layer[L]
        n = d["n"]
        print(f"  L{L:<3d}   {d['a_content']/n:.3f}      {d['a_all']/n:.3f}   "
              f"{d['mf']/n:.3f}     {d['cm']/n:.3f}      {d['ps']/n:.3f}")
        for k in ("a_content", "a_all", "mf", "cm", "ps"):
            agg[k] += d[k]
        agg["n"] += n
    n = agg["n"]
    print(f"  ALL     {agg['a_content']/n:.3f}      {agg['a_all']/n:.3f}   "
          f"{agg['mf']/n:.3f}     {agg['cm']/n:.3f}      {agg['ps']/n:.3f}")
    print(f"\n=> teacher content-argmax accuracy {agg['a_content']/n:.3f}; "
          f"student p(fact) {agg['ps']/n:.3f}; "
          f"teacher mass on fact {agg['mf']/n:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
