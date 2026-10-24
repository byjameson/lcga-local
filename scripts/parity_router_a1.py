"""Exactness check for A1 router optimizations (cache + vectorize in concepts.py).

Saves idx / probs / scores for fixed deterministic inputs BEFORE the change
(mode=save), recomputes AFTER (mode=check) and asserts bitwise equality.

Usage:
  python scripts\parity_router_a1.py save
  python scripts\parity_router_a1.py check
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from lcsa.concepts import ConceptGraphRouter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_router(d=896, seed=0):
    torch.manual_seed(seed)
    r = ConceptGraphRouter(d, 64, 16, rel_bias=True).eval().float()
    for p in r.parameters():
        p.requires_grad_(False)
    return r


def run_case(T, budget, data_seed):
    r = make_router()
    g = torch.Generator().manual_seed(data_seed)
    h = torch.randn(1, T, 896, generator=g)
    with torch.no_grad():
        ql, kl = r.concept_logits(h)
        sc = r.block_scores(ql, kl, h, block=64)
        idx, probs_sel = r.select_blocks(sc, block=64, budget=budget)
        mp = r.masked_probs(sc, block=64)
        torch.manual_seed(77)  # deterministic gumbel noise
        idxg, _ = r.select_blocks(sc, block=64, budget=budget, gumbel_sigma=0.5)
    return {"scores": sc.clone(), "idx": idx.clone(), "probs": mp.clone(),
            "probs_sel": probs_sel.clone(), "idx_gumbel": idxg.clone()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["save", "check"])
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "parity_a1.pt"))
    args = ap.parse_args()

    cases = {}
    for T, budget, dseed in [(1024, 0.25, 1), (2048, 0.5, 2)]:
        cases[f"T{T}_b{budget}"] = run_case(T, budget, dseed)

    if args.mode == "save":
        torch.save(cases, args.out)
        print("saved", args.out)
        return

    ref = torch.load(args.out, map_location="cpu", weights_only=False)
    ok = True
    for k, v in cases.items():
        for name, t in v.items():
            r0 = ref[k][name]
            eq = torch.equal(r0, t)
            if not eq:
                ok = False
                d = (r0.float() - t.float()).abs().max().item()
                n = int((r0 != t).sum())
                print(f"[FAIL] {k}.{name}: equal=False maxdiff={d:.3e} n_diff={n}")
            else:
                print(f"[ok]   {k}.{name}")
    print("PARITY:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
