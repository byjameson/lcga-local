"""P9a: yeni bir rastgele-kontrol bolunmesi (partition-robustluk testi).

Mevcut `results/p8_clusters.npz` icindeki **rel bolunmesi ve W aynen korunur**;
`rnd` bolunmesi exp14 `mode_corr`'daki AYNI algoritmayla (rel profiline eslesli
rastgele permutasyon + denge kisitlari) **farkli bir RNG tohumuyla** yeniden
cek ilir. Boylece "rnd sonucu tek bir cekilise mi bagli?" sorusu test edilir.

Kullanim:
  python scripts/make_rnd_partition.py --seed 2024
Cikti: results/p8_clusters_rnd2.npz (rel/W ayni, rnd yeni)
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from lcsa.metrics import now


def cut_of(W, labels):
    n = len(labels)
    s = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] != labels[j]:
                s += float(W[i, j])
    return s


def draw_rnd(rel, sizes, total, seed, bal_lo, bal_hi, min_frac):
    """exp14 mode_corr ile birebir ayni rastgele-kontrol cekilisi."""
    N = len(rel)
    rng = np.random.default_rng(seed)
    rnd = None
    for _ in range(2000):
        cand = rel[rng.permutation(N)]
        f = float(sizes[cand == 0].sum() / total)
        if bal_lo <= f <= bal_hi and bal_lo <= 1 - f <= bal_hi:
            rnd = cand
            break
    if rnd is None:                                   # exp14 ile ayni kademeli gevsetme
        for _ in range(2000):
            cand = rel[rng.permutation(N)]
            f = float(sizes[cand == 0].sum() / total)
            if min(f, 1.0 - f) >= min_frac:
                rnd = cand
                break
    return rnd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(ROOT, "results", "p8_clusters.npz"))
    ap.add_argument("--seed", type=int, default=2024,
                    help="yeni rastgele bolunme icin RNG tohumu")
    ap.add_argument("--scan", type=int, default=0,
                    help=">0: seed..seed+N-1 icin kesim/denge taramasi yaz ve cik")
    ap.add_argument("--out", default=os.path.join(ROOT, "results",
                                                  "p8_clusters_rnd2.npz"))
    ap.add_argument("--bal_lo", type=float, default=0.35)
    ap.add_argument("--bal_hi", type=float, default=0.65)
    ap.add_argument("--min_frac", type=float, default=0.1)
    args = ap.parse_args()

    z = np.load(args.npz)
    names = [str(s) for s in z["names"]]
    rel = z["rel"].astype(int)
    sizes = z["sizes"].astype(float)
    W = z["W"]
    total = float(sizes.sum())
    N = len(names)

    if args.scan > 0:
        print(f"[scan] rel: frac0={float(sizes[rel == 0].sum() / total):.3f} "
              f"cut={cut_of(W, rel):.4f}")
        for sd in range(args.seed, args.seed + args.scan):
            cand = draw_rnd(rel, sizes, total, sd, args.bal_lo, args.bal_hi,
                            args.min_frac)
            if cand is None:
                print(f"[scan] seed={sd}: (cekilis yok)")
                continue
            f = float(sizes[cand == 0].sum() / total)
            print(f"[scan] seed={sd}: frac0={f:.3f} cut={cut_of(W, cand):.4f}")
        return

    rnd = draw_rnd(rel, sizes, total, args.seed, args.bal_lo, args.bal_hi,
                   args.min_frac)
    if rnd is None:
        raise RuntimeError("rastgele bolunme bulunamadi")

    fr = float(sizes[rnd == 0].sum() / total)
    fr_rel = float(sizes[rel == 0].sum() / total)
    print(f"[rnd2] seed={args.seed} N={N}")
    print(f"[rnd2] rel : frac0={fr_rel:.3f} cut={cut_of(W, rel):.4f}")
    print(f"[rnd2] rnd2: frac0={fr:.3f} cut={cut_of(W, rnd):.4f} "
          f"(eski rnd: frac0={float(sizes[z['rnd'] == 0].sum() / total):.3f} "
          f"cut={float(z['cut_rnd']):.4f})")
    for lbl, arr in (("related", rel), ("rnd2", rnd)):
        c0 = [names[i] for i in range(N) if arr[i] == 0]
        c1 = [names[i] for i in range(N) if arr[i] == 1]
        print(f"[rnd2] {lbl}: C0={c0} | C1={c1}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, names=z["names"], rel=z["rel"], rnd=rnd, W=W, C=z["C"],
             sizes=z["sizes"], cut_rel=z["cut_rel"], cut_rnd=cut_of(W, rnd),
             eps=z["eps"], corr_batches=z["corr_batches"],
             corr_batch=z["corr_batch"], rnd_seed=np.array(args.seed), ts=now())
    print(f"[rnd2] -> {args.out}")


if __name__ == "__main__":
    main()
