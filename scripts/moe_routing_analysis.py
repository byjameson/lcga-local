"""MoE uzmanlasma analizi — moe4_best.pt router davranisi (exp15 takip).

48 valid penceresi (seed 999) uzerinde her katmanin top-1 uzman atamalari
toplanir; token ozellikleriyle iliskisi olculur:

  - kullanim (denge) ve uzman basina en sik tokenlar
  - kategori bazli kullanim: kelime-basi, noktalama, rakam, uzun token,
    frekans kovalari, pozisyon kovalari
  - NMI (normalize karsilikli bilgi) — karisik (shuffle) taban cizgisiyle
  - token-ici atama entropisi (ayni token hep ayni uzmana mi gidiyor?)
  - katmanlar arasi NMI matrisi (ortak bolumleme yapisi)

Cikti: results/moe_routing.json + konsol ozeti. CPU'da calisir (GPU'ya dokunmaz).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import string
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch

from lcsa.mini_lm import MiniLM

STRIP = string.punctuation + string.whitespace


def load_tokens(path):
    obj = torch.load(path, weights_only=True)
    return obj["tokens"] if isinstance(obj, dict) else obj


def entropy(counts):
    p = np.asarray(counts, dtype=float)
    p = p / max(p.sum(), 1e-12)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum()) if len(p) else 0.0


def mi_nmi(x, y, kx, ky):
    tab = np.zeros((kx, ky), dtype=np.float64)
    np.add.at(tab, (x, y), 1.0)
    n = tab.sum()
    pxy = tab / n
    px = pxy.sum(1)
    py = pxy.sum(0)
    mi = 0.0
    for i in range(kx):
        for j in range(ky):
            if pxy[i, j] > 0:
                mi += pxy[i, j] * math.log(pxy[i, j] / (px[i] * py[j]))
    den = math.sqrt(entropy(px * n) * entropy(py * n))
    return float(mi / den) if den > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints", "moe4_best.pt"))
    ap.add_argument("--valid", default=os.path.join(ROOT, "data", "ts_valid.pt"))
    ap.add_argument("--tok", default=os.path.join(ROOT, "data", "ts_bpe8k.json"))
    ap.add_argument("--n_win", type=int, default=48)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "moe_routing.json"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg") or {}
    kw = dict(vocab=cfg.get("vocab", 8192), dim=cfg.get("dim", 384),
              layers=cfg.get("layers", 8), heads=cfg.get("heads", 6),
              ctx=cfg.get("ctx", 512), mlp_hidden=cfg.get("mlp_hidden", 1024),
              loops=cfg.get("loops", 1), moe_experts=cfg.get("moe", 0))
    model = MiniLM(**kw)
    model.load_state_dict(ck["model"])
    model.eval()
    del ck
    L, E, ctx = len(model.blocks), model.moe_experts, model.ctx
    assert E > 0, "ckpt MoE degil"
    print(f"[moe-routing] L={L} E={E} ctx={ctx} params="
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)

    valid = load_tokens(args.valid)
    gen = torch.Generator().manual_seed(args.seed)
    wins = []
    for _ in range(args.n_win):
        s = int(torch.randint(0, len(valid) - ctx - 1, (1,), generator=gen))
        wins.append(valid[s: s + ctx].long())
    wins = torch.stack(wins)
    print(f"[moe-routing] {args.n_win} pencere x {ctx} = {wins.numel()} token",
          flush=True)

    stores = [[] for _ in range(L)]
    confs = [[] for _ in range(L)]
    hooks = []
    for i, blk in enumerate(model.blocks):
        def hook(_m, _inp, out, i=i):
            p = torch.softmax(out.float(), dim=-1)
            stores[i].append(p.argmax(-1))
            confs[i].append(p.max(-1).values)
        hooks.append(blk.mlp.router.register_forward_hook(hook))
    with torch.no_grad():
        for i in range(0, args.n_win, args.batch):
            model(wins[i: i + args.batch], return_hidden=True)
    for h in hooks:
        h.remove()
    top = torch.stack([torch.cat(s) for s in stores])          # (L, Ntok)
    conf = torch.stack([torch.cat(c) for c in confs])          # (L, Ntok)
    ids = wins.reshape(-1)
    N = ids.numel()
    assert top.shape[1] == N, (top.shape, N)
    ids_np = ids.numpy()
    top_np = top.numpy().astype(np.int64)
    pos = torch.arange(ctx).repeat(args.n_win).numpy()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tok)
    scache = {}

    def s_of(i):
        if i not in scache:
            scache[i] = tok.decode([int(i)])
        return scache[i]

    freq = torch.bincount(valid.long(), minlength=model.vocab).numpy()
    f = freq[ids_np]

    s_all = [s_of(i) for i in ids_np.tolist()]
    word_init = np.array([s.startswith(" ") or s.startswith("Ġ") for s in s_all])
    punct = np.array([bool(s.strip()) and all(ch in string.punctuation for ch in s.strip())
                      for s in s_all])
    digit = np.array([s.strip().isdigit() for s in s_all])
    long_tok = np.array([len(s.strip()) >= 6 for s in s_all])
    fb = np.digitize(f, [50, 500, 5000])                       # 0..3
    pb = np.digitize(pos, [64, 256])                           # 0..2

    feats = {
        "word_init": word_init.astype(np.int64),
        "punct": punct.astype(np.int64),
        "digit": digit.astype(np.int64),
        "long_tok": long_tok.astype(np.int64),
        "freq_bucket": fb,
        "pos_bucket": pb,
    }
    feat_names = {"word_init": ["cont", "word_start"],
                  "punct": ["other", "punct"],
                  "digit": ["other", "digit"],
                  "long_tok": ["short", "long"],
                  "freq_bucket": ["<50", "50-500", "500-5k", ">5k"],
                  "pos_bucket": ["0-63", "64-255", "256-511"]}

    # ---- kullanim ----
    usage = np.zeros((L, E))
    for l in range(L):
        usage[l] = np.bincount(top_np[l], minlength=E) / N
    print("\n[usage] katman | uzman kullanimlari")
    for l in range(L):
        print(f"  L{l}: " + " ".join(f"{u:.3f}" for u in usage[l]))

    print("\n[guven] top-1 olasiligi (ortalama | p<0.6 orani)")
    conf_mean = [round(float(conf[l].mean()), 3) for l in range(L)]
    lowconf = [round(float((conf[l] < 0.6).float().mean()), 3) for l in range(L)]
    for l in range(L):
        print(f"  L{l}: {conf_mean[l]:.3f} | {lowconf[l]:.3f}")

    # ---- NMI + shuffle taban cizgisi ----
    rng = np.random.default_rng(args.seed + 7)
    nmi = {k: [] for k in feats}
    nmi_shuf = {k: [] for k in feats}
    for l in range(L):
        sh = rng.permutation(top_np[l])
        for k, v in feats.items():
            kx = E
            ky = int(v.max()) + 1
            nmi[k].append(mi_nmi(top_np[l], v, kx, ky))
            nmi_shuf[k].append(mi_nmi(sh, v, kx, ky))

    print("\n[NMI] ozellik | gozlenen (katman ort.) | karisik taban (ort.)")
    nmi_tab = {}
    for k in feats:
        obs = float(np.mean(nmi[k]))
        base = float(np.mean(nmi_shuf[k]))
        nmi_tab[k] = {"per_layer": [round(x, 4) for x in nmi[k]],
                      "mean": round(obs, 4),
                      "shuffle_mean": round(base, 4)}
        print(f"  {k:12s}: {obs:.4f} | {base:.4f}")

    # ---- token-ici entropi (ayni token ayni uzmana mi gidiyor?) ----
    ent_stats = []
    for l in range(L):
        cnt = np.zeros((model.vocab, E))
        np.add.at(cnt, (ids_np, top_np[l]), 1.0)
        n = cnt.sum(1)
        sel = n >= 20
        w = n[sel].sum()
        h = 0.0
        for i in np.where(sel)[0]:
            h += n[i] * entropy(cnt[i]) / math.log(E)
        ent_stats.append(round(float(h / max(w, 1)), 4))
        sh = rng.permutation(top_np[l])
        cnt2 = np.zeros((model.vocab, E))
        np.add.at(cnt2, (ids_np, sh), 1.0)
        h2 = 0.0
        for i in np.where(sel)[0]:
            h2 += n[i] * entropy(cnt2[i]) / math.log(E)
        ent_stats[l] = {"observed": ent_stats[l],
                        "shuffle": round(float(h2 / max(w, 1)), 4)}
    print("\n[token-ici entropi] (dusuk = token hep ayni uzman; 1.0 = tamamen serbest)")
    for l in range(L):
        print(f"  L{l}: {ent_stats[l]['observed']:.3f} (karisik {ent_stats[l]['shuffle']:.3f})")

    # ---- kategori bazli kullanim ----
    cat_tab = {}
    for k, v in feats.items():
        names = feat_names[k]
        rows = []
        for c in range(int(v.max()) + 1):
            m = v == c
            if m.sum() == 0:
                continue
            use = np.stack([np.bincount(top_np[l][m], minlength=E) / m.sum()
                            for l in range(L)])
            rows.append({"cat": names[c] if c < len(names) else str(c),
                         "n": int(m.sum()),
                         "usage_mean": [round(float(x), 3) for x in use.mean(0)],
                         "usage_L4": [round(float(x), 3) for x in use[4]]})
        cat_tab[k] = rows
    print("\n[kategori kullanimi] (L4; E=4 ise 0.25 = fark yok)")
    for k in ("word_init", "punct", "digit", "long_tok", "freq_bucket", "pos_bucket"):
        for r in cat_tab[k]:
            u = " ".join(f"{x:.3f}" for x in r["usage_L4"])
            print(f"  {k:11s} {r['cat']:8s} n={r['n']:6d} | {u}")

    # ---- uzman basina en sik tokenlar ----
    samp_counts = np.bincount(ids_np, minlength=model.vocab)
    top_tok = {}
    for l in range(L):
        rows = []
        for e in range(E):
            m = top_np[l] == e
            c = np.bincount(ids_np[m], minlength=model.vocab)
            order = np.argsort(c)[::-1][:8]
            rows.append([{"tok": s_of(int(i)).replace("\n", "\\n"),
                          "count": int(c[i]),
                          "p_expert": round(float(c[i] / max(samp_counts[i], 1)), 3)}
                         for i in order if c[i] > 0])
        top_tok[f"L{l}"] = rows
    print("\n[uzman basina en sik tokenlar] (L2/L4/L6)")
    for l in (2, 4, 6):
        print(f"  --- L{l} ---")
        for e in range(E):
            toks = ", ".join(f"'{r['tok']}'(%{100*r['p_expert']:.0f})"
                             for r in top_tok[f"L{l}"][e][:5])
            print(f"    E{e}: {toks}")

    # ---- katmanlar arasi NMI ----
    mat = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            mat[i, j] = 1.0 if i == j else mi_nmi(top_np[i], top_np[j], E, E)
    off = (mat.sum() - L) / (L * L - L)
    print(f"\n[katmanlar arasi NMI] ort (ofdiyagonal) = {off:.3f}")
    print("  " + " ".join(f"L{j}" for j in range(L)))
    for i in range(L):
        print(f"  L{i} " + " ".join(f"{mat[i, j]:.2f}" for j in range(L)))

    out = {"ckpt": os.path.basename(args.ckpt), "n_win": args.n_win, "n_tok": N,
           "usage": [[round(float(x), 4) for x in row] for row in usage],
           "router_conf_mean": conf_mean, "router_lowconf_frac": lowconf,
           "nmi": nmi_tab, "token_entropy": ent_stats,
           "cat_usage": cat_tab, "top_tokens": top_tok,
           "layer_nmi": [[round(float(x), 3) for x in row] for row in mat],
           "layer_nmi_offdiag_mean": round(float(off), 4)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"[moe-routing] -> {args.out}")


if __name__ == "__main__":
    main()
