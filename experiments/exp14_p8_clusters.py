"""P8 (exp14): ilgililik kumeli parcali egitim mini deneyi.

Soru: "Ilgili parametreleri ayni kumeye toplayip kumeleri sirayla
(round-robin) egitmek, rastgele kumelere gore daha az hasar verir mi?"

--mode corr : step-3000 ckpt'ten gradyan-etkilesim matrisi cikarir.
    Birim u'ya dtheta=-eps*g_u/||g_u|| adimi; s_vu=||dg_v||/(eps*||g_v||).
    N=18 birim (tok / 8x attn / 8x mlp / norm) -> W=(C+C.T)/2 ->
    dengeli 2'li kesim (brute force, param-frac ~0.5+-0.15) + rastgele kontrol.

--mode train : ckpt'ten devam; --schedule related|random|full.
    related/random: bir kume aktif (requires_grad), digerleri donuk.
    --phase > 0: kume her `phase` adimda degisir (parcali egitim hedefi);
    --phase 0: kume her adimda degisir (P8v1; lr 9.5e-4'te kararsiz cikti).
    LR + eval pencereleri P3 (exp11) ile birebir.

Kullanim:
  python experiments/exp14_p8_clusters.py --mode corr --resume checkpoints/p3_ckpt_s3000.pt
  python experiments/exp14_p8_clusters.py --mode train --schedule related \
      --resume checkpoints/p3_ckpt_s3000.pt --tag p8_rel --steps 3000
Cikti: results/p8_clusters.npz, checkpoints/{tag}_best|_ckpt.pt, results.jsonl.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from lcsa.metrics import append_jsonl, now
from lcsa.mini_lm import MiniLM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
RESULTS = os.path.join(ROOT, "results", "results.jsonl")


def get_batch(data, ctx, batch, gen, device):
    max_start = len(data) - ctx - 1
    starts = torch.randint(0, max_start, (batch,), generator=gen).tolist()
    xs = torch.stack([data[s: s + ctx + 1].long() for s in starts])
    return xs.to(device)


def save_atomic(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_split(prefix, split):
    p = os.path.join(DATA, f"{prefix}_{split}.pt")
    if not os.path.exists(p) and "smoke" in prefix:
        p = os.path.join(DATA, f"ts_{split}_smoke.pt")
    obj = torch.load(p, weights_only=True)
    if isinstance(obj, dict):
        return obj["tokens"], obj.get("n_bytes"), p
    return obj, None, p


def unit_groups(model):
    """Parametre birimleri (tied embedding tek birim)."""
    groups = [("tok", [model.tok.weight])]
    for i, b in enumerate(model.blocks):
        groups.append((f"L{i}.attn", [b.n1.weight, b.attn.qkv.weight, b.attn.proj.weight]))
        groups.append((f"L{i}.mlp", [b.n2.weight, b.mlp.w12.weight, b.mlp.w3.weight]))
    groups.append(("norm", [model.norm.weight]))
    return groups


def model_cfg_from(ck, args):
    ca = (ck or {}).get("args", {}) or {}

    def pick(k, d):
        return int(ca.get(k, getattr(args, k, d)))
    return dict(vocab=pick("vocab", 8192), dim=pick("dim", 384),
                layers=pick("layers", 8), heads=pick("heads", 6),
                ctx=pick("ctx", 512), mlp_hidden=pick("mlp_hidden", 1024),
                loops=pick("loops", 1))


def lr_at(step, warmup, total, min_frac):
    if step < warmup:
        return (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * prog))


def _flat_grads(groups):
    return [torch.cat([p.grad.detach().reshape(-1).float() for p in ps])
            for _, ps in groups]


def mode_corr(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    train, _, tr_path = load_split(args.data_prefix, "train")
    print(f"[corr] data={len(train)} tok ({os.path.basename(tr_path)}) device={device}", flush=True)

    ck = torch.load(args.resume, map_location="cpu", weights_only=False)
    model = MiniLM(**model_cfg_from(ck, args)).to(device)
    model.load_state_dict(ck["model"])
    del ck
    if device == "cuda":
        torch.cuda.empty_cache()
    model.train()

    groups = unit_groups(model)
    names = [n for n, _ in groups]
    N = len(groups)
    sizes = np.array([sum(p.numel() for p in ps) for _, ps in groups], float)
    print(f"[corr] N={N} birim | params={int(sizes.sum())} eps={args.eps}", flush=True)

    eps = args.eps
    acc_d = np.zeros((N, N))
    acc_n = np.zeros(N)
    gen = torch.Generator().manual_seed(args.seed + 5)
    t0 = time.time()
    for bi in range(args.corr_batches):
        xb = get_batch(train, args.ctx, args.corr_batch, gen, device)
        model.zero_grad(set_to_none=True)
        loss = model.loss(xb)
        loss.backward()
        base = _flat_grads(groups)
        norms = [float(g.norm()) for g in base]
        acc_n += np.array(norms)
        for u, (_, ps) in enumerate(groups):
            if norms[u] < 1e-12:
                continue
            diru = base[u] / norms[u]
            off = 0
            with torch.no_grad():
                for p in ps:
                    n = p.numel()
                    p.add_(diru[off: off + n].view_as(p), alpha=-eps)
                    off += n
            model.zero_grad(set_to_none=True)
            loss2 = model.loss(xb)
            loss2.backward()
            gp = _flat_grads(groups)
            for v in range(N):
                acc_d[v, u] += float((gp[v] - base[v]).norm()) / eps
            off = 0
            with torch.no_grad():
                for p in ps:
                    n = p.numel()
                    p.add_(diru[off: off + n].view_as(p), alpha=+eps)
                    off += n
        print(f"  [corr] batch {bi + 1}/{args.corr_batches} ({time.time() - t0:.1f}s)", flush=True)

    B = args.corr_batches
    gbar = acc_n / B
    floor = max(1e-6, 1e-3 * float(np.median(gbar)))
    C = (acc_d / B) / np.maximum(gbar, floor)[:, None]
    W = 0.5 * (C + C.T)
    np.fill_diagonal(W, 0.0)

    total = float(sizes.sum())

    def cut_of(labels):
        s = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                if labels[i] != labels[j]:
                    s += float(W[i, j])
        return s

    m = 1 << (N - 1)
    bits = np.zeros((m, N), np.int8)
    idx_arr = np.arange(m, dtype=np.uint32)
    for k in range(N - 1):
        bits[:, k + 1] = (idx_arr >> k) & 1
    cut = np.zeros(m)
    for i in range(N):
        for j in range(i + 1, N):
            w = float(W[i, j])
            if w > 0:
                cut += w * (bits[:, i] ^ bits[:, j])
    frac = (bits @ sizes) / total
    ok = (frac >= args.bal_lo) & (frac <= args.bal_hi)
    if not ok.any():
        ok = (frac >= 0.25) & (frac <= 0.75)
    if not ok.any():
        ok = (frac >= args.min_frac) & (frac <= 1 - args.min_frac)
    if not ok.any():
        raise RuntimeError("dengeli bolunme bulunamadi (--min_frac dusur)")
    rel = bits[int(np.argmin(np.where(ok, cut, np.inf)))].astype(int)
    if len(set(rel.tolist())) < 2:
        raise RuntimeError("dejenere bolunme (tek kume)")

    rnd_seed = args.rnd_seed if getattr(args, "rnd_seed", None) is not None else args.seed
    rng = np.random.default_rng(rnd_seed)
    rnd = None
    for _ in range(2000):
        cand = rel[rng.permutation(N)]
        f = float(sizes[cand == 0].sum() / total)
        if args.bal_lo <= f <= args.bal_hi and args.bal_lo <= 1 - f <= args.bal_hi:
            rnd = cand
            break
    if rnd is None:                      # kademeli gevşeme
        for _ in range(2000):
            cand = rel[rng.permutation(N)]
            f = float(sizes[cand == 0].sum() / total)
            if min(f, 1.0 - f) >= args.min_frac:
                rnd = cand
                break
    if rnd is None:
        raise RuntimeError("rastgele kontrol bolunmesi bulunamadi")

    cut_rel, cut_rnd = cut_of(rel), cut_of(rnd)
    fr_rel = float(sizes[rel == 0].sum() / total)
    fr_rnd = float(sizes[rnd == 0].sum() / total)
    print(f"[corr] kesim skoru: related={cut_rel:.4f} (frac0={fr_rel:.3f}) | "
          f"random={cut_rnd:.4f} (frac0={fr_rnd:.3f})", flush=True)
    for lbl, arr in (("related", rel), ("random", rnd)):
        c0 = [names[i] for i in range(N) if arr[i] == 0]
        c1 = [names[i] for i in range(N) if arr[i] == 1]
        print(f"[corr] {lbl}: C0={c0} | C1={c1}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, names=np.array(names), rel=rel, rnd=rnd, W=W, C=C,
             sizes=sizes, cut_rel=cut_rel, cut_rnd=cut_rnd, eps=eps,
             corr_batches=B, corr_batch=args.corr_batch, ts=now())
    print(f"[corr] -> {args.out}", flush=True)


def mode_train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    train, _, tr_path = load_split(args.data_prefix, "train")
    valid, _, va_path = load_split(args.data_prefix, "valid")
    print(f"[p8] data train={len(train)} valid={len(valid)} device={device}", flush=True)

    ck = torch.load(args.resume, map_location=device, weights_only=False)
    model = MiniLM(**model_cfg_from(ck, args)).to(device)
    model.load_state_dict(ck["model"])

    decay, no_decay = [], []
    for _, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95))
    if "opt" in ck:
        opt.load_state_dict(ck["opt"])
    else:
        print("[p8][warn] ckpt'te opt yok; optimizer sifirdan", flush=True)
    start_step = int(ck["step"])
    del ck
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"[p8] resume step={start_step} schedule={args.schedule}", flush=True)

    lab = None
    K = 1
    if args.schedule in ("related", "random"):
        z = np.load(args.cluster_file)
        zn = [str(s) for s in z["names"]]
        zl = z["rel"] if args.schedule == "related" else z["rnd"]
        lab = {n: int(v) for n, v in zip(zn, zl.tolist())}
        K = int(max(lab.values())) + 1

    groups = unit_groups(model)
    if lab is not None:
        missing = [n for n, _ in groups if n not in lab]
        assert not missing, f"cluster dosyasinda eksik birim: {missing}"
        for c in range(K):
            print(f"[p8] C{c} = {[n for n, _ in groups if lab[n] == c]}", flush=True)

    gen_e = torch.Generator().manual_seed(999)
    wins = []
    for _ in range(args.n_eval):
        s = int(torch.randint(0, len(valid) - args.ctx - 1, (1,), generator=gen_e))
        wins.append(valid[s: s + args.ctx + 1].long())
    wins = torch.stack(wins)
    amp = device == "cuda"

    def do_eval(step):
        model.eval()
        tot, n = 0.0, 0.0
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=amp):
            for i in range(0, args.n_eval, 16):
                xb = wins[i: i + 16].to(device)
                l = model.loss(xb)
                tot += float(l) * xb.size(0) * (xb.size(1) - 1)
                n += xb.size(0) * (xb.size(1) - 1)
        model.train()
        return tot / n

    v0 = do_eval(start_step)
    print(f"[p8] start_val @{start_step} = {v0:.4f} (ppl={math.exp(v0):.2f})", flush=True)

    def check_invariants(active):
        for n, ps in groups:
            want = (lab[n] == active)
            for p in ps:
                if want != (p.grad is not None):
                    raise RuntimeError(f"invariant ihlali: {n} active={active}")

    gen = torch.Generator().manual_seed(args.seed)
    model.train()
    best_val = float("inf")
    evals = []
    step_ms = []
    tok_seen = 0
    t_wall0 = time.time()
    ckpt_path = os.path.join(CKPT_DIR, f"{args.tag}_ckpt.pt")
    best_path = os.path.join(CKPT_DIR, f"{args.tag}_best.pt")

    for k in range(args.steps):
        gstep = start_step + 1 + k
        if gstep % args.eval_every == 0 or k == args.steps - 1:
            v = do_eval(gstep)
            ppl = math.exp(v)
            ev = {"step": gstep, "val_loss": round(v, 4), "ppl": round(ppl, 3)}
            evals.append(ev)
            print(f"  [p8 eval @{gstep}] val_loss={v:.4f} ppl={ppl:.2f}", flush=True)
            if v < best_val:
                best_val = v
                state_cpu = {kk: t.detach().cpu().clone()
                             for kk, t in model.state_dict().items()}
                save_atomic({"model": state_cpu, "cfg": model.cfg(),
                             "meta": {"tag": args.tag, "step": gstep, "kind": "best",
                                      "val_loss": v, "schedule": args.schedule,
                                      "ts": now()}}, best_path)
                print(f"  [p8 best] -> {best_path}", flush=True)
            if k == args.steps - 1:
                break

        if lab is None:
            active = -1
        elif args.phase > 0:                      # fazli parcali egitim (P8v2)
            active = (gstep // args.phase) % K
        else:
            active = gstep % K
        if lab is not None:
            for n, ps in groups:
                req = (lab[n] == active)
                for p in ps:
                    p.requires_grad_(req)

        for pg in opt.param_groups:
            pg["lr"] = args.lr * lr_at(gstep, args.warmup, args.steps_total,
                                       args.min_lr_frac)

        xb = get_batch(train, args.ctx, args.batch, gen, device)
        if amp:
            torch.cuda.synchronize()
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.bfloat16, enabled=amp):
            loss = model.loss(xb)
        loss.backward()
        if lab is not None and (k < 3 or args.check_invariants):
            check_invariants(active)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        if lab is not None:
            for n, ps in groups:
                for p in ps:
                    p.requires_grad_(True)
        if amp:
            torch.cuda.synchronize()
        dt = time.time() - t0
        tok_seen += args.batch * args.ctx
        if k % 20 == 0:
            step_ms.append(dt * 1000)
        if (k + 1) % 100 == 0 or k == args.steps - 1:
            vram = f" vram={torch.cuda.memory_reserved() / 1e9:.2f}G" if amp else ""
            print(f"  [p8 step {gstep}] loss={float(loss.detach()):.4f} "
                  f"lr={opt.param_groups[0]['lr']:.2e} {dt * 1000:.0f}ms{vram}", flush=True)
        if gstep % args.ckpt_every == 0:
            save_atomic({"model": model.state_dict(), "opt": opt.state_dict(),
                         "step": gstep, "best_val": best_val, "evals": evals,
                         "args": vars(args), "ts": now()}, ckpt_path)
            print(f"  [p8 ckpt] step {gstep} -> {ckpt_path}", flush=True)

    wall_min = (time.time() - t_wall0) / 60
    med_ms = sorted(step_ms)[len(step_ms) // 2] if step_ms else 0.0
    row = {"exp": "exp14_p8_clusters", "tag": args.tag, "schedule": args.schedule,
           "phase": args.phase,
           "k": K, "cluster_file": os.path.basename(args.cluster_file),
           "base_step": start_step, "steps": args.steps, "batch": args.batch,
           "ctx": args.ctx, "lr": args.lr, "warmup": args.warmup,
           "steps_total": args.steps_total, "n_eval": args.n_eval,
           "tokens_seen": tok_seen, "evals": evals,
           "start_val": round(v0, 4),
           "best_val_loss": round(best_val, 4),
           "final_val_loss": evals[-1]["val_loss"] if evals else None,
           "step_med_ms": round(med_ms, 1), "wall_min": round(wall_min, 1),
           "vram_peak_gb": (round(torch.cuda.max_memory_reserved() / 1e9, 2)
                            if amp else None),
           "cluster_membership": ({n: lab[n] for n, _ in groups} if lab else None),
           "ts": now()}
    append_jsonl(RESULTS, row)
    print(f"[p8 done] {args.tag} final={row['final_val_loss']} "
          f"best={row['best_val_loss']} wall={wall_min:.1f}min med={med_ms:.0f}ms",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["corr", "train"], required=True)
    ap.add_argument("--resume", required=True)
    ap.add_argument("--data_prefix", default="ts")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--rnd_seed", type=int, default=None,
                    help="rastgele-kontrol bolunmesi icin ayri RNG tohumu "
                         "(partition-robustluk testi)")
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--mlp_hidden", type=int, default=1024)
    ap.add_argument("--loops", type=int, default=1)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "p8_clusters.npz"))
    ap.add_argument("--corr_batches", type=int, default=6)
    ap.add_argument("--corr_batch", type=int, default=16)
    ap.add_argument("--eps", type=float, default=1e-3)
    ap.add_argument("--bal_lo", type=float, default=0.35)
    ap.add_argument("--bal_hi", type=float, default=0.65)
    ap.add_argument("--min_frac", type=float, default=0.1)
    ap.add_argument("--schedule", choices=["related", "random", "full"],
                    default="related")
    ap.add_argument("--phase", type=int, default=0)   # 0=her adim; >0=faz uzunlugu (P8v2)
    ap.add_argument("--cluster_file",
                    default=os.path.join(ROOT, "results", "p8_clusters.npz"))
    ap.add_argument("--tag", default="p8")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--steps_total", type=int, default=18000)
    ap.add_argument("--min_lr_frac", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--n_eval", type=int, default=128)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--check_invariants", action="store_true")
    args = ap.parse_args()

    if args.mode == "corr":
        mode_corr(args)
    else:
        mode_train(args)


if __name__ == "__main__":
    main()
