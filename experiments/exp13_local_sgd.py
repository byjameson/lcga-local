"""P6 (exp13): senkron maliyeti -- "parcali/senkronsuz" egitimde kalite bedeli.

Soru: N=4 isci modeli veriye bolup egitsek; GPU'lar arasi gradyan senkronu
(all-reduce) silinirse/hafifletilirse kalite (bpb) ne olur?

Kollar (hepsi AYNI toplam butce: steps x shards x batch x ctx token):
  full    : TEK model; her optimizer adimi 4 mikro-batch (her shard'dan 1)
            gradyanini biriktirip ortalar -> DDP+Adam'in matematiksel esdegeri.
  local   : 4 replica; her biri kendi shard'inda K adim; her K adimda parametre
            ortalamasi (local SGD; Adam durumlari replica-bazli kalir).
  nosync  : 4 replica; hic ortalanmaz. Sonunda tek tek + "soup" (parametre
            ortalamasi) degerlendirilir; sapma (divergence) egrisi kaydedilir.
  phase   : TEK model + 4 mikro-batch (full gibi), ama her fazda yalnizca bir
            parametre kumesi egitilir (digeri donuk) -> yalnizca AKTIF kumenin
            gradyanlari all-reduce edilir. Kume dosyasi: results/p8_clusters.npz.
            Iletisim muhasebesi: sync_frac = aktif-param orani (adim ort.).
            Tek GPU'da gercek ag trafigi olculmez; olculen: kalite + iletilen
            gradyan-parametre hacmi.

Durust not: tek GPU'da GERCEK ag trafigi olculmez; olculen sey "iletisimi
kesmenin algoritmik kalite bedeli"dir. Veri: {prefix}_{train,valid}.pt (dict).

Cikti: results/results.jsonl'e tek satir (exp13_local_sgd); ckpt:
checkpoints/{tag}.pt (nosync'ta soup + 4 replica state'i birlikte).
"""
from __future__ import annotations

import argparse
import copy
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


def ascii_safe(s, n=300):
    return s[:n].replace("\n", " ").encode("ascii", "replace").decode()


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


def shard_bounds(n, shards):
    size = n // shards
    b = [(i * size, (i + 1) * size) for i in range(shards)]
    b[-1] = (b[-1][0], n)
    return b


def get_batch_shard(data, lo, hi, ctx, batch, gen, device):
    max_start = (hi - lo) - ctx - 1
    assert max_start > 0, f"shard cok kucuk: {hi - lo}"
    starts = torch.randint(0, max_start, (batch,), generator=gen).tolist()
    xs = torch.stack([data[lo + s: lo + s + ctx + 1].long() for s in starts])
    return xs.to(device)


def unit_groups(model):
    """exp14 ile ayni parametre birimleri (cluster npz adlariyla eslesir)."""
    groups = [("tok", [model.tok.weight])]
    for i, b in enumerate(model.blocks):
        groups.append((f"L{i}.attn", [b.n1.weight, b.attn.qkv.weight, b.attn.proj.weight]))
        groups.append((f"L{i}.mlp", [b.n2.weight, b.mlp.w12.weight, b.mlp.w3.weight]))
    groups.append(("norm", [model.norm.weight]))
    return groups


def make_model_parts(args):
    model = MiniLM(vocab=args.vocab, dim=args.dim, layers=args.layers,
                   heads=args.heads, ctx=args.ctx, mlp_hidden=args.mlp_hidden,
                   loops=args.loops)
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95))
    return model, opt


def average_into(dst_model, src_models):
    """src_models parametrelerinin ortalamasini dst_model'e yaz."""
    with torch.no_grad():
        for ps, dp in zip(zip(*[m.parameters() for m in src_models]),
                          dst_model.parameters()):
            acc = ps[0].detach().clone()
            for p in ps[1:]:
                acc += p.detach()
            acc /= len(ps)
            dp.copy_(acc)


def sync_models(models):
    """Tum replica'larin parametrelerini ortalamaya esitle (local SGD sync)."""
    average_into(models[0], models)
    with torch.no_grad():
        for ps in zip(*[m.parameters() for m in models]):
            for p in ps[1:]:
                p.copy_(ps[0])


def divergence(models):
    """Ortalama ikili rel-L2 (parametre uzayinda); 0 = ozdes."""
    flats = [torch.cat([p.detach().reshape(-1) for p in m.parameters()])
             for m in models]
    d, n = 0.0, 0
    for i in range(len(flats)):
        for j in range(i + 1, len(flats)):
            denom = 0.5 * (flats[i].norm() + flats[j].norm()) + 1e-12
            d += float((flats[i] - flats[j]).norm() / denom)
            n += 1
    return d / n if n else 0.0


def eval_model(model, wins, device, chunk=16):
    """Ortalama CE (val_loss) -- exp11 ile ayni pencere/agirliklandirma."""
    model.eval()
    tot, n = 0.0, 0.0
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16,
                                         enabled=(device == "cuda")):
        for i in range(0, wins.size(0), chunk):
            xb = wins[i: i + chunk].to(device)
            l = model.loss(xb)
            tot += float(l) * xb.size(0) * (xb.size(1) - 1)
            n += xb.size(0) * (xb.size(1) - 1)
    model.train()
    return tot / n


def lr_at(step, args):
    if step < args.warmup:
        return (step + 1) / args.warmup
    prog = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog))


def bpb_of(v, args, valid_len, valid_bytes):
    if not valid_bytes:
        return None
    return round(v * valid_len / (math.log(2) * valid_bytes), 4)


def train_step(model, opt, xb, args, amp, step_idx):
    for pg in opt.param_groups:
        pg["lr"] = args.lr * lr_at(step_idx, args)
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", torch.bfloat16, enabled=amp):
        loss = model.loss(xb)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    opt.step()
    return float(loss.detach())


def run_selftest(device):
    """DDP-esdeger: 2x(mikro-batch/2) birikim == tek buyuk batch adimi; sync dogru mu."""
    torch.manual_seed(0)
    m1 = MiniLM(vocab=512, dim=64, layers=2, heads=4, ctx=128, mlp_hidden=128).to(device)
    torch.manual_seed(0)
    m2 = MiniLM(vocab=512, dim=64, layers=2, heads=4, ctx=128, mlp_hidden=128).to(device)
    g = torch.Generator().manual_seed(1)
    xb1 = torch.randint(0, 512, (2, 129), generator=g).to(device)
    xb2 = torch.randint(0, 512, (2, 129), generator=g).to(device)
    big = torch.cat([xb1, xb2], 0)
    o1 = torch.optim.AdamW(m1.parameters(), lr=1e-3)
    o2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    o1.zero_grad(set_to_none=True)
    (m1.loss(xb1) / 2).backward()
    (m1.loss(xb2) / 2).backward()
    o1.step()
    o2.zero_grad(set_to_none=True)
    m2.loss(big).backward()
    o2.step()
    md = max(float((a - b).abs().max()) for a, b in zip(m1.parameters(), m2.parameters()))
    print(f"[selftest] birikim(2x2)/2 vs tek(4) adim sonrasi maks param farki={md:.2e}")
    assert md < 1e-4, "DDP-esdeger birikim testi BASARISIZ!"

    m3 = copy.deepcopy(m1)
    with torch.no_grad():
        for p, q in zip(m1.parameters(), m3.parameters()):
            q.copy_(p + 1.0)
    sync_models([m1, m3])
    d2 = max(float((a - b).abs().max()) for a, b in zip(m1.parameters(), m3.parameters()))
    chk = 0.5 * (float(m1.tok.weight.detach().abs().sum()))  # sigorta: sifir degil
    print(f"[selftest] sync sonrasi m1-m3 fark={d2:.2e} (ozdes olmali); agirlik normu={chk:.1f}")
    assert d2 < 1e-6, "sync testi BASARISIZ!"
    print("[SELFTEST OK]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="e13")
    ap.add_argument("--mode", choices=["full", "local", "nosync", "phase"],
                    default="full")
    ap.add_argument("--cluster_file",
                    default=os.path.join(ROOT, "results", "p8_clusters.npz"))
    ap.add_argument("--phase", type=int, default=300,
                    help="phase modunda faz uzunlugu (adim)")
    ap.add_argument("--steps", type=int, default=1000)      # isci basina optimizer adimi
    ap.add_argument("--block", type=int, default=250)       # olcum/blogu (ve varsayilan sync araligi)
    ap.add_argument("--sync_every", type=int, default=250)  # local modu icin
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)        # isci basina
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--min_lr_frac", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--mlp_hidden", type=int, default=1024)
    ap.add_argument("--loops", type=int, default=1)
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--n_eval", type=int, default=128)
    ap.add_argument("--gen_tokens", type=int, default=160)
    ap.add_argument("--data_prefix", default="tok2_ctrl8k")
    ap.add_argument("--tok_path", default="")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    device = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu")
    amp = device == "cuda"

    if args.selftest:
        run_selftest(device)
        return

    torch.manual_seed(args.seed)
    train, _, tr_path = load_split(args.data_prefix, "train")
    valid, valid_bytes, va_path = load_split(args.data_prefix, "valid")
    bounds = shard_bounds(len(train), args.shards)
    print(f"[data] train={len(train)} tok | valid={len(valid)} tok "
          f"| valid_bytes={valid_bytes}", flush=True)
    print(f"[shards] {args.shards} x ~{(bounds[0][1]-bounds[0][0])/1e6:.1f}M tok "
          f"({os.path.basename(tr_path)})", flush=True)
    tot_tok = args.steps * args.shards * args.batch * args.ctx
    print(f"[plan] {args.mode}: {args.steps} adim/isci x {args.shards} isci x "
          f"{args.batch}x{args.ctx} = {tot_tok/1e6:.0f}M token toplam", flush=True)

    # tokenizer (ornek uretim)
    tok = None
    if args.tok_path and os.path.exists(args.tok_path):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(args.tok_path)
    prompt = torch.tensor([tok.encode("Once upon a time").ids]) if tok else None

    # sabit eval pencereleri
    gen_e = torch.Generator().manual_seed(999)
    wins = []
    for _ in range(args.n_eval):
        s = int(torch.randint(0, len(valid) - args.ctx - 1, (1,), generator=gen_e))
        wins.append(valid[s: s + args.ctx + 1].long())
    wins = torch.stack(wins)

    def do_sample(model, temp=0.8, top_k=100):
        if tok is None:
            return "", 0.0
        t0 = time.time()
        ids = model.generate(prompt, args.gen_tokens, temp=temp, top_k=top_k,
                             eos_id=tok.token_to_id("<|endoftext|>"))
        dt = time.time() - t0
        text = tok.decode(ids[0].tolist())
        print(f"  [sample {dt:.1f}s] {ascii_safe(text)}", flush=True)
        model.train()
        return text, dt

    evals = []
    n_params = None
    cfg = None
    t_wall0 = time.time()
    med_ms = 0.0
    samples = []
    ckpt_obj = None

    if args.mode in ("full", "phase"):
        model, opt = make_model_parts(args)
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        cfg = model.cfg()
        print(f"[model] params={n_params/1e6:.2f}M cfg={cfg}", flush=True)
        if args.mode == "phase":
            z = np.load(args.cluster_file)
            zn = [str(s) for s in z["names"]]
            zl = z["rel"]
            lab = {n: int(v) for n, v in zip(zn, zl.tolist())}
            K = int(max(lab.values())) + 1
            groups = unit_groups(model)
            missing = [n for n, _ in groups if n not in lab]
            assert not missing, f"cluster eksik birim: {missing}"
            for c in range(K):
                print(f"  [phase] C{c} = "
                      f"{[n for n, _ in groups if lab[n] == c]}", flush=True)
            total_params = sum(p.numel() for p in model.parameters())
            csize = {c: sum(p.numel() for n, ps in groups if lab[n] == c
                            for p in ps) for c in range(K)}
            print("  [phase] aktif-param orani: "
                  + ", ".join(f"C{c}={csize[c] / total_params:.3f}"
                              for c in range(K)), flush=True)
        v = eval_model(model, wins, device)
        evals.append({"step": 0, "val_loss": round(v, 4),
                      "bpb": bpb_of(v, args, len(valid), valid_bytes)})
        print(f"  [eval @0] val_loss={v:.4f} bpb={evals[-1]['bpb']}", flush=True)
        gens = [torch.Generator().manual_seed(args.seed + 100 + r)
                for r in range(args.shards)]
        sync_frac = 0.0        # phase: iletilen gradyan-parametre orani toplami
        step_ms = []
        done = 0
        while done < args.steps:
            blk = min(args.block, args.steps - done)
            for k in range(blk):
                if amp:
                    torch.cuda.synchronize()
                t0 = time.time()
                if args.mode == "phase":
                    active = ((done + k) // args.phase) % K
                    for n, ps in groups:
                        req = (lab[n] == active)
                        for p in ps:
                            p.requires_grad_(req)
                for pg in opt.param_groups:
                    pg["lr"] = args.lr * lr_at(done + k, args)
                opt.zero_grad(set_to_none=True)
                for r, (lo, hi) in enumerate(bounds):
                    xb = get_batch_shard(train, lo, hi, args.ctx, args.batch,
                                         gens[r], device)
                    with torch.autocast("cuda", torch.bfloat16, enabled=amp):
                        loss = model.loss(xb) / args.shards
                    loss.backward()
                if args.mode == "phase" and k < 3:
                    for n, ps in groups:
                        want = (lab[n] == active)
                        for p in ps:
                            if want != (p.grad is not None):
                                raise RuntimeError(f"phase invariant ihlali: {n}")
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step()
                if args.mode == "phase":
                    for n, ps in groups:
                        for p in ps:
                            p.requires_grad_(True)
                    sync_frac += csize[active] / total_params
                if amp:
                    torch.cuda.synchronize()
                step_ms.append((time.time() - t0) * 1000)
            done += blk
            v = eval_model(model, wins, device)
            rec = {"step": done, "val_loss": round(v, 4),
                   "bpb": bpb_of(v, args, len(valid), valid_bytes)}
            evals.append(rec)
            msg = f"  [eval @{done}] val_loss={v:.4f} bpb={rec['bpb']}"
            if args.mode == "phase":
                msg += f" sync_frac_avg={sync_frac / max(done, 1):.3f}"
            print(msg, flush=True)
        ckpt_obj = {"model": {k: t.detach().cpu().clone()
                              for k, t in model.state_dict().items()},
                    "cfg": cfg, "args": vars(args), "evals": evals}

    else:  # local | nosync
        models, opts = [], []
        for r in range(args.shards):
            torch.manual_seed(args.seed)          # tum replica'lar AYNI init
            m, o = make_model_parts(args)
            models.append(m.to(device))
            opts.append(o)
        n_params = sum(p.numel() for p in models[0].parameters())
        cfg = models[0].cfg()
        print(f"[model] params={n_params/1e6:.2f}M x {args.shards} replica "
              f"cfg={cfg}", flush=True)
        gens = [torch.Generator().manual_seed(args.seed + 100 + r)
                for r in range(args.shards)]
        v = eval_model(models[0], wins, device)
        evals.append({"step": 0, "val_loss": round(v, 4),
                      "bpb": bpb_of(v, args, len(valid), valid_bytes),
                      "divergence": 0.0})
        print(f"  [eval @0] val_loss={v:.4f} bpb={evals[-1]['bpb']} div=0", flush=True)
        step_ms = []
        done = 0
        while done < args.steps:
            blk = min(args.block, args.steps - done)
            for r in range(args.shards):
                for k in range(blk):
                    if amp:
                        torch.cuda.synchronize()
                    t0 = time.time()
                    lo, hi = bounds[r]
                    xb = get_batch_shard(train, lo, hi, args.ctx, args.batch,
                                         gens[r], device)
                    train_step(models[r], opts[r], xb, args, amp, done + k)
                    if amp:
                        torch.cuda.synchronize()
                    step_ms.append((time.time() - t0) * 1000)
            done += blk
            div = divergence(models)
            rec = {"step": done, "divergence_before_sync": round(div, 6)}
            if args.mode == "local" and args.sync_every > 0 and done % args.sync_every == 0:
                sync_models(models)
                rec["synced"] = True
            if args.mode == "nosync":
                losses = [eval_model(m, wins, device) for m in models]
                rec["replica_val_loss"] = [round(x, 4) for x in losses]
                rec["replica_bpb"] = [bpb_of(x, args, len(valid), valid_bytes)
                                      for x in losses]
                rec["replica_bpb_mean"] = round(sum(rec["replica_bpb"]) / len(losses), 4)
                rec["val_loss"] = round(sum(losses) / len(losses), 4)
                rec["bpb"] = rec["replica_bpb_mean"]
                print(f"  [eval @{done}] replica bpb={rec['replica_bpb']} "
                      f"mean={rec['replica_bpb_mean']} div={rec['divergence_before_sync']}",
                      flush=True)
            else:
                v = eval_model(models[0], wins, device)
                rec["val_loss"] = round(v, 4)
                rec["bpb"] = bpb_of(v, args, len(valid), valid_bytes)
                print(f"  [eval @{done}] val_loss={v:.4f} bpb={rec['bpb']} "
                      f"div_pre_sync={rec['divergence_before_sync']}"
                      + (" (sync edildi)" if rec.get("synced") else ""), flush=True)
            evals.append(rec)

        soup = copy.deepcopy(models[0])
        average_into(soup, models)
        if args.mode == "nosync":
            v = eval_model(soup, wins, device)
            soup_bpb = bpb_of(v, args, len(valid), valid_bytes)
            evals.append({"step": done, "soup_val_loss": round(v, 4),
                          "soup_bpb": soup_bpb})
            print(f"  [soup] val_loss={v:.4f} bpb={soup_bpb} "
                  f"div_final={divergence(models):.4f}", flush=True)
        final_model = soup if args.mode == "nosync" else models[0]
        ckpt_obj = {"model": {k: t.detach().cpu().clone()
                              for k, t in final_model.state_dict().items()},
                    "cfg": cfg, "args": vars(args), "evals": evals}
        if args.mode == "nosync":
            ckpt_obj["soup_model"] = ckpt_obj.pop("model")
            ckpt_obj["replicas"] = [
                {k: t.detach().cpu().clone() for k, t in m.state_dict().items()}
                for m in models]

    final_model = final_model if args.mode not in ("full", "phase") else model
    s1, _ = do_sample(final_model, temp=0.8, top_k=100)
    s2, _ = do_sample(final_model, temp=0.6, top_k=100)
    samples = [s1, s2]

    wall_min = (time.time() - t_wall0) / 60
    med_ms = sorted(step_ms)[len(step_ms) // 2] if step_ms else 0.0

    final_bpb = next((e["bpb"] for e in reversed(evals)
                      if e.get("bpb") is not None), None)
    row = {"exp": "exp13_local_sgd", "tag": args.tag, "mode": args.mode,
           "steps": args.steps, "block": args.block,
           "sync_every": args.sync_every if args.mode == "local" else None,
           "phase": (args.phase if args.mode == "phase" else None),
           "cluster_file": (os.path.basename(args.cluster_file)
                            if args.mode == "phase" else None),
           "sync_frac_avg": (round(sync_frac / args.steps, 4)
                             if args.mode == "phase" else None),
           "shards": args.shards, "batch": args.batch, "ctx": args.ctx,
           "lr": args.lr, "warmup": args.warmup, "params": n_params, "cfg": cfg,
           "tokens_total": tot_tok, "data_prefix": args.data_prefix,
           "vocab": args.vocab, "n_eval": args.n_eval, "seed": args.seed,
           "evals": evals, "samples": samples,
           "final_bpb": final_bpb,
           "wall_min": round(wall_min, 1), "med_step_ms": round(med_ms, 1),
           "valid_bytes": valid_bytes, "ts": now()}
    append_jsonl(RESULTS, row)

    if "smoke" not in args.tag:
        ckpt_path = os.path.join(CKPT_DIR, f"{args.tag}.pt")
        save_atomic(ckpt_obj, ckpt_path)
        print(f"[ckpt] -> {ckpt_path}", flush=True)

    print(f"[done] tag={args.tag} mode={args.mode} final_bpb={row['final_bpb']} "
          f"wall={wall_min:.1f}min med_step={med_ms:.0f}ms", flush=True)


if __name__ == "__main__":
    main()
