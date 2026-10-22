"""P3 (exp11) + P5 asama-2 kosucusu: mini LM'i sifirdan egitir (TinyStories).

Kullanim:
  python experiments/exp11_own_lm.py --tag p3 --steps 20000 --batch 48
  python experiments/exp11_own_lm.py --tag p3 --steps 20000 --batch 48 --resume checkpoints/p3_ckpt.pt
  # P5 asama-2 (tokenizer arenasi; dict veri -> bpb):
  python experiments/exp11_own_lm.py --tag a2_sw8k --steps 4000 --batch 48 --vocab 8192 ^
      --data_prefix tok2_sw8k --tok_path data/tok2/bpe_sw_8k.json --n_eval 128

Veri: {data_prefix}_{train,valid}.pt
  - duz uint16 tensor (P3) veya {"tokens", "n_bytes", ...} dict (tok2; bpb icin)
Cikti: checkpoints/{tag}_best.pt (en iyi valid), {tag}_ckpt.pt (resume icin,
optimizer'li, her --ckpt_every); results/results.jsonl'e tek satir (exp11_own_lm).
n_bytes meta'si varsa bpb (bit/byte) da hesaplanir (tokenizer-bagimsiz metrik).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from lcsa.metrics import append_jsonl, now
from lcsa.mini_lm import MiniLM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
RESULTS = os.path.join(ROOT, "results", "results.jsonl")


def ascii_safe(s, n=400):
    return s[:n].replace("\n", " ").encode("ascii", "replace").decode()


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
    """{prefix}_{split}.pt yukle; eski 'ts_smoke' konvansiyonuna geri-dus.

    Duz tensor -> (tensor, None) ; dict -> (tokens, n_bytes)."""
    p = os.path.join(DATA, f"{prefix}_{split}.pt")
    if not os.path.exists(p) and "smoke" in prefix:
        p = os.path.join(DATA, f"ts_{split}_smoke.pt")
    obj = torch.load(p, weights_only=True)
    if isinstance(obj, dict):
        return obj["tokens"], obj.get("n_bytes"), p
    return obj, None, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p3")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--min_lr_frac", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--mlp_hidden", type=int, default=1024)
    ap.add_argument("--loops", type=int, default=1)           # P4: dongulu derinlik
    ap.add_argument("--moe_experts", type=int, default=0)     # exp15: E uzmanli top-1 MoE
    ap.add_argument("--loops_rand", type=int, default=0)      # P4.2: egitim L ~ U{1..Lmax}
    ap.add_argument("--loop_sweep", action="store_true")      # eval'de L=1/2/4/8 tara
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--n_eval", type=int, default=32)
    ap.add_argument("--gen_every", type=int, default=2000)
    ap.add_argument("--gen_tokens", type=int, default=160)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--resume", default="")
    ap.add_argument("--data_prefix", default="ts")
    ap.add_argument("--tok_path", default="")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    train, _, tr_path = load_split(args.data_prefix, "train")
    valid, valid_bytes, va_path = load_split(args.data_prefix, "valid")
    print(f"[data] train={len(train)} tok ({os.path.basename(tr_path)}) | "
          f"valid={len(valid)} tok ({os.path.basename(va_path)})"
          + (f" | valid_bytes={valid_bytes}" if valid_bytes else ""), flush=True)
    plan_tok = args.steps * args.batch * args.ctx
    print(f"[plan] {args.steps} adim x {args.batch}x{args.ctx} = {plan_tok/1e6:.0f}M token "
          f"(~{plan_tok/max(len(train),1):.2f} epoch)", flush=True)

    mlp_hidden_eff = (args.mlp_hidden // args.moe_experts
                      if args.moe_experts > 0 else args.mlp_hidden)
    if args.moe_experts > 0:
        assert args.mlp_hidden % args.moe_experts == 0, "mlp_hidden / moe_experts tam bolunmeli"
    model = MiniLM(vocab=args.vocab, dim=args.dim, layers=args.layers,
                   heads=args.heads, ctx=args.ctx, mlp_hidden=mlp_hidden_eff,
                   loops=args.loops, moe_experts=args.moe_experts).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params/1e6:.2f}M cfg={model.cfg()}", flush=True)

    decay, no_decay = [], []
    for _, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.wd},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95))

    start_step, best_val, evals = 0, float("inf"), []
    ckpt_path = os.path.join(CKPT_DIR, f"{args.tag}_ckpt.pt")
    best_path = os.path.join(CKPT_DIR, f"{args.tag}_best.pt")
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"] + 1
        best_val = ck["best_val"]
        evals = ck.get("evals", [])
        del ck  # P3-resume fix: ckpt GPU'da tutulup ~0.4G ekstra rezerve (WDDM tasmasi) yaratiyordu
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"[resume] step={start_step} best_val={best_val:.4f} "
              + (f"vram={torch.cuda.memory_reserved()/1e9:.2f}G (bosaltildi)"
                 if device == "cuda" else ""), flush=True)

    # tokenizer (ornek uretim icin; --tok_path verilirse o kullanilir)
    tok_suffix = "_smoke" if "smoke" in args.data_prefix else ""
    tok_path = args.tok_path or os.path.join(DATA, f"ts_bpe{args.vocab // 1000}k{tok_suffix}.json")
    tok = None
    if os.path.exists(tok_path):
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(tok_path)

    def lr_at(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog))

    # sabit eval pencereleri (best-olcumu kosular arasi karsilastirilabilir)
    gen_e = torch.Generator().manual_seed(999)
    wins = []
    for _ in range(args.n_eval):
        s = int(torch.randint(0, len(valid) - args.ctx - 1, (1,), generator=gen_e))
        wins.append(valid[s: s + args.ctx + 1].long())
    wins = torch.stack(wins)  # (n_eval, ctx+1)

    def do_eval(step):
        model.eval()
        tot, n = 0.0, 0.0
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16,
                                             enabled=(device == "cuda")):
            for i in range(0, args.n_eval, 16):
                xb = wins[i: i + 16].to(device)
                l = model.loss(xb)
                tot += float(l) * xb.size(0) * (xb.size(1) - 1)
                n += xb.size(0) * (xb.size(1) - 1)
        model.train()
        return tot / n

    prompt = None
    if tok is not None:
        prompt = torch.tensor([tok.encode("Once upon a time").ids])

    def do_sample(step, temp=0.8, top_k=100):
        if tok is None or prompt is None:
            return "", 0.0
        t0 = time.time()
        ids = model.generate(prompt, args.gen_tokens, temp=temp, top_k=top_k,
                             eos_id=tok.token_to_id("<|endoftext|>"))
        dt = time.time() - t0
        text = tok.decode(ids[0].tolist())
        print(f"  [sample @{step} {dt:.1f}s] {ascii_safe(text, 300)}", flush=True)
        model.train()
        return text, dt

    gen = torch.Generator().manual_seed(args.seed)
    gen_lr = torch.Generator().manual_seed(args.seed + 777)   # P4.2 (loops_rand)
    loop_hist = ({L: 0 for L in range(1, args.loops_rand + 1)}
                 if args.loops_rand > 0 else {})
    model.train()
    step_ms = []
    tok_seen = 0
    t_wall0 = time.time()
    amp = device == "cuda"

    for step in range(start_step, args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            v = do_eval(step)
            ppl = math.exp(v)
            ev = {"step": step, "val_loss": round(v, 4), "ppl": round(ppl, 3)}
            if valid_bytes:
                ev["bpb"] = round(v * len(valid) / (math.log(2) * valid_bytes), 4)
                print(f"  [eval @{step}] val_loss={v:.4f} ppl={ppl:.2f} bpb={ev['bpb']:.4f}",
                      flush=True)
            else:
                print(f"  [eval @{step}] val_loss={v:.4f} ppl={ppl:.2f}", flush=True)
            if args.loop_sweep:
                base_l = model.loops
                sw = {}
                for L in (1, 2, 4, 8):
                    model.loops = L
                    vl = do_eval(step)
                    sw[str(L)] = {
                        "val_loss": round(vl, 4), "ppl": round(math.exp(vl), 3),
                        "bpb": (round(vl * len(valid) / (math.log(2) * valid_bytes), 4)
                                if valid_bytes else None)}
                model.loops = base_l
                ev["loop_sweep"] = sw
                print("  [loop-sweep @%d] " % step + " ".join(
                    (f"L={L} bpb={sw[str(L)]['bpb']:.4f}" if sw[str(L)]["bpb"] is not None
                     else f"L={L} val={sw[str(L)]['val_loss']:.4f}")
                    for L in (1, 2, 4, 8)), flush=True)
            if args.moe_experts > 0:
                print("  [moe @%d] " % step + " ".join(
                    "L%d[" % i + ",".join("%.2f" % float(u) for u in
                                          blk.mlp.last_usage.tolist()) + "]"
                    for i, blk in enumerate(model.blocks)), flush=True)
            evals.append(ev)
            if v < best_val:
                best_val = v
                state_cpu = {k: t.detach().cpu().clone()
                             for k, t in model.state_dict().items()}
                save_atomic({"model": state_cpu, "cfg": model.cfg(),
                             "meta": {"tag": args.tag, "step": step, "kind": "best",
                                      "val_loss": v, "ppl": ppl, "ts": now()}},
                            best_path)
                print(f"  [best] -> {best_path}", flush=True)
        if step == args.steps:
            break
        if step > 0 and step % args.gen_every == 0:
            do_sample(step)

        for pg in opt.param_groups:
            pg["lr"] = args.lr * lr_at(step)

        if args.loops_rand > 0:               # P4.2: dongu sayisi her adimda rastgele
            model.loops = int(torch.randint(1, args.loops_rand + 1, (1,),
                                            generator=gen_lr).item())
            loop_hist[model.loops] += 1
        xb = get_batch(train, args.ctx, args.batch, gen, device)
        if amp:
            torch.cuda.synchronize()
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.bfloat16, enabled=amp):
            loss = model.loss(xb, add_aux=(args.moe_experts > 0))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step()
        if args.loops_rand > 0:               # ornek/eval icin taban donguye don
            model.loops = args.loops
        if amp:
            torch.cuda.synchronize()
        dt = time.time() - t0
        tok_seen += args.batch * args.ctx
        if step % 20 == 0:
            step_ms.append(dt * 1000)
        if step % 100 == 0:
            g = model.tok.weight.detach().norm().item()
            vram = f" vram={torch.cuda.memory_reserved()/1e9:.2f}G" if amp else ""
            print(f"  [step {step}] loss={float(loss.detach()):.4f} lr={opt.param_groups[0]['lr']:.2e} "
                  f"{dt*1000:.0f}ms emb_norm={g:.2f}{vram}", flush=True)
        if step > 0 and step % args.ckpt_every == 0:
            save_atomic({"model": model.state_dict(), "opt": opt.state_dict(),
                         "step": step, "best_val": best_val, "evals": evals,
                         "args": vars(args), "ts": now()}, ckpt_path)
            print(f"  [ckpt] step {step} -> {ckpt_path}", flush=True)

    wall_min = (time.time() - t_wall0) / 60
    med_ms = sorted(step_ms)[len(step_ms) // 2] if step_ms else 0.0

    # en iyi agirliklarla final ornekler
    samples = []
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device)["model"])
        s1, gt1 = do_sample(args.steps, temp=0.8, top_k=100)
        s2, gt2 = do_sample(args.steps + 1, temp=0.6, top_k=100)
        samples = [s1, s2]
    else:
        gt1 = gt2 = 0.0

    final_val = evals[-1]["val_loss"] if evals else None
    best_bpb = min((e["bpb"] for e in evals if "bpb" in e), default=None)
    row = {"exp": "exp11_own_lm", "tag": args.tag, "params": n_params,
           "cfg": model.cfg(), "steps": args.steps, "batch": args.batch,
           "ctx": args.ctx, "lr": args.lr, "warmup": args.warmup,
           "loops": args.loops,
           "loops_rand": (args.loops_rand or None),
           "moe_experts": (args.moe_experts or None),
           "moe_hidden": (mlp_hidden_eff if args.moe_experts else None),
           "router_usage_final": ([[round(float(u), 4) for u in
                                    blk.mlp.last_usage.tolist()]
                                   for blk in model.blocks]
                                  if args.moe_experts else None),
           "looprand_hist": loop_hist,
           "loop_sweep_final": (evals[-1].get("loop_sweep") if evals else None),
           "data_prefix": args.data_prefix, "vocab": args.vocab, "n_eval": args.n_eval,
           "tokens_seen": tok_seen, "evals": evals,
           "best_val_loss": round(best_val, 4), "best_ppl": round(math.exp(best_val), 3),
           "best_bpb": best_bpb, "valid_bytes": valid_bytes,
           "final_val_loss": final_val,
           "step_med_ms": round(med_ms, 1),
           "tokens_per_s": round(tok_seen / max(wall_min * 60, 1e-9)),
           "wall_min": round(wall_min, 1),
           "gen_sec_160": round((gt1 + gt2) / 2, 2),
           "vram_peak_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2) if amp else None,
           "sample_temp08": samples[0] if samples else "",
           "sample_temp06": samples[1] if len(samples) > 1 else "",
           "ts": now()}
    append_jsonl(RESULTS, row)
    print(f"[done] best_val={best_val:.4f} ppl={math.exp(best_val):.2f} "
          + (f"bpb={best_bpb:.4f} " if best_bpb is not None else "")
          + f"wall={wall_min:.1f}min tokens/s={row['tokens_per_s']}", flush=True)


if __name__ == "__main__":
    main()
