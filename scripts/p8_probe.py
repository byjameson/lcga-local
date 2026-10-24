"""P8 protokol probe: hangi rejim stabil? (faz-uzunlugu alternasyonu testi)

3 config x 200 adim (batch 48, ctx 512), s3000 ckpt'ten, warm Adam:
  full      : tum parametreler (P3 devami replika)
  step_alt  : kume her ADIMDA degisir (mevcut konfig)
  phase300  : kume her 300 ADIMDA degisir ("parcali egitim" hedefi)
Cikti: results/p8_probes.txt + results.jsonl (exp14_probe satirlari)
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import numpy as np
import torch

import exp14_p8_clusters as e14
from lcsa.mini_lm import MiniLM
from lcsa.metrics import append_jsonl, now

CKPT = os.path.join(ROOT, "checkpoints", "p3_ckpt_s3000.pt")
RESULTS = os.path.join(ROOT, "results", "results.jsonl")
OUT_TXT = os.path.join(ROOT, "results", "p8_probes.txt")


def build(device):
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    ns = argparse.Namespace(vocab=8192, dim=384, layers=8, heads=6, ctx=512,
                            mlp_hidden=1024, loops=1)
    model = MiniLM(**e14.model_cfg_from(ck, ns)).to(device)
    model.load_state_dict(ck["model"])
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=1e-3, betas=(0.9, 0.95))
    opt.load_state_dict(ck["opt"])
    step = int(ck["step"])
    del ck
    torch.cuda.empty_cache()
    return model, opt, step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337)
    train, _, _ = e14.load_split("ts", "train")
    valid, _, _ = e14.load_split("ts", "valid")
    print(f"[probe] data train={len(train)} valid={len(valid)} device={device}", flush=True)

    z = np.load(os.path.join(ROOT, "results", "p8_clusters.npz"))
    zn = [str(s) for s in z["names"]]
    lab = {n: int(v) for n, v in zip(zn, z["rel"].tolist())}

    gen_e = torch.Generator().manual_seed(999)
    wins = []
    for _ in range(32):
        s = int(torch.randint(0, len(valid) - 513, (1,), generator=gen_e))
        wins.append(valid[s: s + 513].long())
    wins = torch.stack(wins)
    amp = device == "cuda"

    sched = {"full": None, "step_alt": lambda g: g % 2,
             "phase300": lambda g: (g // 300) % 2}
    lines = []

    def w(s):
        print(s, flush=True)
        lines.append(s)

    for name in ["full", "step_alt", "phase300"]:
        model, opt, start = build(device)
        groups = e14.unit_groups(model)

        def ev():
            model.eval()
            tot = n = 0.0
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=amp):
                for i in range(0, 32, 16):
                    xb = wins[i:i + 16].to(device)
                    l = model.loss(xb)
                    tot += float(l) * xb.size(0) * (xb.size(1) - 1)
                    n += xb.size(0) * (xb.size(1) - 1)
            model.train()
            return tot / n

        v0 = ev()
        gen = torch.Generator().manual_seed(1337)
        losses = []
        t0 = time.time()
        for k in range(args.steps):
            g = start + 1 + k
            if sched[name] is not None:
                la = sched[name](g)
                for n, ps in groups:
                    req = (lab[n] == la)
                    for p in ps:
                        p.requires_grad_(req)
            for pg in opt.param_groups:
                pg["lr"] = 1e-3 * e14.lr_at(g, 500, 18000, 0.1)
            xb = e14.get_batch(train, 512, 48, gen, device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.bfloat16, enabled=amp):
                loss = model.loss(xb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if (k + 1) % 50 == 0:
                losses.append(round(float(loss.detach()), 3))
        for _, ps in groups:
            for p in ps:
                p.requires_grad_(True)
        v1 = ev()
        wall = time.time() - t0
        flag = ("EXPLODE" if v1 > v0 + 0.2 else
                "STABIL" if abs(v1 - v0) <= 0.2 else "??")
        w(f"[{name}] v0={v0:.4f} -> v1={v1:.4f} | train50={losses} | {wall:.0f}s | {flag}")
        append_jsonl(RESULTS, {"exp": "exp14_probe", "tag": f"probe_{name}",
                               "base_step": start, "steps": args.steps, "batch": 48,
                               "v0": round(v0, 4), "v1": round(v1, 4),
                               "train50": losses, "flag": flag,
                               "wall_s": round(wall, 1), "ts": now()})
        del model, opt
        torch.cuda.empty_cache()

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    w("PROBE BITTI")


if __name__ == "__main__":
    main()
