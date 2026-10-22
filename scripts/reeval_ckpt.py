"""Kontrol-pencereli yeniden degerlendirme (MoE analizi icin).

a2 ailesi (a2_ctrl8k, p4_flat4, p4_loop2x4) `tok2_ctrl8k_valid` uzerinde
seed 999 + ilk 128 pencere ile degerlendirildi. moe4 (ts verisi) farkli
pencerelerde oldugundan adil kiyas icin AYNI pencerelerle yeniden degerlendirilir.

Tum isler tek scriptte: a2 ailesi teyit + moe4 [a2-pencere/adil] + moe4 [ts-pencere/teyit].
Cikti: konsol + results.jsonl (exp="reeval").
"""
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch

from lcsa.mini_lm import MiniLM
from lcsa.metrics import append_jsonl, now

RESULTS = os.path.join(ROOT, "results", "results.jsonl")
CTX, N_EVAL = 512, 128


def load_valid(which):
    if which == "a2":
        d = torch.load(os.path.join(ROOT, "data", "tok2_ctrl8k_valid.pt"),
                       weights_only=True)
        return d["tokens"], d.get("n_bytes")
    d = torch.load(os.path.join(ROOT, "data", "ts_valid.pt"), weights_only=True)
    return d, None


def eval_on(model, valid, device):
    gen = torch.Generator().manual_seed(999)
    wins = []
    for _ in range(N_EVAL):
        s = int(torch.randint(0, len(valid) - CTX - 1, (1,), generator=gen))
        wins.append(valid[s: s + CTX + 1].long())
    wins = torch.stack(wins)
    tot = n = 0.0
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16,
                                         enabled=(device == "cuda")):
        for i in range(0, N_EVAL, 16):
            xb = wins[i:i + 16].to(device)
            l = model.loss(xb)
            tot += float(l) * xb.size(0) * (xb.size(1) - 1)
            n += xb.size(0) * (xb.size(1) - 1)
    return tot / n


def main():
    if torch.cuda.is_available():
        free_mb = torch.cuda.mem_get_info()[0] / 1e6
        device = "cuda" if free_mb > 1500 else "cpu"
    else:
        device = "cpu"
    print(f"[reeval] device={device}", flush=True)
    v_a2, nb_a2 = load_valid("a2")
    v_ts, _ = load_valid("ts")
    ratio_a2 = len(v_a2) / (math.log(2) * nb_a2)      # bpb katsayisi

    jobs = [
        ("a2_ctrl8k [a2-pencere/teyit]", "checkpoints/a2_ctrl8k_best.pt", "a2"),
        ("p4_flat4 [a2-pencere/teyit]", "checkpoints/p4_flat4_best.pt", "a2"),
        ("p4_loop2x4 [a2-pencere/teyit]", "checkpoints/p4_loop2x4_best.pt", "a2"),
        ("moe4 [a2-pencere/ADIL]", "checkpoints/moe4_best.pt", "a2"),
        ("moe4 [ts-pencere/teyit]", "checkpoints/moe4_best.pt", "ts"),
    ]
    for name, rel, which in jobs:
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p):
            print(f"[reeval] ATLA (yok): {rel}", flush=True)
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        cfg = ck.get("cfg") or {}
        kw = dict(vocab=cfg.get("vocab", 8192), dim=cfg.get("dim", 384),
                  layers=cfg.get("layers", 8), heads=cfg.get("heads", 6),
                  ctx=cfg.get("ctx", 512), mlp_hidden=cfg.get("mlp_hidden", 1024),
                  loops=cfg.get("loops", 1), moe_experts=cfg.get("moe", 0))
        model = MiniLM(**kw).to(device)
        model.load_state_dict(ck["model"])
        del ck
        v = eval_on(model, v_a2 if which == "a2" else v_ts, device)
        line = f"[reeval] {name}: val_loss={v:.4f} ppl={math.exp(v):.3f}"
        if which == "a2":
            line += f" bpb={v * ratio_a2:.4f}"
        print(line, flush=True)
        append_jsonl(RESULTS, {"exp": "reeval", "tag": name.split()[0],
                               "ckpt": os.path.basename(rel), "windows": which,
                               "val_loss": round(v, 4), "ppl": round(math.exp(v), 3),
                               "bpb": (round(v * ratio_a2, 4) if which == "a2" else None),
                               "ts": now()})
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
