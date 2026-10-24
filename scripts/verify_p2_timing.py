"""P2 timing verification: sync'li vs sync'siz (WDDM enqueue artefaktı) adım süreleri.

Amaç: exp10'da flat modelinin step_med_ms=16.55 değeri CPU-enqueue ölçümü olabilir
(statik graf, adım içinde sync yok); patch'te triton/gather yolları sync zorladığı
için ölçüm gerçeğe yakın. Bu script, her iki model için HER adımda senkronize ederek
gerçek adım süresini ölçer ve eval_bpb'nin gerçek maliyetini doğrular.

Kullanım: python scripts/verify_p2_timing.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from lcsa.byte_lm import BytePatchLM, FlatByteLM, eval_bpb, load_bytes, sample_windows

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bench(model, tb, device, n=40, sync_every_step=True, W=512, batch=32):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-7)
    gen = torch.Generator().manual_seed(0)
    for _ in range(3):  # warmup
        xb = sample_windows(tb, W, batch, gen).to(device)
        opt.zero_grad(set_to_none=True)
        nll, cnt = model.byte_loss(xb)
        (nll / cnt).backward()
        opt.step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        xb = sample_windows(tb, W, batch, gen).to(device)
        opt.zero_grad(set_to_none=True)
        nll, cnt = model.byte_loss(xb)
        (nll / cnt).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if sync_every_step:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    t1 = time.time()
    return (t1 - t0) / n * 1000.0


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tb = load_bytes(os.path.join(ROOT, "data", "bytes_wt2_train.pt"))
    vb = load_bytes(os.path.join(ROOT, "data", "bytes_wt2_valid.pt"))
    teb = load_bytes(os.path.join(ROOT, "data", "bytes_wt2_test.pt"))

    specs = [
        ("flat", FlatByteLM, dict(W=512, dim=256, layers=6, heads=8)),
        ("patch", BytePatchLM, dict(P=4, W=512, byte_dim=64, dim=256, layers=6,
                                    heads=8, lo_layers=2, lo_heads=4)),
    ]
    out = {}
    for name, cls, kw in specs:
        ck = torch.load(os.path.join(ROOT, "checkpoints", f"p2_{name}.pt"),
                        map_location=dev)
        m = cls(**kw).to(dev)
        m.load_state_dict(ck["state"])
        ms_sync = bench(m, tb, dev, n=40, sync_every_step=True)
        ms_async = bench(m, tb, dev, n=40, sync_every_step=False)
        torch.cuda.synchronize()
        t0 = time.time()
        v = eval_bpb(m, vb, 512, 64, 32, dev)
        t1 = time.time()
        vt = eval_bpb(m, teb, 512, 128, 32, dev)
        out[name] = {"ms_sync": round(ms_sync, 1), "ms_nosync": round(ms_async, 1),
                     "eval64_sec": round(t1 - t0, 3), "valid_bpb_check": round(v, 4),
                     "test_bpb_check": round(vt, 4)}
        print(f"{name}: sync {ms_sync:.1f} ms/step | nosync {ms_async:.1f} ms/step | "
              f"eval(64) {t1 - t0:.3f}s | valid_bpb={v:.4f} | test_bpb={vt:.4f}",
              flush=True)

    return out


if __name__ == "__main__":
    main()
