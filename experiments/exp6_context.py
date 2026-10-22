"""EXP6: null-scaling bug quantification + clean-config quality study.

Smoke discovery: the null-key scaling  out = num * sigmoid(lse - nl)  with the
trained constant nl=-3 multiplies every attention output row by a gain in (0,1)
that is far below 1 for a large fraction of rows. This alone raised PPL 4x at
rho=0.25 (87.3 -> 21.4 when disabled). The trained coarse path was compensating
for that bug, and the behavior at rho=1.0 (PPL 32.4 vs dense 12.5) confirms the
exact path itself is otherwise faithful.

This experiment:
  Part 0 (T=1024, 16 chunks): null sweep + gate sweep -> best post-hoc config.
  Part 1 (T=1024, 32 chunks): clean config (exact-only, null off) quality
      curve rho in {0.125, 0.25, 0.5, 1.0} + controls (oracle/window/random/
      local) + per-layer recall/MSE stats + lse/scale probe (torch exact path).
  Part 2 (T=2048/4096): clean-config curve + oracle/window controls at rho=0.25
      (the regime where sparsity pays off).
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from lcsa.data import load_tokens, chunks_from_tokens
from lcsa.harness import load_model, quick_ppl
from lcsa.metrics import append_jsonl, now, routing_diagnostics
from lcsa.model import (patch_model, set_sparse_enabled, begin_capture, end_capture,
                        set_selection_mode, set_torch_exact)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "Qwen2.5-0.5B")
DATA_DIR = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results", "results.jsonl")
DEV = "cuda"
NL_OFF = -1e9


def load_ckpt(model, path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for i, layer in enumerate(model.model.layers):
        own = dict(layer.self_attn.named_parameters())
        for n, t in ckpt["state"][i].items():
            if n in own:
                own[n].data.copy_(t.to(own[n].device))
    return ckpt.get("meta", {})


def _logit(p):
    if p >= 1.0:
        return 30.0
    if p <= 0.0:
        return -30.0
    return math.log(p / (1.0 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--tag", type=str, default="v2")
    ap.add_argument("--mc1024", type=int, default=32)
    ap.add_argument("--mc2048", type=int, default=32)
    ap.add_argument("--mc4096", type=int, default=16)
    args = ap.parse_args()

    model = load_model(MODEL_DIR, device=DEV)
    meta = load_ckpt(model, args.ckpt)
    margs = meta.get("args", {})
    wrappers = patch_model(model, budget=margs.get("budget", 0.25), block=64,
                           n_concepts=margs.get("n_concepts", 64),
                           graph_rank=margs.get("graph_rank", 16),
                           dynamic=margs.get("dynamic", False),
                           content_rank=margs.get("content_rank", 0),
                           rel_bias=not margs.get("no_rel_bias", False),
                           capture_dense=True)
    load_ckpt(model, args.ckpt)
    print(f"[{now()}] loaded {args.ckpt}", flush=True)

    toks = load_tokens(os.path.join(DATA_DIR, "wt2_test.pt"))
    c1024 = chunks_from_tokens(toks, 1024, max_chunks=args.mc1024)
    c1024s = chunks_from_tokens(toks, 1024, max_chunks=16)
    c2048 = chunks_from_tokens(toks, 2048, max_chunks=args.mc2048)
    c4096 = chunks_from_tokens(toks, 4096, max_chunks=args.mc4096)

    out = {"exp": "exp6_context", "ckpt": args.ckpt, "tag": args.tag, "ts": now(),
           "budget": margs.get("budget", 0.25), "ppl": {}, "sweep": {}, "stats": {},
           "probe": {}}

    def reset():
        for w in wrappers.values():
            w.budget = margs.get("budget", 0.25)
            w.use_coarse = True
        set_selection_mode("topk")
        set_sparse_enabled(False)

    def dense(name, chunks, where):
        set_sparse_enabled(False)
        t0 = time.time()
        r = quick_ppl(model, chunks, device=DEV)
        where[name] = {"ppl": r["ppl"], "sec": round(time.time() - t0, 1)}
        print(f"[{name}] ppl={r['ppl']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        return r["ppl"]

    def stats_now(chunks, max_chunks=4, torch_exact=False):
        set_sparse_enabled(True)
        if torch_exact:
            set_torch_exact(True)
        rec, mse, lse = [], [], []
        for i in range(min(max_chunks, len(chunks))):
            begin_capture()
            model(input_ids=chunks[i].to(DEV).unsqueeze(0))
            cap = end_capture()
            diag = routing_diagnostics(cap)
            rec.append([d["recall_mean"] for d in diag])
            mse.append([float(((e["student"].float() - e["dense"].float()) ** 2).mean())
                        for e in cap if "dense" in e])
            lse.append([e.get("exact_stats") for e in cap if "exact_stats" in e])
        set_torch_exact(False)
        set_sparse_enabled(False)
        rt, mt = torch.tensor(rec), torch.tensor(mse)
        st = {"recall_mean": float(rt.mean()), "recall_first": float(rt[:, 0].mean()),
              "recall_last": float(rt[:, -1].mean()),
              "mse_mean": float(mt.mean()), "mse_first": float(mt[:, 0].mean()),
              "mse_last": float(mt[:, -1].mean())}
        if lse and lse[0] and lse[0][0]:
            n = len(lse[0])
            st["lse_mean_first"] = sum(e["lse_mean"] for e in lse[0]) / n
            st["lse_mean_last"] = sum(e["lse_mean"] for e in lse[0][-1:]) / 1
            if "scale_mean" in lse[0][0]:
                st["scale_mean"] = sum(e["scale_mean"] for e in lse[0]) / n
                st["scale_min"] = min(e["scale_min"] for e in lse[0])
        return st

    def run(name, chunks, rho=None, coarse=True, mode="topk", nl=None, g=None,
            do_stats=0, torch_exact=False, where=None):
        t0 = time.time()
        need_save = nl is not None or g is not None
        saves = []
        if need_save:
            for w in wrappers.values():
                saves.append((w.null_logit.data.clone(), w.gate.data.clone()))
        for w in wrappers.values():
            if rho is not None:
                w.budget = rho
            w.use_coarse = coarse
            if nl is not None:
                w.null_logit.data.fill_(nl)
            if g is not None:
                w.gate.data.fill_(_logit(g))
        set_selection_mode(mode)
        set_sparse_enabled(True)
        r = quick_ppl(model, chunks, device=DEV)
        st = None
        if do_stats:
            st = stats_now(chunks, max_chunks=do_stats, torch_exact=torch_exact)
        if need_save:
            for w, (nl0, gt0) in zip(wrappers.values(), saves):
                w.null_logit.data.copy_(nl0)
                w.gate.data.copy_(gt0)
        reset()
        ppl = r["ppl"]
        target = where if where is not None else out["sweep"]
        target[name] = {"ppl": ppl, "sec": round(time.time() - t0, 1)}
        if st:
            out["stats"][name] = st
        msg = f"[{name}] ppl={ppl:.3f}"
        if st:
            msg += f" recall={st['recall_mean']:.3f} mse={st['mse_mean']:.5f}"
            if "lse_mean_first" in st:
                msg += f" lse[0]={st['lse_mean_first']:.2f}"
                if "scale_mean" in st:
                    msg += f" scale={st['scale_mean']:.3f}"
        msg += f" ({time.time()-t0:.0f}s)"
        print(msg, flush=True)
        return ppl

    # ---------------- probe: as-trained null scaling per layer ----------------
    fund = run("probe_null3", c1024s, rho=0.25, coarse=False, do_stats=1,
               torch_exact=True, where=out["probe"])
    st = out["stats"].get("probe_null3", {})
    print(f"[probe] lse_first={st.get('lse_mean_first')} scale_mean={st.get('scale_mean')}"
          f" scale_min={st.get('scale_min')}", flush=True)

    # ---------------- Part 0: null + gate sweeps (16 chunks) ----------------
    w0 = out["sweep"]
    b16 = dense("dense16", c1024s, w0)
    run("ref_as_trained16", c1024s, rho=0.25, where=w0)
    for nl in [NL_OFF, -12.0, -8.0, -5.0, -3.0]:
        run(f"exact_nl{nl}_rho0.25", c1024s, rho=0.25, coarse=False, nl=nl, where=w0)
    for nl in [NL_OFF, -8.0, -3.0]:
        run(f"exact_nl{nl}_rho0.125", c1024s, rho=0.125, coarse=False, nl=nl, where=w0)
    for nl in [NL_OFF, -8.0, -3.0]:
        run(f"exact_nl{nl}_rho0.5", c1024s, rho=0.5, coarse=False, nl=nl, where=w0)
    for nl in [NL_OFF, -3.0]:
        run(f"exact_nl{nl}_rho1.0", c1024s, rho=1.0, coarse=False, nl=nl, where=w0)
    for g in [0.0, 0.3, 0.5, 0.7, 1.0]:
        run(f"mix_g{g}_nlOFF_rho0.25", c1024s, rho=0.25, g=g, nl=NL_OFF, where=w0)
    for g in [0.3, 0.7, 1.0]:
        run(f"mix_g{g}_nl3_rho0.25", c1024s, rho=0.25, g=g, where=w0)
    for k, v in w0.items():
        if "ppl" in v:
            v["delta"] = v["ppl"] - b16

    # ---------------- Part 1: clean config, T=1024, 32 chunks ----------------
    w1 = out["ppl"]["1024"] = {}
    clean = dict(coarse=False, nl=NL_OFF)
    b = dense("dense", c1024, w1)
    for rho in (0.125, 0.25, 0.5, 1.0):
        run(f"clean_rho{rho}", c1024, rho=rho, where=w1, **clean)
    run("clean_oracle_rho0.25", c1024, rho=0.25, mode="oracle", where=w1, **clean)
    run("clean_window_rho0.25", c1024, rho=0.25, mode="window", where=w1, **clean)
    run("clean_random_rho0.25", c1024, rho=0.25, mode="random", where=w1, **clean)
    run("clean_local", c1024, rho=0.25, mode="local", where=w1, **clean)
    run("clean_rho0.25_stats", c1024, rho=0.25, do_stats=4, where=w1, **clean)
    run("clean_oracle_rho0.25_stats", c1024, rho=0.25, mode="oracle", do_stats=4,
        where=w1, **clean)
    run("ref_as_trained", c1024, rho=0.25, where=w1)
    for k, v in w1.items():
        if "ppl" in v:
            v["delta"] = v["ppl"] - b

    # ---------------- Part 2: context length, clean config ----------------
    for T, chunks in [(2048, c2048), (4096, c4096)]:
        wt = out["ppl"][str(T)] = {}
        b = dense("dense", chunks, wt)
        for rho in (0.125, 0.25, 0.5):
            run(f"clean_rho{rho}", chunks, rho=rho, do_stats=(2 if rho == 0.25 else 0),
                where=wt, **clean)
        run("clean_oracle_rho0.25", chunks, rho=0.25, mode="oracle", do_stats=(0),
            where=wt, **clean)
        run("clean_window_rho0.25", chunks, rho=0.25, mode="window", where=wt, **clean)
        for k, v in wt.items():
            if "ppl" in v:
                v["delta"] = v["ppl"] - b

    gates = [round(float(torch.sigmoid(w.gate).detach().mean()), 3) for w in wrappers.values()]
    out["gate_per_layer"] = gates
    out["gate_mean"] = sum(gates) / len(gates)
    print(f"[gate] mean={out['gate_mean']:.3f} min={min(gates)} max={max(gates)}", flush=True)

    with open(os.path.join(ROOT, "results", f"exp6_{args.tag}.json"), "w") as f:
        json.dump(out, f, indent=1)
    append_jsonl(RESULTS, out)
    print(f"[{now()}] exp6 done", flush=True)


if __name__ == "__main__":
    main()
