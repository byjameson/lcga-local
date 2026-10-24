"""EXP1b: verify the GQA fix (parity) and re-measure dense speed with fused SDPA.

The as-loaded transformers sdpa path passes enable_gqa=True; on this torch/GPU
build that has no fused kernel and falls back to the math backend (measured
~104ms/layer at T=4096 vs ~2.5ms for the fused path). load_model(fast_dense=True)
expands GQA heads instead, restoring the memory-efficient kernel.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from lcsa.data import load_tokens, chunks_from_tokens
from lcsa.harness import load_model, quick_ppl, enable_fast_dense
from lcsa.metrics import cuda_time_ms, append_jsonl, now

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "Qwen2.5-0.5B")
DATA_DIR = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results", "results.jsonl")
DEV = "cuda"


def main():
    toks = load_tokens(os.path.join(DATA_DIR, "wt2_test.pt"))
    chunks = chunks_from_tokens(toks, 1024, max_chunks=8)

    # --- parity: slow path (as-loaded) vs fast path on the same chunks --------
    model = load_model(MODEL_DIR, device=DEV, fast_dense=False)
    slow = quick_ppl(model, chunks, device=DEV)
    enable_fast_dense()
    fast = quick_ppl(model, chunks, device=DEV)
    d_ce = abs(fast["ce"] - slow["ce"])
    print(f"[parity] slow_ce={slow['ce']:.6f} fast_ce={fast['ce']:.6f} |dce|={d_ce:.2e}", flush=True)
    assert d_ce < 1e-3, "fast dense changes model outputs beyond fp tolerance"
    del model
    torch.cuda.empty_cache()

    # --- speed with fast dense ------------------------------------------------
    model = load_model(MODEL_DIR, device=DEV, fast_dense=True)
    record = {"exp": "exp1b_fast_dense", "ts": now(), "parity": {"slow_ce": slow["ce"],
              "fast_ce": fast["ce"], "abs_diff": d_ce}, "speed_ms": {}}
    for T in [1024, 2048, 4096, 8192]:
        ids = torch.randint(0, 1000, (1, T), device=DEV)

        def fwd():
            with torch.no_grad():
                model(input_ids=ids, logits_to_keep=1)

        try:
            ms = cuda_time_ms(fwd, warmup=3, iters=10)
            record["speed_ms"][f"fast_dense_core_T{T}"] = ms
            print(f"[speed] fast dense core T={T}: {ms['median']} ms", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            record["speed_ms"][f"fast_dense_core_T{T}"] = "OOM"
            print(f"[speed] fast dense core T={T}: OOM", flush=True)
        del ids
        torch.cuda.empty_cache()

    # --- profile at T=4096 to show the new breakdown ---------------------------
    from torch.profiler import profile, ProfilerActivity
    ids = torch.randint(0, 1000, (1, 4096), device=DEV)
    with torch.no_grad():
        for _ in range(2):
            model(input_ids=ids, logits_to_keep=1)
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            model(input_ids=ids, logits_to_keep=1)
    rows = []
    for ev in prof.key_averages():
        if ev.device_type.name == "CUDA" and ev.self_device_time_total > 0:
            rows.append((ev.key[:55], round(ev.self_device_time_total / 1000, 2)))
    rows.sort(key=lambda x: -x[1])
    record["profile_T4096"] = rows[:10]
    print("[profile] T=4096 top ops (ms):", flush=True)
    for k, v in rows[:8]:
        print(f"    {v:8.2f} ms  {k}", flush=True)

    append_jsonl(RESULTS, record)
    with open(os.path.join(ROOT, "results", "exp1b_fast_dense.json"), "w") as f:
        json.dump(record, f, indent=1)
    print(f"[{now()}] exp1b done", flush=True)


if __name__ == "__main__":
    main()
