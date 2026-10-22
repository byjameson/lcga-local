"""EXP1: prepare data, dense baseline PPL on dev/test sets, dense speed, component profile."""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from lcsa.data import build_all, load_tokens, chunks_from_tokens
from lcsa.harness import load_model, quick_ppl
from lcsa.metrics import cuda_time_ms, append_jsonl, now

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "Qwen2.5-0.5B")
DATA_DIR = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results", "results.jsonl")
DEV = "cuda"

os.makedirs(DATA_DIR, exist_ok=True)


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)

    print(f"[{now()}] building datasets...", flush=True)
    paths = build_all(tok, DATA_DIR)
    tokens = {k: load_tokens(v) for k, v in paths.items()}
    for k, v in tokens.items():
        print(f"[data] {k}: {len(v)} tokens")

    model = load_model(MODEL_DIR, device=DEV)
    t_load = time.time()

    # ---- dense PPL on dev + independent test sets ---------------------------
    record = {"exp": "exp1_baseline", "ts": now(), "model": "Qwen2.5-0.5B", "dense": {}}
    sets = {
        "dev_wt2_test_1024": (tokens["wt2_test"], 1024, 128),
        "test_wt103_1024": (tokens["wt103_test"], 1024, 128),
        "test_tinystories_1024": (tokens["tinystories_valid"], 1024, 128),
        "dev_wt2_test_2048": (tokens["wt2_test"], 2048, 64),
    }
    for name, (toks, T, maxc) in sets.items():
        chunks = chunks_from_tokens(toks, T, max_chunks=maxc)
        t0 = time.time()
        res = quick_ppl(model, chunks, device=DEV)
        res["chunks"] = len(chunks)
        res["seq_len"] = T
        res["sec"] = round(time.time() - t0, 1)
        record["dense"][name] = res
        print(f"[baseline] {name}: ppl={res['ppl']:.3f} ce={res['ce']:.4f} ({res['chunks']} chunks, {res['sec']}s)", flush=True)

    # ---- dense speed (logits_to_keep=1 to isolate core forward) --------------
    speed = {}
    for T in [1024, 2048, 4096, 8192]:
        ids = torch.randint(0, 1000, (1, T), device=DEV)

        def fwd():
            with torch.no_grad():
                model(input_ids=ids, logits_to_keep=1)

        try:
            ms = cuda_time_ms(fwd, warmup=3, iters=10)
            speed[f"dense_core_T{T}"] = ms
            print(f"[speed] dense core T={T}: {ms['median']} ms", flush=True)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            speed[f"dense_core_T{T}"] = "OOM"
            print(f"[speed] dense core T={T}: OOM", flush=True)
        del ids
        torch.cuda.empty_cache()
    record["dense_speed"] = speed

    # ---- profiler: where does time go at T=1024 and T=4096? -------------------
    from torch.profiler import profile, ProfilerActivity
    prof_out = {}
    for T in [1024, 4096]:
        ids = torch.randint(0, 1000, (1, T), device=DEV)
        with torch.no_grad():
            for _ in range(3):
                model(input_ids=ids, logits_to_keep=1)
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                model(input_ids=ids, logits_to_keep=1)
        rows = []
        for ev in prof.key_averages():
            if ev.device_type.name == "CUDA" and ev.self_device_time_total > 0:
                rows.append((ev.key[:60], round(ev.self_device_time_total / 1000, 2)))
        rows.sort(key=lambda x: -x[1])
        prof_out[f"T{T}"] = rows[:15]
        print(f"[profile] T={T} top ops (ms):", flush=True)
        for k, v in rows[:12]:
            print(f"    {v:8.2f} ms  {k}")
        del ids
        torch.cuda.empty_cache()
    record["dense_profile_ms"] = prof_out

    append_jsonl(RESULTS, record)
    with open(os.path.join(ROOT, "results", "exp1_baseline.json"), "w") as f:
        json.dump(record, f, indent=1)
    print(f"[{now()}] exp1 done (model load {time.time()-t_load:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
