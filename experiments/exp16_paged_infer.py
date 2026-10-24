"""exp16: paged inference -- run a model larger than VRAM on a 6GB GPU.

Modes
-----
1) Parity (CPU, no GPU needed):
   python experiments/exp16_paged_infer.py --model models/Qwen2.5-0.5B \
       --device cpu --dtype float32 --parity --max_new_tokens 8

2) Paged 3B on 6GB:
   python experiments/exp16_paged_infer.py --model models/Qwen2.5-3B \
       --resident_k auto --pin --max_new_tokens 48

Every run appends a JSON line to results/results.jsonl and text to
results/exp16_paged.log. Console output is ASCII-safe.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lcsa.paged_qwen import PagedQwen, vram_stats  # noqa: E402

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def ascii_safe(s: str) -> str:
    return s.encode("ascii", "replace").decode("ascii")


def parse_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def tokenize(model: str, prompt: str) -> torch.Tensor:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    ids = tok(prompt, return_tensors="pt").input_ids
    return ids, tok


def hf_reference(model: str, ids: torch.Tensor, dtype: torch.dtype, device: str) -> torch.Tensor:
    from transformers import AutoModelForCausalLM
    kw = dict(low_cpu_mem_usage=True)
    try:
        hf = AutoModelForCausalLM.from_pretrained(model, dtype=dtype, **kw)
    except TypeError:
        hf = AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype, **kw)
    hf.eval()
    hf = hf.to(device)
    with torch.no_grad():
        out = hf(ids.to(device)).logits
    del hf
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/Qwen2.5-3B")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--max_seq", type=int, default=0, help="0 = prompt + new + 8")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "float16", "float32"])
    ap.add_argument("--resident_k", default="auto", help="int or 'auto'")
    ap.add_argument("--quant", default="none", choices=["none", "int8", "int4"], help="streamed-layer weight quant")
    ap.add_argument("--state_dict", default="", help="optional torch state_dict checkpoint (HF key names)")
    ap.add_argument("--pin", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--parity", action="store_true", help="compare logits with HuggingFace")
    ap.add_argument("--parity_model", default="", help="default: same as --model (use 0.5B)")
    ap.add_argument("--parity_device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--bench_copy", type=int, default=0, help="time N layer copies (streamed)")
    ap.add_argument("--force", action="store_true", help="run on cuda even if it looks busy")
    ap.add_argument("--tag", default="m1_paged")
    args = ap.parse_args()

    dtype = parse_dtype(args.dtype)
    model = str(ROOT / args.model) if not Path(args.model).is_absolute() else args.model

    ids, tok = tokenize(model, args.prompt)
    prompt_len = int(ids.shape[1])
    max_seq = args.max_seq or (prompt_len + args.max_new_tokens + 8)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("[error] CUDA not available")
            return 2
        free, total = torch.cuda.mem_get_info()
        print(f"[gpu] {torch.cuda.get_device_name(0)} free={free/2**20:.0f}MB/{total/2**20:.0f}MB")
        if free < 5000 * 2**20 and not args.force:
            print("[error] GPU looks busy (<5GB free); wait or use --force")
            return 2
        torch.cuda.reset_peak_memory_stats()

    # ---- parity check (optional) ----
    parity = None
    if args.parity:
        pm = args.parity_model or args.model
        pm = str(ROOT / pm) if not Path(pm).is_absolute() else pm
        print(f"[parity] reference model: {pm} on {args.parity_device}")
        t0 = time.time()
        ref = hf_reference(pm, ids, dtype, args.parity_device)
        print(f"[parity] HF forward {time.time()-t0:.1f}s")
        eng = PagedQwen(pm, device=args.device, dtype=dtype, resident_k=None,
                        max_seq=max(prompt_len, 16), pin=False, verbose=True)
        with torch.no_grad():
            ours = eng.forward(ids, start_pos=0, all_logits=True)
        a = ref.float().cpu()
        b = ours.float().cpu()
        diff = (a - b).abs()
        parity = {
            "model": Path(pm).name,
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "argmax_match": float((a.argmax(-1) == b.argmax(-1)).float().mean()),
            "ref_shape": list(a.shape),
        }
        print(f"[parity] max={parity['max_abs_diff']:.5f} mean={parity['mean_abs_diff']:.6f} "
              f"argmax={parity['argmax_match']:.3f}")
        del ref, ours, a, b, eng
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # ---- paged run ----
    resident_k = None if args.resident_k == "auto" else int(args.resident_k)
    state_dict = None
    if args.state_dict:
        sd_path = Path(args.state_dict)
        if not sd_path.is_absolute():
            sd_path = Path(__file__).resolve().parents[1] / sd_path
        print(f"[state_dict] loading {sd_path} ...")
        state_dict = torch.load(str(sd_path), map_location="cpu", mmap=True)
    t_load = time.time()
    eng = PagedQwen(model, device=args.device, dtype=dtype, resident_k=resident_k,
                    max_seq=max_seq, pin=args.pin, quant=args.quant,
                    state_dict=state_dict, verbose=True)
    load_s = time.time() - t_load

    if args.bench_copy:
        print(f"[copy] timing {args.bench_copy} layer copies ...")
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.bench_copy):
            for i in range(eng.resident_k, eng.n_layers):
                eng._layer_gpu(i)
        torch.cuda.synchronize()
        dt = time.time() - t0
        nbytes = eng.bytes_streamed
        print(f"[copy] {nbytes/2**30:.2f}GB in {dt:.2f}s = {nbytes/dt/2**30:.2f}GB/s")

    t0 = time.time()
    out, stats = eng.generate(
        ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )
    wall = time.time() - t0
    text = tok.decode(out[0], skip_special_tokens=True)

    vs = vram_stats() if args.device == "cuda" else {}
    streamed = eng.n_layers - eng.resident_k
    pass_bytes = eng.streamed_bytes_per_pass()
    passes = 1 + stats["n_new"]
    row = {
        "tag": args.tag,
        "model": Path(model).name,
        "device": args.device,
        "dtype": args.dtype,
        "state_dict": args.state_dict or None,
        "quant": eng.quant,
        "quant_err_max": eng.quant_err_max,
        "layers": eng.n_layers,
        "resident_k": eng.resident_k,
        "streamed_layers": streamed,
        "layer_mb": round(eng.layer_bytes / 2**20, 1),
        "embed_mb": round(eng.embed_bytes / 2**20, 1),
        "prompt_tokens": prompt_len,
        "new_tokens": stats["n_new"],
        "prefill_s": round(stats["prefill_s"], 3),
        "decode_s": round(stats["decode_s"], 3),
        "tok_s": round(stats["tok_s"], 2),
        "wall_s": round(wall, 2),
        "load_s": round(load_s, 2),
        "streamed_mb_per_pass": round(pass_bytes / 2**20, 0),
        "streamed_mb_total": round(eng.bytes_streamed / 2**20, 0),
        "streamed_gb_per_token": round(pass_bytes / 2**30, 3),
        "pin": eng.pinned,
        "temperature": args.temperature,
        "seed": args.seed,
        "peak_alloc_mb": round(vs.get("peak_alloc_mb", 0), 0),
        "peak_reserved_mb": round(vs.get("peak_reserved_mb", 0), 0),
        "free_mb_after": round(vs.get("free_mb", 0), 0),
        "parity": parity,
        "sample": ascii_safe(text[:400]).replace("\n", " "),
    }
    print(json.dumps(row, ensure_ascii=True))
    print("[sample] " + ascii_safe(text[:400]).replace("\n", " "))

    with open(RESULTS / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(RESULTS / "exp16_paged.log", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.write("PROMPT: " + args.prompt + "\nTEXT: " + text + "\n\n")
    if parity is not None:
        with open(RESULTS / "exp16_parity.json", "w", encoding="utf-8") as f:
            json.dump(parity, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
