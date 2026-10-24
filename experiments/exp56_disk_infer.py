"""exp56: disk-streamed **inference** for models that do not fit in VRAM.

Same mmap block loader as `exp54`, but with no training machinery: a KV cache and
one generated token at a time. Each decoder block is materialized on the GPU just
for that step's attention+MLP, then dropped.

Why this is inherently slow: training amortises one weight pass over `seq` tokens
(plus the backward), while generation needs a full pass over *all* weights **per
token**. Measured numbers below; treat them as the honest cost of the tradeoff.

Note on the block window: the LRU window is useless here unless it can hold *all*
layers, because the access pattern is cyclic (0..L-1, 0..L-1, ...) and LRU evicts
exactly the blocks that are needed next. So `--blocks-on-gpu` defaults to 1.

Usage:
  python experiments/exp56_disk_infer.py --model-dir models/Qwen2.5-32B-Instruct-bnb-4bit \
      --prompt "Explain in one sentence why the sky is blue." --max-new 16 --verify
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

from lcsa.disk_quant_store import (  # noqa: E402
    DiskQuantStore,
    LayerCache,
    build_skeleton,
    materialize_shared,
)
from lcsa.metrics import append_jsonl, now  # noqa: E402
from lcsa.stream_train import _layer_out  # noqa: E402

try:
    import psutil
    _proc = psutil.Process()

    def host_rss_gb():
        return round(_proc.memory_info().rss / 1e9, 2)
except Exception:
    def host_rss_gb():
        return None


@torch.no_grad()
def stream_generate(model, store, cache, tokenizer, prompt: str, max_new: int,
                    greedy: bool = True, temperature: float = 0.7,
                    device: str = "cuda"):
    """Token-by-token generation through streamed blocks."""
    base = model.model
    layers = list(base.layers)
    embed, rotary, norm, head = (base.embed_tokens, base.rotary_emb,
                                 base.norm, model.lm_head)

    try:
        from transformers import DynamicCache
    except Exception:  # older layouts
        from transformers.cache_utils import DynamicCache
    past = DynamicCache()

    ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
    prompt_len = len(ids)
    out_tokens: list[int] = []
    times: list[float] = []
    peaks: list[float] = []
    nxt = None

    for step in range(prompt_len + max_new):
        tok = ids[step] if step < prompt_len else nxt
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()

        h = embed(torch.tensor([[tok]], device=embed.weight.device))
        if h.device.type != "cuda":
            h = h.to(device)
        pos = torch.tensor([[step]], device=device)
        pos_emb = rotary(h, pos)

        for i, layer in enumerate(layers):
            cache.stage(layer, i)
            out = layer(h, attention_mask=None, position_ids=pos,
                        past_key_values=past, use_cache=True,
                        position_embeddings=pos_emb)
            h = _layer_out(out)
            cache.release(layer, i)

        if step < prompt_len - 1:
            continue  # still prefilling; nothing to sample yet

        logits = head(norm(h)[:, -1]).float()[0]
        if greedy:
            nxt = int(logits.argmax())
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            nxt = int(torch.multinomial(probs, 1))

        out_tokens.append(nxt)
        times.append(time.time() - t0)
        peaks.append(torch.cuda.max_memory_reserved() / 1e9)
        if nxt == tokenizer.eos_token_id:
            break

    return out_tokens, times, peaks


@torch.no_grad()
def verify_cache(model, store, cache, tokenizer, prompt: str, device="cuda"):
    """Check the KV-cache path against a single full-sequence forward.

    Both use the same streamed blocks; only the caching differs. If the argmax of
    the next token differs, the cache plumbing is wrong.
    """
    base = model.model
    layers = list(base.layers)
    embed, rotary, norm, head = (base.embed_tokens, base.rotary_emb,
                                 base.norm, model.lm_head)
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    seq = ids.shape[1]

    h = embed(ids.to(embed.weight.device))
    if h.device.type != "cuda":
        h = h.to(device)
    pos = torch.arange(seq, device=device).unsqueeze(0)
    pos_emb = rotary(h, pos)
    for i, layer in enumerate(layers):
        cache.stage(layer, i)
        h = _layer_out(layer(h, attention_mask=None, position_ids=pos,
                             past_key_values=None, use_cache=False,
                             position_embeddings=pos_emb))
        cache.release(layer, i)

    full_next = int(head(norm(h)[:, -1]).float().argmax())
    return full_next


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prompt", default="Explain in one sentence why the sky is blue.")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--blocks-on-gpu", type=int, default=1)
    ap.add_argument("--shared-dtype", choices=["fp32", "as-stored"], default="as-stored")
    ap.add_argument("--greedy", action="store_true", default=True)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--tag", default="disk_infer")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    model_dir = str(ROOT / args.model_dir)
    t0 = time.time()
    model, cfg = build_skeleton(model_dir, with_lora=False,
                                low_precision_shared=(args.shared_dtype == "as-stored"))
    store = DiskQuantStore(model_dir, device="cuda")
    base = materialize_shared(model, store, cfg, device="cuda")
    layers = list(base.model.layers)
    cache = LayerCache(store, layers, device="cuda", capacity=args.blocks_on_gpu)
    tok = AutoTokenizer.from_pretrained(model_dir)
    print(f"[load] {time.time()-t0:.1f}s layers={len(layers)} "
          f"keys={len(store.weight_map)} rss={host_rss_gb()}GB", flush=True)

    if args.verify:
        t1 = time.time()
        ref = verify_cache(model, store, cache, tok, args.prompt)
        print(f"[verify] full-sequence next token={ref} ({time.time()-t1:.1f}s)",
              flush=True)

    t1 = time.time()
    out, times, peaks = stream_generate(model, store, cache, tok, args.prompt,
                                        args.max_new, greedy=args.greedy,
                                        temperature=args.temperature)
    wall = time.time() - t1
    text = tok.decode(out, skip_special_tokens=True)
    med = sorted(times)[len(times) // 2] if times else None
    print(f"[gen] {len(out)} tokens in {wall:.1f}s  "
          f"median={med:.2f}s/token  peak={max(peaks) if peaks else 0:.2f}GB", flush=True)
    print(f"[text] {text}", flush=True)

    if args.verify:
        first = int(out[0]) if out else None
        print(f"[verify] cached first token={first}  full={ref}  "
              f"{'MATCH' if first == ref else 'MISMATCH'}", flush=True)

    row = {
        "exp": "exp56_disk_infer",
        "tag": args.tag,
        "model": Path(model_dir).name,
        "prompt": args.prompt,
        "max_new": args.max_new,
        "n_new": len(out),
        "text": text,
        "sec_per_token_med": round(med, 3) if med else None,
        "tok_per_s": round(1.0 / med, 4) if med else None,
        "wall_s": round(wall, 1),
        "peak_gb_max": round(max(peaks), 3) if peaks else None,
        "host_rss_gb": host_rss_gb(),
        "blocks_on_gpu": cache.capacity,
        "shared_dtype": args.shared_dtype,
        "verify_match": (int(out[0]) == ref) if (args.verify and out) else None,
        "ts": now(),
    }
    print(json.dumps(row, ensure_ascii=True), flush=True)
    append_jsonl(str(ROOT / "results" / "results.jsonl"), row)
    with open(ROOT / "results" / "exp56_disk_infer.log", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
