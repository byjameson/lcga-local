"""Shared experiment harness: model loading, logits functions, quick PPL."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def enable_fast_dense():
    """Force transformers' sdpa path to expand GQA heads instead of passing
    enable_gqa=True. On this device/torch build the enable_gqa path has no fused
    (mem-efficient/flash) kernel and silently falls back to the math backend,
    which is ~40x slower per layer at T=4096 and OOMs at T=8192."""
    try:
        from transformers.integrations import sdpa_attention as sa

        sa.use_gqa_in_sdpa = lambda attention_mask, key, value: False
        return True
    except Exception:
        return False


def load_model(model_dir, dtype=torch.float16, attn="sdpa", device="cuda", fast_dense=True):
    from transformers import AutoModelForCausalLM
    if fast_dense:
        enable_fast_dense()
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype, attn_implementation=attn)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype, attn_implementation=attn)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def logits_fn(model, device="cuda"):
    def fn(ids):
        return model(input_ids=ids.to(device)).logits
    return fn


@torch.no_grad()
def quick_ppl(model, chunks, device="cuda", max_chunks=None, dtype=torch.float32,
              pos_chunk: int = 1024):
    """Mean NLL per token and PPL over chunks (1,T).

    The CE is accumulated in position chunks: a full T=8192 logits tensor is
    T*vocab*2 bytes (2.5 GB for Qwen's 152k vocab), which OOMs a 6 GB card on its
    own. Chunking keeps the result identical while bounding peak memory.
    """
    n_chunks = len(chunks) if max_chunks is None else min(len(chunks), max_chunks)
    total, n = 0.0, 0
    for i in range(n_chunks):
        ids = chunks[i].to(device).unsqueeze(0)
        logits = model(input_ids=ids).logits
        T = ids.shape[1]
        for a in range(0, T - 1, pos_chunk):
            b = min(a + pos_chunk, T - 1)
            ce = F.cross_entropy(logits[0, a:b].float(), ids[0, a + 1:b + 1],
                                 reduction="sum").item()
            total += ce
            n += b - a
        del logits
    mean_ce = total / max(n, 1)
    return {"ce": mean_ce, "ppl": float(torch.exp(torch.tensor(mean_ce))), "ntok": n}
