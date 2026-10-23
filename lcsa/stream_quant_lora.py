"""Streaming QLoRA: train LoRA adapters on a 4-bit base while blocks stream CPU<->GPU.

This is the "our own engine" counterpart to the HF in-memory QLoRA attempt. The
base model is loaded in 4-bit (NF4) on CPU with `device_map="cpu"`; only one
decoder block is on the GPU at a time. That keeps VRAM bounded by one block +
activations + embedding/head, independent of model depth, so 7B can be trained
on a 6 GB card even though the full 4-bit model + compute temporaries do not fit
(HF's whole-model QLoRA peaked at 8.4-8.7 GB and spilled).

The backward pass re-streams blocks in reverse order and accumulates LoRA
gradients on CPU; the base 4-bit weights are never updated. This is the same
recompute strategy as `lcsa/stream_train.py` (P1), but with a PEFT LoRA on a
bitsandbytes 4-bit base instead of the minimal local LoRA on a bf16 base.
"""
from __future__ import annotations

import time

import torch

from .stream_train import _layer_out, chunked_lm_loss_and_backward


def chunked_lm_loss_and_backward_masked(head, h, ys, chunk=128, ignore_index=-100):
    """Chunked CE with an ignore_index (for SFT prompt masking)."""
    B, T, _ = h.shape
    total = torch.zeros((), device=h.device, dtype=torch.float32)
    valid_total = 0
    for a in range(0, T, chunk):
        b = min(a + chunk, T)
        logits = head(h[:, a:b])
        tgt = ys[:, a:b].reshape(-1).to(h.device)
        valid = int((tgt != ignore_index).sum())
        if valid == 0:
            continue
        loss_sum = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), tgt,
            reduction="sum", ignore_index=ignore_index,
        )
        (loss_sum / valid).backward()
        with torch.no_grad():
            total = total + loss_sum.detach()
        valid_total += valid
    denom = max(valid_total, 1)
    return total / denom, total / denom


def base_model_of(model):
    """Return the underlying HF causal-LM (the module owning `.model.layers` + `.lm_head`).

    Works for plain HF models and for peft/wrapper models alike: wrappers delegate
    attributes, so we take the *deepest* module that owns both the decoder stack
    and the LM head.
    """
    found = None
    stack = [model]
    seen = set()
    while stack:
        m = stack.pop()
        if id(m) in seen:
            continue
        seen.add(id(m))
        if hasattr(m, "lm_head") and hasattr(getattr(m, "model", None), "layers"):
            found = m
        stack.extend(m.children())
    return found if found is not None else model


class StreamingQuantLoRATrainer:
    """LoRA fine-tuning on a 4-bit base with block streaming."""

    def __init__(self, model, device: str = "cuda", loss_chunk: int = 128,
                 lr: float = 1e-4, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0, clip: float = 0.0,
                 verbose: bool = False, stage=None, release=None):
        self.device = device
        self.loss_chunk = loss_chunk
        self.clip = clip
        self.verbose = verbose
        self.model = model
        # How a decoder block is brought to the GPU and taken away again. The
        # defaults assume the whole 4-bit base sits in host RAM; a disk-backed
        # store can pass its own load/release callables instead.
        self.stage = stage if stage is not None else (lambda layer, i: layer.to(device))
        self.release = release if release is not None else (lambda layer, i: layer.to("cpu"))
        base = base_model_of(model)
        self.base = base
        self.layers = list(base.model.layers)
        self.embed = base.model.embed_tokens
        self.rotary = base.model.rotary_emb
        self.norm = base.model.norm
        self.lm_head = base.lm_head

        # Small/shared modules stay resident on the GPU. The input embedding is
        # kept on CPU (4-bit/bf16) and only the gathered activation is moved to
        # the GPU; for 14B the embedding alone is ~1.5 GB bf16. Exception: when
        # the LM head is *tied* to the embedding (e.g. Qwen2.5-3B/0.5B) the same
        # tensor feeds the head, so it has to live where the head runs.
        self.rotary.to(device)
        self.norm.to(device)
        self.tied = self.lm_head.weight is self.embed.weight
        if self.tied:
            self.embed.to(device)
        else:
            self.lm_head.to(device)

        self.trainables = [p for p in model.parameters() if p.requires_grad]
        self.n_lora = sum(p.numel() for p in self.trainables)
        self.opt = torch.optim.AdamW(self.trainables, lr=lr, betas=betas, eps=eps,
                                     weight_decay=weight_decay)
        self._acc = {id(p): torch.zeros_like(p, device="cpu", dtype=torch.float32)
                     for p in self.trainables}

    # ------------------------------------------------------------------ forward
    @torch.no_grad()
    def _forward_cached(self, ids):
        dev = self.device
        # Embedding usually lives on CPU (only the gathered activation moves to
        # the GPU); for tied models it lives on the GPU together with the head.
        emb_dev = self.embed.weight.device
        h = self.embed(ids.to(emb_dev))
        if h.device != torch.device(dev):
            h = h.to(dev)
        pos_ids = torch.arange(ids.shape[1], device=dev).unsqueeze(0)
        pos = self.rotary(h, pos_ids)
        cache = []
        for i, layer in enumerate(self.layers):
            self.stage(layer, i)
            cache.append(h)
            h = _layer_out(layer(h, position_embeddings=pos))
            self.release(layer, i)
            if self.verbose and (i + 1) % 8 == 0:
                print(f"    [fwd] {i + 1}/{len(self.layers)} blocks "
                      f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)
        return h, cache, pos

    # ----------------------------------------------------------------- backward
    def _backward_blocks(self, cache, pos, grad):
        dev = self.device
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            self.stage(layer, i)
            h_in = cache[i]
            h_in.requires_grad_(True)
            out = _layer_out(layer(h_in, position_embeddings=pos))
            out.backward(grad)
            grad = h_in.grad
            h_in.grad = None
            h_in.requires_grad_(False)
            for p in layer.parameters():
                if p.requires_grad and p.grad is not None:
                    self._acc[id(p)].add_(p.grad.detach().float().cpu())
                    p.grad = None
            self.release(layer, i)
            if self.verbose and (len(self.layers) - i) % 8 == 0:
                print(f"    [bwd] {len(self.layers) - i}/{len(self.layers)} blocks "
                      f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)

    # --------------------------------------------------------------------- step
    def step(self, xs, ys, ignore_index=None):
        torch.cuda.synchronize()
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            h_blocks, cache, pos = self._forward_cached(xs)
        h_b = h_blocks.detach().requires_grad_(True)
        z_g = self.norm(h_b)
        z = z_g.detach().requires_grad_(True)
        if ignore_index is None:
            _, loss = chunked_lm_loss_and_backward(self.lm_head, z, ys, self.loss_chunk)
        else:
            _, loss = chunked_lm_loss_and_backward_masked(
                self.lm_head, z, ys, self.loss_chunk, ignore_index=ignore_index
            )
        z_g.backward(z.grad)
        grad = h_b.grad
        self._backward_blocks(cache, pos, grad)
        for p in self.trainables:
            p.grad = self._acc[id(p)].to(p.dtype)
        if self.clip > 0:
            torch.nn.utils.clip_grad_norm_(self.trainables, self.clip)
        self.opt.step()
        for p in self.trainables:
            p.grad = None
            self._acc[id(p)].zero_()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_reserved() / 1e9
        return float(loss), time.time() - t0, peak
