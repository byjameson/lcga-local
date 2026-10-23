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


def base_model_of(model):
    """Return the underlying HF causal-LM from a (possibly PEFT-wrapped) model."""
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    m = model
    for _ in range(4):
        if hasattr(m, "base_model") and getattr(m, "base_model") is not m:
            m = m.base_model
        else:
            break
    if hasattr(m, "model") and hasattr(m.model, "model") and hasattr(m.model.model, "layers"):
        return m.model
    return m


class StreamingQuantLoRATrainer:
    """LoRA fine-tuning on a 4-bit base with block streaming."""

    def __init__(self, model, device: str = "cuda", loss_chunk: int = 128,
                 lr: float = 1e-4, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0, clip: float = 0.0,
                 verbose: bool = False):
        self.device = device
        self.loss_chunk = loss_chunk
        self.clip = clip
        self.verbose = verbose
        self.model = model
        base = base_model_of(model)
        self.base = base
        self.layers = list(base.model.layers)
        self.embed = base.model.embed_tokens
        self.rotary = base.model.rotary_emb
        self.norm = base.model.norm
        self.lm_head = base.lm_head

        # Small/shared modules stay resident on the GPU. Base layers stay on CPU.
        self.embed.to(device)
        self.rotary.to(device)
        self.norm.to(device)
        self.tied = self.lm_head.weight is self.embed.weight
        if not self.tied:
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
        h = self.embed(ids.to(dev))
        pos_ids = torch.arange(ids.shape[1], device=dev).unsqueeze(0)
        pos = self.rotary(h, pos_ids)
        cache = []
        for i, layer in enumerate(self.layers):
            layer.to(dev)
            cache.append(h)
            h = _layer_out(layer(h, position_embeddings=pos))
            layer.to("cpu")
            if self.verbose and (i + 1) % 8 == 0:
                print(f"    [fwd] {i + 1}/{len(self.layers)} blocks "
                      f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)
        return h, cache, pos

    # ----------------------------------------------------------------- backward
    def _backward_blocks(self, cache, pos, grad):
        dev = self.device
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            layer.to(dev)
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
            layer.to("cpu")
            if self.verbose and (len(self.layers) - i) % 8 == 0:
                print(f"    [bwd] {len(self.layers) - i}/{len(self.layers)} blocks "
                      f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB", flush=True)

    # --------------------------------------------------------------------- step
    def step(self, xs, ys):
        torch.cuda.synchronize()
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            h_blocks, cache, pos = self._forward_cached(xs)
        h_b = h_blocks.detach().requires_grad_(True)
        z_g = self.norm(h_b)
        z = z_g.detach().requires_grad_(True)
        _, loss = chunked_lm_loss_and_backward(self.lm_head, z, ys, self.loss_chunk)
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
