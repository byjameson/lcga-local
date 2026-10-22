"""P1: block-streaming ("parça parça") LoRA trainer for low-VRAM GPUs.

Base weights live in CPU RAM. Each transformer block is moved to the GPU just
for its forward pass, moved back, and re-streamed in reverse order during the
backward pass (per-block activation recompute). Peak VRAM is bounded by
(one block + its activations + embedding/head + loss chunk), independent of
model depth — so a 6GB GPU can fine-tune models whose full weights alone do
not fit (e.g. 3B fp16 = 6.2GB).

Pipeline per step:
  forward   : embed -> [block_i on GPU, run, off] x L -> final norm
  loss      : chunked cross-entropy through lm_head (bounded logits memory)
  backward  : dL/dh_final -> [block_i back on GPU, recompute forward with grad,
              backward, accumulate LoRA grads on CPU, off] x L (reverse)
  step      : AdamW on the (small, CPU-resident) LoRA params

Correctness: the recompute backward is mathematically identical to a normal
autograd backward over the same block list; experiments/exp8_stream.py checks
the loss trajectory against a fully-on-GPU naive trainer on identical batches.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .lora import apply_lora, count_lora_params, lora_parameters


def _layer_out(out):
    return out[0] if isinstance(out, tuple) else out


def chunked_lm_loss_and_backward(head, h, ys, chunk=128):
    """Cross-entropy in position chunks; accumulates dL/dh into h.grad.

    h: (B, T, d) leaf requiring grad (or the output of a small graph).
    Targets are pre-shifted (TrainSampler convention): ys[p] is the target of
    position p, all T positions valid.

    FIX (19.10.2026): the previous version used F.cross_entropy(mean) per chunk
    and called backward(), which scaled the accumulated gradient by the number
    of chunks (T/chunk). Now each chunk contributes a cross-entropy SUM divided
    by n_pos, so the accumulated gradient is exactly d(mean CE)/dh.
    """
    B, T, _ = h.shape
    n_pos = T
    total = torch.zeros((), device=h.device, dtype=torch.float32)
    last = None
    for a in range(0, n_pos, chunk):
        b = min(a + chunk, n_pos)
        logits = head(h[:, a:b])
        tgt = ys[:, a:b].reshape(-1).to(h.device)
        loss_sum = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                                   tgt, reduction="sum")
        (loss_sum / n_pos).backward()
        with torch.no_grad():
            total = total + loss_sum.detach()
        last = loss_sum
    return last.detach() / n_pos, total / n_pos


@torch.no_grad()
def stream_eval_ce(trainer, ids, loss_chunk=128):
    """Mean next-token CE over a (1, T) token chunk, via the streaming forward.

    Positions 0..T-2 predict ids[1..T-1] (standard intra-window LM loss).
    Reuses the trainer's block streaming; no backward, no cache retained.
    """
    h, _cache, _pos = trainer._forward_cached(ids)
    z = trainer.norm(h)
    total = 0.0
    n = 0
    T = ids.shape[1]
    for a in range(0, T - 1, loss_chunk):
        b = min(a + loss_chunk, T - 1)
        logits = trainer.lm_head(z[:, a:b]).float()
        tgt = ids[:, a + 1: b + 1].reshape(-1).to(z.device)
        total += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt,
                                       reduction="sum"))
        n += tgt.numel()
    return total / max(n, 1)


class StreamingLoRATrainer:
    """LoRA fine-tuning while base blocks stream CPU <-> GPU."""

    def __init__(self, model, device: str = "cuda", loss_chunk: int = 128,
                 lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0, r_attn: int = 16, r_mlp: int = 0,
                 clip: float = 0.0, verbose: bool = False):
        self.device = device
        self.loss_chunk = loss_chunk
        self.clip = clip
        self.verbose = verbose
        apply_lora(model, r_attn=r_attn, r_mlp=r_mlp)
        self.model = model
        base = model.model
        self.layers = list(base.layers)
        self.embed = base.embed_tokens
        self.rotary = base.rotary_emb
        self.norm = base.norm
        self.lm_head = model.lm_head

        model.to("cpu")
        for p in model.parameters():
            p.requires_grad_(False)
        # shared small modules stay resident on GPU
        self.embed.to(device)
        self.rotary.to(device)
        self.norm.to(device)
        if self.lm_head.weight is not self.embed.weight:
            self.lm_head.to(device)
        for p in self.lm_head.parameters():
            p.requires_grad_(False)

        self.trainables = list(lora_parameters(model))
        for p in self.trainables:
            p.requires_grad_(True)
        self.n_lora = count_lora_params(model)
        self.opt = torch.optim.AdamW(self.trainables, lr=lr, betas=betas, eps=eps,
                                     weight_decay=weight_decay)
        self._acc = {id(p): torch.zeros_like(p, device="cpu") for p in self.trainables}

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
        z_g = self.norm(h_b)                       # graph h_b -> z_g (elementwise, cheap)
        z = z_g.detach().requires_grad_(True)      # leaf: chunked head backward accumulates z.grad
        _, loss = chunked_lm_loss_and_backward(self.lm_head, z, ys, self.loss_chunk)
        z_g.backward(z.grad)                       # norm graph traversed exactly once
        grad = h_b.grad
        self._backward_blocks(cache, pos, grad)
        for p in self.trainables:
            p.grad = self._acc[id(p)]
        if self.clip > 0:
            torch.nn.utils.clip_grad_norm_(self.trainables, self.clip)
        self.opt.step()
        for p in self.trainables:
            p.grad = None
            self._acc[id(p)].zero_()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_reserved() / 1e9
        return float(loss), time.time() - t0, peak


class NaiveLoRATrainer:
    """Reference: everything on GPU, standard full-graph autograd + full logits."""

    def __init__(self, model, device: str = "cuda", lr: float = 1e-3,
                 betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0,
                 r_attn: int = 16, r_mlp: int = 0, clip: float = 0.0,
                 verbose: bool = False):
        self.device = device
        self.clip = clip
        self.verbose = verbose
        apply_lora(model, r_attn=r_attn, r_mlp=r_mlp)
        self.model = model.to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        self.trainables = list(lora_parameters(model))
        for p in self.trainables:
            p.requires_grad_(True)
        self.n_lora = count_lora_params(model)
        self.opt = torch.optim.AdamW(self.trainables, lr=lr, betas=betas, eps=eps,
                                     weight_decay=weight_decay)

    def step(self, xs, ys):
        torch.cuda.synchronize()
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        dev = self.device
        ys = ys.to(dev)
        h = self.model.model(input_ids=xs.to(dev), use_cache=False).last_hidden_state
        logits = self.model.lm_head(h).float()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ys.reshape(-1))
        loss.backward()
        if self.clip > 0:
            torch.nn.utils.clip_grad_norm_(self.trainables, self.clip)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_reserved() / 1e9
        return float(loss.detach()), time.time() - t0, peak
