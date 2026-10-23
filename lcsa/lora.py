"""Minimal hand-rolled LoRA for the frozen base model (no peft dependency).

LoRALinear wraps an existing nn.Linear and adds a low-rank update
    y = base(x) + scale * (x @ A^T @ B^T),   scale = alpha / r
B starts at zero so the wrapped module is an identity at init.
A/B are kept in fp32 even when the base is fp16 (fp16 updates would vanish
below resolution); the low-rank path is cheap.

merge() folds the update into the base weight (in-place) so that inference
runs at zero extra cost. Call it once, after training.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scale = alpha / r
        w = base.weight
        self.A = nn.Parameter(torch.randn(r, w.shape[1], device=w.device,
                                          dtype=torch.float32) / math.sqrt(w.shape[1]))
        self.B = nn.Parameter(torch.zeros(w.shape[0], r, device=w.device, dtype=torch.float32))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        y = self.base(x)
        h = self.dropout(x).float()
        # A/B live on CPU (grads are accumulated there and the optimizer steps
        # there); hop them to the activation's device for the matmul. The low-rank
        # path is tiny, so this costs a negligible amount of PCIe traffic.
        A = self.A if self.A.device == h.device else self.A.to(h.device)
        B = self.B if self.B.device == h.device else self.B.to(h.device)
        return y + (self.scale * (h @ A.t() @ B.t())).to(y.dtype)

    @torch.no_grad()
    def merge(self):
        self.base.weight.data += (self.scale * (self.B @ self.A)).to(self.base.weight.dtype)
        self.A.data.zero_()
        self.B.data.zero_()


def apply_lora(model, r_attn: int = 16, r_mlp: int = 16, alpha_attn: int = 16,
               alpha_mlp: int = 16, layers=None):
    """Replace q/k/v/o and mlp projections with LoRALinear in-place."""
    base = model.model if hasattr(model, "model") else model
    n = len(base.layers)
    if layers is None:
        layers = list(range(n))
    for i in layers:
        layer = base.layers[i]
        attn = layer.self_attn.orig if hasattr(layer.self_attn, "orig") else layer.self_attn
        for name, r, a in [("q_proj", r_attn, alpha_attn), ("k_proj", r_attn, alpha_attn),
                           ("v_proj", r_attn, alpha_attn), ("o_proj", r_attn, alpha_attn)]:
            mod = getattr(attn, name)
            if not isinstance(mod, LoRALinear):
                setattr(attn, name, LoRALinear(mod, r=r, alpha=a))
        if r_mlp > 0:
            for name in ("gate_proj", "up_proj", "down_proj"):
                mod = getattr(layer.mlp, name)
                if not isinstance(mod, LoRALinear):
                    setattr(layer.mlp, name, LoRALinear(mod, r=r_mlp, alpha=alpha_mlp))


def lora_parameters(model):
    for mod in model.modules():
        if isinstance(mod, LoRALinear):
            yield mod.A
            yield mod.B


def merge_lora(model):
    for mod in model.modules():
        if isinstance(mod, LoRALinear):
            mod.merge()


def count_lora_params(model):
    return sum(p.numel() for p in lora_parameters(model))


def lora_state_dict(model):
    """Adapter-only state dict (A/B tensors), CPU copies — for saving checkpoints."""
    out = {}
    for name, p in model.named_parameters():
        if p.requires_grad and (name.endswith(".A") or name.endswith(".B")):
            out[name] = p.detach().cpu().clone()
    return out


def set_lora_state_dict(model, sd):
    """Load an adapter saved with lora_state_dict (module tree must match)."""
    own = {n for n, p in model.named_parameters() if p.requires_grad}
    missing = [k for k in sd if k not in own]
    if missing:
        raise KeyError(f"adapter keys not found in model: {missing[:4]}")
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in sd:
                p.copy_(sd[name].to(device=p.device, dtype=p.dtype))
