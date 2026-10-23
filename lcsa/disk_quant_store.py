"""Disk-backed (mmap) NF4 weight store: stream weights from SSD into VRAM.

Idea
----
The RAM-resident streaming engine (`stream_quant_lora`) keeps the whole 4-bit
base in host RAM. That is what caps the model size at what fits in RAM
(a 14B NF4 base is ~8 GB; a 32B one would need ~18 GB). This module removes
that cap: the pre-quantized weights stay in the bnb-4bit safetensors shards on
disk and are memory-mapped. `safetensors.safe_open(...).get_tensor()` returns a
*zero-copy* CPU view into the mmap, so

  * host RAM is used only as an OS-managed page cache (evictable, LRU), i.e.
    "use as much RAM as is free, keep the rest on SSD" happens for free, and
  * the model-size limit becomes disk space + step time instead of host RAM.

A decoder block is materialized directly on the GPU (uint8 data copied H2D,
`Params4bit.from_prequantized` re-attaches the saved NF4 quant state) and moved
back to `meta` afterwards, so only one block lives in VRAM at a time.

Nothing here mutates the checkpoint: the base stays frozen 4-bit; only the
LoRA adapters (fp32, CPU) are trained.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

# Components of a bnb NF4 quant state as written by transformers >= 4.4x:
# one packed JSON blob (non-tensor items) + four tensors.
QUANT_SUFFIXES = (
    "weight.quant_state.bitsandbytes__nf4",
    "weight.absmax",
    "weight.quant_map",
    "weight.nested_absmax",
    "weight.nested_quant_map",
)


class DiskQuantStore:
    """Key -> mmap'd tensor lookups over a (possibly sharded) bnb-4bit checkpoint."""

    def __init__(self, model_dir, device: str = "cuda"):
        self.dir = Path(model_dir)
        index = json.loads((self.dir / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = dict(index["weight_map"])
        self.device = device
        self._handles: dict[str, object] = {}

    def _handle(self, filename: str):
        h = self._handles.get(filename)
        if h is None:
            from safetensors import safe_open

            h = safe_open(str(self.dir / filename), framework="pt", device="cpu")
            self._handles[filename] = h
        return h

    def has(self, key: str) -> bool:
        return key in self.weight_map

    def get(self, key: str) -> torch.Tensor:
        """Zero-copy CPU view backed by the memory-mapped shard."""
        return self._handle(self.weight_map[key]).get_tensor(key)


def build_skeleton(model_dir, lora_r: int = 8, lora_alpha: int = 16,
                   targets: str = "attn", head: str = "bf16"):
    """Instantiate the model on the `meta` device with bnb Linear4bit + LoRA.

    No weight memory is allocated here; only the (tiny) LoRA adapters are real.
    Returns (model, config).
    """
    from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
    from transformers.integrations.bitsandbytes import replace_with_bnb_linear

    from .lora import LoRALinear, apply_lora

    cfg = AutoConfig.from_pretrained(str(model_dir))
    cfg.use_cache = False

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)

    qcfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.uint8,
    )
    replace_with_bnb_linear(
        model,
        modules_to_not_convert=None if head == "nf4" else ["lm_head"],
        quantization_config=qcfg,
        pre_quantized=True,
    )

    r_attn, r_mlp = (lora_r, 0) if targets == "attn" else (lora_r, lora_r)
    apply_lora(model, r_attn=r_attn, r_mlp=r_mlp, alpha_attn=lora_alpha,
               alpha_mlp=lora_alpha)

    # LoRA adapters were created on `meta` (that is where the base lives); give
    # them real CPU storage so the optimizer can train them.
    import math

    for mod in model.modules():
        if isinstance(mod, LoRALinear):
            out_f, in_f = mod.base.weight.shape
            mod.A = torch.nn.Parameter(
                torch.randn(mod.r, in_f, dtype=torch.float32) / math.sqrt(in_f))
            mod.B = torch.nn.Parameter(torch.zeros(out_f, mod.r, dtype=torch.float32))
    return model, cfg


def fix_rotary(base, cfg):
    """`rotary_emb.inv_freq` is computed at init, so it is `meta` here; rebuild it."""
    rot = base.model.rotary_emb
    dim = cfg.hidden_size // cfg.num_attention_heads
    inv = 1.0 / (float(cfg.rope_theta) ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    rot.inv_freq = inv
    rot.original_inv_freq = inv.clone()
    if getattr(rot, "attention_scaling", None) is None:
        rot.attention_scaling = 1.0


def materialize_shared(model, store: DiskQuantStore, cfg, device: str = "cuda"):
    """Load embed (CPU, mmap), final norm and lm_head (GPU) once."""
    from .stream_quant_lora import base_model_of

    base = base_model_of(model)
    emb = base.model.embed_tokens
    # Keep the (bf16) embedding as a zero-copy mmap view on CPU: gathering rows
    # only touches the pages for the tokens actually used.
    emb.weight.data = store.get("model.embed_tokens.weight")
    if store.has("model.norm.weight"):
        base.model.norm.weight.data = store.get("model.norm.weight").to(device)
    if store.has("lm_head.weight"):
        # bf16 head stays resident on the GPU (compute needs it every step).
        base.lm_head.weight.data = store.get("lm_head.weight").to(device)
    fix_rotary(base, cfg)
    base.model.rotary_emb.to(device)
    return base


def _layer_prefix(i: int) -> str:
    return f"model.layers.{i}"


def _key_of(prefix: str, module_name: str) -> str:
    """Map a module path inside a layer to its checkpoint prefix."""
    name = module_name
    if name.endswith(".base"):
        name = name[: -len(".base")]
    return f"{prefix}.{name}" if name else prefix


def load_layer(layer, i: int, store: DiskQuantStore, device: str = "cuda") -> None:
    """Materialize decoder block `i` on `device` straight from the mmap'd store."""
    import bitsandbytes as bnb

    prefix = _layer_prefix(i)
    for name, child in layer.named_modules():
        sub = _key_of(prefix, name)
        if isinstance(child, bnb.nn.Linear4bit):
            wkey = f"{sub}.weight"
            if store.has(wkey):
                qs = {s: store.get(f"{sub}.{s}") for s in QUANT_SUFFIXES
                      if store.has(f"{sub}.{s}")}
                child.weight = bnb.nn.Params4bit.from_prequantized(
                    data=store.get(wkey), quantized_stats=qs,
                    requires_grad=False, device=device, module=child,
                )
                child._is_hf_initialized = True
            if child.bias is not None and store.has(f"{sub}.bias"):
                child.bias.data = store.get(f"{sub}.bias").to(device)
        else:
            for pname, p in child.named_parameters(recurse=False):
                key = f"{sub}.{pname}"
                if store.has(key):
                    p.data = store.get(key).to(device)
            for bname, b in child.named_buffers(recurse=False):
                key = f"{sub}.{bname}"
                if store.has(key):
                    b.data = store.get(key).to(device)

    _move_lora(layer, device)


def release_layer(layer, i: int, store: DiskQuantStore) -> None:
    """Drop block `i`'s weights (back to `meta`) so VRAM/RAM are freed."""
    for p in layer.parameters():
        if p.requires_grad:  # LoRA adapters are kept (moved to CPU)
            continue
        p.data = torch.empty(tuple(p.shape), dtype=p.dtype, device="meta")
    _move_lora(layer, "cpu")


def _move_lora(layer, device) -> None:
    from .lora import LoRALinear

    for mod in layer.modules():
        if isinstance(mod, LoRALinear):
            mod.A.data = mod.A.data.to(device)
            mod.B.data = mod.B.data.to(device)
