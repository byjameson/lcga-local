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
                   targets: str = "attn", head: str = "bf16",
                   low_precision_shared: bool = False, with_lora: bool = True):
    """Instantiate the model on the `meta` device with bnb Linear4bit + LoRA.

    No weight memory is allocated here; only the (tiny) LoRA adapters are real.
    Mirrors peft's `prepare_model_for_kbit_training`: every parameter is frozen
    and all non-4bit weights (norms, biases, embedding/head) are cast to fp32 --
    except the embedding/head when `low_precision_shared=True`, which keeps them
    in the checkpoint dtype (needed for very large models).
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
        # `lm_head` stays a plain Linear (bf16) unless we explicitly quantize it;
        # tied models must keep it plain because it *is* the embedding tensor.
        modules_to_not_convert=(
            None if (head == "nf4" and not cfg.tie_word_embeddings) else ["lm_head"]
        ),
        quantization_config=qcfg,
        pre_quantized=True,
    )

    r_attn, r_mlp = (lora_r, 0) if targets == "attn" else (lora_r, lora_r)
    if not with_lora:
        r_attn = r_mlp = 0

    # 1) freeze everything, like prepare_model_for_kbit_training
    for p in model.parameters():
        p.requires_grad = False

    # 2) cast all non-4bit params to fp32 (the RAM path does this, so the forward
    #    matches bit-for-bit). `low_precision_shared` keeps *everything* in the
    #    checkpoint dtype: mixing an fp32 norm with a bf16 head breaks F.linear.
    if not low_precision_shared:
        for name, p in list(model.named_parameters()):
            if p.__class__.__name__ == "Params4bit":
                continue
            if p.dtype not in (torch.float16, torch.bfloat16):
                continue
            mod_name, _, attr = name.rpartition(".")
            target = model.get_submodule(mod_name) if mod_name else model
            target._parameters[attr] = torch.nn.Parameter(
                target._parameters[attr].data.to(torch.float32), requires_grad=False)

    if with_lora:
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

    # Tie the LM head to the input embedding when the config asks for it (the
    # checkpoint then has no separate `lm_head.weight`).
    if getattr(cfg, "tie_word_embeddings", False):
        from .stream_quant_lora import base_model_of

        lm = base_model_of(model)
        lm.lm_head._parameters["weight"] = lm.model.embed_tokens._parameters["weight"]
    return model, cfg


def _rope_theta(cfg) -> float:
    """Theta location moved between transformers versions; look in all of them."""
    rp = getattr(cfg, "rope_parameters", None)
    if isinstance(rp, dict) and "rope_theta" in rp:
        return float(rp["rope_theta"])
    rs = getattr(cfg, "rope_scaling", None)
    if isinstance(rs, dict) and "rope_theta" in rs:
        return float(rs["rope_theta"])
    return float(getattr(cfg, "rope_theta", 1_000_000.0))


def fix_rotary(inner, cfg):
    """`rotary_emb` buffers are computed at init, so they are `meta` here.

    Rebuild the module for real (the constructor knows how to read rope config);
    fall back to filling `inv_freq` by hand.
    """
    rot = inner.rotary_emb
    try:
        fresh = type(rot)(cfg)
        if not fresh.inv_freq.is_meta:
            inner.rotary_emb = fresh
            return
    except Exception:
        pass

    dim = cfg.hidden_size // cfg.num_attention_heads
    theta = _rope_theta(cfg)
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    rot.inv_freq = inv
    rot.original_inv_freq = inv.clone()
    if getattr(rot, "attention_scaling", None) is None:
        rot.attention_scaling = 1.0


def _set_param(module, name: str, tensor, requires_grad: bool = False) -> None:
    """Replace a parameter's storage.

    Works even when the current parameter lives on `meta`: PyTorch forbids
    `meta_param.data = cuda_tensor` ("incompatible tensor type"), so we swap the
    `Parameter` object instead. Base weights are never in an optimizer, so losing
    object identity is fine.
    """
    module._parameters[name] = torch.nn.Parameter(tensor, requires_grad=requires_grad)


def _set_param_cast(module, name: str, tensor) -> None:
    """`_set_param` that keeps the destination dtype (e.g. fp32 norms)."""
    old = module._parameters.get(name)
    if old is not None and old.dtype != tensor.dtype:
        tensor = tensor.to(old.dtype)
    _set_param(module, name, tensor)


def materialize_shared(model, store: DiskQuantStore, cfg, device: str = "cuda"):
    """Load embed, final norm and lm_head once (they are not streamed)."""
    from .stream_quant_lora import base_model_of

    lm = base_model_of(model)
    inner = lm.model
    emb = inner.embed_tokens
    tied = lm.lm_head.weight is emb.weight

    weight = store.get("model.embed_tokens.weight")
    if tied:
        # the same tensor feeds the LM head, so it must sit on the GPU
        _set_param_cast(emb, "weight", weight.to(device))
    else:
        # zero-copy mmap view on CPU: gathering rows only touches the pages for
        # the tokens actually used
        _set_param_cast(emb, "weight", weight)

    if store.has("model.norm.weight"):
        _set_param_cast(inner.norm, "weight", store.get("model.norm.weight").to(device))
    if tied:
        lm.lm_head._parameters["weight"] = emb._parameters["weight"]
    elif store.has("lm_head.weight"):
        _set_param_cast(lm.lm_head, "weight", store.get("lm_head.weight").to(device))

    fix_rotary(inner, cfg)
    inner.rotary_emb.to(device)
    return lm


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
                _set_param_cast(child, "bias", store.get(f"{sub}.bias").to(device))
        else:
            for pname, _p in list(child.named_parameters(recurse=False)):
                key = f"{sub}.{pname}"
                if store.has(key):
                    _set_param_cast(child, pname, store.get(key).to(device))
            for bname, b in list(child.named_buffers(recurse=False)):
                key = f"{sub}.{bname}"
                if store.has(key):
                    t = store.get(key).to(device)
                    if b is not None and b.dtype != t.dtype:
                        t = t.to(b.dtype)
                    child._buffers[bname] = t

    _move_lora(layer, device)


def release_layer(layer, i: int, store: DiskQuantStore) -> None:
    """Drop block `i`'s weights (back to `meta`) so VRAM/RAM are freed."""
    for child in layer.modules():
        for pname, p in list(child.named_parameters(recurse=False)):
            if p.requires_grad:  # LoRA adapters are kept (moved to CPU)
                continue
            child._parameters[pname] = torch.nn.Parameter(
                torch.empty(tuple(p.shape), dtype=p.dtype, device="meta"),
                requires_grad=False,
            )
        for bname, b in list(child.named_buffers(recurse=False)):
            if b is not None:
                child._buffers[bname] = torch.empty(
                    tuple(b.shape), dtype=b.dtype, device="meta")
    _move_lora(layer, "cpu")


def _move_lora(layer, device) -> None:
    """LoRA adapters deliberately stay on CPU.

    Gradients are accumulated in CPU buffers and the optimizer steps on CPU, so
    moving the adapters with the block would put param and grad on different
    devices. `LoRALinear.forward` hops them to the activation's device instead.
    """
    return


class LayerCache:
    """Keep up to `capacity` decoder blocks resident on the GPU (LRU eviction).

    capacity=1 reproduces the plain "one block at a time" streaming. A larger
    capacity is what makes a bigger GPU fast: blocks stay resident between the
    forward and the backward pass, and if capacity >= the number of layers the
    weights are loaded once and never streamed again.
    """

    def __init__(self, store: "DiskQuantStore", layers, device: str = "cuda",
                 capacity: int = 1):
        self.store = store
        self.layers = layers
        self.device = device
        self.capacity = max(1, int(capacity))
        self.order: list[int] = []

    def stage(self, layer, i: int) -> None:
        if i in self.order:  # already resident: refresh its LRU position
            self.order.remove(i)
            self.order.append(i)
            return
        while len(self.order) >= self.capacity:
            j = self.order.pop(0)
            release_layer(self.layers[j], j, self.store)
        load_layer(layer, i, self.store, self.device)
        self.order.append(i)

    def release(self, layer, i: int) -> None:
        # blocks stay resident until the LRU evicts them
        return

    def set_capacity(self, capacity: int) -> None:
        """Change the window at runtime (used by the measured autotuner)."""
        self.capacity = max(1, int(capacity))
        while len(self.order) > self.capacity:
            j = self.order.pop(0)
            release_layer(self.layers[j], j, self.store)

    def stats(self) -> dict:
        return {"resident": len(self.order), "capacity": self.capacity}
