"""Paged Qwen2 inference: run a model larger than VRAM by streaming decoder layers.

Design
------
- All weights are loaded into CPU RAM (safetensors, bf16).
- `embed_tokens` / final norm live on the GPU; the KV cache is preallocated on
  the GPU.
- A prefix of `resident_k` decoder layers stays on the GPU permanently. The
  remaining layers are copied one at a time into a single reusable GPU slot
  (same shapes, `copy_`), used for the forward, then overwritten by the next
  layer. This is the inference counterpart of `lcsa/stream_train.py` (P1).
- Weights of streamed layers can be pinned in host memory for faster copies.

Correctness: the math follows Qwen2 exactly (RMSNorm in fp32, GQA, RoPE theta,
SwiGLU, tied embeddings). `experiments/exp16_paged_infer.py --parity` checks
logits against the HuggingFace implementation.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


# ----------------------------------------------------------------------------- config / weights

def load_config(model_path: str | Path) -> dict:
    with open(Path(model_path) / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)


_SHORT = {
    "input_layernorm.weight": "ln1",
    "post_attention_layernorm.weight": "ln2",
    "self_attn.q_proj.weight": "q_w",
    "self_attn.q_proj.bias": "q_b",
    "self_attn.k_proj.weight": "k_w",
    "self_attn.k_proj.bias": "k_b",
    "self_attn.v_proj.weight": "v_w",
    "self_attn.v_proj.bias": "v_b",
    "self_attn.o_proj.weight": "o_w",
    "self_attn.q_norm.weight": "q_norm",   # Qwen3: per-head RMSNorm
    "self_attn.k_norm.weight": "k_norm",
    "mlp.gate_proj.weight": "gate_w",
    "mlp.up_proj.weight": "up_w",
    "mlp.down_proj.weight": "down_w",
}

# Attention biases (Qwen2) and q/k norms (Qwen3) are optional; a layer needs at
# least the projections and the two block norms.
REQUIRED_LAYER_KEYS = ("ln1", "ln2", "q_w", "k_w", "v_w", "o_w",
                       "gate_w", "up_w", "down_w")
OPTIONAL_LAYER_KEYS = ("q_b", "k_b", "v_b", "q_norm", "k_norm")
LAYER_KEYS = list(REQUIRED_LAYER_KEYS)


def _shard_files(model_path: Path) -> list[Path]:
    idx = model_path / "model.safetensors.index.json"
    if idx.exists():
        with open(idx, "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        names = sorted(set(weight_map.values()))
        return [model_path / n for n in names]
    single = model_path / "model.safetensors"
    if single.exists():
        return [single]
    raise FileNotFoundError(f"no safetensors in {model_path}")


def load_weights(model_path: str | Path, verbose: bool = True, state_dict: dict | None = None):
    """Load Qwen2 weights into (embed, norm, layers). Layers use short keys."""
    model_path = Path(model_path)
    cfg = load_config(model_path)
    n_layers = int(cfg["num_hidden_layers"])
    layers: list[dict[str, torch.Tensor]] = [{} for _ in range(n_layers)]
    embed = None
    norm = None
    t0 = time.time()

    def consume(key: str, t: torch.Tensor):
        nonlocal embed, norm
        if key == "model.embed_tokens.weight":
            embed = t
        elif key == "model.norm.weight":
            norm = t
        elif key.startswith("model.layers."):
            parts = key.split(".")
            i = int(parts[2])
            short = _SHORT[".".join(parts[3:])]
            layers[i][short] = t
        elif key == "lm_head.weight":
            # tied embeddings: keep, but prefer embed_tokens
            if embed is None:
                embed = t
        elif "rotary_emb" in key:
            pass  # non-persistent buffer in some checkpoints
        else:
            raise KeyError(f"unexpected weight: {key}")

    if state_dict is not None:
        for key, t in state_dict.items():
            consume(key, t)
    else:
        for shard in _shard_files(model_path):
            with safe_open(str(shard), framework="pt", device="cpu") as f:
                for key in f.keys():
                    consume(key, f.get_tensor(key))
    missing = [i for i, d in enumerate(layers)
               if any(k not in d for k in REQUIRED_LAYER_KEYS)]
    if missing:
        raise RuntimeError(f"incomplete layers: {missing[:5]}")
    if embed is None or norm is None:
        raise RuntimeError("missing embed/norm")
    if verbose:
        n_params = sum(t.numel() for d in layers for t in d.values()) + embed.numel() + norm.numel()
        print(f"[load] {n_params/1e9:.3f}B params, {time.time()-t0:.1f}s, layers={n_layers}")
    return embed, norm, layers, cfg


# ----------------------------------------------------------------------------- engine

class PagedQwen:
    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        resident_k: int | None = None,
        max_seq: int = 2048,
        pin: bool = False,
        safety_mb: int = 450,
        quant: str = "none",
        state_dict: dict | None = None,
        resident_quant: bool = False,
        quant_cache: str | Path | None = None,
        verbose: bool = True,
    ):
        self.model_path = str(model_path)
        self.device = torch.device(device)
        self.dtype = dtype
        self.verbose = verbose

        embed, norm, layers, cfg = load_weights(model_path, verbose=verbose, state_dict=state_dict)
        self.cfg = cfg
        self.n_layers = len(layers)
        self.hidden = int(cfg["hidden_size"])
        self.n_heads = int(cfg["num_attention_heads"])
        self.n_kv = int(cfg["num_key_value_heads"])
        self.head_dim = int(cfg.get("head_dim") or (self.hidden // self.n_heads))
        self.n_rep = self.n_heads // self.n_kv
        self.eps = float(cfg.get("rms_norm_eps", 1e-6))
        self.max_seq = int(max_seq)
        self.is_cuda = self.device.type == "cuda"

        # CPU-resident weights (bf16/fp32 as loaded)
        self.embed_cpu = embed
        self.norm_cpu = norm
        self.layers_cpu = layers
        # free the raw handles
        embed = norm = layers = None

        self.layer_bytes = sum(t.numel() * t.element_size() for t in self.layers_cpu[0].values())
        self.embed_bytes = self.embed_cpu.numel() * self.embed_cpu.element_size()

        # GPU-resident tensors
        self.embed = self.embed_cpu.to(self.device, dtype=dtype)
        self.norm = self.norm_cpu.to(self.device, dtype=dtype)

        # KV cache (per layer) and RoPE tables
        self.k_cache = [
            torch.zeros(1, self.n_kv, self.max_seq, self.head_dim, dtype=dtype, device=self.device)
            for _ in range(self.n_layers)
        ]
        self.v_cache = [
            torch.zeros(1, self.n_kv, self.max_seq, self.head_dim, dtype=dtype, device=self.device)
            for _ in range(self.n_layers)
        ]
        self.cos, self.sin = self._rope_tables(self.max_seq)

        self.layers_gpu: list[dict[str, torch.Tensor] | None] = [None] * self.n_layers
        self.slot: dict[str, torch.Tensor] | None = None
        self.pinned = False
        self.bytes_streamed = 0
        self.quant = "none"
        self.quant_names: list[str] = []
        self.small_names: list[str] = []
        self.q_cpu: dict[int, dict[str, torch.Tensor]] = {}
        self.q_scale_gpu: dict[int, dict[str, torch.Tensor]] = {}
        self.small_cpu: dict[int, dict[str, torch.Tensor]] = {}
        self.q_gpu: dict[int, dict[str, torch.Tensor]] = {}
        self.small_gpu: dict[int, dict[str, torch.Tensor]] = {}
        self.resident_quant = False
        self.i8_stage: dict[str, torch.Tensor] = {}
        self.slot_spec: dict | None = None
        self._gpu_ready = True
        self.i8_bytes = 0
        self.small_bytes = 0
        self.i8_layer_bytes = 0
        self.small_layer_bytes = 0
        self.streamed_layer_bytes = 0
        self.quant_err_max = 0.0
        self.quant_cache_hit = False
        self.quant_cache_write_s = 0.0
        self.quant_cache_load_s = 0.0

        if not self.is_cuda:
            self.resident_k = self.n_layers  # everything is "resident" on CPU
            self.embed = self.embed_cpu.to(dtype=self.dtype)
            self.norm = self.norm_cpu.to(dtype=self.dtype)
            self.layers_cpu = [{k: v.to(dtype=self.dtype) for k, v in d.items()} for d in self.layers_cpu]
            self.k_cache = [c.to(dtype=self.dtype) for c in self.k_cache]
            self.v_cache = [c.to(dtype=self.dtype) for c in self.v_cache]
        else:
            if quant not in ("none", "int8", "int4"):
                raise ValueError(f"unknown quant: {quant}")
            if resident_quant:
                if quant not in ("int8", "int4"):
                    raise ValueError("resident_quant requires int8 or int4")
                # Keep every decoder layer as int8 on the GPU and dequantize
                # one layer at a time into a shared bf16 slot. This trades
                # PCIe streaming for GPU memory bandwidth, so a 4B model fits
                # in 6GB and still runs faster than CPU paging.
                self.resident_quant = True
                self.resident_k = self.n_layers
                self.quant = quant
                self._prepare_quant_spec()
                self._alloc_slot()
                cache_loaded = False
                if quant_cache is not None:
                    cache_loaded = self._load_quant_cache(Path(quant_cache))
                self.quant_cache_hit = cache_loaded
                if cache_loaded:
                    self._upload_quant()
                else:
                    self._quantize_all()
                    self._upload_quant()
                    if quant_cache is not None:
                        self._save_quant_cache(Path(quant_cache))
            else:
                k = self._auto_resident(safety_mb) if resident_k is None else int(resident_k)
                k = max(0, min(self.n_layers, k))
                self.resident_k = self._move_resident(k)
                if self.resident_k < self.n_layers:
                    self.slot_spec = {name: (tuple(t.shape), t.dtype)
                                      for name, t in self.layers_cpu[-1].items()}
                    self._alloc_slot()
                    if quant in ("int8", "int4"):
                        self.quant = quant
                        self._quantize_streamed()
                    if pin:
                        self._pin_streamed()
            if verbose:
                free, total = torch.cuda.mem_get_info()
                print(
                    f"[setup] resident={self.resident_k}/{self.n_layers} "
                    f"layer={self.layer_bytes/2**20:.0f}MB embed={self.embed_bytes/2**20:.0f}MB "
                    f"quant={self.quant}{' gpu-resident' if self.resident_quant else ''} "
                    f"free={free/2**20:.0f}MB/{total/2**20:.0f}MB"
                )
        self.past_len = 0

    # ------------------------------------------------------------------ setup helpers

    def _rope_tables(self, n: int):
        d = self.head_dim
        # transformers v5 moved rope_theta under `rope_parameters`; older
        # checkpoints (including the Qwen2.5 ones here) keep it at top level.
        rope_cfg = self.cfg.get("rope_parameters") or {}
        theta = float(self.cfg.get("rope_theta") or
                      rope_cfg.get("rope_theta") or 1e6)
        inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
        pos = torch.arange(n, dtype=torch.float32)
        freqs = torch.outer(pos, inv)          # [n, d/2]
        emb = torch.cat([freqs, freqs], -1)    # [n, d]
        return (emb.cos().to(device=self.device, dtype=self.dtype),
                emb.sin().to(device=self.device, dtype=self.dtype))

    def _auto_resident(self, safety_mb: int) -> int:
        free, _ = torch.cuda.mem_get_info()
        safety = safety_mb * 2**20
        slot = self.layer_bytes
        avail = free - safety - slot
        return max(0, int(avail // self.layer_bytes))

    def _move_resident(self, k: int) -> int:
        moved = 0
        try:
            for i in range(k):
                d = {name: t.to(self.device, dtype=self.dtype) for name, t in self.layers_cpu[i].items()}
                self.layers_gpu[i] = d
                moved += 1
        except torch.cuda.OutOfMemoryError:
            if self.layers_gpu[moved] is not None:
                self.layers_gpu[moved] = None
            torch.cuda.empty_cache()
        if self.is_cuda:
            torch.cuda.empty_cache()
        return moved

    def _alloc_slot(self):
        if self.slot_spec:
            self.slot = {name: torch.empty(shape, dtype=dt, device=self.device)
                         for name, (shape, dt) in self.slot_spec.items()}
        else:
            self.slot = {name: torch.empty_like(t, device=self.device, dtype=self.dtype)
                         for name, t in self.layers_cpu[-1].items()}

    # ------------------------------------------------------- resident quant cache
    #
    # Quantizing all 36 layers of Qwen3-4B takes ~17s on the laptop GPU and is
    # repeated on every process start. The result is deterministic, so we can
    # persist the int8 payload (plus per-row scales and the small tensors) and
    # skip quantization on later starts. The cache is validated against the
    # model file names/sizes/mtimes and the exact tensor shapes; a stale or
    # corrupt file is ignored and regenerated.

    CACHE_FORMAT = "lcsa-resident-quant-v1"

    def _prepare_quant_spec(self):
        """Record which tensors are quantized and the shared slot layout."""
        sample = self.layers_cpu[0]
        self.quant_names = [n for n in ("q_w", "k_w", "v_w", "o_w",
                                        "gate_w", "up_w", "down_w")
                            if n in sample]
        self.small_names = [n for n in sample if n not in self.quant_names]
        self.quant_group = 64
        self.slot_spec = {name: (tuple(t.shape), t.dtype)
                          for name, t in sample.items()}

    def _quant_cache_meta(self) -> dict:
        root = Path(self.model_path)
        names = ["config.json"] + [p.name for p in _shard_files(root)]
        files = []
        for name in names:
            st = (root / name).stat()
            files.append([name, int(st.st_size), int(st.st_mtime_ns)])
        spec = {name: [list(shape), str(dt)]
                for name, (shape, dt) in self.slot_spec.items()}
        return {
            "format": self.CACHE_FORMAT,
            "model": str(root.resolve()),
            "files": files,
            "quant": self.quant,
            "quant_group": self.quant_group,
            "dtype": str(self.dtype),
            "n_layers": self.n_layers,
            "hidden": self.hidden,
            "n_heads": self.n_heads,
            "n_kv": self.n_kv,
            "head_dim": self.head_dim,
            "quant_names": list(self.quant_names),
            "small_names": list(self.small_names),
            "slot_spec": spec,
        }

    def _recompute_quant_stats(self):
        self.i8_bytes = sum(t.numel() for d in self.q_cpu.values() for t in d.values())
        self.small_bytes = sum(t.numel() * t.element_size()
                               for d in self.small_cpu.values() for t in d.values())
        first_q = self.q_cpu[0]
        first_small = self.small_cpu[0]
        self.i8_layer_bytes = sum(t.numel() for t in first_q.values())
        self.small_layer_bytes = sum(t.numel() * t.element_size()
                                     for t in first_small.values())
        self.scale_layer_bytes = sum(t.numel() * t.element_size()
                                     for t in self.q_scale_gpu[0].values())
        self.streamed_layer_bytes = 0
        self.i8_stage = {}

    def _load_quant_cache(self, path: Path) -> bool:
        if not path.exists():
            return False
        t0 = time.time()
        try:
            blob = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(blob, dict) or blob.get("format") != self.CACHE_FORMAT:
                raise ValueError("unknown cache format")
            if blob.get("meta") != self._quant_cache_meta():
                raise ValueError("model/config changed")
            layers = blob.get("layers")
            if not isinstance(layers, dict) or len(layers) != self.n_layers:
                raise ValueError("wrong layer count")

            q_cpu, scale_gpu, small_cpu = {}, {}, {}
            qnames = set(self.quant_names)
            snames = set(self.small_names)
            for i in range(self.n_layers):
                d = layers.get(i)
                if d is None:
                    d = layers.get(str(i))
                if not isinstance(d, dict):
                    raise ValueError(f"missing layer {i}")
                qd, sd, sm = d.get("q"), d.get("scale"), d.get("small")
                if not (isinstance(qd, dict) and isinstance(sd, dict)
                        and isinstance(sm, dict)):
                    raise ValueError(f"bad layer {i}")
                if set(qd) != qnames or set(sd) != qnames or set(sm) != snames:
                    raise ValueError(f"tensor names changed in layer {i}")
                q_out, s_out, sm_out = {}, {}, {}
                for name in self.quant_names:
                    t = qd[name]
                    shape, _ = self.slot_spec[name]
                    if self.quant == "int8":
                        expected_q = shape
                    else:
                        expected_q = (shape[0], shape[1] // 2)
                    if tuple(t.shape) != expected_q or t.dtype != torch.int8:
                        raise ValueError(f"bad q tensor {i}/{name}")
                    q_out[name] = t.contiguous()
                    s = sd[name].to(self.device)
                    if s.dtype != self.dtype:
                        raise ValueError(f"bad scale dtype {i}/{name}")
                    if self.quant == "int8":
                        expected = (shape[0], 1)
                    else:
                        expected = (shape[0], shape[1] // self.quant_group)
                    if tuple(s.shape) != expected:
                        raise ValueError(f"bad scale tensor {i}/{name}")
                    s_out[name] = s.contiguous()
                for name in self.small_names:
                    t = sm[name]
                    shape, _ = self.slot_spec[name]
                    if tuple(t.shape) != shape:
                        raise ValueError(f"bad small tensor {i}/{name}")
                    sm_out[name] = t.contiguous()
                q_cpu[i] = q_out
                scale_gpu[i] = s_out
                small_cpu[i] = sm_out

            self.q_cpu = q_cpu
            self.q_scale_gpu = scale_gpu
            self.small_cpu = small_cpu
            self._recompute_quant_stats()
            # The bf16 decoder weights were only needed to build the cache.
            # Drop them before uploading the int8 payload to keep host RAM low.
            self.layers_cpu = [{} for _ in range(self.n_layers)]
            self.quant_err_max = float(
                (blob.get("stats") or {}).get("quant_err_max", 0.0))
            self.quant_cache_load_s = time.time() - t0
            if self.verbose:
                print(f"[cache] hit {path.name}: int8={self.i8_bytes/2**20:.0f}MB "
                      f"in {self.quant_cache_load_s:.1f}s", flush=True)
            return True
        except Exception as e:
            if self.verbose:
                print(f"[cache] miss ({type(e).__name__}: {e}); rebuilding", flush=True)
            return False

    def _save_quant_cache(self, path: Path) -> None:
        if not self.q_cpu:
            return
        t0 = time.time()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            layers = {}
            for i in range(self.n_layers):
                layers[i] = {
                    "q": {n: t.cpu() for n, t in self.q_cpu[i].items()},
                    "scale": {n: t.detach().cpu()
                              for n, t in self.q_scale_gpu[i].items()},
                    "small": {n: t.cpu() for n, t in self.small_cpu[i].items()},
                }
            blob = {"format": self.CACHE_FORMAT,
                    "meta": self._quant_cache_meta(),
                    "stats": {"quant_err_max": self.quant_err_max},
                    "layers": layers}
            tmp = path.with_name(path.name + ".tmp")
            torch.save(blob, tmp)
            os.replace(tmp, path)
            self.quant_cache_write_s = time.time() - t0
            if self.verbose:
                mb = path.stat().st_size / 2**20
                print(f"[cache] wrote {path.name} ({mb:.0f}MB) in "
                      f"{self.quant_cache_write_s:.1f}s", flush=True)
        except Exception as e:
            if self.verbose:
                print(f"[cache] write failed ({type(e).__name__}: {e})", flush=True)

    def _quantize_all(self):
        """Quantize every decoder layer and keep it for GPU-resident use."""
        t0 = time.time()
        sample = self.layers_cpu[0]
        self.quant_names = [n for n in ("q_w", "k_w", "v_w", "o_w",
                                        "gate_w", "up_w", "down_w")
                            if n in sample]
        self.small_names = [n for n in sample if n not in self.quant_names]
        self.quant_group = 64
        err_max = 0.0
        for i in range(self.n_layers):
            layer = self.layers_cpu[i]
            qd, sd = {}, {}
            for name in self.quant_names:
                w = layer.pop(name)
                wf = w.float()
                out, inn = wf.shape
                if self.quant == "int8":
                    scale = wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
                    q = torch.round(wf / scale).clamp_(-127, 127).to(torch.int8)
                    deq = q.float() * scale
                    sd[name] = scale.to(self.dtype).contiguous()
                    qd[name] = q
                else:
                    g = self.quant_group
                    assert inn % g == 0, (inn, g)
                    wg = wf.view(out, inn // g, g)
                    scale = wg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 7.0
                    qg = torch.round(wg / scale).clamp_(-8, 7).to(torch.int8)
                    q = qg.view(out, inn)
                    u = (q + 8).to(torch.uint8)
                    packed = (u[:, 0::2] | (u[:, 1::2] << 4)).contiguous()
                    deq = (qg.float() * scale).view(out, inn)
                    sd[name] = scale.squeeze(-1).to(self.dtype).contiguous()
                    qd[name] = packed
                denom = max(float(wf.abs().max()), 1e-8)
                err_max = max(err_max, float((deq - wf).abs().max()) / denom)
                del w, wf, deq
            self.q_cpu[i] = qd
            self.q_scale_gpu[i] = {n: s.to(self.device) for n, s in sd.items()}
            self.small_cpu[i] = {n: layer.pop(n) for n in self.small_names}
            self.layers_cpu[i] = {}
        self.quant_err_max = err_max
        self.i8_bytes = sum(t.numel() for d in self.q_cpu.values() for t in d.values())
        self.small_bytes = sum(t.numel() * t.element_size()
                               for d in self.small_cpu.values() for t in d.values())
        first = self.q_cpu[0]
        self.i8_layer_bytes = sum(t.numel() for t in first.values())
        self.small_layer_bytes = sum(t.numel() * t.element_size()
                                     for t in self.small_cpu[0].values())
        self.scale_layer_bytes = sum(t.numel() * t.element_size()
                                     for t in self.q_scale_gpu[0].values())
        self.streamed_layer_bytes = 0
        self.i8_stage = {}
        if self.verbose:
            print(f"[quant-all] {self.quant} {self.n_layers} layers: "
                  f"int8={self.i8_bytes/2**20:.0f}MB "
                  f"small={self.small_bytes/2**20:.1f}MB "
                  f"max_rel_err={err_max:.2e}; {time.time()-t0:.1f}s")

    def _upload_quant(self):
        """Move the int8 weights and small tensors to the GPU (resident mode)."""
        if not self.q_gpu:
            self.q_gpu = {i: {n: t.to(self.device) for n, t in d.items()}
                          for i, d in self.q_cpu.items()}
            self.small_gpu = {i: {n: t.to(self.device) for n, t in d.items()}
                              for i, d in self.small_cpu.items()}
        if self.slot is None:
            self._alloc_slot()
        if self.verbose and self.is_cuda:
            free, total = torch.cuda.mem_get_info()
            print(f"[upload] int8={self.i8_bytes/2**20:.0f}MB "
                  f"free={free/2**20:.0f}MB/{total/2**20:.0f}MB")

    def _quantize_streamed(self):
        """Weight-only quantization (per-row int8 or group-wise int4) for streamed layers."""
        t0 = time.time()
        # Names differ slightly between Qwen2 (attention biases) and Qwen3
        # (per-head q/k norms). Quantize the projections, keep the rest exact.
        sample = self.layers_cpu[self.resident_k]
        self.quant_names = [n for n in ("q_w", "k_w", "v_w", "o_w",
                                        "gate_w", "up_w", "down_w")
                            if n in sample]
        self.small_names = [n for n in sample if n not in self.quant_names]
        self.quant_group = 64
        err_max = 0.0
        for i in range(self.resident_k, self.n_layers):
            layer = self.layers_cpu[i]
            qd, sd = {}, {}
            for name in self.quant_names:
                w = layer.pop(name)
                wf = w.float()
                out, inn = wf.shape
                if self.quant == "int8":
                    scale = wf.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
                    q = torch.round(wf / scale).clamp_(-127, 127).to(torch.int8)
                    deq = q.float() * scale
                    sd[name] = scale.to(self.dtype).contiguous()
                    qd[name] = q
                else:  # int4, groups of 64 along input dim
                    g = self.quant_group
                    assert inn % g == 0, (inn, g)
                    wg = wf.view(out, inn // g, g)
                    scale = wg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 7.0
                    qg = torch.round(wg / scale).clamp_(-8, 7).to(torch.int8)
                    q = qg.view(out, inn)
                    u = (q + 8).to(torch.uint8)
                    packed = (u[:, 0::2] | (u[:, 1::2] << 4)).contiguous()
                    deq = (qg.float() * scale).view(out, inn)
                    sd[name] = scale.squeeze(-1).to(self.dtype).contiguous()  # [out, groups]
                    qd[name] = packed
                denom = max(float(wf.abs().max()), 1e-8)
                err_max = max(err_max, float((deq - wf).abs().max()) / denom)
                del w, wf, deq
            self.q_cpu[i] = qd
            self.q_scale_gpu[i] = {n: s.to(self.device) for n, s in sd.items()}
            self.small_cpu[i] = {n: layer.pop(n) for n in self.small_names}
            self.layers_cpu[i] = {}
        self.quant_err_max = err_max
        self.i8_bytes = sum(t.numel() for d in self.q_cpu.values() for t in d.values())
        self.small_bytes = sum(t.numel() * t.element_size() for d in self.small_cpu.values() for t in d.values())
        first = next(iter(self.q_cpu.values()))
        self.i8_layer_bytes = sum(t.numel() for t in first.values())
        self.small_layer_bytes = sum(t.numel() * t.element_size()
                                     for t in self.small_cpu[self.resident_k].values())
        self.scale_layer_bytes = sum(t.numel() * t.element_size()
                                     for t in self.q_scale_gpu[self.resident_k].values())
        self.streamed_layer_bytes = self.i8_layer_bytes + self.small_layer_bytes + self.scale_layer_bytes
        self.i8_stage = {n: torch.empty(t.shape, dtype=t.dtype, device=self.device)
                         for n, t in first.items()}
        if self.verbose:
            bf16_bytes = (self.n_layers - self.resident_k) * self.layer_bytes
            print(f"[quant] {self.quant} {len(self.q_cpu)} layers: {self.i8_bytes/2**20:.0f}MB vs bf16 "
                  f"{bf16_bytes/2**20:.0f}MB; max_rel_err={err_max:.2e}; {time.time()-t0:.1f}s")

    def _pin_streamed(self):
        try:
            if self.quant in ("int8", "int4"):
                for i in range(self.resident_k, self.n_layers):
                    self.q_cpu[i] = {k: v.pin_memory() for k, v in self.q_cpu[i].items()}
                    self.small_cpu[i] = {k: v.pin_memory() for k, v in self.small_cpu[i].items()}
            else:
                for i in range(self.resident_k, self.n_layers):
                    self.layers_cpu[i] = {k: v.pin_memory() for k, v in self.layers_cpu[i].items()}
            self.pinned = True
        except Exception as e:  # pin_memory can fail on locked-down systems
            self.pinned = False
            if self.verbose:
                print(f"[warn] pin_memory failed: {e}")

    def streamed_bytes_per_pass(self) -> int:
        if self.quant != "none":
            return (self.n_layers - self.resident_k) * self.streamed_layer_bytes
        return (self.n_layers - self.resident_k) * self.layer_bytes

    def release_gpu(self) -> float:
        """Free GPU weight copies/caches; CPU weights are kept for a fast reuse.

        Used when the service switches from a paged unit back to dense units so
        the two do not compete for VRAM (otherwise the driver spills to shared
        memory and every model slows down).
        """
        if not self.is_cuda:
            return 0.0
        t0 = time.time()
        self.layers_gpu = [None] * self.n_layers
        self.slot = None
        self.i8_stage = {}
        if self.resident_quant:
            self.q_gpu = {}
            self.small_gpu = {}
        self.embed = self.embed_cpu          # drop the 0.6GB GPU embed copy too
        self.norm = self.norm_cpu
        self._gpu_ready = False
        torch.cuda.empty_cache()
        return time.time() - t0

    # ------------------------------------------------------------------ core

    def _layer_gpu(self, i: int) -> dict[str, torch.Tensor]:
        if not self.is_cuda:
            return self.layers_cpu[i]
        if self.resident_quant:
            qd = self.q_gpu[i]
            for name in self.quant_names:
                dst = self.slot[name]
                dst.copy_(qd[name])                    # int8 -> bf16
                dst.mul_(self.q_scale_gpu[i][name])    # dequantize in place
            out = dict(self.slot)
            out.update(self.small_gpu[i])
            return out
        d = self.layers_gpu[i]
        if d is not None:
            return d
        if i < self.resident_k:
            # re-materialize a resident layer after release_gpu()
            d = {name: t.to(self.device, dtype=self.dtype)
                 for name, t in self.layers_cpu[i].items()}
            self.layers_gpu[i] = d
            return d
        if self.slot is None:
            self._alloc_slot()
        if self.quant in ("int8", "int4") and not self.i8_stage:
            first = next(iter(self.q_cpu.values()))
            self.i8_stage = {n: torch.empty(t.shape, dtype=t.dtype, device=self.device)
                             for n, t in first.items()}
        assert self.slot is not None
        if self.quant in ("int8", "int4"):
            for name in self.quant_names:
                stage = self.i8_stage[name]
                stage.copy_(self.q_cpu[i][name], non_blocking=self.pinned)
                dst = self.slot[name]
                if self.quant == "int8":
                    dst.copy_(stage)                      # int8 -> bf16 on GPU
                    dst.mul_(self.q_scale_gpu[i][name])   # dequantize in place
                else:
                    out, inn = dst.shape
                    dst[..., 0::2].copy_(stage & 0x0F)
                    dst[..., 1::2].copy_(stage >> 4)
                    dst.sub_(8.0)
                    g = self.quant_group
                    dst.view(out, inn // g, g).mul_(
                        self.q_scale_gpu[i][name].unsqueeze(-1))
            for name in self.small_names:
                self.slot[name].copy_(self.small_cpu[i][name], non_blocking=self.pinned)
            self.bytes_streamed += self.streamed_layer_bytes
            return self.slot
        src = self.layers_cpu[i]
        for name, dst in self.slot.items():
            dst.copy_(src[name], non_blocking=self.pinned)
        self.bytes_streamed += self.layer_bytes
        return self.slot

    def _rmsnorm(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (w * x.to(dt))

    def _rope(self, x: torch.Tensor, start: int) -> torch.Tensor:
        T = x.shape[2]
        cos = self.cos[start:start + T].unsqueeze(0).unsqueeze(0)
        sin = self.sin[start:start + T].unsqueeze(0).unsqueeze(0)
        half = self.head_dim // 2
        x1, x2 = x[..., :half], x[..., half:]
        rot = torch.cat((-x2, x1), dim=-1)
        return x * cos + rot * sin

    def _attn(self, w, h, mask, start: int, T: int, i: int) -> torch.Tensor:
        x = self._rmsnorm(h, w["ln1"])
        q = F.linear(x, w["q_w"], w.get("q_b")).view(
            1, T, self.n_heads, self.head_dim)
        k = F.linear(x, w["k_w"], w.get("k_b")).view(
            1, T, self.n_kv, self.head_dim)
        v = F.linear(x, w["v_w"], w.get("v_b")).view(
            1, T, self.n_kv, self.head_dim)
        # Qwen3 normalizes each head before RoPE (Qwen2 has no q/k norm).
        if "q_norm" in w:
            q = self._rmsnorm(q, w["q_norm"])
        if "k_norm" in w:
            k = self._rmsnorm(k, w["k_norm"])
        q = self._rope(q.transpose(1, 2), start)
        k = self._rope(k.transpose(1, 2), start)
        v = v.transpose(1, 2)
        self.k_cache[i][:, :, start:start + T, :] = k
        self.v_cache[i][:, :, start:start + T, :] = v
        K = self.k_cache[i][:, :, :start + T, :]
        V = self.v_cache[i][:, :, :start + T, :]
        try:
            out = F.scaled_dot_product_attention(q, K, V, attn_mask=mask, enable_gqa=self.n_rep > 1)
        except (TypeError, RuntimeError):
            Kr = K.repeat_interleave(self.n_rep, dim=1) if self.n_rep > 1 else K
            Vr = V.repeat_interleave(self.n_rep, dim=1) if self.n_rep > 1 else V
            out = F.scaled_dot_product_attention(q, Kr, Vr, attn_mask=mask)
        out = out.transpose(1, 2).reshape(1, T, self.n_heads * self.head_dim)
        return F.linear(out, w["o_w"])

    def _mlp(self, w, h) -> torch.Tensor:
        x = self._rmsnorm(h, w["ln2"])
        return F.linear(F.silu(F.linear(x, w["gate_w"])) * F.linear(x, w["up_w"]), w["down_w"])

    def reset_cache(self):
        self.past_len = 0

    def _restore_gpu(self) -> None:
        self.embed = self.embed_cpu.to(self.device, dtype=self.dtype)
        self.norm = self.norm_cpu.to(self.device, dtype=self.dtype)
        if self.resident_quant:
            self._upload_quant()
        self._gpu_ready = True

    def forward(self, ids: torch.Tensor, start_pos: int = 0, all_logits: bool = False) -> torch.Tensor:
        if not self._gpu_ready:
            self._restore_gpu()
        ids = ids.to(self.device)
        if ids.ndim == 1:
            ids = ids.view(1, -1)
        T = ids.shape[1]
        if start_pos + T > self.max_seq:
            raise ValueError(f"sequence {start_pos+T} > max_seq {self.max_seq}")
        h = F.embedding(ids, self.embed)
        mask = None
        if T > 1:
            total = start_pos + T
            idx = torch.arange(total, device=self.device)
            allow = idx.unsqueeze(0) <= (start_pos + torch.arange(T, device=self.device)).unsqueeze(1)
            mask = torch.where(allow, 0.0, float("-inf")).to(self.dtype)
            mask = mask.unsqueeze(0).unsqueeze(0)
        for i in range(self.n_layers):
            w = self._layer_gpu(i)
            h = h + self._attn(w, h, mask, start_pos, T, i)
            h = h + self._mlp(w, h)
        h = self._rmsnorm(h, self.norm)
        if not all_logits:
            h = h[:, -1:]
        return F.linear(h, self.embed)

    @torch.no_grad()
    def generate(
        self,
        ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_id: int | tuple[int, ...] | list[int] | None = (151643, 151645),
        seed: int | None = None,
    ):
        if seed is not None:
            torch.manual_seed(seed)
        self.reset_cache()
        ids = ids.to(self.device)
        if ids.ndim == 1:
            ids = ids.view(1, -1)
        prompt_len = ids.shape[1]
        t0 = time.time()
        logits = self.forward(ids, start_pos=0, all_logits=False)
        prefill_s = time.time() - t0
        self.past_len = prompt_len
        out = ids
        t1 = time.time()
        for _ in range(max_new_tokens):
            nxt = self._sample(logits[:, -1], temperature, top_k, top_p)
            out = torch.cat([out, nxt], dim=1)
            self.past_len += 1
            tid = int(nxt.item())
            if eos_id is not None and (
                    tid == eos_id if isinstance(eos_id, int) else tid in eos_id):
                break
            logits = self.forward(nxt, start_pos=self.past_len - 1, all_logits=False)
        decode_s = time.time() - t1
        n_new = out.shape[1] - prompt_len
        return out, {
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "n_new": n_new,
            "tok_s": (n_new / decode_s) if decode_s > 0 else float("inf"),
        }

    @torch.no_grad()
    def generate_stream(
        self,
        ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_id: int | tuple[int, ...] | list[int] | None = (151643, 151645),
        seed: int | None = None,
    ):
        """Yield new token ids one at a time (greedy by default).

        Stats are stored on ``self.last_gen_stats`` when the generator finishes.
        """
        if seed is not None:
            torch.manual_seed(seed)
        self.reset_cache()
        ids = ids.to(self.device)
        if ids.ndim == 1:
            ids = ids.view(1, -1)
        prompt_len = ids.shape[1]
        t0 = time.time()
        logits = self.forward(ids, start_pos=0, all_logits=False)
        prefill_s = time.time() - t0
        self.past_len = prompt_len
        t1 = time.time()
        n_new = 0
        for _ in range(max_new_tokens):
            nxt = self._sample(logits[:, -1], temperature, top_k, top_p)
            n_new += 1
            self.past_len += 1
            tid = int(nxt.item())
            if eos_id is not None and (
                    tid == eos_id if isinstance(eos_id, int) else tid in eos_id):
                break
            yield tid
            logits = self.forward(nxt, start_pos=self.past_len - 1, all_logits=False)
        decode_s = time.time() - t1
        self.last_gen_stats = {
            "prefill_s": prefill_s,
            "decode_s": decode_s,
            "n_new": n_new,
            "tok_s": (n_new / decode_s) if decode_s > 0 else float("inf"),
        }

    def _sample(self, logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits.float() / max(temperature, 1e-6)
        if top_k and top_k > 0:
            k = min(top_k, logits.shape[-1])
            vals, _ = torch.topk(logits, k, dim=-1)
            logits = torch.where(logits < vals[..., -1:], float("-inf"), logits)
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            keep = probs <= top_p
            keep[..., 1:] = keep[..., :-1].clone()
            keep[..., 0] = True
            sorted_logits = torch.where(keep, sorted_logits, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1)


# ----------------------------------------------------------------------------- convenience

def vram_stats() -> dict:
    if not torch.cuda.is_available():
        return {}
    return {
        "peak_alloc_mb": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20,
        "free_mb": torch.cuda.mem_get_info()[0] / 2**20,
    }
