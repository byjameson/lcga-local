"""VRAM/RAM-aware planning so the engine adapts to whatever GPU the user has.

Everything here is derived from the checkpoint metadata alone -- no weights are
read -- so it costs milliseconds:

  * per-decoder-block size *in the quantized format that is actually stored*,
  * the non-block ("shared") tensors: embedding, final norm, LM head,
  * the dequantization workspace a 4-bit LM head needs at the loss step.

From that it decides:
  * how many decoder blocks can stay resident on the GPU at once
    (`blocks_on_gpu`), which is what makes big cards fast,
  * whether the bf16 LM head fits on the GPU or must stay CPU/4-bit,
  * whether the whole 4-bit base fits in host RAM (RAM-resident path) or has to
    be streamed from disk.

The estimates are deliberately conservative: on Windows/WDDM exceeding the VRAM
limit spills silently into shared memory, which is far worse than being a little
slower.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

GB = 1e9


@dataclass
class Plan:
    strategy: str                  # "gpu-full" | "ram-stream" | "disk-stream"
    gpu_name: str
    vram_total_gb: float
    vram_free_gb: float
    vram_budget_gb: float
    host_ram_total_gb: float | None
    host_ram_available_gb: float | None
    model_gb: float
    block_gb: float
    n_layers: int
    blocks_on_gpu: int
    head_gb: float
    head_on_gpu: bool
    embed_on_gpu: bool
    loss_chunk: int
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"[auto] strategy={self.strategy}  gpu={self.gpu_name}",
            f"[auto] VRAM total={self.vram_total_gb:.1f}GB free={self.vram_free_gb:.1f}GB "
            f"budget={self.vram_budget_gb:.1f}GB",
            f"[auto] model={self.model_gb:.1f}GB  block={self.block_gb * 1024:.0f}MB  "
            f"layers={self.n_layers}",
            f"[auto] blocks_on_gpu={self.blocks_on_gpu} "
            f"({self.blocks_on_gpu * self.block_gb:.2f}GB)  head_on_gpu={self.head_on_gpu} "
            f"({self.head_gb:.2f}GB)  loss_chunk={self.loss_chunk}",
        ]
        lines += [f"[auto] note: {n}" for n in self.notes]
        return "\n".join(lines)


def probe_gpu() -> dict | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        p = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        return {"name": p.name, "total_gb": total / GB, "free_gb": free / GB}
    except Exception:
        return None


def probe_ram() -> tuple[float | None, float | None]:
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.total / GB, vm.available / GB
    except Exception:
        return None, None


def checkpoint_stats(model_dir) -> dict:
    """Sizes of the stored tensors, grouped into per-block and shared parts."""
    from safetensors import safe_open

    root = Path(model_dir)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = dict(index["weight_map"])

    handles: dict[str, object] = {}
    per_layer: dict[int, int] = {}
    shared = 0
    head_bytes = embed_bytes = 0

    for key, fname in weight_map.items():
        h = handles.get(fname)
        if h is None:
            h = safe_open(str(root / fname), framework="pt", device="cpu")
            handles[fname] = h
        t = h.get_tensor(key)
        nbytes = t.numel() * t.element_size()
        m = re.match(r"model\.layers\.(\d+)\.", key)
        if m:
            per_layer[int(m.group(1))] = per_layer.get(int(m.group(1)), 0) + nbytes
        elif key == "lm_head.weight":
            head_bytes = nbytes
        elif key == "model.embed_tokens.weight":
            embed_bytes = nbytes
        else:
            shared += nbytes

    n_layers = max(per_layer) + 1 if per_layer else 0
    block_bytes = (sum(per_layer.values()) / max(len(per_layer), 1))
    total = sum(per_layer.values()) + shared + head_bytes + embed_bytes
    return {
        "n_layers": n_layers,
        "block_bytes": block_bytes,
        "layers_bytes": sum(per_layer.values()),
        "head_bytes": head_bytes,
        "embed_bytes": embed_bytes,
        "other_bytes": shared,
        "total_bytes": total,
    }


def plan_for(model_dir, seq: int = 128, lora_r: int = 8, head: str = "bf16",
             hidden: int | None = None, vocab: int | None = None,
             safety: float = 0.78, reserve_gb: float = 0.6,
             loss_chunk: int | None = None) -> Plan:
    """Pick a placement strategy for this checkpoint on this machine.

    `reserve_gb` covers the CUDA caching allocator's overhead and fragmentation:
    the *reserved* peak we measure is noticeably higher than the sum of the
    tensors, and on WDDM an overrun is far more expensive than a few fewer
    resident blocks.
    """
    stats = checkpoint_stats(model_dir)
    gpu = probe_gpu()
    ram_total, ram_avail = probe_ram()
    notes: list[str] = []

    cfg_path = Path(model_dir) / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    hidden = hidden or int(cfg.get("hidden_size", 4096))
    vocab = vocab or int(cfg.get("vocab_size", 152064))
    tied = bool(cfg.get("tie_word_embeddings", False))

    if gpu is None:
        return Plan("disk-stream", "cpu", 0.0, 0.0, 0.0, ram_total, ram_avail,
                    stats["total_bytes"] / GB, stats["block_bytes"] / GB,
                    stats["n_layers"], 0, stats["head_bytes"] / GB, False, False,
                    loss_chunk or 32, ["no CUDA device found"])

    # usable VRAM: prefer the *free* number (other apps may hold memory)
    vram = gpu["free_gb"] if gpu["free_gb"] > 1.0 else gpu["total_gb"]
    budget = vram * safety

    head_gb = stats["head_bytes"] / GB
    if tied:
        head_gb = 0.0
        notes.append("LM head is tied to the embedding; no separate head tensor")

    # a 4-bit LM head needs a bf16 dequantization workspace at the loss step
    head_workspace = (hidden * vocab * 2 / GB) if (head == "nf4" and not tied) else 0.0
    if head_workspace:
        notes.append(f"4-bit head adds a {head_workspace:.2f}GB dequant workspace")

    loss_chunk = loss_chunk or (64 if budget > 4.0 else 32)
    logits_gb = loss_chunk * vocab * 4 / GB
    lora_gb = 3 * lora_r * hidden * 8 * stats["n_layers"] * 4 / GB  # A+B+grads, fp32
    act_gb = max(0.25, seq * hidden * 2 * 24 / GB)

    block_gb = stats["block_bytes"] / GB
    # embed: tied -> must be on GPU; untied -> stay on CPU as an mmap view
    embed_on_gpu = tied
    fixed = head_gb + head_workspace + logits_gb + lora_gb + act_gb + reserve_gb
    fixed += (stats["embed_bytes"] / GB) if embed_on_gpu else 0.0

    room = budget - fixed
    fit = int(max(0, room) // max(block_gb, 1e-9))
    n_layers = stats["n_layers"]

    if fixed + stats["layers_bytes"] / GB <= budget:
        strategy, blocks = "gpu-full", n_layers
        notes.append("the whole 4-bit base fits: no streaming needed")
    else:
        blocks = max(1, min(fit, n_layers))
        total_gb = stats["total_bytes"] / GB
        if ram_avail and total_gb <= ram_avail * 0.85:
            strategy = "ram-stream"
            notes.append(f"4-bit base ({total_gb:.1f}GB) fits in RAM "
                         f"({ram_avail:.1f}GB free): keep it RAM-resident")
        else:
            strategy = "disk-stream"
            notes.append(f"4-bit base ({total_gb:.1f}GB) does not fit comfortably in "
                         f"RAM ({ram_avail:.1f}GB free): stream from disk")
        if blocks == 1:
            notes.append("only one block fits: maximum compatibility, slowest")

    head_on_gpu = bool(head_gb and head_gb + block_gb <= budget * 0.7)
    if head_gb and not head_on_gpu:
        notes.append(f"LM head ({head_gb:.2f}GB) is large vs budget; consider --head nf4")

    return Plan(strategy, gpu["name"], gpu["total_gb"], gpu["free_gb"], budget,
                ram_total, ram_avail, stats["total_bytes"] / GB, block_gb, n_layers,
                blocks, head_gb, head_on_gpu, embed_on_gpu, loss_chunk, notes)


def main() -> int:
    ap = argparse.ArgumentParser(description="Plan a QLoRA placement for this machine")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--head", choices=["bf16", "nf4"], default="bf16")
    ap.add_argument("--safety", type=float, default=0.78)
    ap.add_argument("--reserve_gb", type=float, default=0.6)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    plan = plan_for(args.model_dir, seq=args.seq, lora_r=args.lora_r,
                    head=args.head, safety=args.safety,
                    reserve_gb=args.reserve_gb)
    if args.json:
        print(json.dumps(asdict(plan), indent=2))
    else:
        print(plan.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
