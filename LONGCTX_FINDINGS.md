# Long-context training on 6 GB — findings (first pass)

Engine: `lcsa/stream_quant_lora.py` + `experiments/exp54_disk_qlora.py` — an NF4
(4-bit) base streamed from disk, one transformer block resident on the GPU, recompute
in the backward. Measured on one RTX 3050 Laptop 6 GB, Windows 11, **AC power**.

## Protocol (important)

- **`--steps 1` is not a step-time measurement.** The first step includes a cold read
  of the whole quantized base (19.2 GB at ~300 MB/s ≈ 60–75 s for 32B), so it measures
  I/O. A 1-step probe gave 67.6 s (seq 512) and 75.0 s (seq 1024); those numbers are
  dominated by the model read.
- The numbers below use `--steps 3` and the **median** from the JSON summary
  (`scripts/longctx_sweep.py`), so the base is in the OS page cache.
- **AC only.** A sweep that ran while the machine was on battery (and asleep) gave
  17.4 s / 27.4 s at 512 / 1024; those rows were discarded and re-measured on AC. The
  repo's other speed tables follow the same rule.

## 32B (Qwen2.5-32B-Instruct, nf4, 19.2 GB on disk)

| seq | step s (median) | tok/s | peak VRAM | host RSS | blocks on GPU |
|---:|---:|---:|---:|---:|---:|
| 128 | 10.95 (100-step SFT) | 11.6 | 3.21 GB | 16.6 GB | 6 |
| 512 | 25.35 | 20.2 | 4.39 GB | 16.4 GB | 5 |
| 1024 | 28.57 | 35.8 | 4.61 GB | 19.0 GB | 2 |
| 2048 | **did not finish** | — | ≥ 5.8 GB | — | — |

The 2048 point did not complete a single step within a one-hour timeout: the
activation cache for the recompute-backward (`layers × seq × hidden × 2 B` ≈ 1.34 GB)
plus the head pushes the run into WDDM shared-memory spill, where the step time
collapses. **32B fits at 1k, not at 2k.**

## What this means

- Within the range that fits, the amortisation is real: the step time grows only
  **11 → 25 → 29 s** from 128 → 512 → 1024 while the tokens per step grow 8×, so
  throughput rises **11.6 → 20.2 → 35.8 tok/s (3.1×)**.
- The wall is **memory, not compute**: at 2k the activation cache plus the head exceeds
  the card and the run falls into shared-memory spill. So the plan's "2k at ~110 tok/s"
  is not reachable for 32B here; the honest headline is **32B fine-tuning at 1k
  context on a 6 GB laptop GPU: ~37 tok/s, 4.6 GB peak**.
- The battery/sleep sweep that ran earlier produced 17.4 s / 27.4 s at 512 / 1024; it
  was discarded (power state) and re-measured on AC, as the repo's speed tables
  require.
- Next measurements, in order:
  1. **14B at 1024–3072** (activation cache 1.0 GB at 2k, so 2k should fit): the 14B
     directory here is a plain fp16 model, so it needs
     `exp54 --model models/Qwen2.5-14B-Instruct --prepare --save-quant
     checkpoints/qwen14b_nf4` first; then
     `scripts/longctx_sweep.py --models 14B --seqs 1024,2048,3072`.
  2. **E2: short Dolly SFT at the largest fitting seq** (50 steps, `--train_size 128`)
     for the "32B fine-tuning at 1k context" headline number.
  3. E4 (quality vs context length) needs a long-context eval corpus; not started.

## Reproduce

```bash
python scripts/longctx_sweep.py --models 32B --seqs 512,1024 --steps 3
# 14B (after a one-off quantization pass)
python experiments/exp54_disk_qlora.py --model models/Qwen2.5-14B-Instruct \
    --prepare --save-quant checkpoints/qwen14b_nf4
python scripts/longctx_sweep.py --models 14B --seqs 1024,2048,3072 --steps 3
```
