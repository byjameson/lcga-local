# Long-context training on 6 GB — findings (first pass)

Engine: `lcsa/stream_quant_lora.py` + `experiments/exp54_disk_qlora.py` — an NF4
(4-bit) base streamed from disk, one transformer block resident on the GPU, recompute
in the backward. Measured on one RTX 3050 Laptop 6 GB, Windows 11, **AC power**.

## Result in one line

**32B fine-tunes at 1k context on a 6 GB laptop GPU: 100.7 tok/s (8.6× the seq-128
rate), eval loss 1.8755 → 0.9668 in 50 Dolly-15k steps, typical peak 4.7–5.0 GB.**
The context ceiling is memory, not throughput: 2k does not fit (activation cache +
head spill into shared memory).

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

**Steady-state training run (the number that matters).** 50-step Dolly-15k SFT at
seq 1024 (`--sft --train_size 128`), AC:

| metric | value |
|---|---:|
| eval loss | **1.8755 → 0.9668 (−0.909)** |
| step time (median) | **10.17 s** |
| throughput | **100.7 tok/s** (vs 11.6 tok/s at seq 128 → **8.6×**) |
| peak VRAM | 4.74–5.04 GB typical, 6.17 GB transient (autotune overshoot at 11 blocks, then corrected) |
| host RSS | 17.2 GB |
| wall (50 steps + eval) | 688 s |

**Short-run probe (3 warm steps, no training).** Useful for the memory ceiling, not for
throughput: the pages are only partly cached between separate processes.

| seq | step s (median) | tok/s | peak VRAM | host RSS | blocks on GPU |
|---:|---:|---:|---:|---:|---:|
| 512 | 25.35 | 20.2 | 4.39 GB | 16.4 GB | 5 |
| 1024 | 28.57 | 35.8 | 4.61 GB | 19.0 GB | 2 |
| 2048 | **did not finish** | — | ≥ 5.8 GB | — | — |

The 2048 probe did not complete a single step within a one-hour timeout: the
activation cache for the recompute-backward (`layers × seq × hidden × 2 B` ≈ 1.34 GB)
plus the head pushes the run into WDDM shared-memory spill. **32B fits at 1k, not at
2k.**

## What this means

- **The amortisation claim holds in a real training run.** The step time is ~10 s for
  both 128 and 1024 tokens because the 19.2 GB weight stream is per *step*; tokens/s
  therefore rises from 11.6 to **100.7 (8.6×)** when the context grows 8×. That is the
  pivot's headline, and it is now measured, not projected.
- **The wall is memory, not throughput.** At 2k the activation cache plus the head
  exceeds the card and the run falls into shared-memory spill. So the honest claim is
  **32B fine-tuning at 1k context on a 6 GB laptop GPU at ~100 tok/s**, not 2k.
- Short probes (3 steps, separate processes) under-report throughput by ~3× because
  the page cache is cold each time; quote the 50-step number.
- The battery/sleep sweep that ran earlier produced 17.4 s / 27.4 s at 512 / 1024; it
  was discarded (power state) and re-measured on AC, as the repo's speed tables
  require.
- Next measurements, in order:
  1. **Steady-state 512** (50 steps) to complete the throughput curve at the low end
     (a 3–6 min run).
  2. **14B at 1024–3072** (activation cache 1.0 GB at 2k, so 2k should fit): the 14B
     directory here is a plain fp16 model, so it needs
     `exp54 --model models/Qwen2.5-14B-Instruct --prepare --save-quant
     checkpoints/qwen14b_nf4` first; then
     `scripts/longctx_sweep.py --models 14B --seqs 1024,2048,3072`.
  3. E4 (quality vs context length) needs a long-context eval corpus; not started.

## Reproduce

```bash
python scripts/longctx_sweep.py --models 32B --seqs 512,1024 --steps 3
# 14B (after a one-off quantization pass)
python experiments/exp54_disk_qlora.py --model models/Qwen2.5-14B-Instruct \
    --prepare --save-quant checkpoints/qwen14b_nf4
python scripts/longctx_sweep.py --models 14B --seqs 1024,2048,3072 --steps 3
```
