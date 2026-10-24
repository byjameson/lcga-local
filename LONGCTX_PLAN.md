# Pivot plan — long-context training on 6 GB

The LCGA line is closing as a *research* result (see `LCGA_FINDINGS.md`): learned
block selection helps more as context grows, but it is a compute win, not a memory
win, and it does not beat dense enough to matter in production.

Our proven differentiator is elsewhere: **block-streaming QLoRA trains models that
do not fit**, and this machine's actual gap is *context length* — every run so far
was `seq=128`.

## The claim we want

> *A 32B model can be fine-tuned at 2k+ context on a 6 GB laptop GPU, and the
> streaming cost is amortised: throughput in tokens/s rises almost linearly with
> sequence length because the weights are streamed once per step regardless of T.*

That is a genuinely different pitch from "bigger model on a small card" (crowded)
and from LCGA (compute-only):

1. **Capacity story** the market already believes we can do (32B on 6 GB).
2. **Throughput insight** nobody advertises: weight streaming is per *step*, not
   per *token*, so long sequences are where this engine actually becomes efficient.
3. It exercises the same code (`lcsa/stream_quant_lora.py`, `exp54`) with no new
   research risk.

## Why it should work (and where it breaks)

Per step the engine streams every block once for the forward and once for the
recompute-backward — **independent of `seq`**. Compute and activation memory grow
with `seq`, I/O does not. So:

| seq | weight I/O per step | tokens per step | expected step time | tokens/s |
|---:|---:|---:|---:|---:|
| 128 | ~32 GB (32B model) | 128 | 11.0 s (measured) | 11.6 |
| 1024 | ~32 GB | 1024 | ~13–16 s | ~70 |
| 2048 | ~32 GB | 2048 | ~16–20 s | ~110 |

The hard limit is the **activation cache** we keep for the recompute-backward:
`layers × seq × hidden × 2 bytes` (fp16/bf16).

| model | seq | activation cache | plus one block + head | verdict |
|---|---:|---:|---:|---|
| 32B (64×5120) | 1024 | 0.67 GB | ~3.9 GB | fits |
| 32B | 2048 | 1.34 GB | ~4.6 GB | fits (tight) |
| 32B | 4096 | 2.68 GB | ~5.9 GB | spills — out |
| 14B (48×5120) | 2048 | 1.01 GB | ~5.0 GB | fits |
| 14B | 4096 | 2.01 GB | ~6.0 GB | borderline |

So the experiment is bounded and cheap to explore.

## Experiments (each a few minutes, no training needed for E1)

**E1 — sequence sweep, no training.** 32B and 14B, measure peak VRAM, step time,
tokens/s at `seq ∈ {512, 1024, 2048, 3072}`.

> **Measurement note (25 Oct 2026).** A single-step run is dominated by the **cold read
> of the whole quantized base** (19.2 GB at ~300 MB/s ≈ 60–75 s for 32B), so `--steps 1`
> measures I/O, not the step: the first probe gave 67.6 s at seq 512 and 75.0 s at 1024
> with peak 3.82/4.60 GB, which says nothing about throughput. Use
> `scripts/longctx_sweep.py` (runs `--steps 3`, reports the median of the JSON summary,
> and computes tokens/s) and compare like with like.

`python scripts/longctx_sweep.py --models 32B,14B --seqs 512,1024,2048,3072`
**Decision:** find the largest `seq` with peak ≤ ~5.3 GB (WDDM headroom).

**E2 — long-context SFT.** At the best `seq`, run a short Dolly SFT (e.g. 50 steps,
`--train_size 128`) and report eval loss + tokens/s + peak. This is the headline
number: **32B fine-tuning at Nk context on a 6 GB card**.

**E3 — throughput table.** Combine E1's numbers into the tokens/s-vs-seq curve and
state the amortisation argument explicitly.

**E4 (optional) — does quality improve with context?** Compare eval loss of the
same adapter trained at seq 128 vs seq 2048 on a long-context eval set. That is the
"does it actually learn from longer context" question; it needs a long-context eval
corpus, so it is a stretch goal.

## What would make this fail

- Activation cache + head + block exceeds ~5.3 GB before `seq` 1024 for 32B.
- Step time grows faster than `seq` (it will not: I/O dominates and is constant,
  but the per-token compute does grow linearly).
- Long sequences expose a correctness issue in the masked-loss path (should not —
  `exp53` already used masking at seq 256).

If 32B caps at 1024, the story still holds with 14B at 2048–4096, which is a
perfectly respectable "long-context fine-tuning on 6 GB" result.

## Order of work

1. LCGA wrap-up (current): 16-concept ablation, 16k quality, decode-graph numbers,
   publish.
2. E1 sweep (≈15 min of GPU).
3. E2 SFT at the best `seq` (≈20–30 min).
4. Write `LONGCTX_FINDINGS.md` + README/ROADMAP updates, commit, publish.
