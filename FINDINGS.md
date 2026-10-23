# Fine-tuning 32B on a 6 GB laptop GPU

Findings from building a block-streaming QLoRA engine that keeps the frozen 4-bit
base out of VRAM — and, in the last step, out of RAM as well.

Everything below is measured on one machine, with the exact numbers in
`results/results.jsonl` and the long-form notes in `QLORA_RAPOR.md`.

---

## TL;DR

| Model | Method | Peak VRAM | s / step | Result |
|---|---|---:|---:|---|
| Qwen2.5-3B | HF in-memory QLoRA | 4.25 GB | 2.58 | WT2 CE 2.5154 → 2.3188 (−0.197) |
| Qwen2.5-7B | RAM-resident block streaming | 4.90 GB | 5.9 | WT2 CE 2.2304 → 2.1907 (−0.040) |
| Qwen2.5-7B | + Dolly SFT (mlp targets, 200 st) | 5.66 GB | 5.25 | eval loss −0.911 |
| Qwen2.5-14B | streaming, embedding on GPU | 7.01 GB | 70.2 | spills (WDDM) |
| Qwen2.5-14B | streaming, **embedding on CPU** | 3.90 GB | 8.5 | CE −0.105 (20 st) |
| Qwen2.5-14B | + Dolly SFT (200 st) | 4.14 GB | 8.77 | eval loss −1.276 |
| Qwen2.5-32B | **disk-streamed (mmap)** | **3.11 GB** | **10.95** | CE −0.031 (10 st) |
| Qwen2.5-32B | + Dolly SFT (100 st, ~20 min) | 3.21 GB | 11.0 | **eval loss 3.2600 → 1.7849 (−1.475)** |

Counter-intuitive headline: **the 32B run uses less VRAM than the 14B run**
(3.11 vs 3.90 GB). See insight B.

Hardware: RTX 3050 Laptop **6 GB** (6001 MiB), 34 GB host RAM, WD SN740 NVMe,
Windows 11 (WDDM). All runs are single-GPU, batch size 1, seq 128 unless noted.

---

## The base idea

The base is loaded in 4-bit (NF4, double-quant) and kept **on the CPU**. Exactly
one decoder block is moved to the GPU, run, and moved back; the backward pass
re-streams the blocks in reverse order and accumulates LoRA gradients in CPU
buffers (fp32). The base is frozen, so no base optimizer state is needed and the
only trained tensors are the LoRA adapters.

Consequence: peak VRAM is bounded by *one block + activations + the non-block
modules*, independent of depth. HF's whole-model 4-bit QLoRA needs 8.4–8.7 GB for
the same 7B and spills on this card; this engine needs 4.90 GB.

---

## Insight A — keep the input embedding on the CPU

For a 6 GB card, the bf16 input embedding is not a rounding error: Qwen2.5-14B has
152064 × 5120 = 778M parameters → **1.56 GB**, and the LM head another 1.56 GB.

Moving `embed_tokens` to the CPU and transferring only the *gathered activation*
(seq 128 × 5120 bf16 ≈ 1.3 MB) changed the 14B run from

* 7.01 GB peak, 70.2 s/step (spilling into shared memory), to
* **3.90 GB peak, 8.5 s/step**.

That is a **8× step-time** difference from a ~1.5 GB placement change. On WDDM a
small VRAM overrun is not a little slower; it is catastrophic.

Caveat: this only works when `lm_head` and `embed_tokens` are **not tied**. For
tied models (Qwen2.5-3B/0.5B) the head *is* the embedding tensor, so it must live
on the GPU — a regression we hit and fixed.

## Insight B — a 4-bit LM head can cost more than a bf16 one

The 14B run quantizes `lm_head` to NF4 like every other `Linear`. bitsandbytes
then materializes a **bf16 dequantization workspace** for the head at every loss
step: `hidden × vocab × 2` ≈ 5120 × 152064 × 2 ≈ **1.56 GB**.

The 32B checkpoint (pre-quantized, from `unsloth/Qwen2.5-32B-Instruct-bnb-4bit`)
keeps its head in **bf16**, so no workspace is allocated. Net effect: the 32B
model peaks at 3.11 GB while the 14B peaks at 3.90 GB.

Practical takeaway: when VRAM is the binding constraint, measure the head, don't
assume "4-bit is always cheaper". A bf16 head can be both faster and smaller at
the loss step.

---

## Insight C — stream the weights from disk, not from RAM

The RAM-resident engine caps the model at what fits in host RAM (a 14B NF4 base is
~8 GB; a 32B one would be ~18 GB). The disk-backed variant removes that cap:

* the pre-quantized safetensors shards are **memory-mapped** (`safe_open` returns a
  zero-copy CPU view) and
* each decoder block is materialized **directly on the GPU** with
  `Params4bit.from_prequantized`, which re-attaches the saved NF4 quant state
  (absmax, nested absmax, quant maps), then dropped back to `meta` afterwards.

Host RSS after loading the 32B is **2.44 GB** — the 17.9 GB of weights are never
loaded as a whole. Reading them during a step does populate the OS page cache
(RSS grows to ~19 GB during training; see "limits"), but the engine itself only
ever holds one block's tensors at a time.

**The loader is bit-identical to the in-memory path.** Same 3B checkpoint, same
batch, same seed:

| | RAM path (`exp52`) | Disk path (`exp54`) |
|---|---:|---:|
| eval CE (before) | 2.73626 | **2.73626** |
| step-1 loss | 3.347981 | **3.347981** |
| trainable params | 3,686,400 | 3,686,400 |

That check is what makes the 32B numbers trustworthy: it is the same math, only
the weight residency differs.

## Insight D — measured autotuning beats arithmetic

How many blocks can stay resident on the GPU is the difference between "slow but
compatible" and "fast". A static estimate from free VRAM is unreliable because the
CUDA caching allocator reserves more than the tensors sum to. So the default
(`--blocks-on-gpu adaptive`) measures:

1. start at **1 block** (always safe),
2. read that step's real `max_memory_reserved`,
3. grow toward 80% of the device, sized by the **measured marginal cost** of a
   block (`Δpeak / Δblocks` between two different window sizes — measured at
   186–213 MB/block, vs 258 MB nominal), and
4. shrink if a step crosses 88%, then `empty_cache()`.

On the 32B model this converges 1 → 8 → 10 blocks, peak **5.10 GB**, and
**11.5 → 9.3 s/step**. With a deliberately tight budget (`--vram-hard-gb 4.2`) it
converges 1 → 5 → 6 blocks at 4.19 GB.

A bug worth recording: if growth only respected the soft target, a
`hard < soft` configuration oscillated — measured `1→8→5→9→6→5→…`, growing past
the hard limit and shrinking again forever. Growth is now capped by the hard limit
as well.

---

## Negative results and limits

* **WDDM spills silently.** 14B at 7.01 GB peak ran at 70 s/step instead of 8.5 s.
  Peak VRAM must be kept strictly below the card's total, not "close to" it.
* **The disk path is bounded by I/O, not by VRAM or RAM.** A 32B step reads
  ~2 × 16 GB (forward + recompute-backward). Warm page cache: ~11 s/step; cold:
  ~25 s. With more RAM the cache hits more; with less, the step gets slower — but
  it still runs.
* **RAM accounting is honest but not fully proven.** Startup RSS is 2.44 GB (the
  weights are not loaded), yet RSS reaches ~19 GB during training because the OS
  caches the mapped pages. Windows' commit counter reported ~20 GB as well. We
  could not demonstrate reclaimability with `EmptyWorkingSet` on this machine
  (performance counter sets are disabled). Treat "32B runs on 6 GB VRAM" as
  measured, and "it also fits in 16 GB RAM" as *expected but unproven*.
* **No quality claims.** 100–300 steps on 256–1024 Dolly rows demonstrates that
  the model trains, not that it is well aligned. Eval sets differ between runs
  (256/8 for 32B, 512/32 for 14B, 768/32 and 1024/32 for 7B), so compare each
  run's own delta, not absolute levels.
* **One GPU, one model family.** Everything here is Qwen2.5 on a single RTX 3050.
  No head-to-head against DeepSpeed/Accelerate/Unsloth on equal footing.
* **Merge/serve is still blocked for the big models.** Merging a 14B/32B adapter
  needs a bf16 copy (~28/64 GB), which this machine cannot hold.

---

## Reproduce

```bash
pip install -r requirements-qlora.txt          # bitsandbytes 0.50.2, peft 0.21.2

# plan for your machine (reads only checkpoint metadata; milliseconds)
python -m lcsa.autotune --model-dir models/Qwen2.5-32B-Instruct-bnb-4bit --seq 128

# 32B, weights mmap'd from disk, measured autotune on
python experiments/exp54_disk_qlora.py \
    --model-dir models/Qwen2.5-32B-Instruct-bnb-4bit \
    --steps 10 --seq 128 --eval_chunks 2 --shared-dtype as-stored

# 32B instruction SFT (Dolly-15k, prompt tokens masked)
python experiments/exp54_disk_qlora.py \
    --model-dir models/Qwen2.5-32B-Instruct-bnb-4bit \
    --sft --steps 100 --seq 128 --train_size 256 --eval_size 8 \
    --shared-dtype as-stored --save checkpoints/disk_qlora_32b_sft

# 14B / 7B on the RAM-resident path
python experiments/exp52_stream_qlora_7b.py --model models/Qwen2.5-14B-Instruct \
    --steps 20 --seq 128 --eval_chunks 2
python experiments/exp53_stream_qlora_sft_7b.py --model models/Qwen2.5-14B-Instruct \
    --steps 200 --seq 128 --targets attn
```

## What this is not

It is not a faster trainer than a GPU that actually fits the model. It is a way to
train models that **do not fit**, on hardware people already own, with the cost
moved to PCIe and disk bandwidth. If you have 24 GB of VRAM, put the whole model
on the GPU and use a normal QLoRA stack; `lcsa/autotune.py` will tell you that
(`strategy=gpu-full`) and stream nothing.
