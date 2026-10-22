# Running a 4B model in 6 GB and training 3B parameters on a laptop

**What worked, what didn’t, and what the numbers actually mean.**

I’m Gökhan Çava, an independent developer. I published
[LCGA Local](https://github.com/byjameson/lcga-local), a source-only research
package for local AI on small GPUs. This is the short technical story behind
the release, including the parts that did not work.

## The constraint: 6 GB of VRAM

The test machine is an RTX 3050 Laptop GPU with 6 GB of VRAM. That is enough
for a small model, but not enough to load a 4B model in bf16 alongside a
normal KV cache and a desktop session. The release contains two different
answers to that constraint.

## Serving: Qwen3-4B at a 4.5 GB peak

The serving path uses Qwen3-4B-Instruct-2507 in int8 with layer streaming and
one bf16 layer slot. It is deliberately boring engineering: keep the resident
working set below the limit, move weights in as needed, and avoid allocating
the whole model at once.

Measured on the 6 GB RTX 3050 Laptop:

- **4,513 MB peak VRAM**
- **~3.4 tok/s decode**
- **18.8 s first run, 2.4 s warm start**
- One-command entry point: `python llm.py` (`sohbet.cmd` on Windows)

This is not fast by datacenter standards. It is useful for local chat,
experimentation, and checking whether a 4B model fits at all.

## RAG: citations without a giant vector database

The RAG layer is CPU-only BM25 over a 326-document Wikipedia-derived corpus
(4.57 MiB). It returns a passage and a citation, and it declines when the
corpus does not support an answer.

On the internal exp48 evaluation (50 covered prompts plus 20 controls):

- **Retrieval: 50/50**
- **Single-source answers: 50/50**
- **Keyword overlap: 49/50**
- **Citations: 50/50**
- **Controls answered: 0/20**

The one keyword miss is not a factual failure: the answer used the correct
Turkish synonym `Büyük Okyanus` instead of the literal `Pasifik`. The baseline
keyword score was 42/50.

The important limit: this is a small, hand-built evaluation, not a public
benchmark. The manual review was performed in the session by the agent and
the developer, not by an independent human reviewer. The repository says so
in `results/exp48_manual_review.md`.

## Training: all 3.09B parameters on the same 6 GB GPU

The M5 experiment asks a different question: can the same laptop update the
full parameter set of a 3B model, not a LoRA adapter?

Using block streaming and Adafactor, M5 updates all **3.09B parameters** on
the 6 GB RTX 3050 Laptop:

- **150-step demo**
- **Cross-entropy: 2.5034 → 2.3242**
- **7.67 s/step**
- **4.93 GB peak VRAM**

This is a demo, not a convergence study. A 150-step run is enough to show
that the memory path works; it is not enough to claim a useful fine-tune.
The full report is in `M5_RAPOR.md`.

## What did not work: the sparse-attention result

The repository also contains the LCGA block-sparse attention experiments.
The current quality result is negative: in the measured setup, dense
attention reached **12.46 PPL** and the sparse variant reached **16.56 PPL**.
The release includes that result rather than hiding it. The serving and
training work is independently useful; the sparse-attention idea still needs
more work.

## What is in the release

- Local chat and serving code
- CPU RAG and a 326-document corpus
- M1–M5 and P1 training experiments and reports
- CUDA-graph and research artifacts
- 79 CPU/parity/cache/RAG tests
- Press kit and honest limits

## What is not in the release

Model weights, checkpoints, caches, and `tools/llama.cpp` are excluded.
Qwen3-4B must be downloaded separately. The code is MIT-licensed; the RAG
corpus is CC BY-SA 4.0 with attribution in `data/rag/README.md`.

## Links

- Repository: https://github.com/byjameson/lcga-local
- Release: https://github.com/byjameson/lcga-local/releases/tag/v0.1.0
- Press kit: https://github.com/byjameson/lcga-local/blob/main/PRESS.md

If you run this on a smaller GPU, or if you find a mistake in the evaluation,
open an issue. Negative results and corrections are welcome.
