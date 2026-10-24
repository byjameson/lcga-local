# LCGA Local — press kit

**One-line:** A source-only research package and local chat app that runs
Qwen3-4B-Instruct in 6 GB of VRAM and trains a 3B model with full-parameter
updates on the same laptop GPU.

## What is it?

LCGA Local is the product surface of the LCGA research repository: a
one-command local chat assistant with cited CPU-only RAG. The same repository
contains the experiments, reports, negative results, and serving/training
runtimes.

- **Product:** `python llm.py` (Windows launcher: `sohbet.cmd`).
- **Model:** Qwen3-4B-Instruct-2507, int8 on the GPU, one bf16 layer slot.
- **RAG:** CPU BM25 over 326 Wikipedia-derived documents (4.57 MiB).
- **Research:** learned block-sparse attention, beyond-VRAM training and
  serving milestones (M1–M5, P1), plus an HTTP service demo.

## Verified numbers

- **Qwen3-4B in 6 GB:** 4513 MB peak VRAM, ~3.4 tok/s decode; 18.8 s first
  run, 2.4 s warm start.
- **exp48 RAG evaluation:** 50 covered + 20 control prompts; retrieval
  50/50, single-source 50/50, keyword 49/50, citations 50/50, controls
  0/20. Baseline keyword was 42/50. The one keyword miss is a correct
  Turkish synonym (`Büyük Okyanus` instead of the literal `Pasifik`).
- **M5 full-parameter 3B training:** all 3.09B parameters updated on a
  6 GB RTX 3050 Laptop; Adafactor CE 2.5034 → 2.3242 in a 150-step demo,
  7.67 s/step, 4.93 GB peak.
- **Tests:** 79 CPU/parity/cache/RAG tests pass.

## Honest limits

- This is not SOTA and not a general benchmark.
- The manual RAG review is a session/agent manual review, not independent
  human validation.
- The 3B training result is a short demo, not a convergence study.
- The LCGA sparse-attention result is currently a negative quality result
  (dense 12.46 vs sparse 16.56 PPL on the measured setup); the reports say so.
- Model weights, checkpoints, caches, and `tools/llama.cpp` are not included.
  Qwen3-4B must be downloaded separately.

## Links

- Repository: https://github.com/byjameson/stream-finetune
- Release: https://github.com/byjameson/stream-finetune/releases/tag/v0.1.0
- Main results: `README.md`, `RAPOR.md`, `M5_RAPOR.md`, `SERVICE_RAPOR.md`
- RAG evidence: `results/exp48_rag_eval.json`,
  `results/exp48_retrieval_check.txt`, `results/exp48_manual_review.md`

## Licenses

Code: MIT. RAG corpus: CC BY-SA 4.0 with attribution in
`data/rag/README.md`.

## Contact

Open an issue on the repository. Please do not send model weights or
credentials.
