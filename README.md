# lcga-local — the LCGA research archive

LCGA ("latent concept graph attention") was an attempt to replace dense attention with a
**learned block-sparse router** on one **RTX 3050 Laptop, 6 GB VRAM** (Windows 11, WDDM).
This repository is the archive of that line: every experiment, the raw evidence, the
reports — and **the negative result** in full, including the measurement errors we made
and had to publish corrections for.

> **The practical trainer has moved to its own repository:
> [byjameson/stream-finetune](https://github.com/byjameson/stream-finetune).**
> It is the streaming-NF4 engine that fine-tunes a **32B** model on this card at
> **1024 tokens/step, 36.7 tok/s, 4.61 GB**, plus the measured guide. This repo keeps the
> research; that one keeps the tool. `lcsa/stream_*`, `lcsa/disk_quant_store.py` and
> `experiments/exp54_disk_qlora.py` here are a **frozen snapshot** — the maintained copy
> lives in the trainer repo as the `streamft/` package.

## What we found, in one screen

**On plain language modelling the learned router is worthless.** Trained end-to-end on a
frozen Qwen2.5-0.5B, it learns "sink + the recent window" and **ties a budget-matched
static window** — it is 0.005–0.133 PPL *worse*, while paying for the router. This was the
headline question and the answer is no.

**On retrieval-style data it does learn, and that is the one positive result.** Training at
the *answer* positions with a content-focused target — including a **label-free** target
(the dense teacher's content argmax), with no retrieval labels at all — reaches **dense
exact match at a 25% block budget** (1k → 8k transfer, 3 seeds × 6 prompts, exact 1.00 in
every out-of-window cell; the static window scores 0.00). It replicates on 0.5B and on 3B.
Mixing **1 retrieval step per 7 LM steps** (`mix7`) keeps plain-LM PPL within **+0.6** of
the window while retaining the retrieval, so one router can serve both.

**Query selectivity is negative on three axes.** The router does not choose *which* needle
from the question: teacher-argmax vs. labelled targets, loose vs. binding budget, mean vs.
last-token query pooling — none of them produces query-dependent selection. What remains
is a modelling question (does the base model's hidden state carry "which needle" at all?),
not a tuning question. We say so in §5.5 of the findings instead of hiding it.

### The three measurements that nearly fooled us

1. **We measured a model we never loaded.** The checkpoint was loaded *before* the router
   modules were patched in, so a randomly initialised router was benchmarked for days. The
   published "0.0% content blocks, identical to window" was a bug artefact. Invalid runs
   are kept in [`results/invalid/`](results/invalid/).
2. **"8.6× faster" was a truncation cap.** The SFT path feeds one dataset row per step and
   `--seq` only truncates; Dolly rows are median **134** tokens. Correct LM-path number:
   **3.1×**, not 8.6×.
3. **The baseline was not budget-matched.** The learned router was allowed 4–16 blocks and
   "local" attention 2. Against a **budget-matched window** the router adds nothing.

All three are written up with the corrected numbers in
[`LCGA_FINDINGS.md`](LCGA_FINDINGS.md) §5 and in the DEV article below.

## Read this first

| Document | What it is |
|---|---|
| **[`LCGA_FINDINGS.md`](LCGA_FINDINGS.md)** | The canonical write-up: protocol, negative result, controls, ablations, the one positive regime, scale check, limits. |
| [`RAPOR.md`](RAPOR.md) | Main report (phases A/B/C), Turkish. |
| [`PROGRESS.md`](PROGRESS.md) | Append-only dev journal — every run, every bug, every correction, in order. |
| [`PRESS.md`](PRESS.md) | Press kit / summary for the launch. |
| [`ARTICLE.md`](ARTICLE.md) · [`article.html`](article.html) | Launch article source. |
| [`BASARILAR.md`](BASARILAR.md) | Milestone digest. |
| [`ROADMAP.md`](ROADMAP.md) | Original plan and status. |
| `P1…P9_RAPOR.md`, `M1…M5_RAPOR.md`, `QLORA_RAPOR.md`, `SERVICE_RAPOR.md`, `MoE_RAPOR.md` | Per-milestone reports (block-streaming training, byte-level prototype, paged inference, MoE paging, scheduler, full-parameter 3B training, QLoRA, local chat service). |

Published articles (DEV Community):
[Running a 4B model in 6 GB and training 3B parameters on a laptop](https://dev.to/mycute_97b39c4e34027570f9/running-a-4b-model-in-6-gb-and-training-3b-parameters-on-a-laptop-4al) ·
[Fine-tuning a 32B model on a 6 GB laptop GPU — and the three measurements that nearly fooled me](https://dev.to/mycute_97b39c4e34027570f9/fine-tuning-a-32b-model-on-a-6-gb-laptop-gpu-and-the-three-measurements-that-nearly-fooled-me-2ojm)

## Model artifacts

| Artifact | Where | What it is |
|---|---|---|
| `v3_best.pt` | [HF](https://huggingface.co/byjameson/lcga-v3-0.5b) · [ModelScope](https://modelscope.cn/models/byjameson/lcga-v3-0.5b) | Best LCGA block-sparse checkpoint (Qwen2.5-0.5B base, LoRA + router), the one behind the negative result. |
| `router_needle_mix7.pt` | same repos | Label-free retrieval router, mixed schedule (1 needle : 7 LM steps) — the practical positive result. |
| `router_needle_mix7_nf2.pt` | same repos | Same, trained with paraphrases and two-needle contexts; fixes held-out phrasing at 2k, keeps 5/5 cells. |
| `lcga-p3-17m` | [HF](https://huggingface.co/byjameson/lcga-p3-17m) · [ModelScope](https://modelscope.cn/models/byjameson/lcga-p3-17m) | Byte-level prototype (P3). |

The model cards link back here for the full reports and raw evidence.

## Repository map

```
lcsa/            core library: attention, router (concepts.py), model, decode, streaming
experiments/     one CLI per experiment (exp1 … exp65); evidence rows land in results/
scripts/         probes, aggregation, plots, card/checkpoint publishing
results/         results.jsonl (append-only) + per-experiment JSON; results/invalid/ = bug evidence
plots/           figures used in the reports
dist/            model-card section used by the HF / ModelScope updaters
tests/           attention / decode / paged / RAG / service tests
llm.py serve.py  local chat + HTTP service (the 4B-in-6GB line)
```

## Reproducing the headline

```powershell
pip install -r requirements.txt
python scripts/probe_dense.py --model models/Qwen2.5-0.5B --Ts 1024 --depths 0.1,0.5,0.9   # dense prerequisite
python experiments/exp1_data_baseline.py                                                  # data + dense baselines
python experiments/exp2_train_router.py --tag v1                                          # router training
python experiments/exp3_eval.py --ckpt checkpoints/v3_best.pt --tag v3                    # eval
```

Every experiment appends a timestamped row to `results/results.jsonl`; nothing in the
reports is quoted from memory. The retrieval-router runs are in `exp62`–`exp65` and
`scripts/summarize_needle.py`.

## Limits (stated up front)

- Transfer stops at 8k: 16k is 0.00 for the router (dense 1.00); at 4k the dense teacher
  itself fails, so there is no teacher signal.
- 2k is an unexplained anomaly for the router (dense is fine, kernel parity is fine,
  selection is fine, the answer degrades). We do not have a mechanism for it.
- No KV-memory saving is claimed: the router selects blocks to *attend to*; the cache is
  still dense.
- The positive result is a **retrieval specialist**. Plain-LM perplexity degrades unless it
  is mixed with LM steps, and even then it costs +0.6 PPL at 1k.
- Single GPU, single machine, Windows/WDDM; speed numbers are medians of warm runs on AC
  power only.

## License

Apache-2.0 — see [`LICENSE`](LICENSE). Donation addresses are on the model cards, not here.
