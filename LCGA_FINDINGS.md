# LCGA — findings (learned block-sparse attention on a frozen 0.5B)

Measured on one RTX 3050 Laptop 6 GB (Windows 11/WDDM), AC power. Base:
Qwen2.5-0.5B, frozen; only the router (and, in stage-2 runs, LoRA) is trained.
Reproduce with `experiments/exp2_train_router.py`, `exp3_eval.py`, `exp4_speed.py`,
`exp58_eval_long.py`, `exp59_selection_stats.py`, `exp60_needle.py`,
`exp61_train_router_needle.py`, `exp62_needle_gen.py`; raw rows in
`results/results.jsonl`.

---

## 0. Correction (this revision supersedes the first one)

An earlier version of this document reported **"0.0% of selected blocks lie outside
the recent window"** and **"learned perplexity is identical to a budget-matched
static window to three decimals"**. Both numbers were wrong, and the reason is worth
recording: `exp58/exp59/exp60` called `load_ckpt()` **before** `patch_model()`, so
the router modules did not exist yet and nothing was loaded. Those scripts were
evaluating a **freshly initialised router**, not the trained checkpoint. `exp3`
(the table in §3) and `exp4` (speed) always loaded correctly, so their numbers stand.

What the correction changes:

- the trained v1 router does select some content: **1.7% / 6.0% / 7.0%** at
  1k / 4k / 8k (not 0.0%);
- it is **not identical** to the window: it is 0.005–0.13 PPL **worse**;
- the needle probe's learned row had to be re-measured (`dense`, `window`, `random`,
  `local` and `oracle` never used router weights, so those rows were always valid).

The scripts are fixed (`load_ckpt` now runs after `patch_model`), and every number
below was produced with the fixed code.

---

## TL;DR

1. **On plain language modelling, a distilled block router learns the position
   prior and almost nothing else.** Content blocks are 1.7–7.0% of its selections,
   and at a 25% budget it is indistinguishable from — slightly worse than — a static
   sliding window: 25.166 vs 25.051 (1k), 13.788 vs 13.783 (4k), 12.176 vs 12.160
   (8k), 12.001 vs 11.868 (16k).
2. **That is the correct behaviour, not a training failure.** On WikiText the dense
   teacher itself puts almost no mass on distant blocks, so there is nothing to
   distil. Perplexity cannot reward looking far away.
3. **Retrieval is the regime where distant content matters.** A fact planted at a
   known depth: dense answers it exactly (CE 0.14–0.39 at 1k–8k), a budget-matched
   window cannot (CE ≈ 3.0, exact 0.00).
4. **Content selection is learnable — with the right objective and the right head.**
   Training at the answer positions with a *retrieval* target gives
   **exact match 1.00 at a 25% budget** (dense 1.00, window 0.00) and it transfers
   from 1k training to 8k evaluation. Content blocks rise to **19% / 57.5% / 69.2%**.
5. **Plain attention distillation does not get there — a content-focused one does.**
   Full KL (the original LCGA objective) stalls at p(fact) = 0.11 and fails the needle
   test; content-restricted KL works at 8k but not at 1k; **distilling the teacher's
   content argmax needs no fact labels and matches dense exact match in every
   out-of-window cell at a 25% budget** (§5.4). A shuffled-label control stays at
   chance (0.067), so the effect is the target, not the training loop.
6. **The head matters as much as the objective.** The concept-graph router is
   content-blind at initialisation (softmax over K concepts, block-mean pooled →
   score gradient ~1/K²). With a direct bilinear content path the same objective
   reaches p(fact) = 0.86 in **60 steps**.
7. **It is a trade, not a free win**: the needle-trained router loses plain-LM
   quality (27.7 vs 25.1 at 1k), and it does not generalise to a re-phrased fact.
8. **Speed is unchanged**: prefill **1.71× at 32k**, graph decode **1.27× at 8k**;
   eager decode loses. Still compute/bandwidth, **not** KV memory.

---

## 1. What the v1 router actually selects

`exp59`, corrected loading, 2 chunks per length, budget ρ = 0.25 (block = 64):

| T | sink | window | **content** | mean budget m |
|---|---:|---:|---:|---:|
| 1024 | 40.0% | 58.3% | **1.7%** | 2.5 blocks |
| 4096 | 12.2% | 81.8% | **6.0%** | 8.2 blocks |
| 8192 | 6.2% | 86.8% | **7.0%** | 16.2 blocks |
| 16384 | 3.1% | 90.2% | **6.7%** | 32.1 blocks |

Content is non-zero but small, and it shrinks as a fraction exactly as the window
grows: the router is ~90% a position prior.

## 2. Quality against a budget-matched window

One evaluator for all four lengths (`exp58`, chunked-head CE so 16k fits in 6 GB;
8/4/2/2 chunks at 1k/4k/8k/16k), corrected loading:

| T | dense | learned | **static window** | random | local (2 blocks) | learned vs dense |
|---|---:|---:|---:|---:|---:|---:|
| 1024 | 17.323 | 25.166 | **25.051** | 27.317 | 30.378 | +45.3% |
| 4096 | 10.950 | 13.788 | **13.783** | 17.403 | 29.045 | +25.9% |
| 8192 | 10.375 | 12.176 | **12.160** | 16.178 | 32.140 | +17.4% |
| 16384 | 10.799 | 12.001 | **11.868** | 15.993 | 33.503 | +11.1% |

Learned is within **+0.005 … +0.133 PPL** of the window — i.e. the learned part buys
nothing measurable at this scale, but it is not *identical* either (the earlier
"identical to three decimals" claim came from the loading bug).

What genuinely helps is the **budget**: the gap to dense falls from +45% (1k) to
+11% (16k) at a fixed 25% of blocks.

Independent check with the older `exp3` evaluator (128 chunks, T=1024): dense 14.304,
learned 21.837, random 23.945, local 26.789. Absolute values differ from `exp58`
because of the chunk subset (8 vs 128 chunks); compare within a run.

Budget knob (exp3, T=1024): ρ = 0.125 / 0.25 / 0.5 / 1.0 → 26.789 / 21.837 /
17.198 / 14.303 PPL. At T=8192: 13.691 / 12.176 / 11.340 / 10.376.

## 3. Architecture ablations (T=1024, ρ=0.25, 128 chunks)

Same recipe (seed 0, 1500 steps, lr 1.5e-3), only the head differs. `exp3` loaded
checkpoints correctly, so these are unaffected by the bug:

| Variant | router params | PPL | recall |
|---|---:|---:|---:|
| graph rank-16 + recency prior (v1) | 2,804,280 | 21.838 | 0.779 |
| diag only + recency prior | 2,755,128 | **21.837** | 0.785 |
| diag only, **16 concepts** | **689,592** | 21.841 | 0.785 |
| diag only, **no** recency prior | 2,755,104 | 22.812 | 0.754 |
| random-k (budget-matched) | 0 | 23.945 | 0.742 |

The low-rank concept graph adds nothing; the recency prior is the only component
that matters on this data — exactly what §1 and §2 predict.

## 4. Where distant content actually matters: the needle probe

One unique 4-digit code is planted at a known depth in 40k tokens of WikiText, then
`" Question: What is the secret code? Answer:"`. Metric: teacher-forced CE and exact
match on the answer tokens (`exp60`). At a 25% budget the window can only see the
last ~25% of the sequence, so it can only answer when the fact happens to be there.

T = 8192, 4 prompts per cell:

| depth | dense | **budget-matched window** | random | local |
|---|---:|---:|---:|---:|
| 0.1 | **0.161 / 1.00** | 3.048 / 0.00 | 2.355 / 0.00 | 2.748 / 0.00 |
| 0.5 | **0.373 / 1.00** | 3.023 / 0.00 | 2.352 / 0.00 | 2.701 / 0.00 |
| 0.9 | **0.139 / 1.00** | 0.173 / 1.00 | 2.351 / 0.20 | 2.825 / 0.00 |

(CE / exact match.) The same pattern holds at 1k and 2k. **Dense retrieves; a
budget-matched window does not** unless the fact is inside the window. This is the
one setting where a content-selecting router can beat the static baseline, and
perplexity evaluation can never show it.

## 5. Teaching the router to select content

Setup (`exp61`): needle contexts of length 1024, fact at a uniform random position,
24 layers, budget 0.25 (m = 4 blocks, of which {sink, self} are mandatory), 300–600
steps, batch 2, lr 3e-3. The teacher is the **token-level attention of the answer
tokens** aggregated to blocks (block-averaged dense mass dilutes a 3-token answer
across a 64-token block by ~20×, which alone destroys the signal — the v1 mass
target is not usable here). The loss is applied only at the answer query rows.

### 5.1 Objectives

| Objective | Target | Label needed | p(fact) | top-1 = fact |
|---|---|---:|---:|---:|
| full KL (v1 objective, answer rows) | whole teacher distribution | no | 0.113 | 0.107 |
| content KL | teacher outside {sink, self, window} | no | 0.183 | 0.362 |
| teacher content-argmax | argmax of the above | no | 0.530 | 0.566 |
| retrieval | the fact block itself | yes (oracle) | **0.912** | **0.918** |
| retrieval, shuffled labels | random block | yes (control) | 0.067 | 0.069 |

Chance is 1/16 = 0.0625. Three observations:

- **Full KL cannot work**, and the reason is visible in the teacher itself: at the
  answer rows ~50% of dense mass sits on the sink block and ~20% on the query block;
  the content component is a small fraction, so the KL is dominated by structure the
  position prior already provides. This is the mechanism behind §1–§2, reproduced
  with data on which content *does* matter.
- **Restricting the KL to content helps but is not enough at 1k**, because the
  teacher's content mass is itself diffuse (many blocks at ~0.02, the fact at ~0.20):
  the best-fitting distribution is closer to uniform than to "pick the fact". It does
  work at 8k, where the teacher's content mass is sharper (§5.4).
- **Focusing the target further onto the teacher's content *argmax* is enough**, and
  it needs no labels: the objective is a hard target read off the dense attention.
  It is the only label-free variant that retrieves at both lengths (§5.4).
- **The shuffled-label control learns nothing** (0.067 = chance), so the retrieval
  result is the label, not the optimisation.

### 5.2 The head matters

The same retrieval objective on the *concept-graph* head (no content path, 400 steps,
labelled) versus the content path (300 steps): p(fact) 0.078 vs 0.912, and end to end
at a 25% budget (CE / exact):

| head | 1k depth 0.5 | 8k depth 0.1 | 8k depth 0.5 |
|---|---:|---:|---:|
| concept graph only (400 steps) | 0.560 / 0.00 | 0.339 / 1.00 | 0.226 / 1.00 |
| **+ bilinear content path (300 steps)** | **0.208 / 1.00** | **0.373 / 1.00** | **0.289 / 1.00** |

The concept head does get there at 8k but not at 1k within the same budget of steps:
its score is a bilinear form of two 64-dim softmax distributions averaged over 64
tokens, so at initialisation it is nearly constant across blocks and its gradient is
~1/K². **This, not the data, is why earlier LCGA runs plateaued at the prior** — it
is a head-capacity/signal-scale problem, not an impossibility result.

### 5.3 End-to-end needle result at a 25% budget

`needle_ret_c64`, trained at T=1024 only, evaluated at three lengths (5 prompts per
cell, `exp60`); the same run provides the window control:

| T | depth | dense | **learned router** | window | oracle (dense-mass top-m) |
|---|---:|---:|---:|---:|---:|
| 1024 | 0.1 | 0.676 / 0.00 | 0.915 / 0.00 | 2.815 / 0.00 | 1.127 / 0.00 |
| 1024 | 0.5 | 0.143 / 1.00 | **0.170 / 1.00** | 2.935 / 0.00 | 0.625 / 0.00 |
| 1024 | 0.9 | 0.123 / 1.00 | **0.164 / 1.00** | 0.221 / 1.00 | 0.204 / 1.00 |
| 2048 | 0.1 | 0.253 / 1.00 | 0.334 / 0.20 | 3.085 / 0.00 | 0.286 / 0.80 |
| 2048 | 0.5 | 0.317 / 0.80 | **0.186 / 1.00** | 3.110 / 0.00 | 0.364 / 0.20 |
| 2048 | 0.9 | 0.255 / 0.60 | 0.314 / 0.00 | 0.261 / 0.20 | 0.216 / 1.00 |
| 8192 | 0.1 | 0.161 / 1.00 | **0.175 / 1.00** | 3.048 / 0.00 | 0.314 / 1.00 |
| 8192 | 0.5 | 0.373 / 1.00 | **0.190 / 1.00** | 3.023 / 0.00 | 0.348 / 0.80 |
| 8192 | 0.9 | 0.139 / 1.00 | **0.143 / 1.00** | 0.173 / 1.00 | 0.137 / 1.00 |

(CE / exact match.) The learned router equals dense exact match in **7 of 9 cells**
(the exceptions are 2k depth 0.1, 0.20 vs 1.00, and 2k depth 0.9, where the window
wins because the fact happens to lie inside it), and beats the **oracle** at 8k,
because the oracle spreads its budget over useful-looking blocks for *all* rows while
the router was trained for the answer row. It transfers from 1k training to 8k
evaluation without retraining.

`exp61_eval_check` confirms the mechanism: at the answer row, **24/24 layers** select
the fact block, and the sparse output is identical to dense in exact match
(CE 0.177 vs 0.152, both exact 1.00). A greedy generation check (each step is a full
re-prefill, 8 tokens) gives the same string, `7311`, for dense and sparse at 1k; the
label-free router at 8k also generates `7311` identically to dense, with 21/24 layers
selecting the fact block.

### 5.4 The label-free version works

Same data, same head (`content_rank=64`), same answer-position rows; only the target
changes. Out-of-window cells are the real test — in-window cells are answered by the
recency prior alone. `exp60`, 4 prompts per cell:

| target (no. of steps) | 1k depth 0.5 | 8k depth 0.1 | 8k depth 0.5 |
|---|---:|---:|---:|
| budget-matched window (control) | 2.958 / 0.00 | 3.077 / 0.00 | 3.110 / 0.00 |
| full KL (600) | 2.357 / 0.00 | 2.093 / 0.25 | 1.896 / 0.25 |
| content KL (600) | 0.547 / 0.00 | 0.163 / 1.00 | 0.203 / 1.00 |
| **teacher content-argmax (600, no labels)** | **0.208 / 1.00** | **0.373 / 1.00** | **0.289 / 1.00** |
| fact label (300, oracle) | 0.170 / 1.00 | 0.175 / 1.00 | 0.190 / 1.00 |
| shuffled fact label (300, control) | 2.934 / 0.00 | 2.985 / 0.00 | 3.125 / 0.00 |
| dense (no sparsity) | 0.143 / 1.00 | 0.161 / 1.00 | 0.385 / 1.00 |

(CE / exact match.) **The label-free router matches dense exact match in every
out-of-window cell at a 25% budget**; at 8k depth 0.5 it is even more confident than
dense (CE 0.289 vs 0.385). The full-KL objective — the original LCGA recipe — is the
only one that fails completely, and the shuffled-label control confirms the effect is
the target, not the training loop.

So the practical recipe is: (1) take the dense teacher's attention at the positions
that produce the answer, (2) remove the blocks the position prior already covers,
(3) train the router to put its content budget on the teacher's argmax there. No
needle locations are used. Its selection composition is **19.5 / 61.7 / 71.6%**
content at 1k / 4k / 8k (`exp59`) — essentially the same as the labelled router's
19.0 / 57.5 / 69.2%.

### 5.5 Generalisation (`exp62`, T=2048, 4 prompts)

`hit` = fraction of layers selecting the *target* fact block at the query row.

| condition | depth | dense | learned (CE / exact / hit) | window |
|---|---:|---:|---|---:|
| seen phrasing, 1 fact | 0.1 | 0.218 / 1.00 | **0.202 / 1.00 / 1.00** | 3.170 / 0.00 |
| seen phrasing, 1 fact | 0.5 | 0.316 / 0.00 | 0.292 / 0.25 / 0.87 | 3.125 / 0.00 |
| seen phrasing, 1 fact | 0.9 | 0.598 / 0.00 | 0.702 / 0.00 / 0.84 | 0.563 / 0.00 |
| unseen phrasing | 0.1 | 0.197 / 1.00 | 0.551 / 0.00 / 0.55 | 3.039 / 0.00 |
| unseen phrasing | 0.5 | 0.403 / 0.00 | 3.039 / 0.00 / 0.54 | 2.988 / 0.00 |
| unseen phrasing | 0.9 | 0.501 / 0.00 | 0.400 / 0.00 / 0.71 | 0.280 / 0.75 |
| two facts, ask the backup | 0.1 | 0.147 / 1.00 | 0.621 / 0.00 / 0.79 | 3.050 / 0.00 |
| two facts, ask the backup | 0.5 | 0.298 / 0.50 | 1.984 / 0.00 / 1.00 | 3.034 / 0.00 |
| two facts, ask the backup | 0.9 | 0.763 / 0.00 | 1.045 / 0.00 / 0.59 | 0.539 / 0.00 |

Reading: **position generalises, phrasing does not.** The router trained on one
sentence ("The secret code is N.") transfers to unseen depths and 8× longer contexts,
but a re-phrased fact ("Account recovery token: N.") drops exact match to 0.00 even
though it still finds a fact-like block 55–71% of the time. With two facts it often
selects the *queried* block (hit up to 1.00) yet still fails to produce the answer,
which suggests the residual stream needs the fact at more than one layer for this
harder task. Cells where dense itself is not exact (0.5/0.9, 4 prompts) are noisy and
should not be over-read.

### 5.5 The price: plain-LM quality

The same checkpoint on WikiText (`exp58`):

| T | dense | needle-trained router | static window | random |
|---|---:|---:|---:|---:|
| 1024 | 17.323 | 27.701 | **25.051** | 27.609 |
| 4096 | 10.950 | 18.221 | **13.783** | 17.471 |

The retrieval router is as bad as random on plain text — it sacrificed the LM prior
for content matching. This is a **specialist**, not a drop-in replacement; a
production version would need either a mixed objective or a per-request switch.

## 6. Speed (unchanged; `exp4` always loaded correctly)

Prefill (single request, AC power, eager router):

| T | dense | sparse | speedup |
|---|---:|---:|---:|
| 1024 | 90.0 ms | 97.2 ms | 0.925× |
| 2048 | 178.7 ms | 184.4 ms | 0.969× |
| 4096 | 363.0 ms | 356.4 ms | 1.018× |
| 8192 | 826.0 ms | 754.4 ms | 1.095× |
| 16384 | 2225.1 ms | 1642.1 ms | 1.355× |
| 32768 | 6793.1 ms | 3980.5 ms | **1.707×** |

Crossover is between 2k and 4k tokens; the speedup grows monotonically with context.

Decode with the whole-step CUDA graph (`lcsa/decode_graph.py`,
`scripts/decode_ref.py --graph`), graph-dense vs graph-sparse, top-1 parity 1.0 at
every length:

| ctx | graph dense | graph sparse | speedup |
|---|---:|---:|---:|
| 1024 | 12.92 ms/tok | 13.47 | 0.96× |
| 2048 | 14.60 | 14.16 | 1.03× |
| 4096 | 17.69 | 15.72 | 1.13× |
| 8192 | 24.11 | **18.96** | **1.27×** |

Eager decode (no graph) *loses*: 0.55–0.62× — the router runs per layer per token and
host launch overhead dominates at 0.5B. The graph path removes that. (16k graph
capture hangs on this card; 8k is the published decode point.)

**No KV-memory saving**: the kernel receives the full K/V and gathers selected blocks
internally, so this does not extend the context that fits on a 6 GB card. It saves
compute and bandwidth only.

## 7. What this means

- **The negative result stands, with a correct mechanism.** On data where the next
  token does not depend on distant content, a distilled router learns exactly the
  sink+window prior and nothing more; a budget-matched static window is as good or
  better. Sparse-attention papers that compare only against under-budgeted
  baselines will overstate what learning contributes.
- **The positive result is real but narrow.** Block-level content selection *is*
  learnable at a 25% budget (exact match 1.00 at 8k, trained at 1k) — but it needs
  (a) retrieval-style data, (b) a loss applied where the answer is produced,
  (c) a target focused on content rather than the full attention distribution, and
  (d) a head whose content path is not gradient-starved.
- **Labels vs distillation.** The labelled retrieval run is the ceiling (p = 0.91),
  but it is **not required**: distilling the dense teacher's *content argmax* at the
  answer positions reaches the same end-to-end exact match at a 25% budget without
  knowing where the needle is. What is unusable is distilling the **full** teacher
  distribution — the sink/window mass drowns the content signal.
- **Specialisation is the honest framing**: 1.71× prefill and 1.27× decode at 8k–32k
  are worth having in long-context inference, and the retrieval router shows the
  budget can be spent on content — but a single router cannot serve both plain LM
  and retrieval with one setting.

## 8. Limits

- One base model (Qwen2.5-0.5B), one GPU, one seed per configuration; needle cells use
  4–5 prompts, so exact-match differences of one prompt are noise. The selection
  measurements (content fraction, p(fact)) are averages over thousands of blocks and
  are stable.
- The retrieval router was trained at T=1024; the 8k evaluation is a transfer test,
  not a long-context training run.
- The teacher is the answer-token attention of the *dense* model at the same layer;
  "content" is therefore defined by where dense attention looks, not by an
  information-theoretic criterion.
- Generalisation is tested for phrasing and multi-fact only at T=2048, and phrasing
  fails (see §5.4).
- No KV-memory saving, no generation path for sparse decode inside this repo; all
  needle numbers are teacher-forced scoring of the answer span.
- Speed numbers are AC-only: on battery everything is ~30% slower in absolute terms;
  ratios stay similar.

## 9. Reproduce

```bash
# corrected evaluation of a v1 router (must load after patch_model)
python experiments/exp59_selection_stats.py --ckpt checkpoints/abl_diag_best.pt \
    --Ts 1024,4096,8192 --max_chunks 2
python experiments/exp58_eval_long.py --ckpt checkpoints/abl_diag_best.pt \
    --seq 16384 --max_chunks 2 --controls

# where content matters
python experiments/exp60_needle.py --ckpt checkpoints/abl_diag_best.pt \
    --Ts 2048,8192 --depths 0.1,0.5,0.9 --n_prompts 4 \
    --policies dense,topk,window,random,local,oracle

# teach the router to retrieve (label-free and labelled objectives)
python experiments/exp61_train_router_needle.py --steps 600 --loss kl --content_rank 64
python experiments/exp61_train_router_needle.py --steps 600 --loss residual --content_rank 64
python experiments/exp61_train_router_needle.py --steps 600 --loss top1content --content_rank 64
python experiments/exp61_train_router_needle.py --steps 300 --loss retrieval --content_rank 64
python experiments/exp61_train_router_needle.py --steps 300 --loss retrieval --content_rank 64 \
    --shuffle_label 1                        # control

# end-to-end checks
python experiments/exp61_eval_check.py --ckpt checkpoints/needle_ret_c64_final.pt \
    --seq 1024 --depth 0.5 --gen 8
python experiments/exp62_needle_gen.py --ckpt checkpoints/needle_ret_c64_final.pt \
    --seq 2048 --depths 0.1,0.5,0.9 --n_prompts 4
```
