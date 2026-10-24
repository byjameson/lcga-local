# LCGA — findings (learned block-sparse attention on a frozen 0.5B)

Measured on one RTX 3050 Laptop 6 GB (Windows 11/WDDM), AC power. Base:
Qwen2.5-0.5B, frozen; only the router (and, in stage-2 runs, LoRA) is trained.
Reproduce with `experiments/exp2_train_router.py`, `exp3_eval.py`, `exp4_speed.py`,
`exp58_eval_long.py`, `exp59_selection_stats.py`, `exp60_needle.py`,
`exp61_train_router_needle.py`, `exp62_needle_gen.py`, `exp64_query_swap.py`,
`exp65_teacher_accuracy.py`; raw rows in `results/results.jsonl`, summarised by
`scripts/summarize_needle.py` and `scripts/aggregate_needle.py`. Runs invalidated by
the loader bug are kept for the audit trail in `results/invalid/`.

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
   distil. Perplexity cannot reward looking far away.3. **Retrieval is the regime where distant content matters.** A fact planted at a
   known depth: dense answers it exactly (CE 0.14–0.39 at 1k–8k), a budget-matched
   window cannot (CE ≈ 3.0, exact 0.00).
4. **Content selection is learnable — with the right objective and the right head.**
   Training at the answer positions with a *retrieval* target gives
   **exact match 1.00 at a 25% budget** (dense 1.00, window 0.00) and it transfers
   from 1k training to 8k evaluation. Content blocks rise to **19% / 57.5% / 69.2%**
   (labelled run; the label-free one 19.5 / 61.7 / 71.6%).
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
7. **Generalisation is partial.** Position transfers from 1k training to 8k (three
   seeds, exact 1.00) but **not to 2k or 16k**; a re-phrased fact scores exact 0.00; the
   router selects *fact-like* content, not the block that answers the question
   (§5.5). Multi-fact training helps the multi-fact cases; multi-phrasing training does
   not fix the phrasing gap.
8. **One router can serve both behaviours** (the practical result). Mixing one needle
   step per seven LM steps keeps exact 1.00 in all five cells — including the 1k
   in-window case a needle-only router loses — while plain-LM PPL is only **+0.60
   (1k) / +0.37 (4k)** above the budget-matched window, versus +2.65/+4.44 for
   needle-only training. Seeds 0/1/2 all give 15/15 cells.
9. **Speed is unchanged**: prefill **1.71× at 32k**, graph decode **1.27× at 8k**;
   eager decode loses. Still compute/bandwidth, **not** KV memory.
10. **Scope**: one base model (Qwen2.5-0.5B) for the full ablation grid; the label-free
   recipe is replicated on **Qwen2.5-3B** (dense-level exact match at 1k with a quarter
   of the training tokens, §5.10). The 1.5B host is not usable: its dense attention does
   not retrieve the needle at 1k (§5.10).

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
Because the teacher is computed from the *same* forward pass's q/k (layer-wise
distillation), it is the dense attention over the sparse model's own hidden states,
not an independent dense run — this is the standard setup here and matches v1.

### 5.1 Objectives

| Objective | Target | Label needed | p(fact) | top-1 = fact |
|---|---|---:|---:|---:|
| full KL (v1 objective, answer rows) | whole teacher distribution | no | 0.113 | 0.107 |
| content KL | teacher outside {sink, self, window} | no | 0.183 | 0.362 |
| teacher content-argmax | argmax of the above | no | 0.530 | 0.566 |
| retrieval | the fact block itself | yes (oracle) | **0.912** | **0.918** |
| retrieval, shuffled labels | random block | yes (control) | 0.067 | 0.069 |

Chance is 1/16 = 0.0625. Observations:

**A caveat on p(fact).** Distilling a *hard* argmax target lets the router's learnable
softmax temperature drift upwards, so the reported probability flattens while the
**ranking (which is what top-k selection uses) stays correct**. After training, the
label-free checkpoint shows p(fact) ≈ 0.05–1.00 depending on layer, yet still selects
the fact block in 17/24 layers and answers exactly (below). Treat p(fact) as a
training diagnostic, and **selection rate / exact match as the real metric**.

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

The same comparison with the **label-free** objective (600 steps each) makes the
capacity effect even clearer:

| content path width | p(fact) | 1k depth 0.5 | 8k depth 0.1 | 8k depth 0.5 |
|---:|---:|---:|---:|---:|
| 0 (concept graph only) | 0.066 | 0.764 / 0.00 | 0.185 / 1.00 | 3.150 / 0.00 |
| 16 | 0.430 | 0.206 / 1.00 | 0.320 / 1.00 | 0.543 / 0.00 |
| 64 | 0.530 | 0.208 / 1.00 | 0.361 / 1.00 | 0.285 / 1.00 |

A rank-16 content path is enough at these lengths; the concept-only router cannot
learn the label-free target at all (p = 0.066 with zero gradient movement).

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

**Multi-seed check of the label-free router** (3 seeds × 6 prompts per cell, `exp60`,
25% budget; mean and range of exact match):

| cell | dense | **label-free router** | budget-matched window |
|---|---:|---:|---:|
| 1k, depth 0.5 (out of window) | 1.00 | **1.00 (1.00–1.00)** | 0.00 |
| 8k, depth 0.1 (out of window) | 1.00 | **1.00 (1.00–1.00)** | 0.00 |
| 8k, depth 0.5 (out of window) | 1.00 | **1.00 (1.00–1.00)** | 0.00 |
| 1k, depth 0.9 (**in** window) | 1.00 | 0.39 (0.00–1.00) | 1.00 |
| 8k, depth 0.9 (**in** window) | 1.00 | 1.00 | 1.00 |
| 1k, depth 0.1 (start of context) | 0.00 | 0.00 | 0.00 |

Two honest caveats visible here. At **1k the in-window case is fragile**: the budget
is only m = 4 blocks, of which sink + self are mandatory, so the router has **two free
slots** and must choose between the recent window and distant content — training on
content costs the local case (0.17 exact over 12 prompts for seed 0; 0.39 mean over
three seeds at 6 prompts; the window scores 1.00). At 8k there are 30 free slots and
both fit (1.00/1.00). And the depth-0.1 cell at 1k is one where **dense itself** fails
(CE 0.667, exact 0.00), so it carries no information about the router.

**Transfer stops at 8k.** The same checkpoint (trained at T=1024) at 16k: dense exact
1.00 at depths 0.5/0.9, router exact 0.00 (CE 0.568/0.585). At 32k the 0.5B dense
model itself no longer retrieves (exact 0.00 in every cell), so the probe is
uninformative there. Content selection transfers over an 8× context increase, not 16×.

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

**Where the loss is applied matters as much as what it targets.** The same target and
head trained on *all* query rows (v1-style supervision, block-mass teacher) instead of
the answer rows gives exact **0.75 at 8k and 0.00 at 1k** (p(fact) 0.074, 600 steps) —
the answer-position restriction is worth ~25 points of exact match at 8k and is the
difference between working and not working at 1k.

**The student beats its own teacher's per-instance accuracy.** Measured on the trained
checkpoint (`exp65`, teacher computed from the trained router's own hidden states),
the teacher's content argmax equals the planted fact block in only **15.3%** of
(layer, sample) pairs, yet the student selects the fact block in **71%** of layers on
a probe sample and answers exactly. The student is shared across samples, so it
averages out the teacher's per-instance noise and learns the *consistent* part (the
fact sentence), which is exactly what we want. This also explains why the objective
plateaus in p(fact) while the end-to-end metric saturates: 1% of layers is enough to
route the fact.

### 5.5 Generalisation, and what the router keys on (`exp62`, `exp64`)

**Length.** The 1k-trained router transfers to 8k (three seeds, exact 1.00, §5.3) but
**not to 2k**: with dense at 1.00 the router scores 0.00 (CE 0.49–0.96) at all three
depths, and raising the budget to 0.5 does not fix it. The failure is not in the
selection — at 2k depth 0.1, **16/24 layers still select the fact block** — but in
using it (CE 0.54 vs dense 0.28). Training *at* 2k (`seq2k`, 600 steps) does not fix
2k either (exact 0.00), though it does improve 16k: depth 0.5 goes 0.00 → **0.50**
while 8k stays 1.00. A kernel-parity check at ρ = 1.0 (all blocks selected) shows the
sparse path matches dense at 2k (top-1 agreement 0.998, CE difference 0.0002), so this
is not a sparse-path bug. The 2k anomaly is therefore a property of the router's
selection/use at that length and remains unexplained. At 16k the 1k-trained router fails outright, and at 4k the 0.5B **dense
model itself** stops retrieving (exact 0.00–0.25), so that cell says nothing about the
router.

**Phrasing.** A held-out phrasing ("Account recovery token: N.", `exp62`, T=2048, 6
prompts) drops exact match to **0.00** while the router still finds a fact-like block
31–55% of the time (hit rate). Training on three matched phrasings (`para900`, 900
steps) does **not** fix it — the held-out phrasing is still 0.00 — and it costs
accuracy on the trained phrasings (8k depth 0.5: 0.67 vs 1.00 for the single-phrase
model). At this size and step budget, more phrasings diluted the signal rather than
generalising it.

**Multiple facts and the question.** `exp64` plants two facts with distinct phrasings
in the same context and asks about each in turn. At 2k the budget has 6 free slots, so
the selection is **not query-specific**: `hit(secret) = 0.83` (needle-only model) and
`0.92` (mix7) regardless of which fact the question asks about, while `hit(backup)`
stays at 0.46–0.79. The answer then depends on whether the queried fact happens to be
inside the selection: at depths 0.2 and 0.8 mix7 answers *both* questions exactly, but
at depth 0.5 the backup question fails (CE 0.509) because the backup block is selected
in only 54% of layers. The router learns "select fact-like content", not "select the
block that answers this question"; with the single-fact task (and with as many facts as
the free slots can hold) that is sufficient.

Training with multiple facts per context (`nf2`, paraphrase on, two facts, 900 steps)
is the obvious remedy and it does improve the multi-fact cells: on the same 2k probe
the three-fact condition goes from exact 0.00 to **1.00** at depth 0.1 and the held-out
phrasing reaches **1.00** there as well (10 prompts: seen 0.70, held-out phrasing
**1.00**, two facts **1.00**, three facts **1.00**, hit 0.34–0.80). It does not turn the
router into a query-selective one — it makes the fact-like blocks it already selects
sufficient for the budget. `nf2` is needle-only, so it loses the 1k in-window case
(0.00). Combining the two (`mix7_nf2` = `--lm_every 7 --paraphrase 1 --n_facts 2`)
keeps 5/5 cells at 1.00 and PPL 25.648, and reaches **1.00** on the held-out phrasing
and two facts at 2k depth 0.1 (three facts 0.70) — it is the better general-purpose
configuration, mix7 the cleaner single-task one.

**Query pooling does not fix it either — three axes, same answer.** The router in the
configuration we train (`graph_rank=0`, `content_rank=64`) takes its scores from the
*bilinear content path*, so the concept-query pooling is bypassed entirely; the only
query pooling that matters is `h_b` in that path. Making it use the **last token** of the
block (the one position in a causal block that can attend to the whole block) changes
behaviour but not selectivity:

| run (3 facts, 2 free slots, labels, 900 steps) | three-fact cell @2k d0.1 | single-fact 8k d0.1 | swap hits (secret / backup) |
|---|---:|---:|---|
| mean-pooled query | 0.00 | **1.00** | secret 0.58–0.71, backup 0.83–0.96 **for both questions** |
| last-token query | **0.90** | 0.00 | secret 0.46–0.58, backup 0.88–0.96 **for both questions** |

So the three axes we can control — **objective** (teacher argmax vs labels), **budget**
(loose vs binding) and **query pooling** (mean vs last) — all leave the router selecting
fact-like blocks rather than the queried one. The remaining suspects are upstream of the
router: whether the 0.5B hidden states at the router's input layers encode "which fact is
being asked" strongly enough, and whether a single bilinear block score can express it.
That is a modelling question at this scale, not a tuning knob.

Cells where dense itself is not exact (T=2048 depths 0.5/0.9, 6 prompts) are noisy and
should not be over-read.

### 5.6 The price: plain-LM quality

The same checkpoint on WikiText (`exp58`):

| T | dense | needle-trained router | static window | random |
|---|---:|---:|---:|---:|
| 1024 | 17.323 | 27.701 | **25.051** | 27.609 |
| 4096 | 10.950 | 18.221 | **13.783** | 17.471 |

The retrieval router is as bad as random on plain text — it sacrificed the LM prior
for content matching. **Warm-starting from the LM-trained router does not fix it**:
fine-tuning `abl_diag_best` with the needle objective for 300 steps gives p(fact) =
0.28 and PPL 27.001 (1k) / 17.095 (4k), worse than training from scratch on both axes
— the LM prior fights the new objective. This is a **specialist**, not a drop-in
replacement; serving both behaviours needs a task switch or a mixed objective (§7).

### 5.7 Mixed objective: one router, both behaviours

Mixing the two step types (`--lm_every K`: one needle step per K+1, the rest plain-LM
full-KL steps on WikiText) resolves the specialist trade-off. 900 steps each, same
head (`content_rank=64`), 25% budget:

| schedule | 1k d0.5 | 1k d0.9 (in-win) | 8k d0.1 | 8k d0.5 | PPL 1k | PPL 4k |
|---|---:|---:|---:|---:|---:|---:|
| needle only | 1.00 | 0.17 | 1.00 | 1.00 | 27.70 | 18.22 |
| 1 needle : 2 LM | 0.83 | 0.17 | 1.00 | 1.00 | 25.87 | 14.79 |
| **1 needle : 7 LM** | **1.00** | **1.00** | **1.00** | **1.00** | **25.66** | **14.15** |
| static window (control) | 0.00 | 1.00 | 0.00 | 0.00 | **25.05** | **13.78** |
| dense | 1.00 | 1.00 | 1.00 | 1.00 | 17.32 | 10.95 |

**The 1:7 schedule is the practical answer.** It retrieves exactly where the static
window cannot (four out-of-window cells, 0.00 → 1.00), keeps the in-window case the
needle-only router loses, and costs **+0.60 PPL at 1k / +0.37 at 4k** against the
budget-matched window. The mechanism is the same one that explains the needle-only
failure (§5.8): the LM steps keep the recency prior strong, so at 1k the two free slots
are split between the recent window and distant content; with needle-only training the
content score scale overrides the prior and both slots go to content.

Robustness: seeds 0, 1 and 2 all give exact **1.00 in all five cells (15/15)**, and a
10-prompt re-run of seed 0 gives 1.00 in all five as well. PPL is stable across seeds:
**25.651 / 25.684 / 25.655** at 1k (window 25.051) and **12.459** at 8k (window 12.160).
The 1:2 schedule is included to show the trend (more needle steps → better 1k retrieval
is *not* what happens: fewer LM steps hurt the prior, so 1:2 scores 0.83 where 1:7
scores 1.00). Selection composition for the mixed router (`exp59`): content **14.1 /
38.5 / 40.6%** at 1k / 4k / 8k — lower than the needle-only model's 19.5 / 61.7 /
71.6%, which is exactly what "the LM steps keep the prior" predicts.

**The budget can be much smaller at 8k.** Evaluating the same checkpoint with a reduced
budget (`exp60 --budget`, no retraining): at 8k, ρ = 0.125 (16 of 128 blocks) and even
**ρ = 0.0625 (8 of 128 blocks, i.e. 6.25% of the context)** still give exact **1.00**
at depths 0.1 and 0.5, where the budget-matched window scores 0.00. So the retrieval
result is not an artefact of the 25% budget: a sixteenth of the context is enough at
8k.

**Combining everything** (`mix7_nf2` = mixed schedule + paraphrase + two facts per
context, 900 steps): 5/5 cells at exact 1.00 and PPL **25.648**, i.e. the same LM cost
as mix7, while the held-out phrasing and the two-fact condition at 2k reach **1.00**
(three facts 0.70). It is the better general-purpose configuration; mix7 is the
cleaner single-task one.

### 5.8 Budget and the prior: why more slots did not help

At 1k the label-free router loses the in-window case (0.17 exact over 12 prompts)
while the static window scores 1.00. The intuitive fix — give the router more slots —
does not work: at ρ = 0.375 (m = 6, four free slots) p(fact) *drops* to 0.358 and the
in-window case stays at exact 0.00 (CE 1.978 at 1k depth 0.9, 12 prompts). The reason
is visible in the score composition: training the content path pushes its score scale
up until it overrides the fixed recency prior, so the router stops ranking recent
blocks highly at all; extra slots simply go to more content blocks. The fix has to be
in the objective or the data:

- adding a KL term on the **non-content** part of the teacher (`--lam_window 1.0`,
  600 steps) restores the local case (1k depth 0.9 exact 1.00 over 12 prompts) but
  costs out-of-window retrieval (8k depths 0.1/0.5: 0.50 / 0.00) — the two terms
  compete on every needle step;
- mixing the **data** instead (§5.7, 1 needle : 7 LM) gives 1.00 in *every* cell while
  keeping PPL within +0.6 of the window: the LM steps calibrate the prior without
  competing with content on the needle steps.

### 5.9 Recipe (the practical takeaway)

```text
data      needle/QA contexts PLUS a majority of plain-LM steps (1 : 7). Needle-only
          training works but is a specialist (PPL 27.7); the LM steps keep the
          position prior calibrated and cost 0.6 PPL instead of 2.7
loss      answer positions only; target = teacher's content argmax outside
          {sink, self, last m-1 blocks}, renormalised; skip rows with content mass < 0.03
head      bilinear content path (content_rank 16-64); the concept graph alone is
          gradient-starved (~1/K^2) and does not learn the label-free target
hyper     900 mixed steps, batch 2, seq 1024, lr 3e-3, budget 0.25 (m=4, 2 free slots)
expect    needle exact match = dense in every tested cell at 1k and 8k, where the
          budget-matched window scores 0.00 out of window;
          plain-LM PPL 25.66 / 14.15 (window 25.05 / 13.78)
limit     transfer stops at 8k (16k fails); re-phrased and multi-fact contexts are
          partial; no KV-memory saving
```

The code is `experiments/exp61_train_router_needle.py`; the three ingredients can be
switched independently (`--loss kl|residual|top1content|retrieval --rows answer|all
--content_rank N`) so the contribution of each is measurable, and `exp60/exp62/exp64`
evaluate them without training.

### 5.10 Scale: the same recipe on a 3B model

**Prerequisite.** A scale test is only meaningful if the *teacher* is informative at the
training length. `scripts/probe_dense.py` checks that with no checkpoint and no router:
Qwen2.5-3B retrieves the needle densely at 1k (exact 1.00 at depths 0.1/0.5) while the
budget-matched window scores 0.00 — so 3B is a valid host. (The 1.5B is not: its dense
model scores exact 0.00 at 1k, which is why that test was inconclusive and why the 2k
retry was confounded by the 2k anomaly.)

**Result.** Same label-free objective (`top1content`), `content_rank=64`, seq 1024, but
**batch 1 and 300 steps = 307k tokens** (the 0.5B run used 1.23M — 4× more):

| model | tokens | p(fact) | top-1 | 1k d0.5 (CE / exact) | window | 8k |
|---|---:|---:|---:|---|---:|---:|
| 0.5B | 1.23M | 0.53 | 0.57 | 0.208 / 1.00 | 2.96 / 0.00 | 1.00 |
| **3B** | **0.31M** | 0.45 (at 150 steps) | 0.50 | **0.172 / 1.00** | 3.18 / 0.00 | not measured |

(The 3B 8k eval spills on the 6 GB card — 5.75 GB of weights plus an 8k attention
workspace — so only the 1k point is reported; the 8k transfer itself was established on
the 0.5B.)

So the label-free content router is **not a 0.5B artefact**: on a 6× larger model it
reaches dense-level exact match at a quarter of the training tokens, where the
budget-matched window scores 0.00. Note that the 3B training does **not** fit cleanly on
the 6 GB card (7.4 GB allocated → WDDM spill, ~15 s/step), so its *speed* is not a
result; only the quality is.

## 6. Speed (unchanged; `exp4` always loaded correctly)Prefill (single request, AC power, eager router):

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
- **Scale is partly demonstrated.** The label-free recipe replicates on **Qwen2.5-3B**
  (dense-level exact match at 1k, a quarter of the training tokens, §5.10). The
  **1.5B** host is a dead end: its dense model does not retrieve the needle at 1k
  (exact 0.00), so the teacher is noise there; trained at 2k (where its dense model does
  retrieve) it reports a good training metric (p(fact) 0.52) but exact 0.00 at eval —
  confounded by the same 2k anomaly the 0.5B shows. A kernel-parity check at ρ = 1.0
  rules out a sparse-path bug (sparse ≈ dense, top-1 agreement 0.998–0.999).
- The full ablation grid (head, loss placement, budget, seeds) is 0.5B only; 3B is a
  single confirmation run at 1k.
- The retrieval router was trained at T=1024; the 8k evaluation is a transfer test,
  not a long-context training run. Transfer **fails at 2k and at 16k** and is
  uninformative at 4k (dense itself fails there) — the length curve is not monotone
  and we did not find out why 2k is worse than 1k and 8k (§5.5).
- The router selects fact-like content, **not the block that answers the question**:
  with two facts both are selected regardless of the query; making the budget bind
  (three facts), using labels instead of teacher targets, and switching the content
  path's query pooling to the last token all leave it non-selective (§5.5). The cause is
  upstream of the router's objective. A held-out phrasing scores exact 0.00.
- At 1k the budget leaves only two free slots (m = 4, sink + self mandatory), so
  content selection trades against the recent window: the label-free router scores
  0.39 exact on the in-window case where the static window scores 1.00. At 8k both
  fit and the conflict disappears. A budget of 0.375 at 1k is tested in §5.4's
  follow-up runs (see `results/`).
- Plain-LM quality collapses under needle-only training (27.7 vs 25.1 PPL at 1k) and
  warm-starting from the LM router does not rescue it (§5.6). The mixed objective is
  an open item.
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
