## Latest findings (25 Oct 2026) — corrected, extended, and a working recipe

> **Correction first.** An earlier note said the router selects **0.0%** content blocks
> and is *identical* to a budget-matched static window. Both numbers came from
> evaluation scripts that loaded the checkpoint **before** patching the model, so an
> uninitialised router was being measured. The scripts are fixed and every number
> below is re-measured. (The older `exp3` PPL table — 14.30 dense / 21.84 learned —
> was never affected.)

**1. On plain language modelling the negative result stands.** At a 25% block budget
the learned router is a position prior: content blocks are **1.7 / 6.0 / 7.0 / 6.7%**
of its selections at 1k / 4k / 8k / 16k, and perplexity is **25.166 / 13.788 / 12.176
/ 12.001** versus a budget-matched static window's **25.051 / 13.783 / 12.160 /
11.868** — the learning adds nothing (it is 0.005–0.133 PPL *worse*). What helps is the
budget: the gap to dense falls from +45% (1k) to +11% (16k) at a fixed 25% of blocks.
On WikiText the dense teacher itself puts almost no mass on distant blocks, so there is
nothing to distil — the degeneration is correct behaviour, not a training failure.

**2. Retrieval is the regime where content matters.** Needle at 8k, 25% budget: dense
answers exactly (exact **1.00**, CE 0.14–0.39), the budget-matched window cannot
(exact **0.00**, CE ≈ 3.0). Perplexity can never show this.

**3. The router can learn it — and the three ingredients are separable**:

| ingredient | good | bad | needle exact |
|---|---|---|---|
| where the loss is applied | answer positions | all query rows | 1.00 vs 0.75 (8k), 1.00 vs 0.00 (1k) |
| what the target is | teacher's content argmax | full attention distribution | works vs 0.11 p(fact), fails |
| router head | + bilinear content path | concept graph only | works vs p=0.066, fails |

The full distribution fails because ~50% of the teacher's mass at the answer rows sits
on the sink block and ~20% on the query block; the content signal is a few percent of
the gradient. Restricting the KL to content blocks helps at 8k but not at 1k (the
teacher's content mass is still diffuse). Focusing on the teacher's content **argmax**
works at both, and **needs no needle labels**.

**4. Label-free result, 3 seeds × 6 prompts** (trained at T=1024, evaluated at 1k/8k):

| cell | dense | **label-free router** | budget-matched window |
|---|---:|---:|---:|
| 1k, depth 0.5 (out of window) | 1.00 | **1.00 (1.00–1.00)** | 0.00 |
| 8k, depth 0.1 (out of window) | 1.00 | **1.00 (1.00–1.00)** | 0.00 |
| 8k, depth 0.5 (out of window) | 1.00 | **1.00 (1.00–1.00)** | 0.00 |
| 1k, depth 0.9 (**in** window) | 1.00 | 0.39 (0.00–1.00) | 1.00 |
| 8k, depth 0.9 (**in** window) | 1.00 | 1.00 | 1.00 |

Selection: **24/24 layers** pick the fact block at the query row (17/24 on a second
probe); content blocks rise to **19.5 / 61.7 / 71.6%** (1k/4k/8k); greedy generation
with full re-prefill produces the same answer as dense (`7311`). At 8k the budget can
be cut to **6.25%** of blocks (8 of 128) and retrieval is still exact.

**5. Controls and mechanism.** Shuffled-label training learns nothing (p(fact) 0.067 =
chance). The teacher's own per-instance content argmax points at the fact in only
**15.3%** of (layer, sample) pairs, yet the student selects it in **71%** of layers: a
shared router averages out per-instance teacher noise and learns the consistent part
(the fact sentence). The softmax temperature drifts under a hard argmax target, so
p(fact) flattens after training while the **ranking** (what top-k uses) stays correct —
selection rate and exact match are the metrics to trust.

**6. One router for both behaviours (the practical result).** Mixing one needle step
per seven LM steps (`mix7`, 900 steps) keeps retrieval where the window fails **and**
keeps plain-LM quality:

| schedule | 1k d0.5 | 1k d0.9 | 8k d0.1 | 8k d0.5 | PPL 1k | PPL 4k |
|---|---:|---:|---:|---:|---:|---:|
| needle only | 1.00 | 0.17 | 1.00 | 1.00 | 27.70 | 18.22 |
| 1 needle : 2 LM | 0.83 | 0.17 | 1.00 | 1.00 | 25.87 | 14.79 |
| **1 needle : 7 LM** | **1.00** | **1.00** | **1.00** | **1.00** | **25.66** | **14.15** |
| static window (control) | 0.00 | 1.00 | 0.00 | 0.00 | **25.05** | **13.78** |
| dense | 1.00 | 1.00 | 1.00 | 1.00 | 17.32 | 10.95 |

Seeds 0/1/2 give **15/15 cells** at 1.00 (PPL 25.65–25.68 across seeds; 12.46 at 8k vs
the window's 12.16). **+0.60 PPL at 1k buys exact retrieval on every out-of-window
cell** at a 25% (or 6.25%) compute budget. Warm-starting the LM router on needles
instead does *not* work: p(fact) 0.28, PPL 27.0.

Adding multi-fact + paraphrase training on top (`mix7_nf2`) keeps 5/5 cells at 1.00 and
PPL **25.648**, and raises the held-out-phrasing and two-fact cells at 2k depth 0.1 to
**1.00** (three facts 0.70). That is the better general-purpose configuration.

**7. Limits, stated plainly.**
- Transfer stops at **8k**: 2k fails (exact 0.00 even when trained at 2k; the sparse
  path itself is bitwise-faithful at 2k, so this is a router effect we did not
  explain), 16k fails, and at 4k the 0.5B dense model itself fails.
- A re-phrased fact scores exact **0.00** with the single-phrase model. Multi-fact +
  paraphrase training (`mix7_nf2`) reaches 1.00 on the held-out phrasing and two facts
  at 2k depth 0.1 (three facts 0.70), and keeps 5/5 cells + PPL 25.65.
- The router selects **fact-like content, not the block that answers the question**:
  with two facts both are selected regardless of the query.
- **Scale is not demonstrated**: the same recipe on 1.5B is inconclusive at 1k (its
  dense model does not retrieve, so the teacher is noise) and scores 0.00 at 2k/8k when
  trained at 2k.
- No KV-memory saving — compute/bandwidth only, so the context that fits on a 6 GB
  card does not grow.
- One base model (0.5B), one task family (4-digit needle); all needle numbers are
  teacher-forced scoring of the answer span plus one greedy-generation check.

**8. Speed is unchanged.** Prefill **1.71× at 32k**; graph decode **1.27× at 8k**;
eager decode loses; no KV-memory saving.

**Trained routers in this repo** (router-only, 22 MB each, fp32):
`router_needle_mix7.pt` (single-task, +0.60 PPL) and `router_needle_mix7_nf2.pt`
(paraphrase + multi-fact). Load them on the frozen base with
`patch_model(..., budget=0.25, n_concepts=64, graph_rank=0, content_rank=64,
rel_bias=True, capture_dense=False, use_coarse=False, null_init=-1e9)` and copy each
`ckpt["state"][i]` key into `model.model.layers[i].self_attn` (see
`experiments/exp61_eval_check.py`). The frozen base weights are not included.

Full tables, ablations, controls, limits and reproduction commands:
[`LCGA_FINDINGS.md`](https://github.com/byjameson/lcga-local/blob/main/LCGA_FINDINGS.md).
The licence, weights and donation section above are unchanged.
