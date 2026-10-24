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
This is correct behaviour, not a training failure: on WikiText the dense teacher
itself puts almost no mass on distant blocks.

**2. Retrieval is the regime where content matters.** Needle at 8k, 25% budget, 4
prompts: dense answers exactly (exact **1.00**, CE 0.14–0.39), the budget-matched
window cannot (exact **0.00**, CE ≈ 3.0). Perplexity can never show this.

**3. The router can learn it — and the three ingredients are separable**:

| ingredient | good | bad | effect (needle exact) |
|---|---|---|---|
| where the loss is applied | answer positions | all query rows | 1.00 vs 0.75 (8k), 1.00 vs 0.00 (1k) |
| what the target is | teacher's content argmax | full attention distribution | works vs 0.11 p(fact), fails |
| router head | + bilinear content path | concept graph only | works vs p=0.066, fails |

The full attention distribution fails because ~50% of the teacher's mass at the answer
rows sits on the sink block and ~20% on the query block; the content signal is a few
percent of the gradient. Restricting the KL to content blocks helps at 8k but not at
1k (the teacher's content mass is still diffuse). Focusing on the teacher's content
**argmax** works at both, and **needs no needle labels** — the target is read off the
dense attention.

**4. Label-free result, 3 seeds × 6 prompts** (trained at T=1024, evaluated at 1k/8k):

| cell | dense | **label-free router** | budget-matched window |
|---|---:|---:|---:|
| 1k, depth 0.5 (out of window) | 1.00 | **1.00 (1.00–1.00)** | 0.00 |
| 8k, depth 0.1 (out of window) | 1.00 | **1.00 (1.00–1.00)** | 0.00 |
| 8k, depth 0.5 (out of window) | 1.00 | **1.00 (1.00–1.00)** | 0.00 |
| 1k, depth 0.9 (**in** window) | 1.00 | 0.39 (0.00–1.00) | 1.00 |
| 8k, depth 0.9 (**in** window) | 1.00 | 1.00 | 1.00 |

Selection: **24/24 layers** pick the fact block at the query row; content blocks rise
to **19.5 / 61.7 / 71.6%** (1k/4k/8k). Greedy generation with full re-prefill produces
the same answer as dense (`7311`). The in-window weakness at 1k is a budget artefact
(2 free slots, see 6); at 8k it disappears.

**5. Controls and mechanism.**
- Shuffled-label training learns nothing (p(fact) 0.067 = chance, exact 0.00).
- The teacher's own per-instance content argmax points at the fact in only **15.3%**
  of (layer, sample) pairs, yet the student selects it in **71%** of layers: a shared
  router averages out per-instance teacher noise and learns the consistent part (the
  fact sentence). This is why the objective plateaus in p(fact) while end-to-end
  accuracy saturates — one routed layer is enough.
- The softmax temperature drifts under a hard argmax target, so p(fact) flattens after
  training while the **ranking** (what top-k uses) stays correct. Selection rate and
  exact match are the metrics to trust.

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

**+0.60 PPL at 1k (2.4%) buys exact retrieval on every out-of-window cell**, at a 25%
compute budget. Warm-starting the LM router on needles instead (300 steps) does *not*
work: p(fact) 0.28 and PPL 27.0 — the LM prior fights the new objective.

**7. Limits, stated plainly.**
- Transfer stops at **8k**: at 16k the same router scores exact 0.00 where dense scores
  1.00; at 32k the 0.5B dense model itself fails, so the probe says nothing.
- A re-phrased fact ("Account recovery token: …") and multi-fact contexts are only
  partially handled (`exp62`); numbers in `LCGA_FINDINGS.md` §5.5.
- No KV-memory saving — this is compute/bandwidth only, so it does not extend the
  context that fits on a 6 GB card.
- One base model (0.5B), one task family (4-digit needle), no serving path: all needle
  numbers are teacher-forced scoring of the answer span plus one greedy-generation
  check.

**8. Speed is unchanged.** Prefill **1.71× at 32k**; graph decode **1.27× at 8k**;
eager decode loses; no KV-memory saving.

Full tables, ablations, controls, limits and reproduction commands:
[`LCGA_FINDINGS.md`](https://github.com/byjameson/lcga-local/blob/main/LCGA_FINDINGS.md).
The licence, weights and donation section above are unchanged.
