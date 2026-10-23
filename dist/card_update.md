## Latest findings (25 Oct 2026) — corrected, and a positive follow-up

> **Correction first.** An earlier note said the router selects **0.0%** content
> blocks and is *identical* to a budget-matched static window. Both numbers came from
> evaluation scripts that loaded the checkpoint **before** patching the model, so an
> uninitialised router was being measured. The scripts are fixed and every number
> below is re-measured. (The older `exp3` PPL table — 14.30 dense / 21.84 learned —
> was never affected.)

**1. On WikiText the learned router is essentially a position prior.**
- Content blocks (outside sink + recent window): **1.7 / 6.0 / 7.0%** at 1k / 4k / 8k.
- PPL, 25% budget: learned **25.166 / 13.788 / 12.176 / 12.001** vs budget-matched
  window **25.051 / 13.783 / 12.160 / 11.868** at 1k / 4k / 8k / 16k.
- So learning adds nothing measurable here; the **budget** does. The gap to dense
  falls from +45% (1k) to +11% (16k) at a fixed 25% of blocks.

**2. Where distant content matters, a static window fails.** Needle-in-a-haystack at
8k, 25% budget, 4 prompts: dense exact **1.00**, budget-matched window exact **0.00**
(CE ≈ 3.0). Perplexity can never show this regime.

**3. The router *can* learn content selection — with a retrieval objective.**
Trained at T=1024 only, evaluated at 1k / 2k / 8k:
- exact match equals dense in **7 of 9** cells (8k: CE **0.175** vs dense 0.161 and
  window 3.048);
- **24/24 layers** select the fact block at the query row;
- content blocks rise to **19% / 57.5% / 69.2%** (1k / 4k / 8k; label-free version:
  19.5 / 61.7 / 71.6);
- it transfers to 8k without retraining.

**4. Distillation works — if the target is focused on content.** Same data, same head,
same answer-position loss:

| objective | probability on the fact block |
|---|---:|
| full KL (the v1 objective, at answer rows) | 0.11 |
| KL restricted to content blocks | 0.18 |
| **teacher content-argmax (no labels)** | **0.53** |
| explicit fact label (upper bound) | 0.91 |
| shuffled label (control) | 0.067 = chance |

End-to-end needle result at a 25% budget, out-of-window cells (CE / exact match):

| policy | 1k depth 0.5 | 8k depth 0.1 | 8k depth 0.5 |
|---|---:|---:|---:|
| budget-matched window | 2.96 / 0.00 | 3.08 / 0.00 | 3.11 / 0.00 |
| full KL (v1 objective) | 2.36 / 0.00 | 2.09 / 0.25 | 1.90 / 0.25 |
| content KL | 0.55 / 0.00 | 0.16 / **1.00** | 0.20 / **1.00** |
| **teacher content-argmax (no labels)** | **0.21 / 1.00** | **0.37 / 1.00** | **0.29 / 1.00** |
| dense (no sparsity) | 0.14 / 1.00 | 0.16 / 1.00 | 0.39 / 1.00 |

So the router learns retrieval **without any needle labels**: the target is read off
the dense teacher's attention at the answer positions, with the sink/window part
removed. The original full-KL objective is the one thing that fails. Full attention
distillation is dominated by the sink + recent-window mass that the position prior
already provides.

**5. The head matters as much as the objective.** The concept-graph head is
content-blind at initialisation (softmax over K concepts, block-mean pooled → score
gradient ≈ 1/K²); with a direct bilinear content path the same label run reaches
p = 0.86 in **60 steps** instead of never.

**6. The price.** The needle-trained router is worse on plain text (27.7 vs 25.1 PPL
at 1k — as bad as random). It is a **specialist**, not a drop-in replacement, and it
does not generalise to a re-phrased fact (exact 0.00) or fully to two-fact contexts.
That is a real limit, stated as such.

**7. Speed is unchanged by any of this.** Prefill **1.71× at 32k**; graph decode
**1.27× at 8k**; eager decode loses; still compute/bandwidth only, **no KV-memory
saving**.

Full tables, limits and reproduction commands:
[`LCGA_FINDINGS.md`](https://github.com/byjameson/lcga-local/blob/main/LCGA_FINDINGS.md).
The licence, weights and donation section below are unchanged.
