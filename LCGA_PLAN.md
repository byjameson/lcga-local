# LCGA — research plan for the differentiating line

> Focus shift: stop chasing parameter count (32B/72B/150B) — "train a big model on a
> small card" is already a crowded space. Put the effort where we are actually
> original: **learned block-sparse attention (LCGA) on a frozen base**.

---

## 1. The claim we want to be able to defend

> *A small, trainable block router on a frozen base can cut attention to a
> user-chosen budget ρ of key blocks while keeping long-context quality within a
> few percent of dense, and it pays off in wall-clock only above ~5k tokens.*

Two halves must both be true: **quality within X%** and **speed above T\***. Right
now we have partial evidence for both and no ablation proving the router's most
opinionated part (the concept graph) actually earns its parameters.

## 2. What is already measured (and what it does *not* prove)

| Axis | Measured | Honest reading |
|---|---|---|
| Quality @T=1024, ρ=0.25, 128 chunks | dense **12.46** / sparse **16.56** PPL (v3) | +33% gap at short context — not competitive yet |
| Budget fidelity | ρ=1.0 reproduces dense (12.854 vs 12.796) | the formulation is faithful, not a source of drift |
| Long-context trend | relative loss +27.8% @1k → **+7.5% @8k** (v2) | the method's regime is long context |
| Short budget | ρ=0.125 selection is **bit-identical** to "sink+local" | content routing cannot act below m≈4 |
| Oracle vs trained @T=4096 | oracle 14.36 vs trained 15.56 | ~1.2 PPL of headroom in block *selection* |
| Prefill speed | crossover ≈**5k** tokens; **1.25×** @16k | speed only in the long-context regime |
| Decode (CUDA graph) | **1.12×** @4k, **1.27×** @8k vs graph-dense; dense bitwise parity | real, but small |

**What this does not prove:** that the low-rank concept graph (`diag + U Vᵀ`) is the
reason it works. A diag-only router, a plain content-bilinear router, or even the
static "sink+local" prior could match it at ρ=0.25. Until that ablation exists, the
"concept graph" is an untested design choice, not a contribution.

## 3. The three gaps

1. **No parameter-matched ablation** isolating the concept graph
   (`--graph_rank 0`, `--content_rank`, `--no_rel_bias`, `--dynamic` already exist
   in `exp2_train_router.py`, so this is cheap to run).
2. **No long-context quality evidence at the scale where the speedup lives**
   (8k–16k PPL, plus a retrieval/needle probe).
3. **No memory accounting** — on a 6 GB card the real prize is the KV working set
   at ρ=0.25 vs dense, which we have never quantified end-to-end.

## 4. Experiment ladder

Each step has a decision rule. If a step fails, we stop the branch rather than
piling more runs on top.

### E1 — Router component ablation (the missing baseline)

**RESULT (measured).** Same test set (128 chunks, T=1024, ρ=0.25), same recipe
(seed 0, 1500 steps, lr 1.5e-3), only the scoring head differs:

| | v1 (graph rank-16) | abl_diag (rank-0) | difference |
|---|---:|---:|---:|
| sparse PPL | 21.838 | **21.837** | −0.001 |
| recall | 0.779 | **0.785** | diag better |
| top-1 agreement | 0.7649 | **0.7674** | diag better |
| TinyStories PPL | 7.963 | **7.941** | diag better |
| ρ=0.125 | 26.788 | 26.788 | identical |

**The concept graph adds nothing** (49k extra parameters/layer total, 0.001 PPL).
The diagonal concept-space bilinear already carries all the signal.

What *does* earn its keep — the learned scorer itself:

| Selection @ρ=0.25 | PPL |
|---|---:|
| dense | 14.30 |
| **learned (diag)** | **21.84** |
| random-k | 23.91 |
| static sink+local | 26.79 |

So learned content selection is real (**−2.1 PPL vs random, −5.0 vs local**), but
the elaborate low-rank routing is not. Revised claim: *a ~64-parameter-per-layer
learned block scorer recovers the signal; the concept graph is dead weight.*

Open follow-up: does the **recency prior** matter (`--no_rel_bias`, running), and
does an extra content shortcut add anything (`--content_rank`)?

**RESULT (recency prior).** Same recipe, diag-only, `--no_rel_bias`:

| Selection @ρ=0.25 (128 chunks) | PPL |
|---|---:|
| dense | 14.30 |
| **learned + recency prior** | **21.84** |
| learned, no prior | 22.81 |
| random-k (budget-matched) | 23.95 |
| static local (sink+diag, 2 blocks) | 26.79 |

Decomposition of the 2.1 PPL gained over random: roughly **half from learned
content (~1.1), half from the recency prior (~1.0)**.

**Methodological caveat (important).** The `local` control always selects only
sink+diagonal (2 blocks), so at ρ=0.25 it is *not* budget-matched (the router uses
~4–8). The fair content baseline is `random-k`, which fills the budget. At ρ=0.125
the two coincide by construction — and there learned == local **exactly** (26.788),
which is the clean degeneration result.

> **UPDATE (25 Oct 2026).** The "vs local" margins above are budget artefacts. Against
> a **budget-matched static window** the learned router is within +0.005…+0.133 PPL
> (slightly *worse*), and it selects content blocks only **1.7 / 6.0 / 7.0%** of the
> time at 1k/4k/8k. See `LCGA_FINDINGS.md` §0–2. Earlier `exp58/exp59/exp60` numbers
> were invalid because the checkpoint was loaded before `patch_model` (so an
> uninitialised router was measured); those scripts are fixed and everything was
> re-measured. The follow-up that *does* work — a retrieval-trained router reaching
> exact match 1.00 at a 25% budget, trained at 1k and evaluated at 8k — is in
> `LCGA_FINDINGS.md` §5.

Train four routers with identical data/steps/lr, differing only in the scoring
head; evaluate all with `exp3_eval.py --max_chunks 128` at ρ=0.25:

```bash
# A: diagonal only (64 params) -- no concept graph
python experiments/exp2_train_router.py --graph_rank 0  --tag abl_diag --steps 1500
# B: LCGA (diag + rank-16 graph) -- current design
python experiments/exp2_train_router.py --graph_rank 16 --tag abl_graph --steps 1500
# C: graph + content shortcut
python experiments/exp2_train_router.py --graph_rank 16 --content_rank 32 --tag abl_content --steps 1500
# D: graph, no recency prior
python experiments/exp2_train_router.py --graph_rank 16 --no_rel_bias --tag abl_norel --steps 1500
```

Evaluate: `python experiments/exp3_eval.py --ckpt <ckpt> --max_chunks 128 --tag <tag>`.

**Decision rule:** if C does not beat B by ≥2% PPL, the concept graph + content
path is not earning its parameters → drop the "concept graph" claim and pivot the
story to *budget-controlled routing with a position prior* (still useful, but a
different paper).

### E2 — Budget generalisation (no training)

Train once at ρ=0.25, evaluate at ρ ∈ {0.125, 0.25, 0.5}; report PPL/recall per
budget. **Rule:** if ρ=0.5 is not close to dense (<5%), the router is
over-specialised to its training budget.

### E3 — Long-context quality (eval only)

**RESULT (learned diag router, ρ=0.25).** Same checkpoint trained at T=1024,
evaluated at longer contexts:

| T | dense PPL | learned PPL | relative loss | recall |
|---|---:|---:|---:|---:|
| 1024 | 14.304 | 21.837 | +52.7% | 0.785 |
| 4096 | 10.950 | 13.788 | +25.9% | 0.836 |
| 8192 | 10.375 | **12.176** | **+17.3%** | **0.859** |

Controls at T=4096 (budget ρ=0.25): **random 17.460** (+59.5%), **static local
29.045** (+165%, recall 0.561). So the learned router's margin over random grows
with context (2.1 PPL @1k → 3.7 PPL @4k) while its gap to dense halves, and the
static recency window *degrades* as context grows.

**This confirms the regime claim quantitatively:** content-based block selection
matters more the longer the context; "just keep the recent blocks" does not scale.

*Measurement note:* the first 8k run OOM'd in the control sweep because a full
`T × vocab` logits tensor is 2.5 GB at T=8192. `lcsa/harness.py::quick_ppl` now
accumulates CE in position chunks (identical result, bounded memory).

### E4 — Cost accounting (eval only) — *corrected after reading the kernel*

`sparse_attention_triton(q, k, v, idx, …)` receives the **full** K/V tensors and
gathers the selected blocks *inside* the kernel. So LCGA reduces **attention
compute and bandwidth**, but it does **not** reduce KV memory: the full cache is
still allocated.

That has a direct consequence for the 6 GB story: LCGA does **not** let you fit a
longer context. It makes a context that already fits *cheaper*. Memory savings
would require not storing the unused KV at all (Quest/MInference-style), which is
a different design and is out of scope here.

Measure (attention-only, T=4k/8k/16k): latency dense vs sparse at fixed ρ, plus
peak VRAM to show the KV bytes are unchanged. **Rule:** the speedup must be
material (≥1.25×) at 8k+, and we state the memory limitation explicitly rather
than implying a capacity win.

## 5. Kill criteria (decide before running, not after)

- E1 shows no component beats diag-only → the "learned content routing" claim is
  out; publish the negative result and keep the budget/position story.
- E3 shows the gap does not shrink by 8k → long context is not our regime; stop.
- E4 shows no meaningful speedup at 8k → the 6 GB angle is gone; stop. (Note: E4
  already established that LCGA is a compute/bandwidth method, **not** a memory
  method — no KV saving is claimed.)

Negative results are publishable here: the ρ=0.125 degeneration to sink+local is
already a clean, reproducible finding.

## 6. Positioning (do not overclaim)

Block-sparse attention is crowded: Longformer/BigBird (static patterns),
Routing/Reformer (hashing), Quest/MInference (KV-cache selection at inference),
NSA (DeepSeek) and MoBA (Moonshot) for long context. **Our claim is narrower and
more practical:** a ~2k-parameter-per-layer router trained *on a frozen 0.5B base*
on a 6 GB laptop GPU, with an explicit budget knob and fully reproducible
single-GPU numbers. The novelty we can defend is *budget-controlled, trainable
selection at this scale with honest negative results*, not "block sparsity" itself.

## 7. What we publish when E1–E4 are done

1. `LCGA_FINDINGS.md` (English): claim, ablations, long-context table, memory
   table, negative results.
2. An HF model card for the winning router checkpoint (`byjameson/lcga-v4-0.5b`).
3. A short blog post; the 6 GB-reproducibility angle is the differentiator.
