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

PPL at T=2048/4096/8192 vs dense, plus one retrieval probe (place a unique fact at
position ~2k in an 8k context and ask for it). **Rule:** quality must be within
~10% at 8k, and the probe must not collapse, or the "long-context regime" claim
dies.

### E4 — Memory accounting (eval only)

Measure KV/working-set bytes for dense vs ρ=0.25 at T=4k/8k/16k, and peak VRAM on
this 6 GB card. **This is the number that makes LCGA matter for small GPUs** — it
has never been measured.

## 5. Kill criteria (decide before running, not after)

- E1 shows no component beats diag-only → the "learned content routing" claim is
  out; publish the negative result and keep the budget/position story.
- E3 shows the gap does not shrink by 8k → long context is not our regime; stop.
- E4 shows no meaningful memory saving at 8k → the 6 GB angle is gone; stop.

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
