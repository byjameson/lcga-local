# LCGA — findings (learned block-sparse attention on a frozen 0.5B)

Measured on one RTX 3050 Laptop 6 GB (Windows 11/WDDM). Base: Qwen2.5-0.5B, frozen;
only the router (and, in stage-2 runs, LoRA) is trained. Everything below is
reproducible from `experiments/exp2_train_router.py`, `exp3_eval.py`,
`exp4_speed.py`; raw rows in `results/results.jsonl`.

---

## TL;DR

1. **A ~64-parameter-per-layer learned block scorer works, and its advantage grows
   with context.** At a fixed 25% block budget the gap to dense shrinks from
   **+52.7%** PPL (T=1024) to **+17.3%** (T=8192), while its margin over random
   selection grows from **2.1** to **3.9** PPL.
2. **Static recency windows do not scale**: sink+diagonal selection degrades from
   26.8 (1k) to **32.1** (8k) PPL — i.e. `local` is not a viable baseline at long
   context.
3. **The "concept graph" is dead weight.** Removing the low-rank off-diagonal term
   (49k params total) changes PPL by **0.001** (21.838 → 21.837); a diagonal
   bilinear scorer is enough.
4. **Speed is modest and decode is not free.** Prefill is only **1.11×** at 8k
   (0.92× at 1k); eager decode is **slower** (0.60–0.63×) because the router adds
   ~per-token launch overhead. Earlier 1.12–1.27× decode numbers require the
   CUDA-graph path (`lcsa/decode_graph.py`), not the eager harness.
5. **No memory saving.** The Triton kernel receives the **full** K/V and gathers
   selected blocks internally, so LCGA reduces *compute and bandwidth*, not KV
   memory. It does not let a 6 GB card fit a longer context.

---

## 1. Quality vs context (ρ = 0.25)

Same checkpoint trained at T=1024, evaluated at longer contexts (128 chunks at
1k, 4 at 4k, 2 at 8k; single seed — treat 4k/8k as low-n).

| T | dense | **learned** | random-k | static local | learned vs dense | recall (learned / random / local) |
|---|---:|---:|---:|---:|---:|---|
| 1024 | 14.304 | **21.837** | 23.945 | 26.789 | +52.7% | 0.785 / 0.742 / 0.721 |
| 4096 | 10.950 | **13.788** | 17.460 | 29.045 | +25.9% | 0.836 / 0.687 / 0.561 |
| 8192 | 10.375 | **12.176** | 16.054 | 32.140 | **+17.3%** | 0.859 / 0.668 / 0.440 |

The learned-vs-random recall margin grows **0.043 → 0.149 → 0.191**: the router
is finding content that random selection misses, and it matters more as the
context gets longer. The router was trained with 16 blocks (T=1024) and still
works with 128 blocks (T=8192) — no retraining.

## 2. Budget knob (learned router)

| ρ | 1k PPL (rel.) | 8k PPL (rel.) |
|---|---|---|
| 0.125 | 26.789 (+87%) | 13.691 (+31.9%) |
| 0.25 | 21.837 (+52.7%) | 12.176 (+17.3%) |
| 0.5 | 17.198 (+20.2%) | 11.340 (+9.3%) |
| 1.0 | 14.303 | 10.376 |

Monotone and smooth: the budget is a real knob, and both the relative gap and the
benefit of selection shrink as the budget grows — as expected.

## 3. Architecture ablation (T=1024, ρ=0.25, 128 chunks)

Identical recipe (seed 0, 1500 steps, lr 1.5e-3); only the scoring head differs:

| Variant | params (all layers) | PPL | recall |
|---|---:|---:|---:|
| graph rank-16 + recency prior (v1) | 2,804,280 | 21.838 | 0.779 |
| **diag only + recency prior** | 2,755,128 | **21.837** | **0.785** |
| diag only, **no** recency prior | 2,755,104 | 22.812 | 0.754 |
| random-k (budget-matched) | 0 | 23.945 | 0.742 |

Reading:
* the **concept graph buys nothing** (0.001 PPL, and slightly worse recall);
* the **recency prior is worth ~1.0 PPL** (21.84 vs 22.81);
* learned **content** routing is worth ~1.1 PPL over random (22.81 vs 23.95);
* so the 2.1 PPL total gain over random splits roughly half content, half prior.

At ρ=0.125 the learned router is **bit-identical** to static sink+diagonal
(26.788 both) — with a 2-block budget the mandatory blocks consume the whole
budget, so content routing cannot express itself.

## 4. Speed (same checkpoint, eager path, budget 0.25)

Prefill (single request):

| T | dense | sparse | speedup | router | sparse kernel |
|---|---:|---:|---:|---:|---:|
| 1024 | 88.4 ms | 96.3 ms | 0.918× | 9.9 ms | 3.3 ms |
| 2048 | 178.7 ms | 184.1 ms | 0.971× | 12.3 ms | 8.6 ms |
| 4096 | 363.6 ms | 357.0 ms | 1.018× | 22.4 ms | 27.3 ms |
| 8192 | 834.3 ms | 753.2 ms | **1.108×** | 45.7 ms | 98.5 ms |

Decode (eager, per token): 0.63× (1k), 0.61× (2k), 0.60× (4k) — **slower**, because
the router runs on every layer for every token and the host-side launch overhead
dominates at 0.5B. The earlier 1.12–1.27× decode numbers come from the CUDA-graph
path compared against a CUDA-graph dense baseline; they are not comparable to this
eager measurement.

## 5. What this is and is not

**Is:** a measured, reproducible study of *learned, budget-controlled block
selection* on a frozen small base, on a 6 GB laptop GPU. The interesting, robust
finding is the interaction with context: learned selection looks progressively
better as context grows, while a static local window collapses.

**Is not:** a speed or memory win at this scale. Prefill breaks even around 4k and
is only 1.11× at 8k; eager decode loses. The concept graph, which was the
"interesting" architectural idea, is unnecessary.

## 6. If we want a real win, the next three things are

1. **Cut router cost.** At 8k the router is 45.7 ms of the 753 ms prefill and it is
   paid per layer per token in decode. A cheaper scorer (fewer concepts, fused
   kernel, or CUDA-graph replay) directly converts into speedup.
2. **Measure where dense is actually quadratic**: 16k/32k. The crossover at
   ~4–5k means the current win is small; the regime where this should matter is
   beyond what we measured.
3. **Decode via the CUDA-graph path** (already implemented) and report it next to
   a graph-dense baseline, not the eager one.

Optional but bigger: **stop storing the unselected KV** (Quest/MInference-style).
That is the only route to a *memory* win, and it is a different design.

## 7. Reproduction

```bash
# router-only ablation, 28 min each on this card
python experiments/exp2_train_router.py --graph_rank 0 --tag abl_diag --steps 1500
python experiments/exp2_train_router.py --graph_rank 16 --tag abl_graph --steps 1500
python experiments/exp2_train_router.py --graph_rank 0 --no_rel_bias --tag abl_norel --steps 1500

# quality at a given context (controls included)
python experiments/exp3_eval.py --ckpt checkpoints/abl_diag_best.pt --seq 8192 --max_chunks 2

# prefill/decode speed
python experiments/exp4_speed.py --ckpt checkpoints/abl_diag_best.pt --Ts 1024,2048,4096,8192
```
