# LCGA — findings (learned block-sparse attention on a frozen 0.5B)

Measured on one RTX 3050 Laptop 6 GB (Windows 11/WDDM), AC power. Base:
Qwen2.5-0.5B, frozen; only the router (and, in stage-2 runs, LoRA) is trained.
Reproduce with `experiments/exp2_train_router.py`, `exp3_eval.py`,
`exp4_speed.py`, `exp58_eval_long.py`, `exp59_selection_stats.py`;
raw rows in `results/results.jsonl`.

---

## TL;DR — the honest result

1. **The learned router degenerates to `sink + a contiguous window`.** Instrumenting
   the selections shows **0.0% of chosen blocks lie outside the recent window** at
   T = 1k/4k/8k. Consequently its perplexity is **identical to a static sliding
   window of the same budget**, to three decimals, at every context length tested.
2. **What actually buys quality is the budget, not the learning.** At a 25% block
   budget the gap to dense falls from **+44.6%** (1k) to **+9.9%** (16k).
3. **The comparison that made the method look good was not budget-matched.** The
   "local" control used 2 blocks regardless of budget; the router used 2.5–16.
   Against a budget-matched window the learned router adds **nothing**.
4. The "concept graph" was never needed either (21.838 vs 21.837 PPL; 16 concepts
   instead of 64 gives 21.841) — consistent with (1): the router only needs a
   position prior.
5. **Speed is real but modest**: prefill **1.36× at 16k**; decode with the dedicated
   CUDA-graph path **1.27× at 8k** (bitwise parity); eager decode *loses*.
6. **It is a compute/bandwidth method, not a memory method**: the full KV cache is
   still stored, so it does not extend the context that fits on a 6 GB card.

So: **a careful negative result.** A trained block selector, given a tight budget
and a distillation objective, learns exactly the classical sink+local pattern and
nothing else. Sparse-attention papers that compare against under-budgeted static
baselines will overstate what learning contributes.

---

## 1. Selection composition — the key measurement

For every query block, each selected key block is classified as *sink* (block 0),
*window* (inside the contiguous recent window of the same size) or *content*
(anything else, i.e. what learning adds). `exp59`, 2 chunks per context:

| T | sink | window | **content** | mean budget m |
|---|---:|---:|---:|---:|
| 1024 | 40.0% | 60.0% | **0.0%** | 2.5 blocks |
| 4096 | 12.2% | 87.8% | **0.0%** | 8.2 blocks |
| 8192 | 6.2% | 93.8% | **0.0%** | 16.2 blocks |

**Zero content selection at every tested length.** The router is a (well-trained)
position prior.

## 2. Quality — learned ≡ budget-matched window

One evaluator for all four contexts (`exp58`, chunked-head CE so 16k fits in 6 GB;
8/4/2/2 chunks at 1k/4k/8k/16k):

| T | dense | **learned** | **static window** | random | local (2 blocks) | learned vs dense |
|---|---:|---:|---:|---:|---:|---:|
| 1024 | 17.323 | **25.051** | **25.051** | 27.604 | 30.378 | +44.6% |
| 4096 | 10.950 | **13.783** | **13.783** | 17.411 | 29.045 | +25.9% |
| 8192 | 10.375 | **12.160** | **12.160** | 15.814 | 32.140 | +17.2% |
| 16384 | 10.799 | **11.867** | **11.868** | 16.045 | 33.503 | +9.9% |

Learned and window are the same to the third decimal everywhere — they are, by
measurement, the same policy.

Independent check with the older `exp3` evaluator (128 chunks, T=1024): dense
14.304, learned 21.837, random 23.945, local 26.789. Absolute values differ from
`exp58` because of the chunk subset (8 vs 128 chunks); **compare within a run**.

Budget knob (exp3, T=1024): ρ = 0.125 / 0.25 / 0.5 / 1.0 → 26.789 / 21.837 /
17.198 / 14.303 PPL. At T=8192: 13.691 / 12.176 / 11.340 / 10.376.

## 3. Architecture ablations (T=1024, ρ=0.25, 128 chunks)

Identical recipe (seed 0, 1500 steps, lr 1.5e-3), only the head differs:

| Variant | router params | PPL | recall |
|---|---:|---:|---:|
| graph rank-16 + recency prior (v1) | 2,804,280 | 21.838 | 0.779 |
| diag only + recency prior | 2,755,128 | **21.837** | 0.785 |
| diag only, **16 concepts** | **689,592** | 21.841 | 0.785 |
| diag only, **no** recency prior | 2,755,104 | 22.812 | 0.754 |
| random-k (budget-matched) | 0 | 23.945 | 0.742 |

Reading: the low-rank concept graph adds nothing (0.001 PPL); shrinking the concept
space 4× costs 0.004 PPL; the **recency prior is the only component that matters**
(21.84 vs 22.81), which is exactly what the selection statistics predict.

## 4. Speed

Prefill (single request, AC power, eager router):

| T | dense | sparse | speedup | router | sparse kernel |
|---|---:|---:|---:|---:|---:|
| 1024 | 90.0 ms | 97.2 ms | 0.925× | 10.2 ms | 3.3 ms |
| 2048 | 178.7 ms | 184.4 ms | 0.969× | 12.9 ms | 8.6 ms |
| 4096 | 363.0 ms | 356.4 ms | 1.018× | 23.0 ms | 27.3 ms |
| 8192 | 826.0 ms | 754.4 ms | 1.095× | 49.1 ms | 97.9 ms |
| 16384 | 2225.1 ms | 1642.1 ms | 1.355× | 83.4 ms | 374.6 ms |
| 32768 | 6793.1 ms | 3980.5 ms | **1.707×** | 166.9 ms | 1459.4 ms |

Crossover is between 2k and 4k tokens; the speedup grows monotonically with context
and reaches **1.71× at 32k** (dense 6.79 s vs sparse 3.98 s).

Decode with the dedicated whole-step CUDA graph (`lcsa/decode_graph.py`,
`scripts/decode_ref.py --graph`), graph-dense vs graph-sparse, top-1 parity 1.0 at
every length:

| ctx | graph dense | graph sparse | speedup |
|---|---:|---:|---:|
| 1024 | 12.92 ms/tok | 13.47 | 0.96× |
| 2048 | 14.60 | 14.16 | 1.03× |
| 4096 | 17.69 | 15.72 | 1.13× |
| 8192 | 24.11 | **18.96** | **1.27×** |

Eager decode (no graph) *loses*: 0.55–0.62× — the router runs per layer per token
and host launch overhead dominates at 0.5B. The graph path removes that.

**Negative detail:** enabling the router's own CUDA-graph replay
(`set_router_graph(True)`) changed nothing measurable (8k: 49.1 vs 46.2 ms; 16k:
83.4 vs 75.4 ms).

## 5. What this means

- The useful primitive here is **"attend to a fixed fraction of the most recent
  context, exactly"**: at a 25% budget it stays within 10% of dense at 16k. That is
  classical sliding-window/sink attention, and our contribution is an *exact,
  measured* implementation plus the budget knob.
- **Learning the selection did not work at this scale**, and the mechanism is clear:
  distillation from dense block mass is satisfied by the position prior, so training
  converges there and stops.
- **Comparisons in this literature should be budget-matched.** Ours were not, until
  this last experiment; that is how a 5-PPL "gain" over `local` appeared and then
  vanished.

## 6. Limits

- One base model (Qwen2.5-0.5B), one GPU, one seed per ablation; 2–8 chunks per
  context at long T (low-n). The selection result (content = 0.0%) is exact, so
  seed noise cannot change it.
- No KV-memory saving: the Triton kernel receives the full K/V and gathers selected
  blocks internally. A memory win would need to stop storing unselected KV
  (Quest/MInference-style) — a different design.
- The router was trained at T=1024 only; it generalises to 128 blocks (T=8192), but
  a router trained at long context might learn something else (untested).
- Speed numbers are AC-only: on battery the same measurement is ~30% slower in
  absolute terms (ratios stay similar).

## 7. Reproduce

```bash
# router-only ablation (28 min each on this card)
python experiments/exp2_train_router.py --graph_rank 0            --tag abl_diag --steps 1500
python experiments/exp2_train_router.py --graph_rank 16           --tag abl_graph --steps 1500
python experiments/exp2_train_router.py --graph_rank 0 --n_concepts 16 --tag abl_c16 --steps 1500
python experiments/exp2_train_router.py --graph_rank 0 --no_rel_bias --tag abl_norel --steps 1500

# quality vs context with the budget-matched window control (chunked head)
python experiments/exp58_eval_long.py --ckpt checkpoints/abl_diag_best.pt \
    --seq 16384 --max_chunks 2 --controls

# what the router actually selects (sink / window / content)
python experiments/exp59_selection_stats.py --ckpt checkpoints/abl_diag_best.pt \
    --Ts 1024,4096,8192

# prefill and graph-decode speed
python experiments/exp4_speed.py --ckpt checkpoints/abl_diag_best.pt --Ts 1024,4096,8192,16384
python scripts/decode_ref.py --ckpt checkpoints/abl_diag_best.pt --graph --Ts 8192 --steps 32
```
