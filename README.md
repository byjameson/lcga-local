# LCGA — learnable block-sparse attention on a frozen Qwen2.5-0.5B

An honest single-GPU research package about replacing dense O(N²) attention with a
**learned block-sparse attention path** (a lightweight "latent concept graph" router +
exact attention over the selected blocks), trained end-to-end on an
**RTX 3050 Laptop 6 GB** (Windows 11, WDDM). Every number comes from
`results/results.jsonl`; negative results and scope limits are stated explicitly.

**Detailed reports (Turkish):** [`RAPOR.md`](RAPOR.md) — main report (phases A/B/C) ·
[`P1_RAPOR.md`](P1_RAPOR.md) — block-streaming training · `P2_RAPOR.md` — byte-level
prototype · [`BASARILAR.md`](BASARILAR.md) — milestone results digest ·
[`M1_RAPOR.md`](M1_RAPOR.md)…[`M5_RAPOR.md`](M5_RAPOR.md) — beyond-VRAM milestones ·
[`ROADMAP.md`](ROADMAP.md) — plan & status ·
[`PROGRESS.md`](PROGRESS.md) — chronological dev journal (full history of
what was tried, including the bugs).

**Launch article (English):** [DEV Community](https://dev.to/mycute_97b39c4e34027570f9/running-a-4b-model-in-6-gb-and-training-3b-parameters-on-a-laptop-4al) · [Markdown](ARTICLE.md) · [HTML](article.html)

**Research checkpoints (Hugging Face):** [LCGA v3 0.5B](https://huggingface.co/byjameson/lcga-v3-0.5b) · [LCGA P3 17M](https://huggingface.co/byjameson/lcga-p3-17m) — model cards there link back to the full reports and raw evidence in this repository.

## Product: LCGA Local — one-command local chat

The product surface is not an API. It is a local chat assistant:

```powershell
python llm.py          # or double-click sohbet.cmd on Windows
```

- **Qwen3-4B-Instruct-2507 in 6 GB**: bf16 is ~8 GB, so all 36 layers are kept
  as **int8 on the GPU** and dequantized one layer at a time into a single bf16
  slot. Measured: **4513 MB peak, ~3.4 tok/s decode**; the first run takes
  **18.8 s** and writes a 3.4 GB int8 cache, after which a process start takes
  **2.4 s** (warm OS file cache; cold-disk read not separately measured).
  `llm.py` streams the answer and keeps the last 8 messages.
- `quality` is the default route: general/code → `chat4b`, short social
  phrases → 1.5B `chat`, with automatic fallback to `chat3b`/`chat` when 4B is
  missing.
- Measured end-to-end product path (`exp43`, 60 prompts): keyword **94.2%**,
  exact **84.6%**, list-format **87.5%**, **3.23 s** average, **2.99 tok/s**,
  peak **4526 MB**. A raw 95-prompt resident-vs-paged ablation produced
  **identical** quality metrics (60: 92.3/84.6/12.5; holdout: 83.3/90.0/40.0)
  while cutting generation time from 5.65/7.06 s to **3.21/3.96 s**. The
  95-prompt 3B-vs-4B comparison is a tie overall (4B better on Turkish facts,
  3B better on some instructions); see [`URUN.md`](URUN.md) for the honest
  limits.
- Memory probe (`exp46`, 4B): name, preference, previous topic and correction
  **4/4** correct inside the 8-message window; a unique fact pushed outside the
  window was **not** recalled (0/1). This is a bounded probe, not general memory.
- Local RAG: CPU-only BM25 over **326 Wikipedia-derived docs / 4.57 MiB**
  (`data/rag/world_facts.jsonl`); measured index load **1.55–2.14 s**. Precise
  fact questions get one cited passage; broad questions may get a second only
  when it independently supports the subject. It is opt-in in the service and
  on by default in `llm.py`; `--no-rag` disables it.
  `exp48` (50 covered + 20 controls, `max_new=64`): CPU retrieval found the
  expected fact in context **50/50**, with **50/50 single sources**; GPU RAG
  answers had keyword **49/50** and citations **50/50**, and all **50/50** were
  factually correct on manual review. The one keyword miss is a synonym
  (`tr-ocean` answers **Büyük Okyanus**, not the expected literal “Pasifik”).
  Controls with RAG/citations: **0/20**; baseline keyword was **42/50**
  (**24/30** on the 30 new questions). Strict exact is not a meaningful metric
  once a citation is required. This is not web search or a general benchmark.
  Raw answers are in `results/exp48_rag_eval.json`; the source
  audit is in `results/exp48_retrieval_check.txt`; review notes are in
  `results/exp48_manual_review.md`.
- Tests: **79** CPU/parity/cache/RAG tests (`test_*service*.py`,
  `test_paged_qwen.py`, `test_paged_quant_cache.py`, `test_rag.py`).

**Files:** [`URUN.md`](URUN.md) product guide · `llm.py` REPL · `sohbet.cmd`
launcher · `scripts/download_qwen3_4b.py` downloader · `lcsa/paged_qwen.py`
int8 GPU-resident engine.

## Headline results

| Axis | Result | Where |
|---|---|---|
| Quality (`wt2_test`, T=1024, ρ=0.25, 128 chunks) | dense 12.46 / sparse **16.56** PPL — **v3** is best of 4 trained models (v1: 14.30/21.84; v2: 12.88/16.73; v2ctl: 12.80/17.26) | RAPOR §3.1, §3.7 |
| OOD (`TinyStories` valid) | v3 dense 6.09 / sparse 9.16 (v2 had 10.37/17.22 → forgetting largely fixed by LR 1e-3 → 1e-4) | RAPOR §3.7 |
| Prefill speed (single request, fp16) | crossover **≈5k tokens**; **1.25×** sparse-vs-dense @16k (A2 CUDA-graph router) | RAPOR §4.1 |
| Decode (whole-step CUDA graph) | **~4–7×** faster than the eager harness on the same machine; dense **bitwise parity**; graph-sparse **1.12–1.27×** faster than graph-dense @4k–8k | RAPOR §4.6 |
| P1 — block-streaming LoRA trainer | Qwen2.5-**3B** trained at **1.40 GB constant VRAM** (naive loader: OOM); numerical parity ≤ 9e-4; held-out CE 2.330 → 2.214 (60-step demo) | P1_RAPOR.md |
| P2 — byte-level prototype | wt2 byte test **bpb: flat 1.645 < patch 2.051** (~5M params, same 196.6M-byte budget; flat wins quality, patch ~1.7× faster per step — two-sided result) | P2_RAPOR.md |

## Beyond-VRAM milestones (M1–M5, Oct 2026)

Second line of work: **train and serve models whose weights do not fit in 6 GB**
(full-parameter, no LoRA). Weight blocks stream CPU↔GPU; the question each milestone
answers is stated with its honest framing.

| # | What | Key result | Report |
|---|---|---|---|
| M1 | Paged inference: Qwen2.5-**3B**, top-k layers resident, streamed layers int8 | 36-layer 3B runs in 6 GB: **4.18 tok/s** (bf16 2.75), paged≡dense CE parity 3.3e-5. Honest arena: **llama.cpp Q4 CUDA 57.6 / CPU-8t 12.4 tok/s** — a 3B-Q4 already fits in 6 GB, so paging's value is >VRAM models and *training*, not beating Q4 | `M1_RAPOR.md` |
| M2 | MoE expert paging (top-1 experts in host RAM, GPU slots) | CE identical (**1.37843**) in paged / LRU / resident modes; **15.4 MB/pass** streamed; hit rate 0.6%. Scale tests: E=16/E=64 demand **34.3/54.0 MB/pass**, prefetch cuts **4.07×/3.46×** (hit 75.5%/71.1%), VRAM saving 3.35×/10.7× | `M2_RAPOR.md` |
| M3 | Transition-matrix prefetch (2 slots, predict the next expert) | Demand copies **12.4× lower** (15.37→1.24 MB/pass), hit **0.6%→92%**, prefetch precision 92.7%; CE and greedy outputs identical | `M3_RAPOR.md` |
| M4 | Multi-unit scheduler: plan groups by declared VRAM, parallel CUDA streams, sequential fallback, merge policy, LRU model pool | 0.5B+1.5B in one group: generation **1.74× vs summed solo runs (1.53× vs the sequential group)**; seq fallback peak 2996 MB; `merge_outputs` supports an answer-selection policy (“best” by continuation logprob); **pool**: 3 requests 25.2s → **9.2s (2.74×)**, repeat-request loads 0.00s, evict+host-reload 0.35s | `M4_RAPOR.md` |
| M5 | **Full-parameter 3B training in 6 GB** (block streaming; all 3.09B params updated, no LoRA) | SGD 150 steps: CE **2.5034→2.4072 (−0.096)**, 6.64 s/step, peak 2.43 GB. **Adafactor** (lr 1e-5): CE→**2.3242 (−0.179, 1.86× the SGD drop)**, 7.67 s/step, peak 4.93 GB. Both checkpoints served back through the paged engine (3.9 tok/s, coherent text). **bf16 quantization wall diagnosed**: streaming updates below 1 weight ULP freeze most weights (`scripts/step_diag.py`); an fp32 CPU master option fixes it (+~10% CE on 0.5B) but needs +4 B/param host RAM. Step profile (3B, 3 steps): **71% of a step is PCIe** (H2D 2.44 s + D2H 2.26 s, ~5.3 GB/s, 4 passes ≈24.7 GB); Adafactor update 0.29→1.50 s; **`--prefetch` gives no speedup** (6.56→6.57 s, pageable copies block the host); **`--double-buffer`** (two pinned pools + copy stream) overlaps next-block H2D with compute: **5.28→4.40 s/step (1.20×) SGD, 6.69→5.91 s (1.13×) Adafactor**, losses bit-identical, +0.17 GB peak (§7g). **`--buffers`** (per-block GPU buffers, no forward copy-back): **6.56→5.20 s/step (1.26×)**, losses bit-identical; `--pinned` alone is slower (host staging 1.55 s) | `M5_RAPOR.md` §7 |

**QLoRA path (`exp49`, 23.10.2026):** the same 6 GB card can fine-tune Qwen2.5-**3B-Instruct** with an **NF4 4-bit frozen base + LoRA** adapter: **29.93M trainable params (1.73%)**, peak **4.25 GB**, **2.56 s/step** at seq 512, WikiText-2 eval CE **2.5154→2.3188 (−0.197)** in 10 steps with `paged_adamw8bit` (load 12.6 s, host RSS 1.83 GB). `exp50` merges the adapter into a bf16 base (`checkpoints/qwen3b_qlora_merged`, 6.17 GB) and serves it with the int8 paged engine at **2.74 tok/s / 4731 MB peak**. `exp51` is instruction SFT on Dolly-15k: 200 steps / 512 train rows gave eval loss **2.4405→1.9040 (−0.537)** at **3.90 GB peak / 1.74 s per step** (20-step smoke: 1.9528→1.2480), with coherent sample answers. The SFT adapter also merges and serves through the int8 paged engine at **2.84 tok/s / 4733 MB peak** (`exp50`). This is an adapter path, not a replacement for M5 full-parameter training. HF's whole-model 4-bit QLoRA does **not** fit 7B on this card (peak 8.4–8.7 GB, shared-memory spill), but our own block-streaming QLoRA (`lcsa/stream_quant_lora.py` + `exp52`) does: 7B NF4 base on CPU, one block on GPU, **4.90 GB peak**, 5.9 s/step, eval CE **2.2304→2.1907** (10 steps) with 5.05M LoRA params. A 100-step Dolly-15k SFT (`exp53`) reaches eval loss **2.6850→1.8710 (−0.814)**; 200–300-step runs with mlp LoRA targets reach **−0.911 / −0.855** at **5.5–5.7 GB peak / 5.3–5.6 s per step** (20.19M LoRA params). A **14B** NF4 streaming attempt with the embedding on GPU spilled (7.01 GB, 70 s/step); after moving `embed_tokens` to CPU (only the gathered activation goes to the GPU) it fits: **3.90 GB peak, 8.5 s/step** on the same 6 GB card, and a 20-step run actually learns: eval CE **2.3270→2.2215 (−0.105)**. So the block-streaming approach scales past 7B once the non-block modules are placed carefully. See `QLORA_RAPOR.md` §9–10. Reproduce: `pip install -r requirements-qlora.txt` then `python experiments/exp49_qlora_3b.py --steps 10 --seq 512 --eval_chunks 2 --tag qlora3b_seq512_t10`, then `python experiments/exp50_qlora_merge.py` and `python experiments/exp16_paged_infer.py --model checkpoints/qwen3b_qlora_merged --quant int8 --pin`. See `QLORA_RAPOR.md`.

## Service (optional developer shell)

The research pieces are wired into one service: **route → plan → run → merge**.
The product itself is `python llm.py` (see above); this HTTP demo is optional.

```powershell
python serve.py --port 8000        # demo UI: http://127.0.0.1:8000/
```

- Units: `fast` Qwen2.5-0.5B (1.1 GB, dense), `balanced` Qwen2.5-1.5B (3.3 GB,
  dense), `chat` Qwen2.5-1.5B-Instruct (3.3 GB, dense; auto-registered when
  `models/Qwen2.5-1.5B-Instruct` is present, `--no-chat` disables), `deep`
  Qwen2.5-3B (**int8 paged**, 4.7 GB peak), `chat3b` Qwen2.5-3B-Instruct
  (**int8 paged**, 4.7 GB peak), and `chat4b` Qwen3-4B-Instruct-2507
  (**int8 GPU-resident**, 4.5 GB peak; auto-registered when
  `models/Qwen3-4B-Instruct-2507` is complete). The deep unit serves our
  Adafactor checkpoint `checkpoints/qwen3b_full_adaf_t300_best.pt` when it
  exists (`--deep-ckpt auto`, the default; `--deep-ckpt none` serves base). An
  optional experimental `alt` unit serves **SmolLM2-1.7B-Instruct** (non-Qwen,
  dense) when `models/SmolLM2-1.7B-Instruct` is complete; it exists to test the
  model-agnostic adapter path, not to claim better quality.
- Routing is rule-based: greetings and question-like prompts → `chat`;
  code/reasoning cues → `deep`; otherwise fast+balanced in parallel with
  `best`-of merge. `mode=quality` is an explicit task router. With 4B present,
  general/code/list → `chat4b`, short social phrases → 1.5B, and `chat3b` is a
  fallback. Without 4B the older 3B policy applies: Turkish lists go to the 3B
  paged model with a numbered-line instruction, English lists to the 1.5B
  instruct model plus a deterministic formatter, short social phrases to 1.5B,
  and the rest to 3B. The reason is returned with every answer — it is not a
  learned router. `mode=alt` is an explicit adapter test, never selected by
  `auto` or `quality`.
- `scheduler.plan` packs units into the VRAM budget: one parallel group when
  they fit, sequential groups otherwise; `ModelPool` keeps dense models for
  LRU reload, the paged engine is released before dense work. Paged units
  always run alone, so `units: ["fast","deep"]` becomes two sequential groups.
- Endpoints: `GET /health`, `/units`, `/stats`, `/metrics`, `/v1/models`,
  `POST /v1/answer` (`{"query" | "messages","mode","max_new_tokens","merge","budget_gb","units"}`),
  `POST /v1/stream` (SSE-like `start`/`token`/`final` events; single-unit token
  streaming, multi-unit falls back to one final event; errors after the stream
  starts arrive as an `error` event inside HTTP 200), `POST /v1/preload`
  (`{"units":["deep"]}`) to warm a unit before the first query, and
  `POST /v1/chat/completions` (OpenAI-compatible, non-stream + SSE). Requests
  are appended to `results/service_requests.jsonl` (`--no-log` disables this).
  An LRU response cache (`--cache`, default 32) serves repeated deterministic
  queries in microseconds; `/metrics` exposes Prometheus counters.

Measured end-to-end (`exp22b`, 3 units, 12 new tokens, budget 5.0 GB, 8 queries):
warm fast requests **0.6–1.5 s**, fast+balanced **3.0–6.1 s**, deep 3B paged
**3.5–8.0 s** (engine load 10.6 s once); median **5.42 s**; peak VRAM
**4646 MB** (no spill). First run exposed a real bug: the paged engine stayed
resident, dense models spilled to shared memory (peak 8555 MB, 18.8 s for the
same request). `PagedQwen.release_gpu()` fixed it; both runs are kept in
`results.jsonl`. With a **matched warm start**, parallel grouping is **1.30×**
faster than sequential (0.75 s vs 0.98 s, 3 reps); the first run's 0.88× was a
reload-time confound (the parallel request was cold). A unit larger than the
declared budget still runs alone and the response carries a `warnings` entry.

**Serving our checkpoint (`exp23`, same int8 paged engine):** held-out WT2 test
CE **2.611 → 2.370 (−9.2%)** on 4×256 tokens; load 6.2 s, peak 4897 MB,
generation unchanged (8 tokens: 2.63 vs 2.51 s). It is a WT2 fine-tune, not an
instruction model: a 24-probe OOD check (`exp24b`) shows a small overall CE
regression (**+0.061**; Turkish +0.10, facts +0.045, code −0.032, top-1
unchanged) — in-domain gain, mild out-of-domain cost, n=24 single seed.
HTTP integration (`fast` preload + `fast,deep` override) passed: fast preload
4.9 s cold, warm fast answer 0.54 s, deep with checkpoint in an 8.9 s group,
peak 4763 MB.

**Chat unit (`exp25`/`exp25b`, historical):** base vs instruct 1.5B on 24 short
prompts raised keyword hit **54.5% -> 68.2%** and exact-answer **25% -> 75%**.
The Turkish system prompt (now the default) raised keyword hit to **81.8%**,
exact to **100%**, and Turkish facts **33% -> 83%**; shorter answers flatter
these scores, so this is directional.

**Larger chat check (`exp26`, 60 prompts, greedy 64 tokens):** the same 24
prompts plus 36 new ones, scored with Turkish ASCII folding. Instruct vs base:
keyword **82.7% vs 55.8%**, exact **84.6% vs 15.4%**, strict three-line list
format **62.5% vs 50%**. By category (chat): Turkish facts **75% vs 25%**,
English facts **100% vs 83.3%**, Turkish instructions **81.8% vs 27.3%**,
English instructions **81.8% vs 90.9%** (keyword down; exact **100% vs 50%**),
math **66.7% vs 50%** (exact **60% vs 0%**). Remaining chat errors include
Istanbul/Ankara, Pacific/Atlantic, the gold symbol, the largest organ, one
translation, "dozen" and "half", and inline lists. This is still keyword
scoring with n=60, not a human evaluation. A targeted list-format hint made
strict list following worse (`exp27`: plain 5/6, user suffix 2/6, system hint
3/6), so it is not integrated.

**3B Instruct quality mode (`exp30`-`exp41`):** the optional `chat3b` unit
runs Qwen2.5-3B-Instruct through the same int8 paged engine (total shard size
6.02 GB decimal, ~4.4–4.8 GB peak VRAM). The paged generator now stops on
Qwen's `<|im_end|>` as well as EOT; before that fix the 3B model answered
`Paris'dır.` and then continued with a fabricated `user ...` turn.

| Prompt set | Unit | Keyword | Exact | Strict list | Avg gen | Peak |
|---|---|---:|---:|---:|---:|---:|
| `exp26`, n=60 | `chat` 1.5B | 82.7% | 84.6% | 62.5% | 1.114 s | 3248 MB |
| `exp26`, n=60 | `chat3b` 3B | **88.5%** | **92.3%** | 12.5% | 3.025 s | 4411 MB |
| `exp34` holdout, n=35 | `chat` 1.5B | 73.3% | 70.0% | **80.0%** | 1.085 s | 3231 MB |
| `exp34` holdout, n=35 | `chat3b` 3B | **86.7%** | **90.0%** | 20.0% | 3.461 s | 4411 MB |

The 3B unit is better on most facts, instructions and arithmetic, but **worse
on strict three-line lists**, and still makes Turkish factual errors
(İstanbul/Ankara-style confusions, gold symbol, ocean, organ). This is
keyword/exact scoring on small sets, not human evaluation.

`mode=quality` is a rule-based hybrid, not a learned router:

- Turkish list requests (`listele`) → `chat3b` with an explicit “three numbered
  lines” instruction; this keeps the 3B model’s better list content and fixes
  its usual inline formatting.
- English/other list requests → `chat` (1.5B) plus a deterministic formatter
  that only turns an already-correct inline answer into three numbered lines.
- Short social phrases → `chat`; everything else → `chat3b`.

This was evaluated offline on the already-generated rows (`exp40`), not as a
fresh end-to-end benchmark; `avg gen` excludes model load/switching:

| Set | Policy | Quality | Avg gen | Units |
|---|---|---:|---:|---|
| `exp26`, n=60 | `chat` / `chat3b` | 78.3% / 78.3% | 1.11 / 3.02 s | — |
| `exp26`, n=60 | **hybrid quality** | **91.7%** | **2.87 s** | 5 / 55 |
| `exp34` holdout, n=35 | `chat` / `chat3b` | 71.4% / 74.3% | 1.09 / 3.46 s | — |
| `exp34` holdout, n=35 | **hybrid quality** | **85.7%** | **3.60 s** | 3 / 32 |

The 60 result rose from 86.7% to 88.3% after removing the over-broad `maddede`
cue, then to 91.7% with the list policy. These are **format/keyword scores on
small sets**, not human quality. The list score only checks three numbered
items; manual inspection found 11/12 list answers clearly correct and one
partial (a holdout “vegetables” answer included olives). A separate probe
(`exp39`) showed why format alone is insufficient: 3B with the instruction
scored 11/11 on format but produced degenerate text for an English
“primary colors” prompt; English lists therefore stay on 1.5B. The holdout was
inspected while diagnosing this, so its 85.7% is **diagnostic, not untouched
generalization**.

A learned hashed-n-gram logistic router (`exp36`, trained only on the 60 rows)
reached **82.9% / 3.03 s** on the old holdout, the same quality as the previous
rule router; it was **not integrated**. The confidence-cascade attempt (`exp31`)
was also a **negative result**: lower chat confidence did not identify the cases
where 3B was better. The old `auto` router was weak on the 24-prompt probe
(`exp32`: 81.8% keyword but 0% strict list). `quality` is opt-in. Live HTTP
smoke (`exp41`, cold cache, not a benchmark): Turkish list 11.98 s, English list
6.40 s, fact 3.67 s, explicit `alt` 6.86 s, OpenAI English list 3.30 s; a warm
streamed English list returned the repaired final answer in 0.91 s. These walls
include model load/switching and are much higher than the per-generation
averages above.

**OpenAI-compatible API:** `GET /v1/models` and
`POST /v1/chat/completions` accept OpenAI-style `messages` (roles `system`,
`user`, `assistant`; a system message overrides the default assistant prompt);
`stream: true` returns SSE `chat.completion.chunk` deltas and `data: [DONE]`.
`model=lcga-chat` uses the 1.5B instruct unit, `lcga-chat3b` forces the 3B
paged unit, `lcga-quality` uses the measured task router, and `lcga-alt` uses
the optional non-Qwen adapter test; unsupported model ids fall back to `chat`.
`model` is echoed for client compatibility,
`temperature` is accepted but generation is deterministic, and unsupported
tools/`n>1` return HTTP 400 in OpenAI error shape. Live smoke (`exp28`):
non-stream cold 6.96 s (includes first model load), cached 0.018 s; stream TTFT
**0.443 s**, total **2.697 s**, 12 content chunks, `[DONE]` present. A live
system-message request returned `Paris.` with `finish_reason=stop`.

**Multi-turn / UI:** `/v1/answer` and `/v1/stream` accept an OpenAI-style
history (last 8 messages, last role `user`), forced to `chat` by default
(`chat3b`/`alt` only when explicitly selected). The demo UI now
has an optional **"gecmis (deneysel)"** checkbox (default off), a visible
conversation log, and a clear button. A 5-scenario two-turn probe (`exp25c`)
recalled the stated fact in **2/5** cases; an explicit memory instruction did
not improve it (also 2/5). A follow-up (`exp29`) used an `"Anladim."` ack and a
labeled `[Bilinen bilgiler: ...]` prefix on the final user turn: raw history
recalled **3/5**, prefix **4/5**. A second name-wording probe (`exp29b`) fixed
the name in **3/4** phrasings. The UI uses this as a deterministic **five-template
memory layer** (name, pet, color, weather, meeting) plus a name-question rewrite;
a browser 3-turn test recalled "Ali" in both recall turns (`exp29c`, n=2). This
is template engineering, not a general memory model: other phrasings and
open-ended facts still fall back to the weak 1.5B history behavior.

**Streaming + ops:** warm first token **0.249 s** fast / **0.449 s** deep
(4 tokens 0.484 s, 5 tokens 2.003 s); cold start is load-bound (9.8 s / 8.2 s).
Repeated deterministic query: **9.089 s -> 0.00005 s** (LRU cache); two
concurrent fast requests: second waited **8.814 s** (single GPU lock, reported
in `timings.queue_s`). CPU-only service/HTTP tests: **53 pass**
(`python -m unittest discover -s tests -p "test_*service*.py" -v`).
Honest limits: single in-flight request; the deployed routers are rule-based
(a learned n-gram router tied and was not integrated); no auth/CORS; OpenAI
endpoint is deterministic; the 3B unit costs ~2.7-3.5 s generation and
~4.4-4.8 GB peak, is worse than 1.5B on *default* strict-list formatting, and
still has Turkish factual errors; the hybrid quality result was measured on 60
seen + 35 inspected holdout prompts with keyword/format scoring, not human
evaluation. The optional `alt` SmolLM2 unit proves the non-Qwen dense path and
is explicitly selectable, but is not in `auto`/`quality`; its measured scores
are low (holdout keyword 50%, exact 20%) and its list format score does not mean
correct list content. Streaming is token-level for a single unit; English list
repair in `quality` is emitted as one final event rather than tokens. See
`SERVICE_RAPOR.md`.

## Method in 60 seconds

- **Frozen base + learned router.** Qwen2.5-0.5B weights stay frozen; a wrapper
  (`lcsa/model.py`) gives every attention layer a router (`lcsa/concepts.py`):
  K=64 learnable concept embeddings score the sequence, concept probabilities are
  aggregated into per-block scores, and the top-M blocks per layer
  (M = round(ρ · n_blocks), default ρ=0.25) are selected. Selection is trained with a
  gathered/straight-through formulation (report §2.1–2.2).
- **Exact sparse attention.** Softmax attention runs *exactly* over the selected
  blocks' K/V (no approximation inside the attention itself): Triton kernel for
  training/eval, gathered masked SDPA for the CUDA-graph decode path. Slot-level
  causality (`kpos ≤ qpos`) is the reference semantics in every path.
- **Two-stage training.** v1: router only (frozen base). v2/v3: router + LoRA with
  LM + KL + MSE + recall-gain losses (wikitext-2). **v3 = LoRA LR 1e-4** fixes OOD
  forgetting and is the best on-domain model.
- **Speed engineering.** A1 exact-math caches + vectorized block selection (bitwise
  parity vs eager); A2 CUDA-graph router replay (inference-only, opt-in); C: the whole
  decode step (24 layers) captured as a single CUDA graph, leaving only 2 tiny H2D
  copies + one replay per token. Clean inference config: null-key disabled
  (`null_logit = -1e9`), coarse path off.
- **A decoder bug found by the graph.** The eager sparse decode harness silently
  attended to unwritten future slots of the partially-filled current block (zero keys
  → score 0 → softmax weight exp(0)); training semantics — and the graph decoder —
  mask them. Fixed in `lcsa/decode.py` / `decode_fast.py`; parity is checked by
  `scripts/check_sparse_graph.py` (per-layer selection equality + logits). See RAPOR §4.6.

## Repository layout

```
lcsa/                  core library (package name: lcsa)
  model.py             patched Qwen attention + LCGA wrapper (router attach, dense control)
  concepts.py          router: concept graph, block scores, top-M selection
  attention.py         gather_sparse_attention (reference) + Triton sparse kernel + dense ref
  router_graph.py      A2: CUDA-graph router replay (inference-only)
  decode.py            eager decode harness (naive reference)
  decode_fast.py       eager fast decode (expanded/GQA SDPA, sparse gather)
  decode_graph.py      C: whole-decode-step CUDA graph (dense / sparse modes)
  lora.py              minimal LoRA (attach / merge / load-save round-trip)
  harness.py           model loading, fast-dense guard, quick PPL
  data.py              wikitext-2 / TinyStories tokenization (+ byte data for P2)
  metrics.py           timing helpers, JSONL logging
  stream_train.py      P1: block-streaming (CPU-offload) LoRA trainer + naive reference
  stream_full.py       M5: full-parameter block-streaming trainer (SGD/Adafactor, schedules, fp32 master)
  paged_qwen.py        M1: paged Qwen inference (resident top-k + int8 streamed layers)
  mini_lm.py           tiny MoE-lite LM (top-1 Switch experts) for M2/M3 paging
  paged_moe.py         M2/M3: expert paging + transition-matrix prefetch
  scheduler.py         M4: multi-unit scheduler (plan / parallel groups / merge policies)
  pool.py              M4: model pool with LRU eviction (weights stay warm across requests)
  service.py           product shell: route -> VRAM plan -> parallel/sequential run -> merge
  byte_lm.py           P2: FlatByteLM + BytePatchLM (BLT-lite, fixed patches)
serve.py               stdlib HTTP API + demo UI (GET /health, POST /v1/answer, /v1/stream)
experiments/           one CLI per experiment (exp1..exp34)
scripts/               benchmarks & tooling (speed_ref, decode_ref, check_sparse_graph,
                       summarize_results, make_plots, router_profile, parity checks, ...)
results/               results.jsonl (append-only evidence log) + exp*.json details
checkpoints/           trained artifacts (v1/v2/v3/v2ctl bests, P1 3B adapter, P2)
data/                  tokenized datasets (built by exp1)
plots/                 report figures
tests/                 runnable correctness/speed checks for attention/decode/model patching
```

## Setup

Python 3.11 + CUDA-enabled PyTorch (see `requirements.txt`; Windows/WDDM notes in the reports).

```powershell
pip install -r requirements.txt
python download_model.py                   # Qwen2.5-0.5B -> models/ (fp16, frozen base)
python experiments/exp1_data_baseline.py   # builds data/*.pt + dense baselines (GPU)

# Optional product path: 4B local chat (~8 GB download; first run builds a 3.4 GB int8 cache)
python scripts/download_qwen3_4b.py        # Qwen3-4B-Instruct-2507 -> models/
python llm.py                              # local chat (uses 4B when present, 1.5B/3B otherwise)
```

## Reproduce the core pipeline

```powershell
# 1) v1: router only (defaults: 1500 steps, batch 4, seq 1024)
python experiments/exp2_train_router.py --tag v1

# 2) v3: stage-2 router + LoRA (single change vs v2: lr_lora 1e-4)
python experiments/exp7_stage2.py --steps 1250 --lr_lora 1e-4 --tag v3 --ckpt_gather

# 3) quality: 3 test sets + rho sweep + controls (128 chunks, clean config)
python experiments/exp3_eval.py --ckpt checkpoints/v3_best.pt --tag v3 --v2

# 4) speed: prefill reference and whole-step CUDA-graph decode (parity + timing)
python scripts/speed_ref.py --ckpt checkpoints/v3_best.pt --tag v3_speed --v2 --merge
python scripts/decode_ref.py --tag c_phase --Ts 1024,2048,4096,8192 --steps 64 --graph
python scripts/check_sparse_graph.py

# 5) P1 / P2 (details and exact commands in their reports)
python experiments/exp8_stream.py --help
python experiments/exp10_byte.py --tag p2 --steps 12000 --batch 32 --W 512 --P 4

# 6) M1-M5 beyond-VRAM demos (details in their reports; Qwen2.5-3B/1.5B under models/)
python experiments/exp16_paged_infer.py --model models/Qwen2.5-3B --quant int8 --pin   # M1
python experiments/exp18_moe_paging.py --modes paged,pf2,resident                       # M2/M3
python experiments/exp20_scheduler.py --merge best                                      # M4
python experiments/exp17_full_train.py --model models/Qwen2.5-3B --steps 150 --seq 512 --lr 1e-5 --opt adafactor --eval_chunks 16 --tag demo_m5   # M5
python experiments/exp17_full_train.py --model models/Qwen2.5-3B --steps 3 --seq 512 --lr 5e-4 --profile --buffers --tag prof3b_buf   # M5: 5.20 s/step, 1.26x
python experiments/exp17_full_train.py --model models/Qwen2.5-3B --steps 3 --seq 512 --lr 1e-5 --double-buffer --tag dbuf3b      # M5: 4.1-4.4 s/step, 1.20-1.33x

# 7) product shell: M1/M4/M5 pieces wired into one service; demo UI at http://127.0.0.1:8000/
python serve.py --port 8000                     # serves our 3B checkpoint + chat unit
python serve.py --port 8000 --deep-ckpt none    # base 3B instead
python serve.py --port 8000 --no-chat --cache 0 # minimal: no instruct unit, no cache
python experiments/exp22_service_bench.py --n_new 12 --tag exp22b                   # 8 real queries, 3 units
python experiments/exp22_scen_fair.py --n_new 12 --reps 3 --tag exp22d              # warm parallel vs sequential
python experiments/exp23_ckpt_eval.py --chunks 4 --seq 256 --n_new 8                # held-out CE: base vs checkpoint
python experiments/exp24_ood_probe.py --tag exp24b                                 # 24 OOD probes
python experiments/exp25_chat_eval.py --tag exp25                                  # base vs instruct 1.5B chat probe
python experiments/exp25b_system_prompt.py --tag exp25b                           # EN vs TR system prompt ablation
python experiments/exp25c_multiturn_probe.py --tag exp25c                         # 2-turn memory probe (API history)
python experiments/exp26_quality60.py --tag exp26                                 # 60-prompt base vs chat check
python experiments/exp27_list_prompt.py                                           # list-format hint probe (negative)
python experiments/exp28_openai_smoke.py                                          # live OpenAI endpoint smoke test
python experiments/exp29_memory_prefix.py                                         # memory prefix probe (live server)
python experiments/exp29b_memory_name.py                                          # name wording probe (live server)
python experiments/exp30_chat3b_quality.py --tag exp30_60 --set exp26 --max-new 64  # 1.5B vs 3B, 60 prompts
python experiments/exp31_router_eval.py --input results/exp30_60_chat3b_quality.json --tag exp31_60  # cascade (negative)
python experiments/exp32_auto_eval.py --tag exp32 --max-new 64                     # old auto baseline
python experiments/exp33_quality_route_eval.py --input results/exp34_holdout_chat3b_quality.json --tag exp33_holdout  # quality router
python experiments/exp36_learned_router.py                                             # learned router (tie, not deployed)
python experiments/exp37_alt_model_eval.py --set holdout --tag exp37_holdout          # optional non-Qwen adapter check
python experiments/exp39_list_format_probe.py                                          # 3B numbered-list probe
python experiments/exp40_hybrid_list_router.py                                         # hybrid list policy (offline)
python scripts/smoke_quality_router.py                                                 # live hybrid route smoke
python scripts/check_rag_facts.py                                                  # CPU RAG retrieval/source check
python experiments/exp48_rag_eval.py                                               # 50 covered + 20 controls (GPU)
python experiments/rescore_turkish.py                                             # re-score stored text with TR folding
python -m unittest discover -s tests -v                                           # 79 tests
python -m unittest discover -s tests -p "test_*service*.py" -v                   # 53 service/HTTP tests
```

## Measurement discipline (why the numbers are trustworthy)

- Timings: ≥3 warmups, **median**, `torch.no_grad()`, all parameters
  `requires_grad_(False)`, SDPA backends pinned; dense baseline measured **first in a
  fresh process**.
- On 6 GB WDDM: reservations above ~6 GB silently spill into shared memory (no OOM) —
  the streaming trainer is validated with hard per-process memory caps.
- One GPU process at a time; never measure while training runs.
- Every experiment appends a timestamped row to `results/results.jsonl`
  (`scripts/summarize_results.py` renders the log).

## Honest scope & limits

- Scale: 0.5B model, batch=1, one 6 GB laptop GPU. No SOTA or "dense-quality at a
  fraction of the cost" claims; "quality preserved" means the measured gap on
  wikitext-2 (plus one OOD probe), not lossless attention.
- Eager decode timings are host-load sensitive (launch-bound, WDDM; back-to-back runs
  differ by 40–80%). CUDA-graph timings are stable (±0.2%). The structural claims are
  graph↔eager and graph-sparse↔graph-dense.
- Attribution (v2): ~83% of the quality-gap reduction is generic LoRA adaptation,
  ~17% sparsity-specific (dense_control ablation, RAPOR §3.6).
- OOD: a single probe set (TinyStories). v3 largely recovers v1-level OOD, but
  "never forgets" is not claimed.
- P2 is a small fixed-patch prototype (single seed): at this scale the flat byte
  transformer wins quality while the patch model is ~1.7× faster per step; no dynamic
  patching, no KV-cache decode, no inference-speed claims.
- M5 scope: one 512-token sequence per step, a fixed 16×512 eval window, single seed.
  "Adafactor beats SGD here" is a measured statement for this setup, not a general
  recipe. The bf16-update quantization finding (updates below 1 ULP largely freeze the
  weights) and the fp32-master fix are quantified in `M5_RAPOR.md` §7; the 3B fp32
  master did not fit this machine's free host RAM. Step time is transfer-bound: 71% of a
  3B step is PCIe (H2D 2.44 s + D2H 2.26 s, ~5.3 GB/s); `--prefetch` does not help with
  pageable memory (6.56 → 6.57 s/step). Pinned double-buffering is implemented as
  `--double-buffer` and measured at **5.28→4.40 s/step (SGD) / 6.69→5.91 s (Adafactor)**
  on 3B with bit-identical losses (§7g); the D2H write-back is still synchronous.
- M1–M3 speed numbers are for the 6 GB laptop GPU (WDDM). The llama.cpp comparison uses
  a bartowski Q4_K_M GGUF with the vendor CUDA build on the same machine.

## Support the work / Destek

This project is developed on a 6 GB laptop GPU. Donations go toward a larger GPU
and more compute so the experiments can scale; there is no paid support, token,
or investment offer.

- **EVM-compatible networks** (Ethereum, BNB Smart Chain, Polygon, Optimism,
  Arbitrum One, Base, Avalanche C-Chain):
  `0xaf33F66221C98AF02D00a2a694628B9071944f5F`
- **Solana:**
  `GBRT87EBorFGjL4zLPGQg1fc6dG9een1grHbF1HEMimu`
- **Bitcoin:**
  `bc1qmjvu0c7skkuph5pva6l5uuxpzu3z6fw8nyy0f2`

Please double-check the network and the full address before sending; crypto
transfers are irreversible. Donations are voluntary and do not buy influence
over results or reviews.

## License

MIT — see [LICENSE](LICENSE).
