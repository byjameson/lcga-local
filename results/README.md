# results/ — raw evidence

`results.jsonl` is the append-only log written by every experiment (`append_jsonl`).
The per-experiment `*.json` files are the full payloads the summaries are built from.

## Invalid files (kept for the audit trail, do not cite)

`invalid/` holds runs that used an evaluation script which loaded the checkpoint
**before** `patch_model()`. The router modules did not exist at that point, so
`load_ckpt` silently copied nothing and the run measured a **freshly initialised
router**. The scripts are fixed (`load_ckpt` runs after `patch_model`) and the
affected measurements were re-run into files with a `_reload` / `_fixed` suffix.

Affected (all from 24 Oct 2026, morning):

- `exp60_diag.json`, `exp60_ret_c64.json` — "learned" needle rows
- `exp58_T1024.json`, `exp58_T4096.json`, `exp58_T8192.json`, `exp58_T16384.json`
- `exp59_diag.json`, `exp59_norel.json`, `exp59_v3.json`

Rows in `results.jsonl` with `"ts"` before ~2026-10-24 02:20 and `"exp"` in
{`exp58_eval_long`, `exp59_selection`, `exp60_needle`} are affected the same way.
Everything from `exp3_*` (PPL ablation table) and `exp4_*` (speed) always loaded
correctly and is **not** affected.

## Valid counterparts

- quality vs budget-matched window: `exp58_T*_reload.json`, `exp59_diag_reload.json`,
  `exp59_diag_16k.json`
- needle (dense/window/random/local/oracle are router-independent and were always
  valid): `exp60_ret_c64_fixed.json`, `exp60_top1_c64.json`, `exp60_kl_c64.json`,
  `exp60_res_c64.json`, `exp60_shuf_c64.json`, `exp60_concept_head.json`
- label-free router selection composition: `exp59_top1_c64.json`,
  `exp59_ret_c64_rerun.json`
- generalisation: `exp62_ret_c64.json`
- retrieval router on plain text: `exp58_ret_T1024.json`, `exp58_ret_T4096.json`

`scripts/summarize_needle.py` prints all needle runs as one table.
