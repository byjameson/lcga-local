"""exp40: offline evaluation of the hybrid list policy.

The policy is:

  - Turkish list requests -> chat3b with an explicit numbered-line instruction
  - English/other list requests -> chat (1.5B) + deterministic format repair
  - social phrases -> chat
  - everything else -> chat3b

This is a *policy simulation* over already measured rows, not a new GPU run.
Rows for the Turkish 3B instruction variant come from exp39; all other rows
come from exp30 (60 prompts) and exp34 (35-prompt holdout). The list score is
format-only, so the report also records the generated text for manual audit.

The holdout was already inspected while diagnosing list semantics; its number
here is diagnostic, not an untouched generalization estimate.

Usage:
  python experiments/exp40_hybrid_list_router.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from exp25_chat_eval import score  # noqa: E402
from exp31_router_eval import quality  # noqa: E402
from lcsa.service import list_kind, repair_list_text, route  # noqa: E402

SETS = (
    ("exp26", "results/exp30_60_chat3b_quality.json",
     "results/exp39_list_format_probe.json"),
    ("holdout", "results/exp34_holdout_chat3b_quality.json",
     "results/exp39_list_format_probe.json"),
)


def load_variants(path: str) -> dict[str, list[dict]]:
    data = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return {v["label"]: v["rows"] for v in data["variants"]}


def main() -> int:
    alt_rows = json.loads(
        (ROOT / "results/exp39_list_format_probe.json").read_text(
            encoding="utf-8"))["rows"]
    alt_map = {(r["set"], r["prompt"]): r for r in alt_rows}
    output = {"exp": "exp40_hybrid_list_router",
              "tag": "exp40_summary",
              "note": ("offline policy simulation; list metric is format-only; "
                       "holdout was inspected during error analysis"),
              "sets": {}}
    for set_name, base_path, _ in SETS:
        variants = load_variants(base_path)
        chat_rows, b_rows = variants["chat"], variants["chat3b"]
        picked = []
        total_q = total_gen = 0.0
        units = {"chat": 0, "chat3b": 0}
        for c, b in zip(chat_rows, b_rows):
            prompt = c["prompt"]
            keys, reason = route(prompt, "quality", chat=True, chat3b=True,
                                 alt=True)
            key = keys[0]
            if key == "chat3b" and (set_name, prompt) in alt_map:
                source = alt_map[(set_name, prompt)]
                row = {
                    "category": c["category"], "prompt": prompt,
                    "text": source["text"], "gen_s": source["gen_s"],
                    "list_hit": source["list_hit"],
                    "keyword_hit": None, "exact_hit": None,
                }
            elif key == "chat":
                row = {
                    "category": c["category"], "prompt": prompt,
                    "text": repair_list_text(c["text"]) if list_kind(prompt)
                            else c["text"],
                    "gen_s": c["gen_s"], "keyword_hit": c["keyword_hit"],
                    "exact_hit": c["exact_hit"], "list_hit": c["list_hit"],
                }
                if list_kind(prompt):
                    row["list_hit"] = score(
                        (c["category"], prompt, [], None, True),
                        row["text"])["list_hit"]
            else:
                row = {
                    "category": b["category"], "prompt": prompt,
                    "text": b["text"], "gen_s": b["gen_s"],
                    "keyword_hit": b["keyword_hit"],
                    "exact_hit": b["exact_hit"], "list_hit": b["list_hit"],
                }
            row["unit"] = key
            row["reason"] = reason
            picked.append(row)
            total_q += quality(row)
            total_gen += float(row["gen_s"])
            units[key] += 1
        n = len(picked)
        summary = {
            "n": n,
            "quality": round(total_q / n, 4),
            "avg_gen_s": round(total_gen / n, 3),
            "units": units,
        }
        output["sets"][set_name] = {"summary": summary, "rows": picked}
        print(json.dumps({set_name: summary}, ensure_ascii=True), flush=True)
        for r in picked:
            if r["category"] == "list" or list_kind(r["prompt"]):
                safe = r["text"].replace("\n", " / ").encode(
                    "ascii", "backslashreplace").decode()
                print(f"  {r['unit']:6s} {r['prompt'][:42]!r} -> {safe}",
                      flush=True)
    out_path = ROOT / "results" / "exp40_hybrid_list_router.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    with (ROOT / "results/results.jsonl").open("a", encoding="utf-8") as f:
        for set_name, data in output["sets"].items():
            f.write(json.dumps({
                "exp": "exp40_hybrid_list_router",
                "tag": f"exp40_{set_name}_summary",
                **data["summary"],
                "note": output["note"],
            }, ensure_ascii=False) + "\n")
    print(f"[exp40] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
