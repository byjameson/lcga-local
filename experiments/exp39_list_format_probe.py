"""exp39: probe the 3B list-format instruction.

This is the evidence behind the Turkish-list part of the hybrid router. It
runs only the list prompts from exp26 and the 35-prompt holdout through the real
`Service` with an explicit three-numbered-line instruction. The list score is
format-only; generated text is stored for manual audit. The instruction is not
used for English lists because it produced degenerate output on the
"primary colors" prompt.

Usage:
  python experiments/exp39_list_format_probe.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from exp25_chat_eval import score  # noqa: E402
from exp26_quality60 import PROMPTS as PROMPTS_60  # noqa: E402
from lcsa.service import LIST_FORMAT_INSTRUCTION, Service  # noqa: E402
from prompts_holdout import PROMPTS as PROMPTS_HOLDOUT  # noqa: E402


def main() -> int:
    svc = Service(budget_gb=5.0, log_requests=False, cache_size=0,
                  enable_chat=True)
    if "chat3b" not in svc.specs:
        raise SystemExit("chat3b is not available")
    rows = []
    for set_name, prompts in (("exp26", PROMPTS_60),
                              ("holdout", PROMPTS_HOLDOUT)):
        for item in prompts:
            category, prompt, _keywords, _exact, needs_list = item
            if category != "list":
                continue
            out = svc.answer(prompt + LIST_FORMAT_INSTRUCTION,
                             mode="chat3b", max_new_tokens=64)
            text = str(out["per_unit"]["chat3b"]["text"]).strip()
            sc = score(item, text)
            rows.append({
                "set": set_name, "prompt": prompt, "text": text,
                "gen_s": round(out["per_unit"]["chat3b"]["gen_s"], 3),
                "list_hit": sc["list_hit"],
                "keyword_hit": sc["keyword_hit"],
            })
            safe = text.replace("\n", " / ").encode(
                "ascii", "backslashreplace").decode()
            print(f"{set_name} | {prompt!r} -> {safe}", flush=True)
    summary = {
        "exp": "exp39_list_format_probe",
        "tag": "exp39_summary",
        "instruction": LIST_FORMAT_INSTRUCTION.strip(),
        "n": len(rows),
        "list_hit": round(sum(bool(r["list_hit"]) for r in rows) / len(rows), 4),
        "avg_gen_s": round(sum(float(r["gen_s"]) for r in rows) / len(rows), 3),
        "note": ("format-only score; inspect text for semantics; the English "
                 "primary-colors prompt produced degenerate output"),
        "rows": rows,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_path = ROOT / "results" / "exp39_list_format_probe.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    with (ROOT / "results/results.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({k: v for k, v in summary.items() if k != "rows"},
                           ensure_ascii=False) + "\n")
    print(f"[exp39] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
