"""Re-score and re-summarize the saved exp47 rows without touching the GPU.

The raw answers and timings in ``results/exp47_rag_eval.json`` are valid. This
script exists because the first printed aggregate accidentally mixed the
covered and control rows, and because the short ``au`` keyword needed word
boundaries. It updates only the aggregate scores and the exp47 line in
``results/results.jsonl``; it never re-runs the model.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "exp47_rag_eval.json"
RESULTS = ROOT / "results" / "results.jsonl"


def load_eval_module():
    path = ROOT / "experiments" / "exp47_rag_eval.py"
    spec = importlib.util.spec_from_file_location("exp47_rag_eval", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    ev = load_eval_module()
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    rows = payload["rows"]
    definitions = {r["id"]: r for r in ev.COVERED + ev.UNCOVERED}

    for row in rows:
        definition = definitions[row["id"]]
        scores = ev.score_row(row["answer"], definition)
        for key, value in scores.items():
            row[key] = value

    summaries = []
    for covered in (True, False):
        for mode in ("baseline", "rag"):
            summaries.append(ev.summarize(rows, mode, covered))
    payload["summaries"] = summaries
    SUMMARY.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                       encoding="utf-8")

    new_line = json.dumps({
        "exp": payload["exp"],
        "tag": payload["tag"],
        "n_covered": payload["n_covered"],
        "n_uncovered": payload["n_uncovered"],
        "max_new": payload["max_new"],
        "summaries": summaries,
        "note": payload["note"],
    }, ensure_ascii=False)
    lines = RESULTS.read_text(encoding="utf-8").splitlines()
    replaced = 0
    output = []
    for line in lines:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            output.append(line)
            continue
        if obj.get("exp") == payload["exp"]:
            output.append(new_line)
            replaced += 1
        else:
            output.append(line)
    if replaced != 1:
        raise SystemExit(f"expected exactly one exp47 line, found {replaced}")
    RESULTS.write_text("\n".join(output) + "\n", encoding="utf-8")
    print(json.dumps(summaries, ensure_ascii=True, indent=2))
    print(f"[done] rescored {len(rows)} rows; replaced {replaced} jsonl line")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
