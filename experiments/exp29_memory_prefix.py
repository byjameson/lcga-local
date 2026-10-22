"""exp29: can a deterministic memory prefix fix simple two-turn recall?

exp25c showed 2/5 recall with raw history. This probe keeps the same five
scenarios but compares:

  plain  : [user fact, assistant "Anladim.", user question]
  prefix : same, with "[Bilinen bilgiler: ...]" prepended to the final user turn

The prefix is extracted by a product-side rule, not learned by the model. It is
a small, honest test of whether explicit re-injection helps at 1.5B.

Requires serve.py on 127.0.0.1:8000. Uses the live chat unit; no extra GPU load.

Usage:
  python experiments/exp29_memory_prefix.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from exp25_chat_eval import norm  # noqa: E402

BASE = "http://127.0.0.1:8000"
SCENARIOS = [
    ("isim", "Benim adim Ali.", "Benim adim ne?", "ali",
     "Kullanicinin adi Ali."),
    ("kedi", "Kedimin adi Pamuk.", "Kedimin adi ne?", "pamuk",
     "Kedinin adi Pamuk."),
    ("renk", "En sevdigim renk mavi.", "En sevdigim renk ne?", "mavi",
     "Kullanicinin en sevdigi renk mavi."),
    ("hava", "Bugun hava cok guzel.", "Bugun hava nasil?", "guzel",
     "Bugun hava cok guzel."),
    ("toplanti", "Yarin sabah toplantim var.", "Yarin ne var?", "toplanti",
     "Yarin sabah toplanti var."),
]


def ask(messages: list[dict]) -> str:
    req = urllib.request.Request(
        BASE + "/v1/answer",
        data=json.dumps({"messages": messages, "mode": "chat",
                         "max_new_tokens": 24}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return str(json.loads(r.read()).get("answer", ""))


def main() -> int:
    rows = []
    for label in ("plain", "prefix"):
        for name, fact, question, keyword, memory in SCENARIOS:
            last = (f"[Bilinen bilgiler: {memory}]\n{question}"
                    if label == "prefix" else question)
            messages = [
                {"role": "user", "content": fact},
                {"role": "assistant", "content": "Anladim."},
                {"role": "user", "content": last},
            ]
            t0 = time.perf_counter()
            try:
                answer = ask(messages)
                err = None
            except Exception as e:
                answer, err = "", f"{type(e).__name__}: {e}"
            elapsed = time.perf_counter() - t0
            hit = bool(answer) and norm(keyword) in norm(answer)
            rows.append({"variant": label, "scenario": name, "fact": fact,
                         "question": question, "memory": memory,
                         "answer": answer, "keyword_hit": hit,
                         "elapsed_s": round(elapsed, 3), "error": err})
            safe = answer[:90].encode("ascii", "backslashreplace").decode()
            print(f"[{label}] {name:8s} hit={hit} {safe!r}", flush=True)

    summary = {}
    for label in ("plain", "prefix"):
        rs = [r for r in rows if r["variant"] == label]
        summary[label] = {
            "n": len(rs),
            "keyword_recall": round(sum(r["keyword_hit"] for r in rs) / len(rs), 3),
            "errors": sum(1 for r in rs if r["error"]),
            "avg_elapsed_s": round(sum(r["elapsed_s"] for r in rs) / len(rs), 3),
        }
    out = {"exp": "exp29_memory_prefix", "n": len(SCENARIOS),
           "summary": summary, "rows": rows,
           "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    path = ROOT / "results" / "exp29_memory_prefix.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    with open(ROOT / "results" / "results.jsonl", "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"exp": "exp29_memory_prefix",
                                "tag": "exp29_memory_prefix", **r},
                               ensure_ascii=False) + "\n")
        f.write(json.dumps({"exp": "exp29_memory_prefix",
                            "tag": "exp29_summary", **summary,
                            "ts": out["ts"]}, ensure_ascii=False) + "\n")
    print("[summary]", json.dumps(summary, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
