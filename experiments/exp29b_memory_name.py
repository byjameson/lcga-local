"""exp29b: name-recall wording variants for the deterministic memory prefix.

exp29's prefix fixed color/weather/meeting but not the name. This probe checks
four explicit re-phrasings on the same two-turn history. Live chat unit only.

Usage:
  python experiments/exp29b_memory_name.py
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
FACT = "Benim adim Ali."
QUESTION = "Benim adim ne?"
VARIANTS = [
    ("user_fact", "[Bilinen bilgiler: Benim adim Ali.]\nBenim adim ne?"),
    ("user_fact_q", "[Bilinen bilgiler: Benim adim Ali.]\nKullanicinin adi ne?"),
    ("repeat", "Benim adim Ali. Benim adim ne?"),
    ("label_q", "Kullanicinin adi Ali. Kullanicinin adi ne?"),
]


def ask(messages):
    req = urllib.request.Request(
        BASE + "/v1/answer",
        data=json.dumps({"messages": messages, "mode": "chat",
                         "max_new_tokens": 24}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return str(json.loads(r.read()).get("answer", ""))


def main() -> int:
    rows = []
    for label, last in VARIANTS:
        messages = [{"role": "user", "content": FACT},
                    {"role": "assistant", "content": "Anladim."},
                    {"role": "user", "content": last}]
        t0 = time.perf_counter()
        answer = ask(messages)
        hit = norm("ali") in norm(answer)
        rows.append({"variant": label, "answer": answer, "keyword_hit": hit,
                     "elapsed_s": round(time.perf_counter() - t0, 3)})
        print(f"[{label}] hit={hit} {answer[:90]!r}", flush=True)
    out = {"exp": "exp29b_memory_name", "n": len(rows),
           "summary": {"hits": sum(r["keyword_hit"] for r in rows),
                       "n": len(rows)},
           "rows": rows, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    path = ROOT / "results" / "exp29b_memory_name.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    with open(ROOT / "results" / "results.jsonl", "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"exp": "exp29b_memory_name",
                                "tag": "exp29b_memory_name", **r},
                               ensure_ascii=False) + "\n")
        f.write(json.dumps({"exp": "exp29b_memory_name",
                            "tag": "exp29b_summary", **out["summary"],
                            "ts": out["ts"]}, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
