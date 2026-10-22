"""exp46: small multi-turn memory probe for the Qwen3-4B product path.

The CLI keeps only the last 8 messages. This probe checks four short cases
(name, preference, previous topic, correction) and one case where the first
fact is intentionally pushed out of the 8-message window. It is a behavioural
probe, not a general memory benchmark.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lcsa.service import Service  # noqa: E402


def ask(svc: Service, history: list[dict], text: str, max_new: int = 48) -> dict:
    messages = history + [{"role": "user", "content": text}]
    out = svc.answer(text, mode="chat4b", max_new_tokens=max_new,
                     messages=messages)
    answer = str(out["answer"]).strip()
    history.extend([{"role": "user", "content": text},
                    {"role": "assistant", "content": answer}])
    # Same 8-message window as llm.py/Service.
    del history[:-8]
    return {"q": text, "a": answer,
            "gen_s": out["timings"]["gen_s"],
            "load_s": out["timings"]["load_s"]}


def main() -> int:
    svc = Service(budget_gb=5.0, log_requests=False, cache_size=0,
                  enable_chat=True)
    if "chat4b" not in svc.specs:
        raise SystemExit("chat4b unit is not available")

    cases = []
    # A: name
    h: list[dict] = []
    turns = [ask(svc, h, "Benim adim Ali."),
             ask(svc, h, "Benim adim ne? Sadece adi yaz.")]
    cases.append({"case": "name", "turns": turns,
                  "hit": "ali" in turns[-1]["a"].lower()})

    # B: preference
    h = []
    turns = [ask(svc, h, "En sevdigim renk mavi."),
             ask(svc, h, "En sevdigim renk ne? Sadece rengi yaz.")]
    cases.append({"case": "preference", "turns": turns,
                  "hit": "mavi" in turns[-1]["a"].lower()})

    # C: previous topic
    h = []
    turns = [ask(svc, h, "Python'da liste nedir? Kisa acikla."),
             ask(svc, h, "Az once ne hakkinda konusuyorduk? Tek cumle.")]
    cases.append({"case": "topic", "turns": turns,
                  "hit": ("liste" in turns[-1]["a"].lower()
                          or "python" in turns[-1]["a"].lower())})

    # D: correction
    h = []
    turns = [ask(svc, h, "Benim adim Ali."),
             ask(svc, h, "Aslinda adim Veli."),
             ask(svc, h, "Benim adim ne? Sadece adi yaz.")]
    cases.append({"case": "correction", "turns": turns,
                  "hit": "veli" in turns[-1]["a"].lower()})

    # E: first fact pushed out of the 8-message window. A distinctive number
    # avoids the model simply guessing a common Turkish name.
    h = []
    turns = [ask(svc, h, "En sevdigim sayi 4731.")]
    for q in ("Bir elma kac kalori?", "Bir armut kac kalori?",
              "Bir muz kac kalori?", "Bir portakal kac kalori?"):
        turns.append(ask(svc, h, q))
    turns.append(ask(svc, h, "En sevdigim sayi kac? Sadece sayiyi yaz."))
    cases.append({"case": "window_overflow", "turns": turns,
                  "hit": "4731" in turns[-1]["a"],
                  "expected": "failure (first fact is outside the 8-message window)"})

    n_hit = sum(1 for c in cases if c["hit"])
    summary = {
        "exp": "exp46_multiturn_4b",
        "tag": "exp46",
        "n": len(cases),
        "short_cases_hit": f"{sum(1 for c in cases[:4] if c['hit'])}/4",
        "window_overflow_hit": cases[-1]["hit"],
        "cases": cases,
        "note": ("short name/preference/topic/correction probes; the last case "
                 "intentionally drops the first fact; not a general memory "
                 "benchmark"),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = ROOT / "results" / "exp46_multiturn_4b.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(ROOT / "results" / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    for c in cases:
        safe = c["turns"][-1]["a"][:80].encode("ascii", "backslashreplace").decode()
        print(f"[{c['case']:16s}] hit={c['hit']} -> {safe!r}", flush=True)
    print(f"[exp46] short={summary['short_cases_hit']} "
          f"overflow={summary['window_overflow_hit']} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
