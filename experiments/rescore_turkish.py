"""Re-score stored chat runs after fixing Turkish ASCII folding.

The original norm() only folded dotless i. Correct answers such as
"Teşekkür ederim" were therefore counted as keyword misses because the stored
prompts use ASCII ("tesekkur"). This script re-scores the raw stored outputs;
it does not load a model or touch the GPU.

It writes *_rescored.json artifacts and appends only summary rows to
results/results.jsonl. The old rows stay in place as historical evidence.

Usage:
  python experiments/rescore_turkish.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from exp25_chat_eval import PROMPTS as P25, norm, score, summarize  # noqa: E402
from exp26_quality60 import PROMPTS as P26  # noqa: E402


def prompt_map(prompts):
    out = {}
    for row in prompts:
        cat, prompt, keywords, exact, needs_list = row
        if prompt in out and out[prompt] != row:
            raise ValueError(f"conflicting prompt spec: {prompt}")
        out[prompt] = row
    return out


def rescore_rows(rows, pmap):
    changed = 0
    for r in rows:
        spec = pmap.get(r["prompt"])
        if spec is None:
            raise KeyError(f"no spec for prompt: {r['prompt']!r}")
        before = (r.get("keyword_hit"), r.get("exact_hit"), r.get("list_hit"))
        sc = score(spec, r["text"])
        r.update(sc)
        after = (r["keyword_hit"], r["exact_hit"], r["list_hit"])
        if before != after:
            changed += 1
    return changed


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")


def main() -> int:
    summaries = []

    # exp25: base vs chat, 24 prompts
    p25 = prompt_map(P25)
    d = load(ROOT / "results" / "exp25_chat_eval.json")
    changes = 0
    for v in d["variants"]:
        changes += rescore_rows(v["rows"], p25)
        v["rescored_summary"] = summarize(v)
    d["rescored"] = {"reason": "Turkish ASCII folding", "rows_changed": changes}
    d["summary"] = {
        "base": d["variants"][0]["rescored_summary"],
        "chat": d["variants"][1]["rescored_summary"],
    }
    save(ROOT / "results" / "exp25_chat_eval_rescored.json", d)
    summaries.append({"exp": "exp25_chat_eval_rescore", "tag": "exp25_rescore_summary",
                      "n": 24, "rows_changed": changes,
                      "base": d["variants"][0]["rescored_summary"],
                      "chat": d["variants"][1]["rescored_summary"],
                      "note": "scores recomputed from stored raw text with Turkish ASCII folding"})

    # exp25b: en / tr / tr_list systems, 24 prompts
    p26 = prompt_map(P26)
    d = load(ROOT / "results" / "exp25b_system_prompt.json")
    old_summary = d.get("summary", {})
    changes = 0
    systems = {}
    for v in d["variants"]:
        v.setdefault("model", old_summary.get("model", "Qwen2.5-1.5B-Instruct"))
        v.setdefault("load_s", old_summary.get("load_s", 0.0))
        v.setdefault("peak_vram_mb", old_summary.get("peak_vram_mb", 0.0))
        changes += rescore_rows(v["rows"], p26)
        systems[v["label"]] = summarize(v)
    d["rescored"] = {"reason": "Turkish ASCII folding", "rows_changed": changes}
    d["summary"] = {"systems": systems}
    save(ROOT / "results" / "exp25b_system_prompt_rescored.json", d)
    summaries.append({"exp": "exp25b_system_prompt_rescore",
                      "tag": "exp25b_rescore_summary", "n": 24,
                      "rows_changed": changes, "systems": systems,
                      "note": "scores recomputed from stored raw text with Turkish ASCII folding"})

    # exp25c: 2-turn memory, second answer only
    d = load(ROOT / "results" / "exp25c_multiturn.json")
    kw = {"isim": ["ali"], "kedi": ["pamuk"], "renk": ["mavi"],
          "hava": ["guzel"], "toplanti": ["toplanti"]}
    changes = 0
    for v in d["variants"]:
        hits = 0
        for r in v["rows"]:
            old = r["keyword_hit"]
            r["keyword_hit"] = any(norm(k) in norm(r["second_answer"])
                                   for k in kw[r["scenario"]])
            hits += int(r["keyword_hit"])
            changes += int(old != r["keyword_hit"])
        v["rescored_keyword_recall"] = round(hits / len(v["rows"]), 3)
    d["rescored"] = {"reason": "Turkish ASCII folding", "rows_changed": changes}
    save(ROOT / "results" / "exp25c_multiturn_rescored.json", d)
    summaries.append({"exp": "exp25c_multiturn_rescore",
                      "tag": "exp25c_rescore_summary", "n": 5,
                      "rows_changed": changes,
                      "variants": [{"label": v["label"],
                                    "keyword_recall": v["rescored_keyword_recall"]}
                                   for v in d["variants"]],
                      "note": "second-answer keyword recall with Turkish ASCII folding"})

    # exp26: 60 prompts, base vs chat
    d = load(ROOT / "results" / "exp26_quality60.json")
    changes = 0
    for v in d["variants"]:
        changes += rescore_rows(v["rows"], p26)
        v["rescored_summary"] = summarize(v)
    d["rescored"] = {"reason": "Turkish ASCII folding", "rows_changed": changes}
    d["summary"] = {
        "base": d["variants"][0]["rescored_summary"],
        "chat": d["variants"][1]["rescored_summary"],
    }
    save(ROOT / "results" / "exp26_quality60_rescored.json", d)
    summaries.append({"exp": "exp26_quality60_rescore",
                      "tag": "exp26_rescore_summary", "n": 60,
                      "rows_changed": changes,
                      "base": d["variants"][0]["rescored_summary"],
                      "chat": d["variants"][1]["rescored_summary"],
                      "note": "scores recomputed from stored raw text with Turkish ASCII folding"})

    with open(ROOT / "results" / "results.jsonl", "a", encoding="utf-8") as f:
        for s in summaries:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    for s in summaries:
        print(f"[{s['tag']}] rows_changed={s['rows_changed']}")
        if "chat" in s:
            print("  base:", s["base"]["keyword_hit_rate"],
                  s["base"]["exact_hit_rate"], s["base"]["list_hit_rate"])
            print("  chat:", s["chat"]["keyword_hit_rate"],
                  s["chat"]["exact_hit_rate"], s["chat"]["list_hit_rate"])
        if "systems" in s:
            for k, v in s["systems"].items():
                print(f"  {k}: kw={v['keyword_hit_rate']} ex={v['exact_hit_rate']} "
                      f"list={v['list_hit_rate']}")
        if "variants" in s:
            print("  recall:", [(v["label"], v["keyword_recall"]) for v in s["variants"]])
    print(f"[rescored] results.jsonl rows + {len(summaries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
