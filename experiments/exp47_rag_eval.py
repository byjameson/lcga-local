"""exp47: local RAG vs plain 4B on covered facts + uncovered controls.

This is a product-level, *coverage-limited* evaluation, not a general
benchmark. The covered questions are standard facts whose Wikipedia pages are
in ``data/rag/world_facts.jsonl``. The control questions are deliberately
outside that corpus; they check that retrieval stays off and the normal chat
path still works.

Scores are simple and reproducible:
  - keyword: at least one expected normalized keyword appears
  - exact: normalized answer equals the expected short answer (when defined)
  - citation: a [1]/[2] marker is present
  - rag_used: the service actually injected a retrieved source

Run:
  python experiments/exp47_rag_eval.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results" / "results.jsonl"
SUMMARY = ROOT / "results" / "exp47_rag_eval.json"
MAX_NEW = 64


def norm(text: str) -> str:
    s = str(text or "").translate(str.maketrans(
        "\u00e7\u011f\u0131\u00f6\u015f\u00fc\u00c7\u011e\u0130\u00d6\u015e\u00dc",
        "cgiosuCGIOSU"))
    s = s.casefold()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s)
    return " ".join(s.split())


COVERED = [
    {"id": "tr-eg", "lang": "tr", "q": "Mısır'ın başkenti neresi?",
     "keywords": ["kahire"], "exact": "kahire"},
    {"id": "tr-tr", "lang": "tr", "q": "Türkiye'nin başkenti neresi?",
     "keywords": ["ankara"], "exact": "ankara"},
    {"id": "tr-fr", "lang": "tr", "q": "Fransa'nın başkenti neresi?",
     "keywords": ["paris"], "exact": "paris"},
    {"id": "tr-jp", "lang": "tr", "q": "Japonya'nın başkenti neresi?",
     "keywords": ["tokyo"], "exact": "tokyo"},
    {"id": "tr-au", "lang": "tr", "q": "Avustralya'nın başkenti neresi?",
     "keywords": ["kanberra", "canberra"], "exact": "kanberra"},
    {"id": "tr-de", "lang": "tr", "q": "Almanya'nın başkenti neresi?",
     "keywords": ["berlin"], "exact": "berlin"},
    {"id": "tr-ca", "lang": "tr", "q": "Kanada'nın başkenti neresi?",
     "keywords": ["ottawa"], "exact": "ottawa"},
    {"id": "tr-it", "lang": "tr", "q": "İtalya'nın başkenti neresi?",
     "keywords": ["roma"], "exact": "roma"},
    {"id": "tr-ocean", "lang": "tr", "q": "Dünyanın en büyük okyanusu hangisi?",
     "keywords": ["pasifik"], "exact": None},
    {"id": "tr-au-sym", "lang": "tr", "q": "Altının kimyasal sembolü nedir?",
     "keywords": ["au"], "exact": "au"},
    {"id": "tr-water", "lang": "tr", "q": "Su kaç derecede kaynar?",
     "keywords": ["100"], "exact": None},
    {"id": "tr-heart", "lang": "tr", "q": "İnsan kalbinde kaç odacık vardır?",
     "keywords": ["4", "dort"], "exact": None},
    {"id": "en-eg", "lang": "en", "q": "What is the capital of Egypt?",
     "keywords": ["cairo"], "exact": "cairo"},
    {"id": "en-au", "lang": "en", "q": "What is the capital of Australia?",
     "keywords": ["canberra"], "exact": "canberra"},
    {"id": "en-ca", "lang": "en", "q": "What is the capital of Canada?",
     "keywords": ["ottawa"], "exact": "ottawa"},
    {"id": "en-gold", "lang": "en", "q": "What is the chemical symbol for gold?",
     "keywords": ["au"], "exact": "au"},
    {"id": "en-water", "lang": "en", "q": "At what temperature does water boil?",
     "keywords": ["100"], "exact": None},
    {"id": "en-heart", "lang": "en",
     "q": "How many chambers does the human heart have?",
     "keywords": ["4", "four"], "exact": None},
    {"id": "en-ocean", "lang": "en", "q": "What is the largest ocean?",
     "keywords": ["pacific"], "exact": "pacific"},
    {"id": "en-planet", "lang": "en", "q": "What is the largest planet?",
     "keywords": ["jupiter"], "exact": "jupiter"},
]

UNCOVERED = [
    {"id": "ctl-greet", "q": "Merhaba, nasılsın?", "needs_list": False},
    {"id": "ctl-blue", "q": "Sadece 'mavi' yaz.", "needs_list": False},
    {"id": "ctl-math", "q": "2 + 2 kaç eder?", "needs_list": False},
    {"id": "ctl-list", "q": "Üç meyve listele.", "needs_list": True},
    {"id": "ctl-code", "q": "İki sayıyı toplayan bir Python fonksiyonu yaz.",
     "needs_list": False},
    {"id": "ctl-joke", "q": "Bana kısa bir şaka yap.", "needs_list": False},
    {"id": "ctl-birthday", "q": "Bir arkadaşıma doğum günü mesajı yaz.",
     "needs_list": False},
    {"id": "ctl-motivation", "q": "Bana kısa bir motivasyon sözü yaz.",
     "needs_list": False},
    # Deliberately outside the corpus (a fictional place): the fallback should
    # stay on the normal 4B path instead of forcing an unrelated source.
    {"id": "ctl-narnia", "q": "Narnia'nın başkenti neresi?",
     "needs_list": False},
    {"id": "ctl-narnia-en", "q": "What is the capital of Narnia?",
     "needs_list": False},
]


def score_row(text: str, row: dict) -> dict:
    # A short source marker is useful, but it should not turn an otherwise
    # exact short answer into a miss.
    clean = re.sub(r"\[\d+\]", "", text or "")
    t = norm(clean)
    keywords = row.get("keywords") or []
    def keyword_in(k: str) -> bool:
        # Word boundaries matter for short symbols: "au" is not present in
        # "because", and "4" is not present in "14".
        k = norm(k)
        return bool(re.search(r"(?<![a-z0-9])" + re.escape(k) +
                              r"(?![a-z0-9])", t))
    keyword_hit = (any(keyword_in(k) for k in keywords)
                   if keywords else None)
    exact_hit = (t == norm(row["exact"])
                 if row.get("exact") else None)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    numbered = sum(1 for ln in lines
                   if ln[:2] in ("- ", "* ") or
                   (ln[:1].isdigit() and "." in ln[:3]))
    return {
        "keyword_hit": keyword_hit,
        "exact_hit": exact_hit,
        "list_hit": (numbered >= 3) if row.get("needs_list") else None,
        "citation": bool(re.search(r"\[\d+\]", text or "")),
    }


def run_one(svc, row: dict, use_rag: bool) -> dict:
    t0 = time.perf_counter()
    resp = svc.answer(row["q"], mode="quality", max_new_tokens=MAX_NEW,
                      rag=use_rag)
    text = str(resp.get("answer", ""))
    scores = score_row(text, row)
    rag = resp.get("rag", {})
    return {
        "id": row["id"],
        "lang": row.get("lang", "tr"),
        "query": row["q"],
        "rag_requested": bool(use_rag),
        "rag_used": bool(rag.get("used")),
        "sources": [s.get("title") for s in rag.get("sources", [])],
        "answer": text,
        "answer_chars": len(text),
        "elapsed_s": resp.get("elapsed_s"),
        "timings": resp.get("timings"),
        "peak_vram_mb": resp.get("peak_vram_mb"),
        "route": resp.get("route"),
        **scores,
    }


def summarize(rows: list[dict], mode: str, covered: bool) -> dict:
    wanted_set = "covered" if covered else "uncovered"
    subset = [r for r in rows
              if r.get("set") == wanted_set
              and r["rag_requested"] == (mode == "rag")]
    n = len(subset)
    def rate(key: str) -> str:
        vals = [r[key] for r in subset if r.get(key) is not None]
        if not vals:
            return "n/a"
        return f"{sum(1 for v in vals if v)}/{len(vals)}"
    used = sum(1 for r in subset if r["rag_used"])
    cited = sum(1 for r in subset if r["citation"])
    times = [r["elapsed_s"] for r in subset if r.get("elapsed_s")]
    gens = [r["timings"]["gen_s"] for r in subset
            if isinstance(r.get("timings"), dict)
            and r["timings"].get("gen_s") is not None]
    peaks = [r["peak_vram_mb"] for r in subset if r.get("peak_vram_mb")]
    return {
        "set": "covered" if covered else "uncovered",
        "mode": mode,
        "n": n,
        "keyword_hit": rate("keyword_hit"),
        "exact_hit": rate("exact_hit"),
        "list_hit": rate("list_hit"),
        "citation": f"{cited}/{n}",
        "rag_used": f"{used}/{n}",
        "avg_elapsed_s": round(sum(times) / len(times), 2) if times else None,
        "avg_gen_s": round(sum(gens) / len(gens), 2) if gens else None,
        "peak_vram_mb_max": max(peaks) if peaks else None,
    }


def main() -> int:
    global MAX_NEW
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    args = ap.parse_args()
    MAX_NEW = int(args.max_new)

    from lcsa.rag import DEFAULT_CORPUS
    from lcsa.service import Service
    svc = Service(budget_gb=5.0, log_requests=False, cache_size=0,
                  enable_chat=True, rag=True)
    if svc.rag_store is None or not svc.rag_store.available:
        print(f"[hata] RAG corpus yok: {DEFAULT_CORPUS}", file=sys.stderr)
        return 2
    if "chat4b" not in svc.specs:
        print("[hata] chat4b modeli yok; bu deney 4B urun yolu icin.",
              file=sys.stderr)
        return 2

    # Load the 4B unit once so baseline and RAG timings are both warm.
    svc.answer("Türkiye'nin başkenti neresi?", mode="quality",
               max_new_tokens=4, rag=False)

    rows = []
    for row in COVERED:
        for mode in ("baseline", "rag"):
            r = run_one(svc, row, mode == "rag")
            r["set"] = "covered"
            r["mode"] = mode
            rows.append(r)
            print(f"[covered/{mode}] {row['id']}: kw={r['keyword_hit']} "
                  f"exact={r['exact_hit']} rag={r['rag_used']} "
                  f"cit={r['citation']} {r['elapsed_s']}s", flush=True)
    for row in UNCOVERED:
        for mode in ("baseline", "rag"):
            r = run_one(svc, row, mode == "rag")
            r["set"] = "uncovered"
            r["mode"] = mode
            rows.append(r)
            print(f"[uncovered/{mode}] {row['id']}: rag={r['rag_used']} "
                  f"list={r['list_hit']} {r['elapsed_s']}s", flush=True)

    summaries = []
    for covered in (True, False):
        for mode in ("baseline", "rag"):
            summaries.append(summarize(rows, mode, covered))
    print(json.dumps(summaries, ensure_ascii=True, indent=2))
    payload = {
        "exp": "exp47_rag_eval",
        "tag": "exp47",
        "note": ("Covered sorular korpustaki Wikipedia sayfalariyla ortusur; "
                 "genel benchmark degildir. Kontroller korpus disidir."),
        "n_covered": len(COVERED),
        "n_uncovered": len(UNCOVERED),
        "max_new": MAX_NEW,
        "summaries": summaries,
        "rows": rows,
    }
    SUMMARY.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "exp": "exp47_rag_eval",
            "tag": "exp47",
            "n_covered": len(COVERED),
            "n_uncovered": len(UNCOVERED),
            "max_new": MAX_NEW,
            "summaries": summaries,
            "note": payload["note"],
        }, ensure_ascii=False) + "\n")
    print(f"[done] {SUMMARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
