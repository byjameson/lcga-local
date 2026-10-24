"""CPU-only retrieval diagnostics for exp47's covered/control prompts.

This does not run the LLM. It checks which source pages the local BM25 store
selects, whether the expected fact appears in the injected context, and which
control prompts accidentally trigger retrieval. Useful before spending GPU
time on ``exp47_rag_eval.py``.

    python experiments/exp47_retrieval_check.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.exp47_rag_eval import COVERED, UNCOVERED, norm  # noqa: E402
from lcsa.rag import DEFAULT_CORPUS  # noqa: E402
from lcsa.service import Service  # noqa: E402


KEY_TITLES = [
    "Türkiye", "Ankara", "Fransa", "Paris", "Mısır", "Kahire",
    "Japonya", "Tokyo", "Avustralya", "Canberra", "Kanada", "Ottawa",
    "İtalya", "Roma", "Almanya", "Berlin", "Su", "Altın", "Büyük Okyanus",
    "Kalp", "İnsan vücudu", "Jüpiter",
    "Turkey", "Ankara", "France", "Paris", "Egypt", "Cairo", "Japan",
    "Tokyo", "Australia", "Canberra", "Canada", "Ottawa", "Italy", "Rome",
    "Germany", "Berlin", "Water", "Gold", "Pacific Ocean", "Heart",
    "Human body", "Jupiter",
]


def titles_for(store, q: str) -> list[str]:
    return [h.doc.title for h in store.search(q)]


def main() -> int:
    path = ROOT / DEFAULT_CORPUS
    t0 = time.perf_counter()
    svc = Service(budget_gb=5.0, log_requests=False, cache_size=0,
                  enable_chat=False, rag=True)
    index_s = time.perf_counter() - t0
    if svc.rag_store is None or not svc.rag_store.available:
        print(f"[hata] corpus yok: {path}", file=sys.stderr)
        return 2
    store = svc.rag_store
    titles = {d.title for d in store.docs}
    missing = [t for t in KEY_TITLES if t not in titles]
    print(json.dumps({
        "corpus": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "docs": len(store.docs),
        "index_s": round(index_s, 2),
        "missing_key_titles": missing,
    }, ensure_ascii=True))

    covered_ok = covered_ctx = 0
    for row in COVERED:
        q = row["q"]
        ctx = svc._rag_for(q, "quality", ["chat4b"], True)
        expected = row.get("keywords") or []
        in_ctx = (any(norm(k) in norm(ctx.system) for k in expected)
                  if ctx is not None and expected else None)
        covered_ctx += int(ctx is not None)
        covered_ok += int(bool(in_ctx))
        print(json.dumps({
            "set": "covered",
            "id": row["id"],
            "q": q,
            "rag": ctx is not None,
            "expected_in_ctx": in_ctx,
            "sources": ([s.get("title") for s in ctx.sources]
                        if ctx is not None else []),
            "top": titles_for(store, q),
            "ctx_chars": len(ctx.system) if ctx is not None else 0,
        }, ensure_ascii=True))

    clean = 0
    for row in UNCOVERED:
        q = row["q"]
        ctx = svc._rag_for(q, "quality", ["chat4b"], True)
        clean += int(ctx is None)
        print(json.dumps({
            "set": "control",
            "id": row["id"],
            "q": q,
            "rag": ctx is not None,
            "sources": ([s.get("title") for s in ctx.sources]
                        if ctx is not None else []),
            "top": titles_for(store, q),
            "ctx_chars": len(ctx.system) if ctx is not None else 0,
        }, ensure_ascii=True))

    print(json.dumps({
        "covered_with_context": f"{covered_ctx}/{len(COVERED)}",
        "covered_expected_in_context": f"{covered_ok}/{len(COVERED)}",
        "controls_without_rag": f"{clean}/{len(UNCOVERED)}",
    }, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
