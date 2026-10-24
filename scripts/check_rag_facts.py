"""Check exp48 retrieval without loading a language model.

This is a *format and context-presence* check, not a semantic evaluation. It
confirms that the service gate retrieves a context for covered questions, that
the expected answer words occur in that context, and that controls stay on the
normal chat path. Final answer quality is checked separately by the GPU run.

Run:
  python scripts/check_rag_facts.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lcsa.rag import DEFAULT_CORPUS, fold, tokenize  # noqa: E402
from lcsa.service import Service  # noqa: E402


def norm(text: str) -> str:
    s = str(text or "").translate(str.maketrans(
        "\u00e7\u011f\u0131\u00f6\u015f\u00fc\u00c7\u011e\u0130\u00d6\u015e\u00dc",
        "cgiosuCGIOSU"))
    s = s.casefold()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s)
    return " ".join(s.split())


def has_term(text: str, term: str) -> bool:
    t = norm(text)
    k = norm(term)
    return bool(re.search(r"(?<![a-z0-9])" + re.escape(k) +
                          r"(?![a-z0-9])", t))


def load_exp48():
    path = ROOT / "experiments" / "exp48_rag_eval.py"
    spec = importlib.util.spec_from_file_location("exp48_rag_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" /
                                        "exp48_retrieval_check.txt"))
    args = ap.parse_args()

    exp = load_exp48()
    svc = Service(budget_gb=5.0, enable_chat=False, rag=True,
                  log_requests=False, cache_size=0)
    if svc.rag_store is None or not svc.rag_store.available:
        print(f"[hata] RAG corpus yok: {DEFAULT_CORPUS}", file=sys.stderr)
        return 2

    out_lines = []
    header = {
        "corpus": str(DEFAULT_CORPUS),
        "docs": len(svc.rag_store.docs),
        "note": ("Each covered row should retrieve; each expected term should "
                 "occur in the retrieved context. This is not a semantic score."),
    }
    out_lines.append(json.dumps(header, ensure_ascii=False))

    covered_ok = 0
    covered_ctx = 0
    covered_single = 0
    controls_ok = 0
    for row in exp.COVERED:
        ctx = svc._rag_for(row["q"], "quality", ["chat4b"], True)
        if ctx is not None:
            covered_ctx += 1
            sources = [s.get("title") for s in ctx.sources]
            text = ctx.system
            hits = {k: has_term(text, k) for k in row.get("keywords", [])}
            expected = any(hits.values()) if hits else True
            if expected:
                covered_ok += 1
            if len(sources) == 1:
                covered_single += 1
        else:
            sources = []
            hits = {k: False for k in row.get("keywords", [])}
            expected = False
        rec = {
            "kind": "covered",
            "id": row["id"],
            "lang": row.get("lang", "tr"),
            "q": row["q"],
            "rag": ctx is not None,
            "expected_in_ctx": expected,
            "keyword_hits": hits,
            "sources": sources,
            "source_count": len(sources),
            "ctx_chars": len(ctx.system) if ctx else 0,
        }
        out_lines.append(json.dumps(rec, ensure_ascii=False))
        print(json.dumps(rec, ensure_ascii=True))

    for row in exp.UNCOVERED:
        ctx = svc._rag_for(row["q"], "quality", ["chat4b"], True)
        ok = ctx is None
        if ok:
            controls_ok += 1
        rec = {
            "kind": "control",
            "id": row["id"],
            "q": row["q"],
            "rag": ctx is not None,
            "sources": [s.get("title") for s in ctx.sources] if ctx else [],
            "ctx_chars": len(ctx.system) if ctx else 0,
        }
        out_lines.append(json.dumps(rec, ensure_ascii=False))
        print(json.dumps(rec, ensure_ascii=True))

    summary = {
        "covered_with_context": f"{covered_ctx}/{len(exp.COVERED)}",
        "covered_expected_in_context": f"{covered_ok}/{len(exp.COVERED)}",
        "covered_single_source": f"{covered_single}/{len(exp.COVERED)}",
        "controls_without_rag": f"{controls_ok}/{len(exp.UNCOVERED)}",
    }
    out_lines.append(json.dumps(summary, ensure_ascii=False))
    Path(args.out).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True))
    print(f"[done] {args.out}")
    return 0 if (covered_ctx == len(exp.COVERED) and
                 covered_ok == len(exp.COVERED) and
                 controls_ok == len(exp.UNCOVERED)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
