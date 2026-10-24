"""Add specific Wikipedia pages to an existing RAG corpus.

The full corpus builder fetches every configured topic and can occasionally
skip a page after repeated rate-limit retries. This small utility adds only the
missing pages, without deleting existing rows:

    python scripts/rag_add_pages.py en:Water en:Heart tr:Su

It reuses the same fetch/format logic as ``build_rag_corpus.py`` and writes
the JSONL atomically.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_rag_corpus import OUT, _fetch_one  # noqa: E402


def load_rows(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("id"):
                rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("titles", nargs="+", help="lang:Title entries")
    ap.add_argument("--corpus", default=str(OUT))
    args = ap.parse_args()

    path = Path(args.corpus)
    rows = load_rows(path)
    seen = {r["id"] for r in rows}
    added = skipped = 0
    for spec in args.titles:
        if ":" not in spec:
            print(f"[skip] not lang:Title -> {spec}", file=sys.stderr)
            skipped += 1
            continue
        lang, title = spec.split(":", 1)
        lang = lang.strip().lower()
        title = title.strip()
        if not lang or not title:
            print(f"[skip] empty lang/title -> {spec}", file=sys.stderr)
            skipped += 1
            continue
        row = _fetch_one(lang, title)
        if not row:
            print(f"[skip] fetch failed -> {spec}", file=sys.stderr)
            skipped += 1
            continue
        if row["id"] in seen:
            print(f"[have] {row['id']}", flush=True)
            skipped += 1
            continue
        seen.add(row["id"])
        rows.append(row)
        added += 1
        safe = row["title"].encode("ascii", "backslashreplace").decode()
        print(f"[add] {row['lang']}:{safe} {len(row['text'])} chars",
              flush=True)

    rows.sort(key=lambda r: (r.get("lang", ""), r.get("title", "").casefold()))
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    print(f"[done] added {added}, skipped {skipped}, total {len(rows)} -> {path}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
