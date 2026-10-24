"""Live smoke test for the measured `quality` router.

Verifies that the same API mode switches between the 3B paged unit and the
1.5B dense unit, and records the observed latency/VRAM. This is a smoke test,
not a benchmark.

Usage:
  python scripts/smoke_quality_router.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lcsa.service import Service  # noqa: E402

CASES = [
    ("Fransa'nin baskenti neresi? Kisa cevap ver.", "chat3b"),
    ("Uc meyve listele.", "chat3b"),
    ("List three fruits.", "chat"),
    ("8 * 9 = ? Sadece sayiyi yaz.", "chat3b"),
    ("Tesekkur eden bir cumle yaz.", "chat"),
]


def main() -> int:
    svc = Service(budget_gb=5.0, log_requests=False, cache_size=0,
                  enable_chat=True)
    if "chat3b" not in svc.specs or "chat" not in svc.specs:
        raise SystemExit("chat and chat3b must both be available")
    rows = []
    t0 = time.perf_counter()
    for prompt, expected in CASES:
        out = svc.answer(prompt, mode="quality", max_new_tokens=24)
        rows.append({
            "prompt": prompt,
            "expected_unit": expected,
            "units": out["route"]["units"],
            "reason": out["route"]["reason"],
            "answer": out["answer"],
            "gen_s": out["timings"]["gen_s"],
            "elapsed_s": out["elapsed_s"],
            "peak_vram_mb": out["peak_vram_mb"],
        })
        safe = str(out["answer"])[:80].encode("ascii", "backslashreplace").decode()
        print(f"[quality] {prompt[:36]!r} -> {safe!r} "
              f"units={out['route']['units']}", flush=True)
    summary = {
        "exp": "exp38_quality_smoke",
        "tag": "exp38_quality_smoke_summary",
        "n": len(rows),
        "all_routes_ok": all(r["units"] == [r["expected_unit"]] for r in rows),
        "wall_s": round(time.perf_counter() - t0, 3),
        "rows": rows,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(ROOT / "results" / "results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"},
                     ensure_ascii=True, indent=2), flush=True)
    return 0 if summary["all_routes_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
