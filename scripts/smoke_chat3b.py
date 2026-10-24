"""One-shot chat3b smoke check: load, generate, report latency/VRAM."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lcsa.service import Service  # noqa: E402


def main() -> int:
    svc = Service(budget_gb=5.0, log_requests=False, cache_size=0,
                  enable_chat=True)
    print("units=" + ",".join(svc.specs), flush=True)
    if "chat3b" not in svc.specs:
        raise SystemExit("chat3b not available (model incomplete?)")
    t0 = time.perf_counter()
    out = svc.answer("Fransa'nin baskenti neresi? Kisa cevap ver.",
                     mode="chat3b", max_new_tokens=16)
    print(json.dumps({
        "answer": out["answer"],
        "route": out["route"],
        "plan": out["plan"],
        "timings": out["timings"],
        "elapsed_s": out["elapsed_s"],
        "peak_vram_mb": out["peak_vram_mb"],
        "free_vram_gb": out["free_vram_gb"],
        "wall_s": round(time.perf_counter() - t0, 3),
        "per_unit": out["per_unit"],
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
