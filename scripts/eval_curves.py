"""exp7_eval eğrilerini yazdır (dense/sparse ppl, adım başına) — v2/v3 kıyası için.

Kullanım: python scripts/eval_curves.py [tag ...]   (varsayılan: v1 v2 v3)
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tags = sys.argv[1:] or ["v1", "v2", "v3"]
rows = [json.loads(l) for l in open(os.path.join(ROOT, "results", "results.jsonl"),
                                    encoding="utf-8")]
for tag in tags:
    ev = [r for r in rows if r.get("exp") == "exp7_eval" and r.get("tag") == tag]
    if not ev:
        print(f"--- {tag}: (kayıt yok)")
        continue
    print(f"--- {tag} ({len(ev)} satır)")
    for r in ev:
        d = r.get("dense_ppl", float("nan"))
        s = r.get("sparse_ppl", float("nan"))
        print(f"  step {r.get('step'):>5} dense={d:.3f} sparse={s:.3f} "
              f"delta={s - d:+.3f} recall0={r.get('recall0')} gain={r.get('gain_mean')}")
