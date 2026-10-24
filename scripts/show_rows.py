"""Dev: results.jsonl icinden ozet satir dok (tag onekine gore).

Kullanim:
  python scripts\show_rows.py a3_ p4
"""
import json
import sys

pref = tuple(sys.argv[1:]) or ("a3_",)
for line in open("results/results.jsonl", encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    try:
        r = json.loads(line)
    except Exception:
        continue
    t = str(r.get("tag", ""))
    if t.startswith(pref):
        bpb = r.get("best_bpb") or r.get("final_bpb")
        val = r.get("best_val") or r.get("val_loss")
        extra = ""
        if r.get("loops_rand"):
            extra = f" loops_rand={r.get('loops_rand')} hist={r.get('looprand_hist')}"
        print(f"{t:16s} exp={r.get('exp', '?'):18s} params={r.get('params')} "
              f"bpb={bpb} val={val} loops={r.get('loops')}{extra}")
        if r.get("exp") == "exp13_local_sgd":
            parts = []
            for e in (r.get("evals") or [])[-4:]:
                if e.get("replica_bpb_mean") is not None:
                    parts.append(f"@{e.get('step')} rep={e['replica_bpb_mean']}")
                elif e.get("soup_bpb") is not None:
                    parts.append(f"soup={e['soup_bpb']}")
                elif e.get("bpb") is not None:
                    parts.append(f"@{e.get('step')} bpb={e['bpb']}")
            if parts:
                print("                 " + " | ".join(parts))
        if r.get("loop_sweep_final"):
            sw = r["loop_sweep_final"]
            parts = [f"L{L}={sw[str(L)].get('bpb') if sw.get(str(L)) else None}" for L in (1, 2, 4, 8)]
            print("                 sweep: " + " ".join(parts))
