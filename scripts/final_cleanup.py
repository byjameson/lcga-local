"""Final JSONL cleanup before reporting.

1) Archive rows whose tag starts with 'smoke' (smoke-test noise) ->
   results_archive.jsonl
2) Backfill missing 'step' on legacy exp7_eval rows with tag=='v2'
   (appended in order -> steps 0, 250, 500, ..., per eval_every=250)

Run ONLY when no experiment is writing to results.jsonl.
Use --dry to preview without touching anything.

Usage: python scripts\\final_cleanup.py [--dry]
"""
import datetime
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "results", "results.jsonl")
ARCHIVE = os.path.join(ROOT, "results", "results_archive.jsonl")


def main():
    dry = "--dry" in sys.argv
    lines = [l.strip() for l in open(P, encoding="utf-8") if l.strip()]
    rows = [json.loads(l) for l in lines]

    keep, arch = [], []
    backfilled, v2_idx = 0, 0
    for line, r in zip(lines, rows):
        tag = str(r.get("tag", ""))
        if tag.startswith("smoke"):
            arch.append(line)
            continue
        if r.get("exp") == "exp7_eval" and tag == "v2" and "step" not in r:
            r = dict(r)
            r["step"] = v2_idx * 250
            v2_idx += 1
            backfilled += 1
            line = json.dumps(r, ensure_ascii=False)
        keep.append(line)

    print(f"total={len(lines)} keep={len(keep)} "
          f"smoke_archive={len(arch)} v2_step_backfill={backfilled}")
    if dry:
        return
    if not arch and not backfilled:
        print("nothing to do")
        return

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copyfile(P, os.path.join(ROOT, "results", f"results_backup_{ts}.jsonl"))
    if arch:
        with open(ARCHIVE, "a", encoding="utf-8") as f:
            for line in arch:
                f.write(line + "\n")
    tmp = P + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for line in keep:
            f.write(line + "\n")
    os.replace(tmp, P)
    print(f"done; backup results_backup_{ts}.jsonl")


if __name__ == "__main__":
    sys.exit(main())
