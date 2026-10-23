"""Upload the staged model folders to ModelScope and verify."""
import os
import sys
from pathlib import Path

from modelscope.hub.api import HubApi

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dist" / "modelscope"
REPOS = ["byjameson/lcga-v3-0.5b", "byjameson/lcga-p3-17m"]

token = os.environ.get("MS_TOKEN", "")
if not token:
    print("MS_TOKEN missing", file=sys.stderr)
    raise SystemExit(2)

api = HubApi()
api.login(token)

for rid in REPOS:
    folder = OUT / rid.split("/")[-1]
    ga = folder / ".gitattributes"
    if ga.exists():
        ga.unlink()  # HF-style LFS rules; let ModelScope use its own
    print(f"[upload] {rid} <- {folder}", flush=True)
    for f in sorted(folder.iterdir()):
        print(f"         {f.name}  {f.stat().st_size} bytes", flush=True)
    try:
        api.upload_folder(repo_id=rid, folder_path=str(folder),
                          commit_message="Mirror from Hugging Face (LCGA)")
        print(f"[ok] {rid}", flush=True)
    except Exception as exc:
        print(f"[ERR] {rid}: {type(exc).__name__}: {exc}", flush=True)

print("\n[verify]", flush=True)
for rid in REPOS:
    try:
        files = api.get_model_files(rid, recursive=True)
        names = [(f.get("Path") or f.get("Name"), f.get("Size")) for f in files]
        print(f"  {rid}: {names}", flush=True)
    except Exception as exc:
        print(f"  {rid}: verify failed {type(exc).__name__}: {exc}", flush=True)
