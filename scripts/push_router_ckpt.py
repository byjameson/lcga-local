"""Upload a router checkpoint to the HF / ModelScope model repos.

The GitHub repository versions code and results, not weights (`checkpoints/` is
git-ignored), so the trained routers are published as model-repo artifacts.

Usage:
  python scripts/push_router_ckpt.py --ckpt checkpoints/mix7_final.pt \
      --name router_needle_mix7.pt --hf --ms
Tokens come from HF_TOKEN / MS_TOKEN (never written to disk).
"""
import argparse
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF_REPO = "byjameson/lcga-v3-0.5b"
MS_REPO = "byjameson/lcga-v3-0.5b"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--name", required=True, help="filename inside the model repo")
    ap.add_argument("--commit", default="weights: add needle-trained router")
    ap.add_argument("--hf", action="store_true")
    ap.add_argument("--ms", action="store_true")
    args = ap.parse_args()

    src = Path(args.ckpt)
    if not src.is_absolute():
        src = ROOT / src
    if not src.exists():
        print(f"[err] {src} does not exist")
        return 1
    mb = src.stat().st_size / 1e6
    print(f"[ckpt] {src.name} ({mb:.1f} MB) -> {args.name}")

    if args.hf:
        tok = os.environ.get("HF_TOKEN", "")
        if not tok:
            print("[hf] HF_TOKEN missing -> skipped")
        else:
            from huggingface_hub import HfApi
            HfApi(token=tok).upload_file(path_or_fileobj=str(src), path_in_repo=args.name,
                                         repo_id=HF_REPO, repo_type="model",
                                         commit_message=args.commit)
            print(f"[hf] {HF_REPO}:{args.name} uploaded")

    if args.ms:
        tok = os.environ.get("MS_TOKEN", "")
        if not tok:
            print("[ms] MS_TOKEN missing -> skipped")
        else:
            from modelscope.hub.api import HubApi
            api = HubApi()
            api.login(tok)
            staged = ROOT / "dist" / "modelscope" / MS_REPO.split("/")[-1] / args.name
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, staged)
            api.upload_file(path_or_fileobj=str(staged), path_in_repo=args.name,
                            repo_id=MS_REPO, commit_message=args.commit)
            print(f"[ms] {MS_REPO}:{args.name} uploaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
