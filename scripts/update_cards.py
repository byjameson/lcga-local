"""Append an update section to the model cards on Hugging Face and ModelScope.

Idempotent: if the marker string is already present in a card, that card is
skipped. Tokens are read from the environment (HF_TOKEN / MS_TOKEN) and never
written to disk.

Usage:
  python scripts/update_cards.py --section dist/card_update.md \
      --marker "Latest findings (24.10.2026)"
"""
import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF_REPOS = ["byjameson/lcga-v3-0.5b", "byjameson/lcga-p3-17m"]
MS_REPOS = ["byjameson/lcga-v3-0.5b", "byjameson/lcga-p3-17m"]


def strip_front_matter(text: str) -> tuple[str, str]:
    """HF model cards start with a YAML block; keep it untouched."""
    m = re.match(r"^---\n.*?\n---\n", text, re.S)
    if m:
        return text[:m.end()], text[m.end():]
    return "", text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", required=True, help="markdown file to append")
    ap.add_argument("--marker", required=True, help="idempotency marker")
    ap.add_argument("--hf", action="store_true", default=True)
    ap.add_argument("--no-hf", action="store_false", dest="hf")
    ap.add_argument("--ms", action="store_true", default=True)
    ap.add_argument("--no-ms", action="store_false", dest="ms")
    ap.add_argument("--only", default="",
                    help="comma-separated repo suffixes to update (e.g. lcga-v3-0.5b); "
                         "default = all. Needed because the 17M P3 model is unrelated "
                         "to the LCGA findings.")
    args = ap.parse_args()

    only = {s.strip() for s in args.only.split(",") if s.strip()}

    def wanted(rid: str) -> bool:
        return not only or rid.split("/")[-1] in only

    section = (ROOT / args.section).read_text(encoding="utf-8").strip() + "\n"

    if args.hf:
        hf_token = os.environ.get("HF_TOKEN", "")
        if not hf_token:
            print("[hf] HF_TOKEN missing -> skipped", flush=True)
        else:
            from huggingface_hub import HfApi, hf_hub_download

            api = HfApi(token=hf_token)
            for rid in HF_REPOS:
                if not wanted(rid):
                    print(f"[hf] {rid}: skipped (--only)", flush=True)
                    continue
                try:
                    cur = Path(hf_hub_download(rid, "README.md", token=hf_token,
                                               force_download=True)).read_text(encoding="utf-8")
                    if args.marker in cur:
                        print(f"[hf] {rid}: already updated", flush=True)
                        continue
                    head, body = strip_front_matter(cur)
                    tmp = Path(os.environ.get("TEMP", ".")) / f"_card_{rid.split('/')[-1]}.md"
                    tmp.write_text(head + body.rstrip() + "\n\n" + section, encoding="utf-8")
                    api.upload_file(path_or_fileobj=str(tmp), path_in_repo="README.md",
                                    repo_id=rid, repo_type="model",
                                    commit_message="docs: latest LCGA findings")
                    tmp.unlink(missing_ok=True)
                    print(f"[hf] {rid}: updated", flush=True)
                except Exception as exc:
                    print(f"[hf] {rid}: FAIL {type(exc).__name__}: {exc}", flush=True)

    if args.ms:
        ms_token = os.environ.get("MS_TOKEN", "")
        if not ms_token:
            print("[ms] MS_TOKEN missing -> skipped", flush=True)
        else:
            from modelscope.hub.api import HubApi

            api = HubApi()
            api.login(ms_token)
            for rid in MS_REPOS:
                if not wanted(rid):
                    print(f"[ms] {rid}: skipped (--only)", flush=True)
                    continue
                try:
                    files = api.get_model_files(rid, recursive=True)
                    readme = next((f for f in files
                                   if (f.get("Path") or f.get("Name")) == "README.md"), None)
                    # ModelScope has no simple "read file" helper here: re-upload the
                    # whole folder from dist/modelscope (staged locally).
                    folder = ROOT / "dist" / "modelscope" / rid.split("/")[-1]
                    local = folder / "README.md"
                    if not local.exists():
                        print(f"[ms] {rid}: no staged README", flush=True)
                        continue
                    text = local.read_text(encoding="utf-8")
                    if args.marker in text:
                        print(f"[ms] {rid}: already updated", flush=True)
                        continue
                    local.write_text(text.rstrip() + "\n\n" + section, encoding="utf-8")
                    api.upload_file(path_or_fileobj=str(local), path_in_repo="README.md",
                                    repo_id=rid,
                                    commit_message="docs: latest LCGA findings")
                    print(f"[ms] {rid}: updated", flush=True)
                except Exception as exc:
                    print(f"[ms] {rid}: FAIL {type(exc).__name__}: {exc}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
