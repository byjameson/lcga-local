"""Replace (not append) the "Latest findings" section of the HF / ModelScope cards.

`update_cards.py` appends and skips when the marker is already present, which is what
we want for the first publication. Iterating on the section needs a *replace*: keep
everything before the marker line and append the new section.

Usage:
  python scripts/replace_card_section.py --section dist/card_update.md \
      --marker "Latest findings (25 Oct 2026)" --only lcga-v3-0.5b
Tokens come from HF_TOKEN / MS_TOKEN (never written to disk).
"""
import argparse
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF_REPOS = ["byjameson/lcga-v3-0.5b"]
MS_REPOS = ["byjameson/lcga-v3-0.5b"]


def cut_before_marker(text: str, marker: str) -> str:
    idx = text.find(marker)
    if idx < 0:
        return text.rstrip() + "\n"
    line_start = text.rfind("\n", 0, idx)
    return text[: line_start + 1].rstrip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", required=True)
    ap.add_argument("--marker", required=True)
    ap.add_argument("--hf", action="store_true", default=True)
    ap.add_argument("--no-hf", action="store_false", dest="hf")
    ap.add_argument("--ms", action="store_true", default=True)
    ap.add_argument("--no-ms", action="store_false", dest="ms")
    args = ap.parse_args()

    section = (ROOT / args.section).read_text(encoding="utf-8").strip() + "\n"

    if args.hf:
        tok = os.environ.get("HF_TOKEN", "")
        if not tok:
            print("[hf] HF_TOKEN missing -> skipped", flush=True)
        else:
            from huggingface_hub import HfApi, hf_hub_download
            api = HfApi(token=tok)
            for rid in HF_REPOS:
                try:
                    cur = Path(hf_hub_download(rid, "README.md", token=tok,
                                               force_download=True)).read_text(encoding="utf-8")
                    new = cut_before_marker(cur, args.marker) + "\n" + section
                    tmp = Path(tempfile.gettempdir()) / f"_card_{rid.split('/')[-1]}.md"
                    tmp.write_text(new, encoding="utf-8")
                    api.upload_file(path_or_fileobj=str(tmp), path_in_repo="README.md",
                                    repo_id=rid, repo_type="model",
                                    commit_message="docs: refresh LCGA findings section")
                    tmp.unlink(missing_ok=True)
                    print(f"[hf] {rid}: section replaced ({len(new)} B)", flush=True)
                except Exception as exc:
                    print(f"[hf] {rid}: FAIL {type(exc).__name__}: {exc}", flush=True)

    if args.ms:
        tok = os.environ.get("MS_TOKEN", "")
        if not tok:
            print("[ms] MS_TOKEN missing -> skipped", flush=True)
        else:
            from modelscope.hub.api import HubApi
            api = HubApi()
            api.login(tok)
            for rid in MS_REPOS:
                try:
                    local = ROOT / "dist" / "modelscope" / rid.split("/")[-1] / "README.md"
                    if not local.exists():
                        print(f"[ms] {rid}: no staged README", flush=True)
                        continue
                    cur = local.read_text(encoding="utf-8")
                    new = cut_before_marker(cur, args.marker) + "\n" + section
                    local.write_text(new, encoding="utf-8")
                    api.upload_file(path_or_fileobj=str(local), path_in_repo="README.md",
                                    repo_id=rid,
                                    commit_message="docs: refresh LCGA findings section")
                    print(f"[ms] {rid}: section replaced ({len(new)} B)", flush=True)
                except Exception as exc:
                    print(f"[ms] {rid}: FAIL {type(exc).__name__}: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
