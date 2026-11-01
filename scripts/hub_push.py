#!/usr/bin/env python3
"""Upload a LoRA adapter + its model card to the Hugging Face Hub. Called by ftc/publish.py.

Runs in the training venv (.venv-train has huggingface_hub). Reads the token from $HF_TOKEN.
Prints one JSON object on stdout.

    python3 scripts/hub_push.py whoami
    python3 scripts/hub_push.py upload --repo user/name --adapter <dir> --readme model_card.md \
        [--extra final_report.json heldout_answers.jsonl] [--public]
"""
import argparse
import json
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("whoami")
    up = sub.add_parser("upload")
    up.add_argument("--repo", required=True)
    up.add_argument("--adapter", required=True)
    up.add_argument("--readme", required=True)
    up.add_argument("--extra", nargs="*", default=[], help="files uploaded into eval/ in the repo")
    up.add_argument("--public", action="store_true", help="create the repo public (default: private)")
    a = ap.parse_args()
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return out({"ok": False, "error": "huggingface_hub is not installed in this Python (use .venv-train)"})
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        return out({"ok": False, "error": "HF_TOKEN is not set (add it to .env)"})
    api = HfApi(token=token)
    try:
        if a.cmd == "whoami":
            w = api.whoami()
            return out({"ok": True, "user": w.get("name"), "orgs": [o.get("name") for o in w.get("orgs") or []]})
        adapter = Path(a.adapter)
        if not (adapter / "adapter_config.json").exists():
            return out({"ok": False, "error": f"{adapter} has no adapter_config.json: not a LoRA adapter folder"})
        api.create_repo(a.repo, private=not a.public, exist_ok=True, repo_type="model")
        api.upload_folder(repo_id=a.repo, folder_path=str(adapter), commit_message="Upload LoRA adapter",
                          ignore_patterns=["checkpoint-*", "*.pt", "optimizer*", "rng_state*", "scheduler*"])
        api.upload_file(repo_id=a.repo, path_or_fileobj=a.readme, path_in_repo="README.md",
                        commit_message="Model card (ftc/report.py)")
        for f in a.extra:
            if Path(f).exists():
                api.upload_file(repo_id=a.repo, path_or_fileobj=f, path_in_repo=f"eval/{Path(f).name}",
                                commit_message=f"Evaluation: {Path(f).name}")
        private = getattr(api.repo_info(a.repo), "private", not a.public)
        return out({"ok": True, "repo_id": a.repo, "url": f"https://huggingface.co/{a.repo}", "private": private})
    except Exception as e:
        return out({"ok": False, "error": f"{type(e).__name__}: {e}"[:800]})


def out(obj):
    print(json.dumps(obj))
    return 0 if obj.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
