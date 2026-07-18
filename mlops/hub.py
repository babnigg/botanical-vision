"""The team's shared model store — a Hugging Face model repo.

The whole point of the project for the class: everyone trains their own better models,
then shares their best here so we can compare them apples-to-apples. This is the model
equivalent of how the *dataset* already lives on the Hub — weights never go in git, they
go here. Two commands wrap this module:

    python -m mlops.publish      --checkpoint <ckpt> --name my-model   # share yours
    python -m mlops.leaderboard                                        # compare everyone's

First time only: ``huggingface-cli login`` (or set HF_TOKEN).
"""
from __future__ import annotations

import json
import os

# Shared repo everyone reads/writes. Override with BV_MODEL_REPO if the team forks it.
SHARED_REPO = os.environ.get("BV_MODEL_REPO", "dbabnigg/botanical-vision-models")


def _api():
    from huggingface_hub import HfApi

    return HfApi()


def whoami() -> str | None:
    """Logged-in HF username (used as the default author), or None if not logged in."""
    try:
        return _api().whoami().get("name")
    except Exception:
        return None


def upload_model(bundle_path: str, meta: dict) -> str:
    """Upload a lean checkpoint bundle + a metadata sidecar under ``{author}/{name}``."""
    api = _api()
    api.create_repo(SHARED_REPO, repo_type="model", exist_ok=True)
    base = f"{meta['author']}/{meta['name']}"
    api.upload_file(path_or_fileobj=bundle_path, path_in_repo=f"{base}.pt",
                    repo_id=SHARED_REPO, repo_type="model")
    api.upload_file(path_or_fileobj=json.dumps(meta, indent=2).encode(),
                    path_in_repo=f"{base}.json", repo_id=SHARED_REPO, repo_type="model")
    return base


def list_models() -> list[dict]:
    """Metadata for every shared model (read from the tiny ``.json`` sidecars)."""
    from huggingface_hub import hf_hub_download

    try:
        files = _api().list_repo_files(SHARED_REPO, repo_type="model")
    except Exception:
        return []
    out = []
    for f in files:
        if f.endswith(".json"):
            with open(hf_hub_download(SHARED_REPO, f, repo_type="model")) as fh:
                out.append(json.load(fh))
    return out


def pull(author: str, name: str) -> str:
    """Download a shared checkpoint bundle; return the local cached path."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(SHARED_REPO, f"{author}/{name}.pt", repo_type="model")
