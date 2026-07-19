"""The pickable model list — local checkpoints + the shared Hugging Face repo.

Local models are the `*_best.pt` files in `checkpoints/` (species count + best val read
from each, cached by mtime). Shared models come from the team's HF model repo (read from
the tiny `.json` sidecars — no weights downloaded just to list them). Self-contained: this
duplicates a little HF-listing logic so the demo doesn't depend on the core `share/` package.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "checkpoints"
SHARED_REPO = os.environ.get("BV_MODEL_REPO", "dbabnigg/botanical-vision-models")

INDEX_FILE = CKPT_DIR / ".model_index.json"   # cache so we don't reload big .pt on every restart


def _read_index() -> dict:
    try:
        return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_index(idx: dict) -> None:
    try:
        INDEX_FILE.write_text(json.dumps(idx), encoding="utf-8")
    except Exception:
        pass


_INDEX = _read_index()   # {filename: {mtime, species, val_acc, epochs}}


def _rank(items: list[dict]) -> list[dict]:
    """Most-capable first (most species, then highest val acc) — a display aid, not an
    auto-pick. Selection is always the user's; this just surfaces the best candidates."""
    return sorted(items, key=lambda m: (-(m["species"] or 0), -(m["val_acc"] or 0)))


def _local_meta(path: str) -> dict:
    name = os.path.basename(path)
    mt = os.path.getmtime(path)
    e = _INDEX.get(name)
    if e and e.get("mtime") == mt:
        return {"species": e["species"], "val_acc": e["val_acc"], "epochs": e["epochs"]}
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    va = (ck.get("hist") or {}).get("val_acc") or []
    meta = {"species": len(ck.get("labels", [])),
            "val_acc": (round(max(va), 3) if va else None),
            "epochs": len(va)}
    _INDEX[name] = {"mtime": mt, **meta}
    _write_index(_INDEX)
    return meta


def list_local() -> list[dict]:
    out = []
    for p in sorted(glob.glob(str(CKPT_DIR / "*_best.pt"))):
        try:
            m = _local_meta(p)
        except Exception:
            continue
        out.append({"id": f"local:{Path(p).name}", "source": "local", "name": Path(p).stem,
                    "species": m["species"], "val_acc": m["val_acc"], "detail": f"{m['epochs']} epochs"})
    return _rank(out)


def hf_list() -> list[dict]:
    """Raw metadata sidecars for every shared model (empty if offline / repo missing)."""
    from huggingface_hub import HfApi, hf_hub_download
    try:
        files = HfApi().list_repo_files(SHARED_REPO, repo_type="model")
    except Exception:
        return []
    out = []
    for f in files:
        if f.endswith(".json"):
            try:
                with open(hf_hub_download(SHARED_REPO, f, repo_type="model")) as fh:
                    out.append(json.load(fh))
            except Exception:
                pass
    return out


def hf_pull(author: str, name: str) -> str:
    """Download a shared model bundle; return the local cached path."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(SHARED_REPO, f"{author}/{name}.pt", repo_type="model")


def list_shared() -> list[dict]:
    out = []
    for m in hf_list():
        out.append({"id": f"shared:{m['author']}/{m['name']}", "source": "shared",
                    "name": f"{m['author']}/{m['name']}", "species": m.get("num_classes"),
                    "val_acc": m.get("val_acc"), "detail": m.get("notes") or ""})
    return _rank(out)


def list_all(active_id: str | None = None) -> list[dict]:
    items = list_local() + list_shared()
    for it in items:
        it["active"] = (it["id"] == active_id)
    return items
