"""Share your trained model with the team — one command.

Pulls the weights + labels out of a training checkpoint, drops the bulky optimizer
state, and uploads the lean bundle to the shared Hugging Face model repo so teammates
can pull and compare it. First time only: ``huggingface-cli login``.

    python -m mlops.publish --checkpoint checkpoints/resnet50_improved_best.pt \
        --name randaug-v2 --notes "randaug + cosine, 12 epochs"
"""
from __future__ import annotations

import argparse
import tempfile
from datetime import date
from pathlib import Path

from mlops import config, hub
from mlops.model import IMG_SIZE, MEAN, STD


def _lean_bundle(ckpt_path: Path, tmp: str) -> tuple[str, dict]:
    """Write a serving-ready bundle (weights + labels + transform meta) and return its
    path plus a few stats for the sidecar. Drops optimizer/scaler state to halve size."""
    import torch

    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    labels = payload["labels"]
    hist = payload.get("hist") or {}
    val_acc = float(hist["val_acc"][-1]) if hist.get("val_acc") else None
    bundle = {
        "state_dict": payload["state_dict"], "labels": labels,
        "arch": "resnet50", "img_size": IMG_SIZE, "mean": MEAN, "std": STD,
    }
    path = Path(tmp) / "bundle.pt"
    torch.save(bundle, path)
    return str(path), {"num_classes": len(labels), "val_acc": val_acc}


def publish(checkpoint: str, name: str, author: str | None = None, notes: str = "") -> str:
    author = author or hub.whoami()
    if not author:
        raise SystemExit("no author — pass --author or run `huggingface-cli login` first")
    ckpt = Path(checkpoint)
    if not ckpt.is_absolute():
        ckpt = config.ROOT / ckpt
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    with tempfile.TemporaryDirectory() as tmp:
        bundle_path, stats = _lean_bundle(ckpt, tmp)
        meta = {
            "author": author, "name": name, "arch": "resnet50", "notes": notes,
            "created": date.today().isoformat(), "source_checkpoint": ckpt.name, **stats,
        }
        base = hub.upload_model(bundle_path, meta)

    va = f" (val_acc {stats['val_acc']:.3f})" if stats.get("val_acc") is not None else ""
    print(f"shared {base}{va}  ->  https://huggingface.co/{hub.SHARED_REPO}")
    print("teammates can now run `python -m mlops.leaderboard` to compare.")
    return base


def main() -> None:
    ap = argparse.ArgumentParser(description="Share your trained model with the team.")
    ap.add_argument("--checkpoint", required=True, help="path (relative to project root or absolute)")
    ap.add_argument("--name", required=True, help="short label for your model, e.g. 'randaug-v2'")
    ap.add_argument("--author", default=None, help="defaults to your Hugging Face username")
    ap.add_argument("--notes", default="", help="what you changed (shows up on the leaderboard)")
    args = ap.parse_args()
    publish(args.checkpoint, args.name, args.author, args.notes)


if __name__ == "__main__":
    main()
