"""Compare everyone's shared models — one command.

Pulls every model teammates published to the shared Hugging Face repo, scores each on
the *same* held-out test split, and prints a ranked table. This is the payoff of the
whole setup: an apples-to-apples comparison so the team can see whose model is best.

    python -m mlops.leaderboard --limit 300
"""
from __future__ import annotations

import argparse

from mlops import evaluate, hub
from mlops.model import MEAN, STD


def _predictor_from_bundle(bundle_path: str):
    """Build a top-5 predictor from a shared checkpoint bundle (no MLflow needed)."""
    import torch
    from torch import nn
    from torchvision import models, transforms

    b = torch.load(bundle_path, map_location="cpu", weights_only=False)
    labels = b["labels"]
    net = models.resnet50()
    net.fc = nn.Linear(net.fc.in_features, len(labels))
    net.load_state_dict(b["state_dict"])
    net.eval()
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(b.get("img_size", 224)),
        transforms.ToTensor(),
        transforms.Normalize(b.get("mean", MEAN), b.get("std", STD)),
    ])

    def predict_topk(img):
        x = tf(img).unsqueeze(0)
        with torch.no_grad():
            prob = net(x).float().softmax(1)[0]
        return [labels[int(i)] for i in prob.topk(5).indices]

    return predict_topk


def leaderboard(limit: int = 300) -> list[dict]:
    models = hub.list_models()
    if not models:
        print(f"no shared models yet in {hub.SHARED_REPO}.")
        print("be the first: python -m mlops.publish --checkpoint <ckpt> --name <name>")
        return []

    rows = []
    for m in models:
        path = hub.pull(m["author"], m["name"])
        res = evaluate.score(_predictor_from_bundle(path), limit)
        rows.append({**m, **res})
        print(f"  scored {m['author']}/{m['name']}: top1={res['top1']:.3f}")
    rows.sort(key=lambda r: r["top1"], reverse=True)

    print(f"\n{'#':<4}{'model':<30}{'top-1':>8}{'top-5':>8}   notes")
    for i, r in enumerate(rows, 1):
        tag = "   <- champion" if i == 1 else ""
        name = f"{r['author']}/{r['name']}"
        print(f"{i:<4}{name:<30}{r['top1']:>8.3f}{r['top5']:>8.3f}   {r.get('notes', '')}{tag}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Rank the team's shared models on the same test split.")
    ap.add_argument("--limit", type=int, default=300, help="test images per model (higher = slower, more exact)")
    args = ap.parse_args()
    leaderboard(args.limit)


if __name__ == "__main__":
    main()
