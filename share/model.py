"""Rebuild a classifier from a shared bundle — no MLflow, no registry.

A published bundle is just ``{state_dict, labels, arch, img_size, mean, std}``
(see :mod:`share.publish`). This turns one back into a
``predict_topk(pil) -> [species names]`` function, reused by the leaderboard and the
local score CLI. Torch is imported lazily so importing this module is cheap.
"""
from __future__ import annotations

from .constants import IMG_SIZE, MEAN, STD, TOP_K


def load_bundle(path: str) -> dict:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def build_predictor(bundle_or_path, topk: int = TOP_K):
    """Return ``predict_topk(pil) -> [species names]`` for a bundle (path or dict)."""
    import torch
    from torch import nn
    from torchvision import models, transforms

    b = load_bundle(bundle_or_path) if isinstance(bundle_or_path, str) else bundle_or_path
    labels = b["labels"]
    net = models.resnet50()
    net.fc = nn.Linear(net.fc.in_features, len(labels))
    net.load_state_dict(b["state_dict"])
    net.eval()
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(b.get("img_size", IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(b.get("mean", MEAN), b.get("std", STD)),
    ])

    def predict_topk(img):
        x = tf(img.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            prob = net(x).float().softmax(1)[0]
        return [labels[int(i)] for i in prob.topk(topk).indices]

    return predict_topk
