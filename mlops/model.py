"""MLflow pyfunc "flavor" for the Botanical Vision classifier.

The eval transform and the label vocabulary are bundled with the weights, so the
serving side never re-implements preprocessing — this is the course's explicit
anti-train/serve-skew pattern (Session 4, "MLflow Models wraps your model in a
consistent flavor"). The model takes base64-encoded images and returns, per image,
a top-5 list of ``{species, confidence}``.

Torch/torchvision are imported lazily inside the methods so this module can be
imported (e.g. for logging a model) without them loaded at import time.
"""
from __future__ import annotations

import base64
import io

import mlflow.pyfunc

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224
TOP_K = 5


class BotanicalClassifier(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import torch
        from torch import nn
        from torchvision import models, transforms

        bundle = torch.load(context.artifacts["bundle"], map_location="cpu", weights_only=False)
        self._labels = bundle["labels"]
        net = models.resnet50()
        net.fc = nn.Linear(net.fc.in_features, len(self._labels))
        net.load_state_dict(bundle["state_dict"])
        net.eval()
        self._net = net
        self._tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def _open(self, item: str):
        from PIL import Image

        raw = base64.b64decode(item)
        return Image.open(io.BytesIO(raw)).convert("RGB")

    def predict(self, context, model_input, params=None):
        import torch

        out = []
        for item in _as_list(model_input):
            try:
                x = self._tf(self._open(item)).unsqueeze(0)
                with torch.no_grad():
                    prob = self._net(x).float().softmax(1)[0]
                vals, idx = prob.topk(TOP_K)
                out.append([
                    {"species": self._labels[int(i)], "confidence": round(float(v) * 100, 2)}
                    for v, i in zip(vals, idx)
                ])
            except Exception:
                out.append([])
        return out


def _as_list(model_input):
    """Accept a DataFrame column, a list/array, or a single base64 string."""
    try:
        import pandas as pd

        if isinstance(model_input, pd.DataFrame):
            col = "image_b64" if "image_b64" in model_input.columns else model_input.columns[0]
            return list(model_input[col])
    except Exception:
        pass
    if isinstance(model_input, str):
        return [model_input]
    return list(model_input)
