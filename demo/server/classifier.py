"""Classifier serving for the demo — you pick which model to serve (no auto-pick).

The active model is chosen via /api/models/select and persisted to a small file next to
the checkpoints, so it survives restarts. It can be a local checkpoint (checkpoints/*.pt)
or one pulled from the shared Hugging Face repo — either way a {state_dict, labels, ...}
bundle. We rebuild ResNet-50 + the eval transform and return top-5 {species, confidence}.
If torch is missing or nothing is selected, a labeled stub keeps the frontend alive.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

from .traits import taxonomy

ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "checkpoints"
SEL_FILE = CKPT_DIR / ".demo_selected.json"

# ImageNet normalization + 224 crop — must match the notebooks (03/04/05).
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224
TOP_K = 5

_STUB = {
    "coneflower": [("Echinacea purpurea", 62.4), ("Echinacea pallida", 18.1),
                   ("Rudbeckia hirta", 9.7), ("Ratibida pinnata", 4.3),
                   ("Echinacea angustifolia", 2.1)],
    "iris": [("Iris versicolor", 47.8), ("Iris virginica", 24.9),
             ("Iris setosa", 12.1), ("Sisyrinchium montanum", 6.0),
             ("Iris lacustris", 3.4)],
}

_predictor = None
_loaded_id: str | None = None
_active: dict | None = None


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def active_id() -> str | None:
    """The currently-selected model id (persisted), or None if nothing is chosen yet."""
    if SEL_FILE.exists():
        try:
            return json.loads(SEL_FILE.read_text(encoding="utf-8")).get("id")
        except Exception:
            return None
    return None


def set_active(model_id: str) -> None:
    CKPT_DIR.mkdir(exist_ok=True)
    SEL_FILE.write_text(json.dumps({"id": model_id}), encoding="utf-8")
    global _predictor, _loaded_id, _active
    _predictor, _loaded_id, _active = None, None, None   # force a reload on next use


def clear_active() -> None:
    global _predictor, _loaded_id, _active
    _predictor, _loaded_id, _active = None, None, None
    try:
        SEL_FILE.unlink()
    except Exception:
        pass


def _resolve_path(model_id: str) -> Path | None:
    if model_id.startswith("local:"):
        p = CKPT_DIR / model_id.split(":", 1)[1]
        return p if p.exists() else None
    if model_id.startswith("shared:"):
        ref = model_id.split(":", 1)[1]
        if "/" not in ref:
            return None
        author, name = ref.split("/", 1)
        try:
            from .models import hf_pull
            return Path(hf_pull(author, name))
        except Exception:
            return None
    return None


def _build(path: Path):
    import torch
    from torch import nn
    from torchvision import models, transforms

    b = torch.load(str(path), map_location="cpu", weights_only=False)
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
        vals, idx = prob.topk(TOP_K)
        return [(labels[int(i)], float(v) * 100) for v, i in zip(vals, idx)]

    return predict_topk, labels


def _load():
    """Lazily load + cache the predictor for the currently-selected model."""
    global _predictor, _loaded_id, _active
    mid = active_id()
    if mid is None:
        _predictor, _loaded_id, _active = None, None, None
        return None
    if _predictor is not None and _loaded_id == mid:
        return _predictor
    if not torch_available():
        return None
    path = _resolve_path(mid)
    if path is None:
        _predictor, _active = None, None
        return None
    try:
        pred, labels = _build(path)
    except Exception:
        _predictor, _active = None, None
        return None
    name = mid.split(":", 1)[1]
    if mid.startswith("local:"):
        name = Path(name).stem
    _predictor, _loaded_id = pred, mid
    _active = {"id": mid, "source": mid.split(":", 1)[0], "name": name, "species": len(labels)}
    return _predictor


def served_model() -> dict | None:
    """{id, source, name, species} of the model actually loaded, or None."""
    _load()
    return _active


def _decorate(name: str, conf: float) -> dict:
    t = taxonomy(name)
    return {
        "species": name, "common": t["common"], "genus": t["genus"],
        "family": t["family"], "order": t["order"], "iucn": t["iucn"] or "LC",
        "confidence": round(conf, 2),
    }


def _agreement(top: list[dict]) -> dict:
    fam, gen = top[0]["family"], top[0]["genus"]
    return {
        "family": fam, "family_count": sum(1 for t in top if t["family"] == fam),
        "genus": gen, "genus_count": sum(1 for t in top if t["genus"] == gen),
    }


def _stub(sample: str | None, reason: str, note: str | None = None) -> dict:
    rows = _STUB.get(sample or "coneflower", _STUB["coneflower"])
    top = [_decorate(n, c) for n, c in rows]
    return {
        "live": False, "served": "stub", "reason": reason, "model": None,
        "predictions": top, "taxonomy_agreement": _agreement(top),
        "note": note or "Demo result — the classifier isn't running in this environment.",
    }


def predict(image_bytes: bytes | None, sample: str | None = None) -> dict:
    if not torch_available():
        return _stub(sample, "no_torch",
                     "PyTorch isn't installed here — demo result. "
                     "`pip install torch torchvision` to run the model on your image.")
    if active_id() is None:
        return _stub(sample, "no_model", "No model selected — pick one in the model picker above.")
    if _load() is None:
        return _stub(sample, "no_model", "The selected model couldn't be loaded — pick another.")
    if not image_bytes:
        return _stub(sample, "no_image")
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))
        rows = _predictor(img)
        top = [_decorate(n, c) for n, c in rows]
    except Exception:
        return _stub(sample, "infer_failed")
    return {
        "live": True, "served": "model", "reason": None,
        "model": (_active or {}).get("name"),
        "predictions": top, "taxonomy_agreement": _agreement(top), "note": None,
    }
