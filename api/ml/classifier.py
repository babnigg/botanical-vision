"""Classifier serving — loads the production model from the MLflow registry.

The registered model is a packaged pyfunc (eval transform + label vocabulary
bundled with the weights), so this layer only base64-encodes the image, calls
``predict``, and decorates the result with taxonomy. If no production model is
registered — or torch/mlflow can't load it — it returns a labeled stub so the app
still works during development.
"""
from __future__ import annotations

import base64

from mlops import config

from .traits import taxonomy

# Fallback demo top-5s, keyed by sample name, for when no model is available.
_STUB = {
    "coneflower": [("Echinacea purpurea", 62.4), ("Echinacea pallida", 18.1),
                   ("Rudbeckia hirta", 9.7), ("Ratibida pinnata", 4.3),
                   ("Echinacea angustifolia", 2.1)],
    "iris": [("Iris versicolor", 47.8), ("Iris virginica", 24.9),
             ("Iris setosa", 12.1), ("Sisyrinchium montanum", 6.0),
             ("Iris lacustris", 3.4)],
}

_model = None
_info: dict | None = None
_tried = False


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _load():
    """Lazily load + cache the production pyfunc model from the registry."""
    global _model, _info, _tried
    if _model is not None or _tried:
        return _model
    _tried = True
    try:
        mlflow = config.configure()
        from mlflow.tracking import MlflowClient

        mv = MlflowClient().get_model_version_by_alias(config.MODEL_NAME, config.MODEL_ALIAS)
        _model = mlflow.pyfunc.load_model(config.model_uri())
        _info = {"name": config.MODEL_NAME, "version": mv.version, "alias": config.MODEL_ALIAS}
    except Exception:
        _model, _info = None, None
    return _model


def served_model() -> dict | None:
    """{name, version, alias} of the model actually loaded, or None."""
    _load()
    return _info


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
    if _load() is None:
        return _stub(sample, "no_model",
                     "No production model in the registry — showing a demo result.")
    if not image_bytes:
        return _stub(sample, "no_image")
    try:
        import pandas as pd

        b64 = base64.b64encode(image_bytes).decode()
        raw = _model.predict(pd.DataFrame({"image_b64": [b64]}))[0]
        top = [_decorate(p["species"], p["confidence"]) for p in raw]
    except Exception:
        return _stub(sample, "infer_failed")
    return {
        "live": True, "served": "model", "reason": None,
        "model": (_info or {}).get("name"),
        "predictions": top, "taxonomy_agreement": _agreement(top), "note": None,
    }
