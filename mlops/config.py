"""Central MLOps configuration.

Everything is env-driven so the same code runs against a local SQLite-backed
MLflow (the default, laptop-runnable) or a shared Databricks Managed MLflow later
by setting a single variable (``MLFLOW_TRACKING_URI=databricks``).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # code/project
MLOPS_DIR = ROOT / "mlops"

# sqlite URIs need forward slashes, even on Windows.
_DEFAULT_DB = (MLOPS_DIR / "mlflow.db").as_posix()

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{_DEFAULT_DB}")
EXPERIMENT = os.environ.get("BV_EXPERIMENT", "botanical-vision")
MODEL_NAME = os.environ.get("BV_MODEL_NAME", "botanical-classifier")

# MLflow 3 replaced the Staging/Production *stages* the 32021 lectures show with
# version *aliases* (stages are deprecated). Same concept — the served model is the
# version carrying the "production" alias; a candidate carries "staging".
MODEL_ALIAS = os.environ.get("BV_MODEL_ALIAS", "production")

# Registered model name per module — the seam for adding other sections' models.
# When a section gets a trained model registered in MLflow, add its row here and the
# roadmap + status pick it up with no other code change. (Serving a generator model
# still needs its own pyfunc flavor + endpoint; this only drives the registry view.)
MODELS = {
    "identify": MODEL_NAME,
    # "illustrate": os.environ.get("BV_STYLE_MODEL", "botanical-style-lora"),
    # "compose": os.environ.get("BV_RENDER_MODEL", "botanical-landscape-cn"),
}


def configure():
    """Point the MLflow client at the configured tracking server and return it."""
    import mlflow

    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow


def model_uri(alias: str | None = None) -> str:
    return f"models:/{MODEL_NAME}@{alias or MODEL_ALIAS}"
