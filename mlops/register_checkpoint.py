"""Register a training checkpoint into MLflow as a packaged pyfunc model.

Extracts a lean bundle (state_dict + labels + meta) from a training ``.pt``, logs it
with params/metrics + a signature, registers a version of ``botanical-classifier``,
and optionally assigns an alias (``staging``/``production``). Exposes ``register()``
so the Prefect flow can call it directly.

    python -m mlops.register_checkpoint --checkpoint checkpoints/resnet50_baseline.pt --alias production
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from mlops import config
from mlops.model import IMG_SIZE, MEAN, STD, BotanicalClassifier


def _metrics_from_hist(hist: dict | None) -> dict:
    metrics: dict[str, float] = {}
    if not hist:
        return metrics
    for key in ("val_acc", "val_top5", "train_acc", "val_loss", "train_loss"):
        seq = hist.get(key)
        if seq:
            metrics[key] = float(seq[-1])
    return metrics


def register(checkpoint: str, alias: str | None = None, run_name: str | None = None) -> str:
    """Register the checkpoint; return the new version number (as a string)."""
    import torch
    from mlflow.models import ModelSignature
    from mlflow.tracking import MlflowClient
    from mlflow.types.schema import ColSpec, Schema

    mlflow = config.configure()
    mlflow.set_experiment(config.EXPERIMENT)

    ckpt = Path(checkpoint)
    if not ckpt.is_absolute():
        ckpt = config.ROOT / ckpt
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    labels = payload["labels"]
    hist = payload.get("hist", {})

    bundle = {
        "state_dict": payload["state_dict"], "labels": labels,
        "arch": "resnet50", "img_size": IMG_SIZE, "mean": MEAN, "std": STD,
    }
    signature = ModelSignature(
        inputs=Schema([ColSpec("string", "image_b64")]),
        outputs=Schema([ColSpec("string")]),
    )

    with tempfile.TemporaryDirectory() as tmp:
        bundle_path = Path(tmp) / "bundle.pt"
        torch.save(bundle, bundle_path)
        with mlflow.start_run(run_name=run_name or ckpt.stem):
            mlflow.log_params({
                "arch": "resnet50", "img_size": IMG_SIZE,
                "num_classes": len(labels), "source_checkpoint": ckpt.name,
            })
            if payload.get("sig") is not None:
                mlflow.set_tag("config_signature", str(payload["sig"]))
            for name, value in _metrics_from_hist(hist).items():
                mlflow.log_metric(name, value)
            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=BotanicalClassifier(),
                artifacts={"bundle": str(bundle_path)},
                signature=signature,
                pip_requirements=["torch", "torchvision", "pillow"],
                registered_model_name=config.MODEL_NAME,
            )

    client = MlflowClient()
    version = max(
        client.search_model_versions(f"name='{config.MODEL_NAME}'"),
        key=lambda v: int(v.version),
    ).version
    if alias:
        # An alias points to exactly one version, so assigning it rolls the champion over.
        client.set_registered_model_alias(config.MODEL_NAME, alias, version)
    return version


def main() -> None:
    ap = argparse.ArgumentParser(description="Register a checkpoint into MLflow.")
    ap.add_argument("--checkpoint", required=True, help="path (relative to project root or absolute)")
    ap.add_argument("--alias", default=None, choices=["staging", "production"])
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()
    version = register(args.checkpoint, args.alias, args.run_name)
    print(f"registered {config.MODEL_NAME} v{version}"
          + (f" @{args.alias}" if args.alias else ""))


if __name__ == "__main__":
    main()
