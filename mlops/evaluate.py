"""Score a model on the held-out test split.

Streams the HF ``-256`` test split (no full download) and reports top-1 / top-5 on a
bounded sample — the numbers the team leaderboard ranks by and the promotion gate
compares. Kept deliberately small/streamed so it runs on a laptop.

    python -m mlops.evaluate --version 2 --limit 300
"""
from __future__ import annotations

import argparse
import base64
import io

from mlops import config

HF_REPO = "dbabnigg/botanical-vision-256"


def score(predict_topk, limit: int = 300, hf_repo: str = HF_REPO) -> dict:
    """Run ``predict_topk(pil_rgb) -> [species names]`` over the streamed test split.

    Model-agnostic: the caller supplies the predictor, so the same loop scores an MLflow
    model (the gate) or a raw shared checkpoint (the leaderboard).
    """
    from datasets import load_dataset

    ds = load_dataset(hf_repo, split="test", streaming=True)
    label_feature = ds.features.get("label") if getattr(ds, "features", None) else None

    n = t1 = t5 = 0
    for i, ex in enumerate(ds):
        if i >= limit:
            break
        img = ex.get("image")
        if img is None:
            continue
        truth = ex.get("species")
        if truth is None and label_feature is not None and "label" in ex:
            truth = label_feature.int2str(ex["label"])
        names = predict_topk(img.convert("RGB"))
        n += 1
        t1 += int(bool(names) and names[0] == truth)
        t5 += int(truth in names)
    if n == 0:
        return {"n": 0, "top1": 0.0, "top5": 0.0}
    return {"n": n, "top1": t1 / n, "top5": t5 / n}


def evaluate(model_uri: str, limit: int = 300, hf_repo: str = HF_REPO) -> dict:
    """Score a registered MLflow pyfunc version (used by the promotion gate)."""
    import mlflow
    import pandas as pd

    config.configure()
    model = mlflow.pyfunc.load_model(model_uri)

    def predict_topk(img):
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        preds = model.predict(pd.DataFrame({"image_b64": [b64]}))[0]
        return [p["species"] for p in preds]

    return score(predict_topk, limit, hf_repo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-uri", help="e.g. models:/botanical-classifier@production")
    ap.add_argument("--version", help="registered version number (alternative to --model-uri)")
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args()
    uri = args.model_uri or f"models:/{config.MODEL_NAME}/{args.version}"
    m = evaluate(uri, args.limit)
    print(f"{uri}  n={m['n']}  top1={m['top1']:.3f}  top5={m['top5']:.3f}")


if __name__ == "__main__":
    main()
