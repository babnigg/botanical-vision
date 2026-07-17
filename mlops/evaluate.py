"""Evaluate a registered model version on the held-out test split.

Streams the HF ``-256`` test split (no full download), runs the pyfunc model on a
bounded sample, and reports top-1 / top-5 accuracy — the numbers the promotion gate
compares. Kept deliberately small/streamed so it runs on a laptop.

    python -m mlops.evaluate --version 2 --limit 300
"""
from __future__ import annotations

import argparse
import base64
import io

from mlops import config

HF_REPO = "dbabnigg/botanical-vision-256"


def evaluate(model_uri: str, limit: int = 300, hf_repo: str = HF_REPO) -> dict:
    import mlflow
    import pandas as pd
    from datasets import load_dataset

    config.configure()
    model = mlflow.pyfunc.load_model(model_uri)
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
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG")
        preds = model.predict(pd.DataFrame({"image_b64": [base64.b64encode(buf.getvalue()).decode()]}))[0]
        names = [p["species"] for p in preds]
        n += 1
        t1 += int(bool(names) and names[0] == truth)
        t5 += int(truth in names)
    if n == 0:
        return {"n": 0, "top1": 0.0, "top5": 0.0}
    return {"n": n, "top1": t1 / n, "top5": t5 / n}


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
