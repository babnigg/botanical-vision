"""Score a model on the held-out test split.

Streams the HF ``-256`` test split (no full download) and reports top-1 / top-5 on a
bounded sample — the numbers the leaderboard ranks by. Small + streamed so it runs on
a laptop. Handy before publishing, to see your own number:

    python -m share.score --checkpoint checkpoints/resnet50_improved_best.pt
"""
from __future__ import annotations

import argparse

from .constants import HF_DATA_REPO


def score(predict_topk, limit: int = 300, hf_repo: str = HF_DATA_REPO) -> dict:
    """Run ``predict_topk(pil_rgb) -> [species names]`` over the streamed test split.

    Model-agnostic: the caller supplies the predictor, so the same loop scores any
    shared checkpoint the leaderboard pulls.
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


def main() -> None:
    from pathlib import Path

    from .constants import ROOT
    from .model import build_predictor

    ap = argparse.ArgumentParser(description="Score a local checkpoint on the test split.")
    ap.add_argument("--checkpoint", required=True, help="a training .pt or published bundle")
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args()
    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    m = score(build_predictor(str(ckpt)), args.limit)
    print(f"{ckpt.name}  n={m['n']}  top1={m['top1']:.3f}  top5={m['top5']:.3f}")


if __name__ == "__main__":
    main()
