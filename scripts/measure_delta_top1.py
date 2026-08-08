"""
Measure Δtop-1 (and Δtop-5) on the shared test split with and without the
segmenter's soft-mask preprocessing. This is the number that answers "does the
segmentation model actually help the classifier?" for the writeup.

Runs the same soft-mask logic the demo backend uses (Gaussian-blur blend of the
background, NOT hard-mask to black — hard-mask is out-of-distribution for a
classifier trained on natural photos and reliably hurts top-1).

Usage:
    python scripts/measure_delta_top1.py \
        --classifier checkpoints/resnet50_improved_best.pt \
        --segmenter  checkpoints/deeplabv3_mnv3_seg_best.pt \
        [--n 5000]    # cap for a quick check; default = full test split
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image, ImageFilter
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import bvtrain as bv  # noqa: E402

BG_BLUR_RADIUS = 15   # keep in sync with demo/server/classifier.py:BG_BLUR_RADIUS


def _soft_mask(pil_img: Image.Image, prob_mask: torch.Tensor) -> Image.Image:
    """Alpha-blend the image with a strong-Gaussian-blur copy in the background.

    The segmenter (and classifier) run Resize(256, aspect-preserving) + CenterCrop(224),
    so the 224x224 mask corresponds to a centered ``(224 / scale)`` square of the
    original (``scale = 256 / min(w, h)``) — NOT to the full image. Upsample the mask
    to that square and paste it into a zeroed full-size alpha so the composite the
    classifier sees is geometrically aligned with the photo.
    """
    import numpy as np
    w, h = pil_img.size
    scale = 256 / min(w, h)
    side = min(int(round(bv.IMG_SIZE / scale)), w, h)
    arr = (prob_mask.numpy() * 255).astype(np.uint8)
    patch = Image.fromarray(arr, mode="L").resize((side, side), Image.BILINEAR)
    alpha = Image.new("L", (w, h), 0)
    alpha.paste(patch, ((w - side) // 2, (h - side) // 2))
    fg = pil_img.convert("RGB")
    bg = fg.filter(ImageFilter.GaussianBlur(radius=BG_BLUR_RADIUS))
    return Image.composite(fg, bg, alpha)


def _make_segment_fn(seg_model, env):
    from torchvision.transforms import functional as TF

    def _segment(pil_img: Image.Image) -> torch.Tensor:
        x = TF.resize(pil_img.convert("RGB"), 256)
        x = TF.center_crop(x, bv.IMG_SIZE)
        x = TF.to_tensor(x)
        x = TF.normalize(x, bv.MEAN, bv.STD).unsqueeze(0).to(env.device)
        with torch.no_grad():
            probs = torch.sigmoid(seg_model(x)["out"])[0, 0].cpu()
        return probs

    return _segment


def _classify(clf_model, pil_img: Image.Image, label2idx: dict, env) -> tuple[int, list[int]]:
    """Return (true-class-idx-target-not-known-here=-1, top-5 predicted indices)."""
    tf = bv.eval_transforms()
    x = tf(pil_img.convert("RGB")).unsqueeze(0).to(env.device)
    with torch.no_grad():
        logits = clf_model(x).float()
    top5 = logits.topk(5, dim=1).indices[0].cpu().tolist()
    return top5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--classifier", required=True, type=Path,
                   help="path to a classifier checkpoint (resnet50 bundle)")
    p.add_argument("--segmenter", required=True, type=Path,
                   help="path to a segmenter checkpoint (deeplabv3_mnv3 bundle)")
    p.add_argument("--n", type=int, default=None, help="cap examples (default: full test split)")
    args = p.parse_args()

    env = bv.setup()

    print(f"loading classifier: {args.classifier}")
    clf, clf_ck = bv.load_model(str(args.classifier), env)
    labels = clf_ck["labels"]
    label2idx = {s: i for i, s in enumerate(labels)}

    print(f"loading segmenter:  {args.segmenter}")
    seg, _ = bv.load_seg_model(str(args.segmenter), env)
    segment_fn = _make_segment_fn(seg, env)

    from datasets import load_dataset
    print(f"loading test split from {env.hf_repo} ...")
    ds = load_dataset(env.hf_repo, split="test")
    if args.n:
        ds = ds.select(range(min(args.n, len(ds))))
    print(f"scoring {len(ds):,} images (twice each — unmasked & masked)...")

    hits = {"unmasked": {1: 0, 5: 0}, "masked": {1: 0, 5: 0}}
    n = 0
    for ex in tqdm(ds, desc="scoring"):
        y = label2idx.get(ex["species"])
        if y is None:      # species not in this classifier's label set — skip
            continue
        img = ex["image"]

        top5_un = _classify(clf, img, label2idx, env)
        probs = segment_fn(img)
        img_masked = _soft_mask(img, probs)
        top5_m = _classify(clf, img_masked, label2idx, env)

        for kind, top5 in (("unmasked", top5_un), ("masked", top5_m)):
            if top5[0] == y:
                hits[kind][1] += 1
            if y in top5:
                hits[kind][5] += 1
        n += 1

    print(f"\n{'metric':<10}  {'unmasked':>10}  {'masked':>10}  {'Δ':>8}")
    for k in (1, 5):
        a = hits["unmasked"][k] / n
        b = hits["masked"][k] / n
        print(f"top-{k:<5}  {a*100:>9.2f}%  {b*100:>9.2f}%  {(b-a)*100:>+7.2f}pp")
    print(f"\n(n = {n:,} images with a known label)")


if __name__ == "__main__":
    main()
