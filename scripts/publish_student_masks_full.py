"""
Run the distilled student segmenter over the FULL HF training split of
``dbabnigg/botanical-vision-256`` and publish a paired companion dataset
{image, mask, species, split} — the input for the classifier's train-time
background-augmentation A/B (``notebooks/04_train_improved.ipynb`` bg-aug).

This is the HF-path counterpart to ``scripts/precompute_student_masks.py``
(which writes per-image PNGs to disk keyed by md5(local_path), for the local
train path only). The two exist because the classifier training path splits
local vs. HF and neither loader can read the other's mask key.

Pipeline:
    HF source split  ->  student inference (batched on GPU)
                     ->  1-bit PNG per (split, index) cached to disk
                     ->  HF DatasetDict {image, mask, species, split}
                     ->  push_to_hub

Auth: ``huggingface-cli login`` first (or set HF_TOKEN). Both caching and
upload are resumable — safe to Ctrl-C and rerun.

Usage:
    python scripts/publish_student_masks_full.py \
        --repo <username>/botanical-vision-256-student-masks-full \
        --segmenter checkpoints/deeplabv3_mnv3_seg_best.pt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "scripts"))
sys.path.insert(0, str(PROJECT_DIR))
from precompute_student_masks import load_student  # noqa: E402
from upload_to_hf import retry  # noqa: E402

DEFAULT_SOURCE_REPO = "dbabnigg/botanical-vision-256"
DEFAULT_CACHE_DIR = PROJECT_DIR / "data" / "student_masks_full"


def _batched_masks(model, imgs: list[PILImage.Image], dev, mean, std) -> list[PILImage.Image]:
    """Run the student on a batch of PIL images at 224 and return per-image
    L-mode PIL masks resized back to the source image sizes."""
    x = torch.stack([
        torch.from_numpy(np.asarray(im.resize((224, 224)))).permute(2, 0, 1).float() / 255
        for im in imgs
    ]).to(dev)
    x = (x - mean) / std
    with torch.no_grad():
        prob = torch.sigmoid(model(x)["out"])[:, 0]      # (B, 224, 224)
    masks = (prob.cpu().numpy() * 255).astype("uint8")
    return [
        PILImage.fromarray(masks[i], mode="L").resize(im.size, PILImage.BILINEAR)
        for i, im in enumerate(imgs)
    ]


def cache_masks(hf, split: str, model, dev, cache_dir: Path, batch: int = 16) -> None:
    """Iterate the split, write one L-PNG per image to cache_dir/<split>/<i>.png.
    Resumable — existing files are skipped without re-running the segmenter."""
    out = cache_dir / split
    out.mkdir(parents=True, exist_ok=True)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    todo = [i for i in range(len(hf[split])) if not (out / f"{i}.png").exists()]
    if not todo:
        print(f"{split}: all {len(hf[split]):,} masks already cached")
        return
    print(f"{split}: {len(todo):,} to generate ({len(hf[split]) - len(todo):,} cached)")

    for s in tqdm(range(0, len(todo), batch), desc=f"student[{split}]"):
        idxs = todo[s:s + batch]
        imgs = [hf[split][int(i)]["image"].convert("RGB") for i in idxs]
        masks = _batched_masks(model, imgs, dev, mean, std)
        for i, m in zip(idxs, masks):
            dest = out / f"{i}.png"
            tmp = out / f"{i}.png.tmp"
            m.save(tmp, optimize=True)
            os.replace(tmp, dest)  # atomic — a Ctrl-C mid-write leaves .tmp, not a corrupt .png


def build_dataset(hf, splits: list[str], cache_dir: Path):
    """Assemble the companion DatasetDict from source images + cached masks."""
    from datasets import Dataset, DatasetDict, Features, Image, Value

    features = Features({
        "image": Image(),
        "mask": Image(),
        "species": Value("string"),
        "split": Value("string"),
    })

    def gen(split: str):
        # decode=False returns the raw stored JPEG bytes — no decode/re-encode round-trip,
        # so the companion's images are byte-identical to the source dataset and the
        # bg-aug A/B is a true single-toggle comparison (control vs treatment train on
        # the same pixels).
        src = hf[split].cast_column("image", Image(decode=False))
        cdir = cache_dir / split
        for i in range(len(src)):
            mp = cdir / f"{i}.png"
            if not mp.exists():
                raise RuntimeError(
                    f"{split}: mask missing for index {i} at {mp} — rerun cache_masks "
                    "before build_dataset (silently dropping rows would confound the A/B)"
                )
            ex = src[i]
            yield {
                "image": ex["image"],   # {"bytes": ..., "path": ...} pass-through
                "mask": {"bytes": mp.read_bytes(), "path": None},
                "species": ex["species"],
                "split": split,
            }

    ds = {}
    for split in splits:
        ds[split] = Dataset.from_generator(gen, gen_kwargs={"split": split}, features=features)
        print(f"{split:>5}: {len(ds[split]):,} images w/ masks")
    return DatasetDict(ds)


def dataset_card(repo: str, source_repo: str, dsdict) -> str:
    counts = {k: len(dsdict[k]) for k in dsdict}
    return f"""---
license: cc-by-nc-4.0
task_categories:
- image-classification
tags:
- plants
- flowers
- soft-mask
- background-augmentation
---

# Botanical Vision — full-split student masks

Companion dataset for [{source_repo}](https://huggingface.co/datasets/{source_repo}).
Full training-split plant/background masks generated by the distilled student
segmenter (DeepLabV3 + MobileNetV3-Large, `checkpoints/deeplabv3_mnv3_seg_best.pt`,
trained in `notebooks/07_train_segmentation.ipynb`). Consumed by
`notebooks/04_train_improved.ipynb` for train-time background augmentation
(`bg_aug_p`), so blurred-background inputs become in-distribution for the
classifier.

Distinct from `botanical-vision-256-masks` (the teacher/FastSAM masks over a
stratified 20K subset used to distill the student). This dataset uses the
student and covers the full train split 1:1, so the classifier's bg-aug can be
run at full vocabulary.

## Splits

| split | images |
|-------|--------|
{chr(10).join(f"| {k:<5} | {counts.get(k, 0):,} |" for k in ("train", "val", "test") if k in counts)}

## Fields

- `image` — the source photo (JPEG)
- `mask` — soft plant probability, 8-bit L PNG (0 = background, 255 = plant)
- `species` — species name (from the source dataset)
- `split` — one of `train`/`val`/`test`

## Usage

```python
from datasets import load_dataset
ds = load_dataset("{repo}", split="train")
```

Training code: [`notebooks/04_train_improved.ipynb`](https://github.com/babnigg/botanical-vision/blob/main/notebooks/04_train_improved.ipynb) with `BG_AUG=True`.
"""


def main():
    ap = argparse.ArgumentParser(description="Publish full-split student masks as a paired HF dataset")
    ap.add_argument("--repo", required=True, help="target repo, e.g. username/botanical-vision-256-student-masks-full")
    ap.add_argument("--source-repo", default=DEFAULT_SOURCE_REPO)
    ap.add_argument("--segmenter", default="checkpoints/deeplabv3_mnv3_seg_best.pt")
    ap.add_argument("--splits", default="train", help="comma-separated (default: train)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--skip-upload", action="store_true", help="cache masks locally without publishing")
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading source dataset {args.source_repo} ...")
    from datasets import load_dataset
    hf = load_dataset(args.source_repo)

    print(f"Loading student segmenter from {args.segmenter} ...")
    model = load_student(args.segmenter, dev)

    for split in splits:
        cache_masks(hf, split, model, dev, args.cache_dir, batch=args.batch)

    if args.skip_upload:
        print(f"Skipping upload. Cached masks under {args.cache_dir}")
        return

    dsdict = build_dataset(hf, splits, args.cache_dir)

    print(f"\nPushing to {args.repo} ({'private' if args.private else 'public'})...")
    retry(lambda: dsdict.push_to_hub(args.repo, private=args.private), "push")

    from huggingface_hub import HfApi
    retry(lambda: HfApi().upload_file(
        path_or_fileobj=dataset_card(args.repo, args.source_repo, dsdict).encode(),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="dataset",
    ), "card")
    print(f"Done. https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
