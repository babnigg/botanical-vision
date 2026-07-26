"""
Generate pseudo plant/background masks over a stratified subset of the source
dataset (dbabnigg/botanical-vision-256) and publish them as a companion HF
dataset (botanical-vision-256-masks) that notebook 06 trains from.

Pipeline:
    HF source split  ->  stratified subset by species
        ->  FastSAM (teacher, "everything" mode)
        ->  pick largest mask that contains the image center
        ->  cache 1-bit PNG per image on disk
        ->  build HF DatasetDict {image, mask, species, split}
        ->  push_to_hub

Extra dep (maintainer only):
    pip install ultralytics       # brings FastSAM weights + inference

Auth: `huggingface-cli login` first (or set HF_TOKEN). Publishing is resumable
(reuses upload_to_hf.retry) and mask caching is resumable (skips images whose
cache PNG already exists) — safe to Ctrl-C and rerun.

Usage:
    python scripts/generate_pseudo_masks.py \
        --repo <user>/botanical-vision-256-masks \
        --n-train 20000 --n-val 1000 --n-test 1000

YOLO prompt (once the coworker's detector is available): to swap the center-mask
heuristic for a bbox-prompted mask, edit ``segment_one`` — FastSAM accepts a
`bboxes=[[x1,y1,x2,y2]]` kwarg.
"""

from __future__ import annotations

import argparse
import io
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm

# Reuse the resumable HF push logic from the dataset uploader — same auth model,
# same transient-error backoff.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from upload_to_hf import retry  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_REPO = "dbabnigg/botanical-vision-256"
DEFAULT_CACHE_DIR = PROJECT_DIR / "data" / "pseudo_masks"


def stratified_indices(species_col: list[str], n: int, seed: int = 42) -> list[int]:
    """Sample `n` indices, sampling round-robin across species so rare species
    still contribute masks. Deterministic under `seed`."""
    rng = np.random.default_rng(seed)
    by_species: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(species_col):
        by_species[s].append(i)
    for s in by_species:
        rng.shuffle(by_species[s])
    species = list(by_species.keys())
    rng.shuffle(species)

    picked: list[int] = []
    cursors = {s: 0 for s in species}
    while len(picked) < n:
        made_progress = False
        for s in species:
            if len(picked) >= n:
                break
            idxs = by_species[s]
            c = cursors[s]
            if c < len(idxs):
                picked.append(idxs[c])
                cursors[s] = c + 1
                made_progress = True
        if not made_progress:  # exhausted every species
            break
    return picked[:n]


def _pick_center_mask(masks: np.ndarray, h: int, w: int) -> np.ndarray | None:
    """FastSAM in "everything" mode returns many candidate masks. Pick the
    largest one that includes the image center (plant is almost always framed
    at the center in iNat photos); fall back to the largest mask overall."""
    if masks.size == 0:
        return None
    cy, cx = h // 2, w // 2
    center_hits = [i for i in range(len(masks)) if masks[i, cy, cx] > 0]
    if center_hits:
        areas = [(masks[i] > 0).sum() for i in center_hits]
        return masks[center_hits[int(np.argmax(areas))]]
    areas = [(masks[i] > 0).sum() for i in range(len(masks))]
    return masks[int(np.argmax(areas))]


def segment_one(teacher, img: PILImage.Image) -> PILImage.Image:
    """Return a single-channel PIL mask (0/255) for the plant subject."""
    w, h = img.size
    res = teacher(img, imgsz=640, conf=0.4, iou=0.9, verbose=False, retina_masks=True)
    r0 = res[0] if res else None
    if r0 is None or r0.masks is None or r0.masks.data is None:
        return PILImage.new("L", (w, h), 255)   # nothing detected -> everything = plant
    masks = r0.masks.data.cpu().numpy()          # (N, H', W')
    picked = _pick_center_mask(masks, masks.shape[1], masks.shape[2])
    if picked is None:
        return PILImage.new("L", (w, h), 255)
    m = (picked > 0).astype(np.uint8) * 255
    return PILImage.fromarray(m, mode="L").resize((w, h), PILImage.NEAREST)


def load_teacher():
    """FastSAM-x is a bit heavier than -s but the masks are noticeably cleaner
    and download is a one-time ~140MB. Swap to 'FastSAM-s.pt' for the smaller
    one if bandwidth is tight."""
    try:
        from ultralytics import FastSAM
    except ImportError:
        raise SystemExit(
            "Missing ultralytics. Install: pip install ultralytics\n"
            "(kept out of the base requirements — this script is maintainer-only)"
        )
    return FastSAM("FastSAM-x.pt")


def cache_masks(hf, split: str, indices: list[int], teacher, cache_dir: Path):
    """Run the teacher over every sampled image and cache the mask PNG.
    Skips images whose cache file already exists — safe to Ctrl-C and rerun."""
    out = cache_dir / split
    out.mkdir(parents=True, exist_ok=True)
    for i in tqdm(indices, desc=f"segment[{split}]"):
        path = out / f"{i}.png"
        if path.exists():
            continue
        img = hf[split][int(i)]["image"].convert("RGB")
        mask = segment_one(teacher, img)
        mask.save(path, optimize=True)


def build_masks_dataset(hf, indices_by_split: dict[str, list[int]], cache_dir: Path):
    """Assemble the companion HF DatasetDict from source images + cached masks."""
    from datasets import Dataset, DatasetDict, Features, Image, Value

    features = Features({
        "image": Image(),
        "mask": Image(),
        "species": Value("string"),
        "split": Value("string"),
    })

    def gen(split: str, idxs: list[int]):
        for i in idxs:
            src = hf[split][int(i)]
            mask_path = cache_dir / split / f"{i}.png"
            if not mask_path.exists():
                continue
            img_buf = io.BytesIO()
            src["image"].convert("RGB").save(img_buf, format="JPEG", quality=90)
            yield {
                "image": {"bytes": img_buf.getvalue(), "path": None},
                "mask": {"bytes": mask_path.read_bytes(), "path": None},
                "species": src["species"],
                "split": split,
            }

    ds = {}
    for split, idxs in indices_by_split.items():
        ds[split] = Dataset.from_generator(gen, gen_kwargs={"split": split, "idxs": idxs},
                                            features=features)
        print(f"{split:>5}: {len(ds[split]):,} images w/ masks")
    return DatasetDict(ds)


def dataset_card(repo: str, source_repo: str, dsdict) -> str:
    counts = {k: len(dsdict[k]) for k in dsdict}
    return f"""---
license: cc-by-nc-4.0
task_categories:
- image-segmentation
tags:
- plants
- flowers
- pseudo-labels
- fastsam
---

# Botanical Vision — pseudo masks

Companion segmentation dataset for [{source_repo}](https://huggingface.co/datasets/{source_repo}).
Binary plant/background masks generated by FastSAM (largest center-containing mask
per image), used to distill a small student segmenter for the demo's preprocessing
stage.

## Splits

| split | images |
|-------|--------|
| train | {counts.get('train', 0):,} |
| val   | {counts.get('val', 0):,} |
| test  | {counts.get('test', 0):,} |

Stratified subsets of the source dataset; masks are 8-bit L PNGs (0 = background,
255 = plant).

## Fields

- `image` — the source photo (JPEG)
- `mask`  — the pseudo mask (1-channel PNG, thresholded)
- `species` — species name (kept for stratification / joint stats)
- `split` — one of `train`/`val`/`test`

## Usage

```python
from datasets import load_dataset
ds = load_dataset("{repo}")
```

Training code: [`notebooks/06_train_segmentation.ipynb`](https://github.com/babnigg/botanical-vision/blob/main/notebooks/06_train_segmentation.ipynb).
"""


def main():
    parser = argparse.ArgumentParser(description="Generate pseudo-masks and publish a companion HF dataset")
    parser.add_argument("--repo", required=True, help="target repo, e.g. username/botanical-vision-256-masks")
    parser.add_argument("--source-repo", default=DEFAULT_SOURCE_REPO,
                        help="source image dataset to draw from")
    parser.add_argument("--n-train", type=int, default=20000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                        help="local dir for cached mask PNGs (resumable)")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-upload", action="store_true",
                        help="only cache masks locally; don't publish (useful for smoke tests)")
    args = parser.parse_args()

    print(f"Loading source dataset {args.source_repo} ...")
    from datasets import load_dataset
    hf = load_dataset(args.source_repo)

    wanted = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    indices_by_split: dict[str, list[int]] = {}
    for split, n in wanted.items():
        n = min(n, len(hf[split]))
        indices_by_split[split] = stratified_indices(hf[split]["species"], n, seed=args.seed)
        print(f"{split}: sampled {len(indices_by_split[split]):,} of {len(hf[split]):,}")

    teacher = load_teacher()
    for split, idxs in indices_by_split.items():
        cache_masks(hf, split, idxs, teacher, args.cache_dir)

    if args.skip_upload:
        print(f"Skipping upload. Cached masks under {args.cache_dir}")
        return

    dsdict = build_masks_dataset(hf, indices_by_split, args.cache_dir)

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
