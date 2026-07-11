"""
Publish the flowering-plant dataset to the Hugging Face Hub.

Reads the train/val/test assignment from data/splits.csv (written by
02_eda_images.ipynb), attaches taxonomy metadata from data/selected_species.csv,
and pushes an image-classification DatasetDict to the Hub. Teammates then load it
with `datasets.load_dataset("<repo>")` — no re-scraping.

Auth: run `huggingface-cli login` first (or set HF_TOKEN).

The push is resumable and retries through brief internet outages — already-uploaded
shards are skipped, so it can survive connections that drop for a minute at a time.
Safe to re-run at any point.

Usage:
    python scripts/upload_to_hf.py --repo <username>/botanical-vision
    python scripts/upload_to_hf.py --repo <username>/botanical-vision --private
"""

import argparse
import io
import time
from pathlib import Path

import pandas as pd
import requests
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, Value
from PIL import Image as PILImage

PROJECT_DIR = Path(__file__).resolve().parent.parent
SPLITS_CSV = PROJECT_DIR / "data" / "splits.csv"
SPECIES_CSV = PROJECT_DIR / "data" / "selected_species.csv"

GBIF_CITATION = (
    "GBIF.org (8 July 2026) GBIF Occurrence Download "
    "https://doi.org/10.15468/dl.3hragg"
)


def abspath(p: str) -> str:
    """Resolve a stored image path against the project root, independent of the
    current working directory and the OS that wrote the path (handles / and \\)."""
    parts = p.replace("\\", "/").split("/")
    if "data" in parts:
        return str(PROJECT_DIR.joinpath(*parts[parts.index("data"):]))
    return str((PROJECT_DIR / p).resolve())


def _resized_bytes(path: str, max_size: int) -> bytes:
    """Load an image, shrink so its long edge is <= max_size (aspect preserved), re-encode JPEG."""
    img = PILImage.open(path).convert("RGB")
    img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def build_dataset(splits_csv: Path = SPLITS_CSV, max_size: int | None = None) -> DatasetDict:
    splits = pd.read_csv(splits_csv)
    species = pd.read_csv(SPECIES_CSV)

    # attach taxonomy (family/genus/class) by species key.
    # via Int64 first so a stray NaN can't turn keys into "5384891.0" and break the join
    splits["species_key"] = splits["species_key"].astype("Int64").astype(str)
    meta = species[["speciesKey", "genus", "family", "order", "class"]].copy()
    meta["species_key"] = meta["speciesKey"].astype("Int64").astype(str)
    df = splits.merge(meta.drop(columns="speciesKey"), on="species_key", how="left")
    if df["family"].isna().any():
        raise SystemExit(f"taxonomy join failed for {int(df['family'].isna().sum())} rows - check species_key alignment")

    labels = sorted(df["species"].unique())
    features = Features({
        "image": Image(),
        "label": ClassLabel(names=labels),
        "species": Value("string"),
        "species_key": Value("string"),
        "genus": Value("string"),
        "family": Value("string"),
        "order": Value("string"),
        "class": Value("string"),
    })

    keep = ["path", "species", "species_key", "genus", "family", "order", "class"]
    ds = {}
    for split in ("train", "val", "test"):
        part = df[df["split"] == split].copy()
        if max_size:
            # resize each image and embed the shrunk JPEG bytes; from_generator streams
            # to disk so we never hold all images in memory
            rows = part[keep].to_dict("records")

            def gen(rows=rows):
                for r in rows:
                    r = dict(r)
                    r["image"] = {"bytes": _resized_bytes(abspath(r.pop("path")), max_size), "path": None}
                    r["label"] = r["species"]
                    yield r

            ds[split] = Dataset.from_generator(gen, features=features)
        else:
            part["image"] = part["path"].map(abspath)
            part["label"] = part["species"]
            ds[split] = Dataset.from_pandas(
                part[["image", "label", "species", "species_key", "genus", "family", "order", "class"]],
                features=features,
                preserve_index=False,
            )
        print(f"{split:>5}: {len(part):,} images")

    print(f"labels: {len(labels)} species" + (f" | resized to <= {max_size}px" if max_size else ""))
    return DatasetDict(ds)


def dataset_card(repo: str, dsdict: DatasetDict, max_size: int | None = None) -> str:
    n = sum(len(d) for d in dsdict.values())
    n_labels = dsdict["train"].features["label"].num_classes
    res_note = (f"\n\nImages are downscaled so the long edge is at most {max_size}px "
                f"(a smaller, Colab-friendly build of the full-resolution dataset)."
                if max_size else "")
    return f"""---
license: cc-by-nc-4.0
task_categories:
- image-classification
tags:
- plants
- flowers
- inaturalist
- fine-grained
---

# Botanical Vision

Fine-grained flowering-plant classification dataset: {n:,} research-grade
iNaturalist photos across {n_labels:,} species (all flowering plants with at least
2,000 observations). Built for Advanced Computer Vision (UChicago ADSP 32023).{res_note}

## Splits

| split | images |
|-------|--------|
| train | {len(dsdict['train']):,} |
| val   | {len(dsdict['val']):,} |
| test  | {len(dsdict['test']):,} |

Split is stratified within each species (70/15/15). Exact and cross-species
duplicate images were removed before splitting.

## Fields

- `image` — the photo
- `label` / `species` — species name (the classification target)
- `genus`, `family`, `order`, `class` — taxonomy
- `species_key` — GBIF species key

## Source

Species were selected from a GBIF occurrence download (iNaturalist Research-grade
Observations, flowering plants with still images); photos were pulled from the
iNaturalist API. Individual photos retain their own iNaturalist licenses.

> {GBIF_CITATION}

## Usage

```python
from datasets import load_dataset
ds = load_dataset("{repo}")
```
"""


def _is_transient(e) -> bool:
    """Only wait-and-retry on genuinely transient network failures. Auth (401/403),
    missing repo (404), too-large (413) etc. won't fix themselves — fail fast on those."""
    from huggingface_hub.errors import HfHubHTTPError
    if isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return True
    if isinstance(e, HfHubHTTPError):
        code = getattr(getattr(e, "response", None), "status_code", None)
        return code is None or code == 429 or (code is not None and code >= 500)
    return False


def retry(fn, what: str, attempts: int = 100, wait: int = 120):
    """Run fn(), retrying ONLY on transient network errors so a flaky connection can't
    kill a long upload. Non-transient errors (bad token, missing repo, ...) raise at once.
    Resumable uploads skip already-sent shards, so a retry continues where it left off.
    """
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            if i == attempts or not _is_transient(e):
                raise
            print(f"\n[{what}] attempt {i} interrupted ({type(e).__name__}); "
                  f"waiting {wait}s then resuming (uploaded shards are skipped)...")
            time.sleep(wait)


def main():
    parser = argparse.ArgumentParser(description="Push the dataset to the Hugging Face Hub")
    parser.add_argument("--repo", required=True, help="target repo, e.g. username/botanical-vision")
    parser.add_argument("--private", action="store_true", help="create a private dataset")
    parser.add_argument("--splits-csv", default=str(SPLITS_CSV),
                        help="override the splits file (e.g. a small test split)")
    parser.add_argument("--max-size", type=int, default=None,
                        help="resize images so the long edge is <= this many px (e.g. 256) "
                             "for a small, Colab-friendly dataset; omit to keep full resolution")
    args = parser.parse_args()

    splits_csv = Path(args.splits_csv)
    if not splits_csv.exists():
        raise SystemExit(f"Missing {splits_csv}. Run notebooks/02_eda_images.ipynb first.")

    dsdict = build_dataset(splits_csv, max_size=args.max_size)

    print(f"\nPushing to {args.repo} ({'private' if args.private else 'public'})...")
    retry(lambda: dsdict.push_to_hub(args.repo, private=args.private), "push")

    # dataset card
    from huggingface_hub import HfApi
    retry(lambda: HfApi().upload_file(
        path_or_fileobj=dataset_card(args.repo, dsdict, args.max_size).encode(),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="dataset",
    ), "card")
    print(f"Done. https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
