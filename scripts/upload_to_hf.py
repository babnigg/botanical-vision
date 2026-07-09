"""
Publish the flowering-plant dataset to the Hugging Face Hub.

Reads the train/val/test assignment from data/splits.csv (written by
02_eda_images.ipynb), attaches taxonomy metadata from data/selected_species.csv,
and pushes an image-classification DatasetDict to the Hub. Teammates then load it
with `datasets.load_dataset("<repo>")` — no re-scraping.

Auth: run `huggingface-cli login` first (or set HF_TOKEN).

Usage:
    python scripts/upload_to_hf.py --repo <username>/botanical-vision
    python scripts/upload_to_hf.py --repo <username>/botanical-vision --private
"""

import argparse
from pathlib import Path

import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, Value

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


def build_dataset(splits_csv: Path = SPLITS_CSV) -> DatasetDict:
    splits = pd.read_csv(splits_csv)
    species = pd.read_csv(SPECIES_CSV)

    # attach taxonomy (family/genus/class) by species key
    splits["species_key"] = splits["species_key"].astype(str)
    meta = species[["speciesKey", "genus", "family", "order", "class"]].copy()
    meta["species_key"] = meta["speciesKey"].astype(str)
    df = splits.merge(meta.drop(columns="speciesKey"), on="species_key", how="left")

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

    ds = {}
    for split in ("train", "val", "test"):
        part = df[df["split"] == split].copy()
        part["image"] = part["path"].map(abspath)
        part["label"] = part["species"]
        ds[split] = Dataset.from_pandas(
            part[["image", "label", "species", "species_key", "genus", "family", "order", "class"]],
            features=features,
            preserve_index=False,
        )
        print(f"{split:>5}: {len(part):,} images")

    print(f"labels: {len(labels)} species")
    return DatasetDict(ds)


def dataset_card(repo: str, dsdict: DatasetDict) -> str:
    n = sum(len(d) for d in dsdict.values())
    n_labels = dsdict["train"].features["label"].num_classes
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
2,000 observations). Built for Advanced Computer Vision (UChicago ADSP 32023).

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


def main():
    parser = argparse.ArgumentParser(description="Push the dataset to the Hugging Face Hub")
    parser.add_argument("--repo", required=True, help="target repo, e.g. username/botanical-vision")
    parser.add_argument("--private", action="store_true", help="create a private dataset")
    parser.add_argument("--splits-csv", default=str(SPLITS_CSV),
                        help="override the splits file (e.g. a small test split)")
    args = parser.parse_args()

    splits_csv = Path(args.splits_csv)
    if not splits_csv.exists():
        raise SystemExit(f"Missing {splits_csv}. Run notebooks/02_eda_images.ipynb first.")

    dsdict = build_dataset(splits_csv)

    print(f"\nPushing to {args.repo} ({'private' if args.private else 'public'})...")
    dsdict.push_to_hub(args.repo, private=args.private)

    # dataset card
    from huggingface_hub import HfApi
    HfApi().upload_file(
        path_or_fileobj=dataset_card(args.repo, dsdict).encode(),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="dataset",
    )
    print(f"Done. https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
